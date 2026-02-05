#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import requests

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


# ----------------------- AIRTABLE HELPERS -----------------------

def _assert_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def _combine_field_across_rows(records, field_name: str) -> str:
    parts = []
    for r in records:
        fields = r.get("fields") or {}
        v = fields.get(field_name)
        if v is None:
            continue
        t = v.strip() if isinstance(v, str) else str(v).strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)

def _build_system_text_from_case(records) -> str:
    opening = _combine_field_across_rows(records, "Opening Sentence")
    divulge_freely = _combine_field_across_rows(records, "Divulge freely")
    divulge_asked = _combine_field_across_rows(records, "Divulge Asked")
    pmhx = _combine_field_across_rows(records, "PMHx RP")
    social = _combine_field_across_rows(records, "Social History")

    family = (
        _combine_field_across_rows(records, "Family Hiostory")
        or _combine_field_across_rows(records, "Family History")
    )

    ice = _combine_field_across_rows(records, "ICE")
    reaction = _combine_field_across_rows(records, "Reaction")

    rules = """
CRITICAL:
- You MUST NOT invent details.
- Only use information explicitly present in the CASE DETAILS below.
- If something is not stated, say: "I'm not sure" / "I don't remember" / "I haven't been told".
- NEVER substitute another symptom.
- NEVER create symptoms.
- Do Not Hallucinate.
- NEVER swap relatives. If relationship is not explicit, say you're not sure.
- Answer only what the clinician asks.
""".strip()

    case = f"""
CASE DETAILS (THIS IS YOUR ENTIRE MEMORY):

OPENING SENTENCE:
{opening or "[Not provided]"}

DIVULGE FREELY:
{divulge_freely or "[Not provided]"}

DIVULGE ONLY IF ASKED:
{divulge_asked or "[Not provided]"}

PAST MEDICAL HISTORY:
{pmhx or "[Not provided]"}

SOCIAL HISTORY:
{social or "[Not provided]"}

FAMILY HISTORY:
{family or "[Not provided]"}

ICE (Ideas / Concerns / Expectations):
{ice or "[Not provided]"}

REACTION / AFFECT:
{reaction or "[Not provided]"}
""".strip()

    return f"{case}\n\n{rules}"

def _airtable_get_json(url: str, api_key: str) -> dict:
    r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Airtable error {r.status_code}: {r.text[:400]}")
    return r.json()

def fetch_case_system_text(case_id: int) -> str:
    api_key = _assert_env("AIRTABLE_API_KEY")
    base_id = _assert_env("AIRTABLE_BASE_ID")

    table_name = f"Case {case_id}"
    offset = None
    records = []

    while True:
        params = {"pageSize": "100"}
        if offset:
            params["offset"] = offset
        # Airtable REST: /v0/{baseId}/{tableName}
        url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_name)}"
        # requests will encode params separately
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Airtable error {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break

    if not records:
        raise RuntimeError(f"No records found in Airtable table '{table_name}'")

    return _build_system_text_from_case(records)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # --- Services ---
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Case selection (simple for now) ---
    case_id = int(os.getenv("CASE_ID", "1"))

    # Build system prompt from Airtable at startup (or you can move this into on_client_connected)
    try:
        system_text = fetch_case_system_text(case_id)
        logger.info(f"✅ Loaded Airtable system prompt for Case {case_id}")
    except Exception as e:
        # Fail-safe: don't start with a broken / empty system prompt
        logger.error(f"❌ Failed to load Airtable case {case_id}: {e}")
        system_text = (
            "CRITICAL: Airtable case failed to load. "
            "Tell the clinician you haven't been given the case details."
        )

    messages = [
    {
        "role": "system",
        "content": """
You are simulating a real patient in a clinical consultation.

Behaviour rules:
- Respond naturally, conversationally, and realistically.
- Do NOT lecture or explain unless explicitly asked.
- Do NOT give medical advice unless the clinician asks for your understanding.
- Answer briefly by default; expand only if prompted.
- Avoid long monologues
- Show mild anxiety when discussing serious symptoms
- Express guilt or worry only when relevant to the case
- If unsure, say so plainly (e.g. "I'm not sure", "I don't remember").
- Stay emotionally consistent with the case.
- Never mention you are an AI, model, or simulation.
"""
    },
    {
        "role": "system",
        "content": system_text  # ← Airtable case details + hard rules
    }
]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Small opening, but keep the case rules as the main system message above.
        messages.append(
            {"role": "system", "content": "Say hello and stay in character as the patient."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
