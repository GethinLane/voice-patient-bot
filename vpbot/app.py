# vpbot/app.py

import os
import re
import json
import time  # used to measure session duration
import threading
import aiohttp
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("✅ Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
logger.info("✅ Silero VAD model loaded")

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies, ExternalUserTurnStrategies
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

from vpbot.config import BOT_VERSION, GRADING_SUBMIT_URL
from vpbot.google_adc import ensure_google_adc
from vpbot.airtable_case import fetch_case_system_text, extract_opening_sentence
from vpbot.transcript import build_transcript_from_context, submit_grading
from vpbot.tts import build_tts_from_body
from vpbot.stt import (
    choose_stt_primary_first,
    should_failover,
    set_cooldown,
    build_stt_service,
)
from vpbot.policy import build_disclosure_policy


logger.info("✅ All components loaded successfully!")


# ----------------------- MAIN BOT -----------------------

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # Session body from Vercel (fast, no network)
    body = getattr(runner_args, "body", None) or {}
    logger.info(f"📥 runner_args.body={body}")

    def _normalize_mode(value):
        v = str(value or "").strip().lower()
        return "premium" if v == "premium" else "standard"

    mode = _normalize_mode(
        body.get("mode")
        or body.get("botMode")
        or (body.get("metadata") or {}).get("mode")
        or (body.get("meta") or {}).get("mode")
        or (body.get("context") or {}).get("mode")
    )

    logger.info(f"🧩 Session mode resolved: {mode}")

    # STT / LLM / TTS
    logger.info(
        "🔎 STT env check: "
        f"STT_FORCE_PROVIDER={os.getenv('STT_FORCE_PROVIDER')!r} "
        f"STT_PRIMARY={os.getenv('STT_PRIMARY')!r} "
        f"STT_SECONDARY={os.getenv('STT_SECONDARY')!r} "
        f"ASSEMBLYAI_API_KEY_set={bool((os.getenv('ASSEMBLYAI_API_KEY') or '').strip())} "
        f"DEEPGRAM_API_KEY_set={bool((os.getenv('DEEPGRAM_API_KEY') or '').strip())}"
    )

    stt, stt_provider_in_use, stt_other = choose_stt_primary_first()
    use_flux_turns = stt_provider_in_use in ("deepgram", "dg")
    logger.info(f"🎙️ STT selected: {stt_provider_in_use} (secondary={stt_other or 'none'})")

    # Ensure Google credentials exist before any Google client init
    ensure_google_adc()

    aiohttp_session = None

    async def _close_aiohttp_session():
        nonlocal aiohttp_session
        try:
            if aiohttp_session is not None and not aiohttp_session.closed:
                await aiohttp_session.close()
        except Exception as e:
            logger.warning(f"Failed to close aiohttp session: {e}")
        aiohttp_session = None

    # ✅ Make TTS selection loud + safe
    try:
        logger.info(f"🔊 Requested TTS config: {json.dumps(body.get('tts'), ensure_ascii=False)}")

        tts_provider = ""
        if isinstance(body.get("tts"), dict):
            tts_provider = str(body["tts"].get("provider", "")).strip().lower()

        # Only create aiohttp session if we actually need it
        if tts_provider == "inworld":
            aiohttp_session = aiohttp.ClientSession()

        if tts_provider == "elevenlabs":
            logger.info(f"🔑 ELEVENLABS_API_KEY present? {bool(os.getenv('ELEVENLABS_API_KEY'))}")

        tts = build_tts_from_body(body, aiohttp_session=aiohttp_session)

        # 🔍 DIAGNOSTIC: prove which code + pipecat/inworld version this agent is actually running
        import inspect
        import pipecat
        from pipecat.services.inworld.tts import InworldHttpTTSService

        logger.info(f"🔍 BOT_VERSION={BOT_VERSION}")
        logger.info(f"🔍 bot file={__file__}")
        logger.info(f"🔍 pipecat version={getattr(pipecat, '__version__', 'unknown')}")
        logger.info(f"🔍 InworldHttpTTSService file={inspect.getfile(InworldHttpTTSService)}")
        logger.info(f"🔍 TTS impl={tts.__class__.__module__}.{tts.__class__.__name__}")
        logger.info(f"🔍 runner_args.body.tts={body.get('tts')}")

        logger.info(f"TTS class = {tts.__class__.__module__}.{tts.__class__.__name__}")

    except Exception as e:
        logger.error(f"❌ TTS init failed ({body.get('tts')}): {e}")
        logger.error("↩️ Falling back to Cartesia so session can continue")

        # If we created an aiohttp session, close it on failure
        try:
            if aiohttp_session is not None and not aiohttp_session.closed:
                await aiohttp_session.close()
        except Exception as close_err:
            logger.warning(f"Failed to close aiohttp session after TTS init failure: {close_err}")
        aiohttp_session = None

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=os.getenv("CARTESIA_VOICE_ID") or "71a7ad14-091c-4e8e-a314-022ece01c121",
        )

    # --- LLM MODEL SELECTION (strict, conversation-specific; no fallback) ---
    ENV_STD = "OPENAI_CONVERSATION_MODEL_STANDARD"
    ENV_PREM = "OPENAI_CONVERSATION_MODEL_PREMIUM"

    standard_model = (os.getenv(ENV_STD) or "").strip()
    premium_model = (os.getenv(ENV_PREM) or "").strip()

    if mode == "premium":
        if not premium_model:
            raise RuntimeError(
                f"Missing required env var: {ENV_PREM} "
                "(e.g. set it to 'gpt-5.1')."
            )
        selected_model = premium_model
        selected_env = ENV_PREM
    else:
        if not standard_model:
            raise RuntimeError(
                f"Missing required env var: {ENV_STD} "
                "(e.g. set it to 'gpt-4.1-mini')."
            )
        selected_model = standard_model
        selected_env = ENV_STD

    logger.info(
        f"🧠 OpenAI conversation model selected: {selected_model} "
        f"(mode={mode}, env={selected_env})"
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=selected_model,
    )

    # Case selection from session body
    case_id = int(body.get("caseId") or os.getenv("CASE_ID", "1"))

    # Identity passthrough
    user_id = (body.get("userId") or "").strip() or None
    email = (body.get("email") or "").strip().lower() or None

    # Tone passthrough (optional)
    start_tone = (body.get("startTone") or "neutral").strip().lower()
    tone_intensity = (body.get("toneIntensity") or "").strip().lower()

    logger.info(f"📘 Using case_id={case_id} (userId={user_id}, email={email}, startTone={start_tone})")

    # Session timing
    connected_at = None

    # Fetch case prompt from Airtable once at startup
    try:
        system_text = fetch_case_system_text(case_id)
        logger.info(f"✅ Loaded Airtable system prompt for Case {case_id}")
    except Exception as e:
        logger.error(f"❌ Failed to load Airtable case {case_id}: {e}")
        system_text = (
            "CRITICAL: Airtable case failed to load. "
            "Tell the clinician you haven't been given the case details."
        )

    opening_sentence = extract_opening_sentence(system_text)

    disclosure_policy = build_disclosure_policy(start_tone, tone_intensity)

    messages = [
        {
            "role": "system",
            "content": f"""
You are simulating a real patient in a clinical consultation.

{disclosure_policy}
""".strip(),
        },
        {"role": "system", "content": system_text},
    ]

    context = LLMContext(messages)

    if use_flux_turns:
        user_turn_strategies = ExternalUserTurnStrategies()
    else:
        user_turn_strategies = UserTurnStrategies(
            start=[
                MinWordsUserTurnStartStrategy(min_words=2),
            ],
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=1.0)
                    )
                )
            ],
        )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=user_turn_strategies,
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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal connected_at
        connected_at = time.time()
        logger.info(f"Client connected (connected_at={connected_at})")

        if opening_sentence:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Start the consultation now by saying ONLY the OPENING SENTENCE exactly as written, "
                        "as ONE short line. Do not add anything else."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Start the consultation now with a brief greeting as the patient in ONE short line, "
                        "then stop and wait."
                    ),
                }
            )

        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

        # duration seconds
        duration_seconds = None
        try:
            if connected_at is not None:
                duration_seconds = int(max(0, time.time() - connected_at))
        except Exception:
            duration_seconds = None

        transcript = build_transcript_from_context(context)

        session_id = getattr(runner_args, "session_id", None)
        logger.info(
            f"🧾 Transcript built: session_id={session_id} case_id={case_id} turns={len(transcript)} "
            f"duration_seconds={duration_seconds}"
        )

        if not transcript:
            logger.warning("⚠️ Transcript is empty; skipping grading submit.")
        else:
            payload = {
                "sessionId": session_id,
                "caseId": case_id,
                "userId": user_id,
                "email": email,
                "mode": mode,
                "durationSeconds": duration_seconds,
                "transcript": transcript,
            }

            # Fire-and-forget background submit
            try:
                logger.info(f"📤 Queueing transcript submit to {GRADING_SUBMIT_URL}")
                th = threading.Thread(
                    target=submit_grading,
                    args=(GRADING_SUBMIT_URL, payload),
                    daemon=True,
                )
                th.start()
            except Exception as e:
                logger.error(f"❌ Failed to start background submit thread: {e}")

        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    try:
        await runner.run(task)

    except Exception as e:
        logger.error(f"❌ Pipeline error (stt={stt_provider_in_use}): {e}")

        if stt_other and should_failover(stt_provider_in_use, e):
            # Put the failing provider in cooldown so new sessions skip it briefly
            set_cooldown(stt_provider_in_use)

            logger.warning(
                f"🔁 STT failover: {stt_provider_in_use} -> {stt_other} (reason={e})"
            )

            # Rebuild STT + pipeline/task with the other provider
            stt_provider_in_use, stt_other = stt_other, stt_provider_in_use
            stt = build_stt_service(stt_provider_in_use)

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
                params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            )

            runner2 = PipelineRunner(handle_sigint=runner_args.handle_sigint)
            await runner2.run(task)

        else:
            raise

    finally:
        await _close_aiohttp_session()


async def bot(runner_args: RunnerArguments):
    def _peek_provider_from_env() -> str:
        forced = (os.getenv("STT_FORCE_PROVIDER") or "").strip().lower()
        if forced:
            return forced
        return (os.getenv("STT_PRIMARY") or "deepgram").strip().lower()

    primary_provider = _peek_provider_from_env()
    use_flux = primary_provider in ("deepgram", "dg")

    def _silero_vad():
        return SileroVADAnalyzer(
            params=VADParams(
                start_secs=0.35,
                stop_secs=0.3,
                confidence=0.8,
                min_volume=0.65,
                vad_audio_passthrough=True,
            )
        )

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_in_filter=KrispVivaFilter(),  # ✅ Krisp VIVA mic filter
            audio_out_enabled=True,
            # ✅ Keep VAD for AssemblyAI, disable for Flux
            vad_analyzer=None if use_flux else _silero_vad(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # ✅ Keep VAD for AssemblyAI, disable for Flux
            vad_analyzer=None if use_flux else _silero_vad(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)
