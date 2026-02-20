# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import re
import json
import time  # used to measure session duration
import threading
import requests
import aiohttp
import base64
import codecs


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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsHttpTTSService
from pipecat.services.google.tts import GoogleTTSService, Language
from pipecat.services.inworld.tts import InworldHttpTTSService


from pipecat.transcriptions.language import Language as TranscriptLanguage

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.assemblyai import AssemblyAISTTService
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

class SafeInworldHttpTTSService(InworldHttpTTSService):
    async def _process_streaming_response(self, response: aiohttp.ClientResponse, context_id: str):
        buffer = b""
        utterance_duration = 0.0

        async for chunk in response.content.iter_chunked(4096):
            if not chunk:
                continue

            buffer += chunk

            # Process complete newline-delimited JSON records
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                # Decode ONLY a full line (prevents split-multibyte UTF-8 crashes)
                line_str = line.decode("utf-8", errors="strict")

                try:
                    chunk_data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                result = chunk_data.get("result") or {}

                audio_b64 = result.get("audioContent")
                if audio_b64:
                    await self.stop_ttfb_metrics()
                    async for frame in self._process_audio_chunk(
                        base64.b64decode(audio_b64),
                        context_id,
                    ):
                        yield frame

                timestamp_info = result.get("timestampInfo")
                if timestamp_info:
                    word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
                    if word_times:
                        await self.add_word_timestamps(word_times, context_id)
                    utterance_duration = max(utterance_duration, chunk_end_time)

        if utterance_duration > 0:
            self._cumulative_time += utterance_duration
            
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

BOT_VERSION = "2026-02-19-speakingrate-default-1.0"
logger.info(f"✅ BOT_VERSION={BOT_VERSION}")

# Where to submit transcript for grading (ONLY on disconnect)
GRADING_SUBMIT_URL = (
    os.getenv("GRADING_SUBMIT_URL", "").strip()
    or "https://voice-patient-web.vercel.app/api/submit-transcript"
)
logger.info(f"✅ GRADING_SUBMIT_URL={GRADING_SUBMIT_URL}")

# ----------------------- GOOGLE ADC (SERVICE ACCOUNT JSON) -----------------------

def _ensure_google_adc():
    """
    If GOOGLE_APPLICATION_CREDENTIALS isn't set but GOOGLE_SA_JSON is,
    write the service account JSON to /tmp and set GOOGLE_APPLICATION_CREDENTIALS.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    sa_json = os.getenv("GOOGLE_SA_JSON")
    if not sa_json:
        return

    path = "/tmp/google-sa.json"
    try:
        data = json.loads(sa_json)  # validate JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        logger.info(f"✅ Google ADC configured: GOOGLE_APPLICATION_CREDENTIALS={path}")
    except Exception as e:
        logger.error(f"❌ Failed to configure Google ADC from GOOGLE_SA_JSON: {e}")


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
    divulge_freely = _combine_field_across_rows(records, "Divulge Freely")
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
- If something is not stated:
  - If it seems unrelated to why I'm here today, say: "I'm not sure that's relevant to this case."
  - If it seems clinically relevant but isn't stated, say: "I'm not sure" / "I don't know, I'm afraid".
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

        url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_name)}"
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


def extract_opening_sentence(system_text: str) -> str:
    m = re.search(
        r"OPENING SENTENCE:\s*(.*?)(?:\n\s*\n|DIVULGE FREELY:)",
        system_text,
        flags=re.S | re.I,
    )
    if not m:
        return ""
    opening = m.group(1).strip()
    opening = re.sub(r"\s+\n\s+", " ", opening).strip()
    return opening


# ----------------------- TRANSCRIPT HELPERS -----------------------

def build_transcript_from_context(context: LLMContext):
    """
    Build transcript (user+assistant only) from the LLM context.
    Called ONLY on disconnect to avoid any runtime overhead.
    """
    out = []
    for m in context.messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        out.append({"role": role, "text": text})
    return out


def _submit_grading_in_background(url: str, payload: dict):
    """
    Fire-and-forget transcript submit so we do NOT block Pipecat shutdown.
    Uses requests in a background thread.
    """
    try:
        logger.info(f"📤 [BG] POST {url}")
        logger.info(
            f"📤 [BG] payload preview: "
            f"{json.dumps({k: payload[k] for k in payload if k != 'transcript'}, ensure_ascii=False)[:400]}"
        )
        r = requests.post(url, json=payload, timeout=60)
        logger.info(f"📤 [BG] response: {r.status_code} {r.text[:400]}")
    except Exception as e:
        logger.error(f"❌ [BG] submit failed: {e}")


# ----------------------- TTS SELECTION -----------------------

def _safe_lower(x):
    return str(x).strip().lower() if x is not None else ""


def _build_tts_from_body(body: dict, aiohttp_session=None):
    """
    Create the TTS service based on runner_args.body.tts

    Expected shape:
      body.tts = {
        "provider": "cartesia" | "elevenlabs" | "google" | "inworld",
        "voice": "<voice_id_or_voice_name>",
        "model": "<optional>",
        "config": { ...optional... }   # NOTE: not applied yet (avoid runtime crashes)
      }

    Defaults to Cartesia if anything is missing.
    """
    tts_cfg = body.get("tts") if isinstance(body, dict) else None
    tts_cfg = tts_cfg if isinstance(tts_cfg, dict) else {}

    provider = _safe_lower(tts_cfg.get("provider") or "cartesia")
    voice = (tts_cfg.get("voice") or "").strip() or None
    model = (tts_cfg.get("model") or "").strip() or None

    # CARTESIA (default)
    if provider in ("cartesia", ""):
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=voice or os.getenv("CARTESIA_VOICE_ID") or "71a7ad14-091c-4e8e-a314-022ece01c121",
        )

    # ELEVENLABS
    if provider == "elevenlabs":
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY missing (provider=elevenlabs)")
    
        voice_id = voice or os.getenv("ELEVENLABS_VOICE_ID")
        if not voice_id:
            raise RuntimeError(
                "ElevenLabs selected but no voice_id provided. "
                "Set it in Airtable (tts.voice) or ELEVENLABS_VOICE_ID env var."
            )

        model_id = model or os.getenv("ELEVENLABS_MODEL") or "eleven_turbo_v2_5"

        # IMPORTANT: force a playable format for Daily/WebRTC
        # 48000 is the safest default for WebRTC pipelines.
        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=voice_id,
            model=model_id,
            sample_rate=24000,
            # Turn on internal service logging (supported by Pipecat)
            params=ElevenLabsTTSService.InputParams(enable_logging=True),
        )
        
    # INWORLD (HTTP streaming)
    if provider == "inworld":
        api_key = (os.getenv("INWORLD_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "INWORLD_API_KEY missing. Set it to the Inworld Base64 runtime credential "
                "(do NOT include 'Basic ')."
            )

        if aiohttp_session is None:
            raise RuntimeError("Inworld selected but aiohttp_session was not provided.")

        voice_id = (voice or "").strip() or os.getenv("INWORLD_VOICE_ID") or "Ashley"
        model_id = (model or "").strip() or os.getenv("INWORLD_MODEL_ID") or "inworld-tts-1.5-max"

        cfg = tts_cfg.get("config") if isinstance(tts_cfg.get("config"), dict) else {}
        sr_raw = cfg.get("speakingRate", None)

        # Default to 1.0 if missing / null / blank / invalid
        sr = 1.0
        if sr_raw is not None:
            if isinstance(sr_raw, str) and not sr_raw.strip():
                sr = 1.0
            else:
                try:
                    sr_candidate = float(sr_raw)
                    # Reject non-positive values (0 or negative)
                    sr = sr_candidate if sr_candidate > 0 else 1.0
                except Exception:
                    sr = 1.0

        # Clamp only after resolving a valid value
        sr = max(0.5, min(2.0, sr))

        logger.info(
            f"🔊 Inworld TTS voice_id={voice_id!r}, model_id={model_id!r}, "
            f"speakingRate(raw)={sr_raw!r} -> {sr}"
        )

        params = InworldHttpTTSService.InputParams(speaking_rate=sr)


        return SafeInworldHttpTTSService(
            api_key=api_key,                 # Pipecat sends: Authorization: Basic <api_key>
            aiohttp_session=aiohttp_session, # required
            voice_id=voice_id,
            model=model_id,
            streaming=True,                  # uses /tts/v1/voice:stream
            encoding="LINEAR16",
            sample_rate=None,
            params=params, 
        )


    # GOOGLE
    if provider == "google":
        voice_id = (voice or "").strip()
        if not voice_id:
            raise RuntimeError("Google TTS selected but no voice_id provided in Airtable (tts.voice).")

        # Determine language from the voice_id prefix (e.g. "en-GB-...", "en-IN-...")
        lang_code = "en-GB"  # default to GB
        m = re.match(r"^([a-z]{2}-[A-Z]{2})-", voice_id)
        if m:
            lang_code = m.group(1)

        lang_map = {
            "en-GB": Language.EN_GB,
            "en-IN": Language.EN_IN,
            "en-US": Language.EN_US,
            "en-AU": Language.EN_AU,
        }
        lang_enum = lang_map.get(lang_code, Language.EN_GB)

        logger.info(f"🔊 Google TTS voice_id={voice_id!r}, language={lang_enum}")

        return GoogleTTSService(
            voice_id=voice_id,
            params=GoogleTTSService.InputParams(language=lang_enum),
        )
        
    raise RuntimeError(f"Unknown TTS provider: {provider}")

# ----------------------- STT PROVIDER SELECTION + FAILOVER (PRIMARY/SECONDARY) -----------------------

_STT_COOLDOWN_UNTIL = {}  # provider -> unix time until which we should avoid it


def _now() -> float:
    return time.time()


def _cooldown_secs() -> int:
    try:
        return int(os.getenv("STT_FAILOVER_COOLDOWN_SECS") or "60")
    except Exception:
        return 60


def _set_cooldown(provider: str):
    if provider:
        _STT_COOLDOWN_UNTIL[provider] = _now() + _cooldown_secs()


def _in_cooldown(provider: str) -> bool:
    if not provider:
        return False
    until = _STT_COOLDOWN_UNTIL.get(provider)
    return bool(until and until > _now())


def _get_primary_secondary() -> tuple[str, str]:
    """
    Env-driven selection.
      STT_FORCE_PROVIDER (optional) overrides everything for testing.
      STT_PRIMARY / STT_SECONDARY are the normal controls.
    """
    forced = (os.getenv("STT_FORCE_PROVIDER") or "").strip().lower()
    if forced:
        return forced, ""

    primary = (os.getenv("STT_PRIMARY") or "deepgram").strip().lower()
    secondary = (os.getenv("STT_SECONDARY") or "assemblyai").strip().lower()

    if secondary == primary:
        secondary = ""

    return primary, secondary


def _is_transient_network_error(exc: Exception) -> bool:
    s = (str(exc) or "").lower()
    return any(tok in s for tok in [
        "timeout", "timed out",
        "service unavailable", "temporarily unavailable", "503",
        "connection reset", "connection refused",
        "disconnected",
        "cannot connect",
        "websocket",
        "network is unreachable",
    ])


def _is_capacity_error(provider: str, exc: Exception) -> bool:
    """
    Match common capacity/rate-limit signals.
    We match on text because Pipecat may wrap underlying exceptions.
    """
    p = (provider or "").lower()
    s = (str(exc) or "").lower()

    if p in ("deepgram", "dg"):
        # Rate limiting/capacity typically: 429 / TOO_MANY_REQUESTS
        return (
            "429" in s
            or "too many requests" in s
            or "too_many_requests" in s
            or "rate limit" in s
        )

    if p in ("assemblyai", "aai"):
        # Concurrency limit often: WS close code 1008 + "Too many concurrent sessions"
        return (
            ("1008" in s and "too many concurrent sessions" in s)
            or ("too many concurrent sessions" in s)
        )

    return False


def _should_failover(provider: str, exc: Exception) -> bool:
    return _is_capacity_error(provider, exc) or _is_transient_network_error(exc)


def _build_stt_service(provider: str):
    provider = (provider or "").strip().lower()

    if provider in ("deepgram", "dg"):
        api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Missing DEEPGRAM_API_KEY (STT provider=deepgram)")
        return DeepgramSTTService(api_key=api_key)

    if provider in ("assemblyai", "aai"):
        api_key = (os.getenv("ASSEMBLYAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Missing ASSEMBLYAI_API_KEY (STT provider=assemblyai)")

        return AssemblyAISTTService(
            api_key=api_key,
            connection_params=AssemblyAIConnectionParams(
                sample_rate=16000,
                formatted_finals=True,
            ),
            # Keep your existing SmartTurn + Silero VAD as the turn controller:
            vad_force_turn_endpoint=True,
        )

    raise RuntimeError(f"Unknown STT provider: {provider!r}")


def choose_stt_primary_first() -> tuple[object, str, str]:
    """
    Returns: (stt_service, provider_in_use, other_provider)
    Uses cooldown to skip a provider that just rate-limited / errored.
    """
    primary, secondary = _get_primary_secondary()

    if primary and not _in_cooldown(primary):
        return _build_stt_service(primary), primary, secondary

    if secondary and not _in_cooldown(secondary):
        logger.warning(f"⏭️ Primary STT in cooldown; using secondary={secondary}")
        return _build_stt_service(secondary), secondary, primary

    # If both are in cooldown, just retry primary
    logger.warning("⚠️ Both STT providers appear in cooldown; retrying primary anyway.")
    return _build_stt_service(primary), primary, secondary

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
    stt, stt_provider_in_use, stt_other = choose_stt_primary_first()
    logger.info(f"🎙️ STT selected: {stt_provider_in_use} (secondary={stt_other or 'none'})")

    # Ensure Google credentials exist before any Google client init
    _ensure_google_adc()

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

        tts = _build_tts_from_body(body, aiohttp_session=aiohttp_session)
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
    premium_model  = (os.getenv(ENV_PREM) or "").strip()

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

    disclosure_policy = """
DISCLOSURE POLICY (follow exactly):

Definitions:
- "Direct question" = clinician asks specifically about a topic (e.g. chest pain, smoking, meds, allergies, family history, mood, ICE, etc.)
- "Vague/open question" = clinician asks broad prompts (e.g. "general health?", "anything else?", "how have you been?", "tell me more")

Hard rules:
A) NEVER ask the clinician questions like:
   - "Is there anything else you want to know?"
   - "Is there anything specific you wanted to know?"
   - "Anything else?"
   Do not ask for direction. Do not hand the conversation back with a question.

B) For ANY vague/open question (including "anything else?"):
   - Respond with ONLY what is in DIVULGE FREELY.
   - Keep it to 1–2 short, natural sentences max.
   - If the answer is already fully covered by what you've said, reply with a closing-style line like:
     "No, that's everything really." / "Not that I can think of." / "I think that's about it."
   - Then STOP. Do not add extra details.

C) Only reveal information from "DIVULGE ONLY IF ASKED" when a direct question matches it.
   If not directly asked:
   - If it seems unrelated to why I'm here today: "I'm sorry, I don't know; however, I'm sure that's not relevant to what's going on."
   - If it seems relevant but isn't stated: "I don't know, I'm afraid."

D) Never expand into PMHx / social / family / ICE unless directly asked about those topics.

E) Default reply length is short. No lists. No multi-part info dumps.
""".strip()


    messages = [
        {
            "role": "system",
            "content": f"""
You are simulating a real patient in a clinical consultation.

Behaviour rules:
Hard style rules (must follow):
- Never describe the patient's experiences using "you/your" (second person). Use "I/my" for the patient.
  BAD: "You get chest pain when you walk."
  GOOD: "I get chest pain when I walk."
- Do not give the clinician instructions or advice ("you should...", "you need to..."). If asked, give only your understanding as the patient.
- If you catch yourself starting a sentence with "You..." rewrite it before responding.
- Respond naturally, conversationally, and realistically.
- Never ask if there is something else they would like to know.
- Do NOT lecture or explain unless explicitly asked.
- Do NOT give medical advice unless the clinician asks for your understanding.
- Answer briefly by default; expand only if prompted.
- Avoid long monologues.
- Show mild anxiety when discussing serious symptoms.
- Express guilt or worry only when relevant to the case.
- If unsure, say so plainly (e.g. "I'm not sure", "I don't remember").
- Stay emotionally consistent with the case.
- Never mention you are an AI, model, or simulation.
- Start the consultation with this emotional tone: {start_tone}{(" (" + tone_intensity + ")") if tone_intensity else ""}.
- Keep answers consistent with this tone at the beginning, and adjust naturally if the clinician is empathic/reassuring.

{disclosure_policy}
""".strip(),
        },
        {"role": "system", "content": system_text},
    ]

    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=1.0)
                        )
                    )
                ]
            ),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.2,
                    vad_audio_passthrough=True,
                )
            ),
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
                    target=_submit_grading_in_background,
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

        if stt_other and _should_failover(stt_provider_in_use, e):
            # Put the failing provider in cooldown so new sessions skip it briefly
            _set_cooldown(stt_provider_in_use)

            logger.warning(
                f"🔁 STT failover: {stt_provider_in_use} -> {stt_other} (reason={e})"
            )

            # Rebuild STT + pipeline/task with the other provider
            stt_provider_in_use, stt_other = stt_other, stt_provider_in_use
            stt = _build_stt_service(stt_provider_in_use)

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
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_in_filter=KrispVivaFilter(),  # ✅ Krisp VIVA mic filter
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)



if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
