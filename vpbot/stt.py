# vpbot/stt.py
import os
import time
from loguru import logger

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams

_STT_COOLDOWN_UNTIL = {}

def _now() -> float:
    return time.time()

def _cooldown_secs() -> int:
    try:
        return int(os.getenv("STT_FAILOVER_COOLDOWN_SECS") or "60")
    except Exception:
        return 60

def set_cooldown(provider: str):
    if provider:
        _STT_COOLDOWN_UNTIL[provider] = _now() + _cooldown_secs()

def in_cooldown(provider: str) -> bool:
    if not provider:
        return False
    until = _STT_COOLDOWN_UNTIL.get(provider)
    return bool(until and until > _now())

def get_primary_secondary() -> tuple[str, str]:
    forced = (os.getenv("STT_FORCE_PROVIDER") or "").strip().lower()
    if forced:
        return forced, ""

    primary = (os.getenv("STT_PRIMARY") or "deepgram").strip().lower()
    secondary = (os.getenv("STT_SECONDARY") or "assemblyai").strip().lower()
    if secondary == primary:
        secondary = ""
    return primary, secondary

def is_transient_network_error(exc: Exception) -> bool:
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

def is_capacity_error(provider: str, exc: Exception) -> bool:
    p = (provider or "").lower()
    s = (str(exc) or "").lower()

    if p in ("deepgram", "dg"):
        return ("429" in s or "too many requests" in s or "too_many_requests" in s or "rate limit" in s)

    if p in ("assemblyai", "aai"):
        return (("1008" in s and "too many concurrent sessions" in s) or ("too many concurrent sessions" in s))

    return False

def should_failover(provider: str, exc: Exception) -> bool:
    return is_capacity_error(provider, exc) or is_transient_network_error(exc)

def build_stt_service(provider: str):
    provider = (provider or "").strip().lower()

    if provider in ("deepgram", "dg"):
        api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Missing DEEPGRAM_API_KEY (STT provider=deepgram)")

        def _f(name: str, default=None):
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except Exception:
                return default

        def _i(name: str, default=None):
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except Exception:
                return default

        flux_params = DeepgramFluxSTTService.InputParams(
            min_confidence=_f("DG_FLUX_MIN_CONFIDENCE", 0.3),
            eot_threshold=_f("DG_FLUX_EOT_THRESHOLD", None),
            eager_eot_threshold=_f("DG_FLUX_EAGER_EOT_THRESHOLD", None),
            eot_timeout_ms=_i("DG_FLUX_EOT_TIMEOUT_MS", None),
        )

        return DeepgramFluxSTTService(
            api_key=api_key,
            params=flux_params,
        )

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
            vad_force_turn_endpoint=True,
        )

    raise RuntimeError(f"Unknown STT provider: {provider!r}")

def choose_stt_primary_first() -> tuple[object, str, str]:
    primary, secondary = get_primary_secondary()

    if primary and not in_cooldown(primary):
        return build_stt_service(primary), primary, secondary

    if secondary and not in_cooldown(secondary):
        logger.warning(f"⏭️ Primary STT in cooldown; using secondary={secondary}")
        return build_stt_service(secondary), secondary, primary

    logger.warning("⚠️ Both STT providers appear in cooldown; retrying primary anyway.")
    return build_stt_service(primary), primary, secondary
