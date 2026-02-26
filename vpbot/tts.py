# vpbot/tts.py
import os
import re
import aiohttp
import codecs
import base64
from loguru import logger

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.tts import GoogleTTSService, Language
from pipecat.services.inworld.tts import InworldHttpTTSService

class SafeInworldHttpTTSService(InworldHttpTTSService):
    """
    Robust parser for Inworld HTTP streaming TTS responses.
    """

    async def _process_streaming_response(self, response: aiohttp.ClientResponse, context_id: str):
        import json

        decoder = codecs.getincrementaldecoder("utf-8")()
        text_buffer = ""
        utterance_duration = 0.0

        async def handle_json_obj(obj: dict):
            nonlocal utterance_duration

            result = obj.get("result") or {}

            err = obj.get("error")
            if err:
                logger.warning(f"Inworld stream error object: {err}")

            audio_b64 = result.get("audioContent")
            if audio_b64:
                await self.stop_ttfb_metrics()
                audio_bytes = base64.b64decode(audio_b64)
                async for frame in self._process_audio_chunk(audio_bytes, context_id):
                    yield frame

            timestamp_info = result.get("timestampInfo")
            if timestamp_info:
                word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
                if word_times:
                    await self.add_word_timestamps(word_times, context_id)
                utterance_duration = max(utterance_duration, chunk_end_time)

        async def process_frame_text(frame_text: str):
            frame_text = frame_text.strip()
            if not frame_text:
                return

            if "data:" in frame_text:
                data_lines = []
                for line in frame_text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
                payload = "\n".join(data_lines).strip() if data_lines else frame_text
            else:
                payload = frame_text

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return

            async for out in handle_json_obj(obj):
                yield out

        async def drain_complete_frames():
            nonlocal text_buffer

            while True:
                sep = "\n\n" if "data:" in text_buffer else "\n"
                idx = text_buffer.find(sep)
                if idx < 0:
                    return

                frame = text_buffer[:idx]
                text_buffer = text_buffer[idx + len(sep) :]

                async for out in process_frame_text(frame):
                    yield out

        async for chunk in response.content.iter_chunked(4096):
            if not chunk:
                continue
            try:
                text_buffer += decoder.decode(chunk)
            except UnicodeDecodeError as e:
                logger.warning(f"Inworld stream UTF-8 decode error (chunk skipped): {e}")
                continue

            async for out in drain_complete_frames():
                yield out

        try:
            text_buffer += decoder.decode(b"", final=True)
        except UnicodeDecodeError as e:
            logger.warning(f"Inworld stream UTF-8 final decode error: {e}")

        leftover = text_buffer.strip()
        if leftover:
            async for out in drain_complete_frames():
                yield out

            tail = text_buffer.strip()
            if tail:
                async for out in process_frame_text(tail):
                    yield out

        if utterance_duration > 0:
            self._cumulative_time += utterance_duration

def _safe_lower(x):
    return str(x).strip().lower() if x is not None else ""

def build_tts_from_body(body: dict, aiohttp_session=None):
    """
    Create the TTS service based on runner_args.body.tts
    Defaults to Cartesia if anything is missing.
    """
    tts_cfg = body.get("tts") if isinstance(body, dict) else None
    tts_cfg = tts_cfg if isinstance(tts_cfg, dict) else {}

    provider = _safe_lower(tts_cfg.get("provider") or "cartesia")
    voice = (tts_cfg.get("voice") or "").strip() or None
    model = (tts_cfg.get("model") or "").strip() or None

    if provider in ("cartesia", ""):
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=voice or os.getenv("CARTESIA_VOICE_ID") or "71a7ad14-091c-4e8e-a314-022ece01c121",
        )

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

        logger.info(f"🔊 ElevenLabs TTS init: voice_id={voice_id!r}, model_id={model_id!r}, sample_rate=48000")

        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=voice_id,
            model=model_id,
            params=ElevenLabsTTSService.InputParams(enable_logging=True),
        )

    if provider == "inworld":
        api_key = (os.getenv("INWORLD_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "INWORLD_API_KEY missing. Set it to the Inworld Base64 runtime credential (do NOT include 'Basic ')."
            )
        if aiohttp_session is None:
            raise RuntimeError("Inworld selected but aiohttp_session was not provided.")

        voice_id = (voice or "").strip() or os.getenv("INWORLD_VOICE_ID") or "Ashley"
        model_id = (model or "").strip() or os.getenv("INWORLD_MODEL_ID") or "inworld-tts-1.5-max"

        cfg = tts_cfg.get("config") if isinstance(tts_cfg.get("config"), dict) else {}
        sr_raw = cfg.get("speakingRate", None)

        sr = 1.0
        if sr_raw is not None:
            if isinstance(sr_raw, str) and not sr_raw.strip():
                sr = 1.0
            else:
                try:
                    sr_candidate = float(sr_raw)
                    sr = sr_candidate if sr_candidate > 0 else 1.0
                except Exception:
                    sr = 1.0

        sr = max(0.5, min(2.0, sr))

        logger.info(f"🔊 Inworld TTS voice_id={voice_id!r}, model_id={model_id!r}, speakingRate(raw)={sr_raw!r} -> {sr}")

        params = InworldHttpTTSService.InputParams(speaking_rate=sr)

        return SafeInworldHttpTTSService(
            api_key=api_key,
            aiohttp_session=aiohttp_session,
            voice_id=voice_id,
            model=model_id,
            streaming=True,
            encoding="LINEAR16",
            sample_rate=48000,
            params=params,
        )

    if provider == "google":
        voice_id = (voice or "").strip()
        if not voice_id:
            raise RuntimeError("Google TTS selected but no voice_id provided in Airtable (tts.voice).")

        lang_code = "en-GB"
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
