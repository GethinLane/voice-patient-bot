# vpbot/transcript.py
import json
import requests
from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext

def build_transcript_from_context(context: LLMContext):
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

def submit_grading(url: str, payload: dict):
    """
    Intended to be called inside a background thread.
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
