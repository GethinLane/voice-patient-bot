# vpbot/config.py
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

BOT_VERSION = "2026-02-26-splitbot-v1"
logger.info(f"✅ BOT_VERSION={BOT_VERSION}")

GRADING_SUBMIT_URL = (
    os.getenv("GRADING_SUBMIT_URL", "").strip()
    or "https://voice-patient-web.vercel.app/api/submit-transcript"
)
logger.info(f"✅ GRADING_SUBMIT_URL={GRADING_SUBMIT_URL}")
