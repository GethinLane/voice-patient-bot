# vpbot/google_adc.py
import os
import json
from loguru import logger

def ensure_google_adc():
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
