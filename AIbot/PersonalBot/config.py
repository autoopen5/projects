import os
from datetime import timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN          = os.environ["TG_TOKEN"]
MY_CHAT_ID        = int(os.environ["MY_CHAT_ID"])
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

MORNING_HOUR = int(os.getenv("MORNING_HOUR", "8"))
EVENING_HOUR = int(os.getenv("EVENING_HOUR", "21"))

# Москва UTC+3 (DST отменён с 2014)
TZ = timezone(timedelta(hours=3))
