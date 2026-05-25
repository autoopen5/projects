import os
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
PROJECT_DIR = CONFIG_DIR.parent


def load_config() -> dict:
    with open(CONFIG_DIR / "signals.yaml") as f:
        return yaml.safe_load(f)


class Settings:
    def __init__(self):
        self.config = load_config()

        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        # ClickHouse
        self.clickhouse_host = os.getenv("CLICKHOUSE_HOST", "localhost")
        self.clickhouse_port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
        self.clickhouse_db = os.getenv("CLICKHOUSE_DB", "pharma")
        self.clickhouse_user = os.getenv("CLICKHOUSE_USER", "default")
        self.clickhouse_password = os.getenv("CLICKHOUSE_PASSWORD", "")

        # 1C
        self.onec_url = os.getenv("ONEC_URL", "")
        self.onec_user = os.getenv("ONEC_USER", "")
        self.onec_password = os.getenv("ONEC_PASSWORD", "")

        # CRM
        self.crm_url = os.getenv("CRM_URL", "")
        self.crm_token = os.getenv("CRM_TOKEN", "")

        # Dev mode - использовать mock данные
        self.use_mock = os.getenv("USE_MOCK_DATA", "true").lower() == "true"

        # API
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))


settings = Settings()
