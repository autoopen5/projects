from pathlib import Path
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# загружаем .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class ReportConfig:
    date_from: str
    date_to: str
    visits_threshold: float
    sales_threshold: float


def load_clickhouse_config() -> ClickHouseConfig:
    return ClickHouseConfig(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
        user=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE"),
    )


def load_report_config() -> ReportConfig:
    return ReportConfig(
        date_from=os.getenv("REPORT_DATE_FROM"),
        date_to=os.getenv("REPORT_DATE_TO"),
        visits_threshold=float(os.getenv("VISITS_THRESHOLD", 1)),
        sales_threshold=float(os.getenv("SALES_THRESHOLD", 1)),
    )