"""
Планировщик задач: утренний дайджест + мониторинг критических сигналов.
"""
import asyncio
import logging
from datetime import datetime, time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from aiogram import Bot

from signals.core.engine import SignalEngine
from signals.core.signal import Priority
from signals.delivery.telegram_bot import send_digest, send_critical_alert
from config.settings import settings

logger = logging.getLogger(__name__)

_sent_critical_ids: set = set()  # чтобы не дублировать алерты


async def job_morning_digest(bot: Bot, engine: SignalEngine, chat_id: str):
    logger.info("Запуск утреннего дайджеста")
    try:
        await send_digest(bot, chat_id)
    except Exception as e:
        logger.error(f"Ошибка дайджеста: {e}")


async def job_critical_check(bot: Bot, engine: SignalEngine, chat_id: str):
    try:
        signals = engine.run_critical_only()
        for signal in signals:
            if signal.id not in _sent_critical_ids:
                _sent_critical_ids.add(signal.id)
                await send_critical_alert(bot, chat_id, signal)
    except Exception as e:
        logger.error(f"Ошибка проверки критических сигналов: {e}")


def setup_scheduler(bot: Bot, engine: SignalEngine, chat_id: str) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")

    digest_time = settings.config.get("schedule", {}).get("digest_time", "08:00")
    hour, minute = map(int, digest_time.split(":"))

    scheduler.add_job(
        job_morning_digest,
        CronTrigger(hour=hour, minute=minute, timezone="Europe/Moscow"),
        args=[bot, engine, chat_id],
        id="morning_digest",
        name="Утренний дайджест",
    )

    critical_interval = settings.config.get("schedule", {}).get("critical_check_interval", 15)
    scheduler.add_job(
        job_critical_check,
        IntervalTrigger(minutes=critical_interval),
        args=[bot, engine, chat_id],
        id="critical_check",
        name="Проверка критических сигналов",
    )

    return scheduler
