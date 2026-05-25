"""
Точка входа. Запускает Telegram бот + планировщик.
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from aiogram import Bot, Dispatcher
from signals.core.engine import SignalEngine
from signals.delivery.telegram_bot import router, run_bot
from signals.delivery.scheduler import setup_scheduler
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    config = settings.config
    engine = SignalEngine(config)

    if not settings.telegram_token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан — запускаю в режиме CLI")
        signals = engine.run_all()
        summary = engine.get_summary(signals)
        print(f"\n{'='*50}")
        print(f"Сигналов: {summary['total']}")
        print(f"По приоритету: {summary['by_priority']}")
        print(f"По доменам: {summary['by_domain']}")
        print(f"{'='*50}\n")
        for s in signals:
            print(s.to_telegram())
            print()
        return

    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(router)

    # Передаём engine в хендлеры через модуль
    import signals.delivery.telegram_bot as tg_module
    tg_module.engine = engine

    scheduler = setup_scheduler(bot, engine, settings.telegram_chat_id)
    scheduler.start()
    logger.info(f"Планировщик запущен. Дайджест в {settings.config['schedule']['digest_time']}")

    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
