"""
Telegram бот для доставки управленческих сигналов.
Использует aiogram v3.
"""
import asyncio
import logging
from datetime import datetime
from typing import List

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.utils.markdown import hbold, hcode

from signals.core.signal import Signal, Priority, Domain
from signals.core.engine import SignalEngine
from config.settings import settings

logger = logging.getLogger(__name__)

router = Router()
engine: SignalEngine = None  # инициализируется в main


DOMAIN_LABELS = {
    Domain.FINANCE: "💰 Финансы",
    Domain.SALES: "📈 Продажи",
    Domain.PRODUCTION: "🏭 Производство",
    Domain.HR: "👥 Персонал",
    Domain.SEASONAL: "🌡 Сезонность",
}

PRIORITY_EMOJIS = {
    Priority.CRITICAL: "🔴",
    Priority.HIGH: "🟠",
    Priority.MEDIUM: "🟡",
    Priority.INFO: "🔵",
}


def format_digest(signals: List[Signal], title: str = "📊 Управленческий дайджест") -> List[str]:
    """Разбивает сигналы на сообщения с учётом лимита Telegram (4096 символов)."""
    if not signals:
        return [f"{title}\n\nСигналов нет — всё в норме ✅"]

    summary_parts = [f"<b>{title}</b>", f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}", ""]

    # Шапка с итогом
    from signals.core.engine import PRIORITY_ORDER
    critical = sum(1 for s in signals if s.priority == Priority.CRITICAL)
    high = sum(1 for s in signals if s.priority == Priority.HIGH)
    medium = sum(1 for s in signals if s.priority == Priority.MEDIUM)

    summary_parts.append(f"Итого сигналов: <b>{len(signals)}</b>")
    if critical:
        summary_parts.append(f"🔴 Критических: <b>{critical}</b>")
    if high:
        summary_parts.append(f"🟠 Высоких: <b>{high}</b>")
    if medium:
        summary_parts.append(f"🟡 Средних: <b>{medium}</b>")
    summary_parts.append("─" * 30)

    messages = []
    current = "\n".join(summary_parts)

    for signal in signals:
        block = format_signal_block(signal)
        if len(current) + len(block) > 3800:
            messages.append(current)
            current = block
        else:
            current += "\n\n" + block

    if current:
        messages.append(current)

    return messages


def format_signal_block(signal: Signal) -> str:
    priority_emoji = PRIORITY_EMOJIS[signal.priority]
    domain_label = DOMAIN_LABELS[signal.domain]
    lines = [
        f"{priority_emoji} <b>{signal.title}</b>",
        f"<i>{domain_label}</i>",
        signal.body,
    ]
    if signal.deviation_pct is not None:
        sign = "+" if signal.deviation_pct > 0 else ""
        lines.append(f"Отклонение: {sign}{signal.deviation_pct:.1f}%")
    if signal.action:
        lines.append(f"💡 <i>{signal.action}</i>")
    return "\n".join(lines)


@router.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "<b>Система управленческих сигналов</b> запущена.\n\n"
        "Доступные команды:\n"
        "/status — текущие сигналы прямо сейчас\n"
        "/critical — только критические сигналы\n"
        "/finance — финансовый блок\n"
        "/sales — продажи\n"
        "/production — производство\n"
        "/hr — персонал\n"
        "/seasonal — сезонность\n"
        "/digest — полный дайджест",
        parse_mode="HTML"
    )


@router.message(Command("status", "digest"))
async def cmd_status(message: types.Message):
    signals = engine.run_all()
    for msg_text in format_digest(signals):
        await message.answer(msg_text, parse_mode="HTML")


@router.message(Command("critical"))
async def cmd_critical(message: types.Message):
    signals = engine.run_critical_only()
    if not signals:
        await message.answer("✅ Критических сигналов нет")
        return
    for s in signals:
        await message.answer(format_signal_block(s), parse_mode="HTML")


@router.message(Command("finance"))
async def cmd_finance(message: types.Message):
    from signals.analyzers.finance import FinanceAnalyzer
    signals = FinanceAnalyzer(engine.config).analyze()
    msgs = format_digest(signals, "💰 Финансы")
    for m in msgs:
        await message.answer(m, parse_mode="HTML")


@router.message(Command("sales"))
async def cmd_sales(message: types.Message):
    from signals.analyzers.sales import SalesAnalyzer
    signals = SalesAnalyzer(engine.config).analyze()
    msgs = format_digest(signals, "📈 Продажи")
    for m in msgs:
        await message.answer(m, parse_mode="HTML")


@router.message(Command("production"))
async def cmd_production(message: types.Message):
    from signals.analyzers.production import ProductionAnalyzer
    signals = ProductionAnalyzer(engine.config).analyze()
    msgs = format_digest(signals, "🏭 Производство")
    for m in msgs:
        await message.answer(m, parse_mode="HTML")


@router.message(Command("hr"))
async def cmd_hr(message: types.Message):
    from signals.analyzers.hr import HRAnalyzer
    signals = HRAnalyzer(engine.config).analyze()
    msgs = format_digest(signals, "👥 Персонал")
    for m in msgs:
        await message.answer(m, parse_mode="HTML")


@router.message(Command("seasonal"))
async def cmd_seasonal(message: types.Message):
    from signals.analyzers.seasonal import SeasonalAnalyzer
    signals = SeasonalAnalyzer(engine.config).analyze()
    msgs = format_digest(signals, "🌡 Сезонность")
    for m in msgs:
        await message.answer(m, parse_mode="HTML")


async def send_digest(bot: Bot, chat_id: str):
    """Отправляет утренний дайджест."""
    signals = engine.run_all()
    for msg_text in format_digest(signals):
        await bot.send_message(chat_id, msg_text, parse_mode="HTML")


async def send_critical_alert(bot: Bot, chat_id: str, signal: Signal):
    """Мгновенный алерт о критическом сигнале."""
    text = f"🚨 <b>КРИТИЧЕСКИЙ СИГНАЛ</b>\n\n{format_signal_block(signal)}"
    await bot.send_message(chat_id, text, parse_mode="HTML")


async def run_bot(signal_engine: SignalEngine):
    global engine
    engine = signal_engine

    if not settings.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN не задан")
        return

    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Запуск Telegram бота...")
    await dp.start_polling(bot)
