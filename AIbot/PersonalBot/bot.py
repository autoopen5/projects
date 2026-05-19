"""
Личный Telegram-ассистент.

Запуск: python bot.py
"""
import logging
from datetime import time as dtime

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

import config
import db
from ai import get_ai_response, morning_message, evening_message, week_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

ME = filters.Chat(chat_id=config.MY_CHAT_ID)


# ── Команды ───────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я твой личный ассистент 👋\n\n"
        "/weight 89.5 — записать вес\n"
        "/done йога 20мин — записать активность\n"
        "/read 20 — страниц прочитано\n"
        "/status — прогресс за неделю\n"
        "/week — итоги недели (Claude)\n\n"
        "Или просто пиши — отвечу как коуч."
    )


async def cmd_weight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        weight = float(context.args[0].replace(",", "."))
    except (IndexError, ValueError):
        await update.message.reply_text("Укажи вес: /weight 89.5")
        return
    db.log_weight(weight)
    left = round(weight - 80, 1)
    emoji = "🔥" if left <= 5 else "💪"
    await update.message.reply_text(
        f"✅ Записал: {weight} кг\n"
        f"{emoji} До цели (80 кг): {left} кг"
    )


async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Что сделал? /done йога 20мин")
        return
    activity = " ".join(context.args)
    db.log_activity(activity)
    await update.message.reply_text(f"✅ {activity} 💪")


async def cmd_read(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        pages = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("Сколько страниц? /read 20")
        return
    db.log_pages(pages)
    await update.message.reply_text(f"📚 +{pages} страниц записано")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    recent = db.get_recent_logs(7)
    last_weight = db.get_last_weight()

    lines = ["📊 *Неделя:*\n"]
    if last_weight:
        left = round(last_weight - 80, 1)
        lines.append(f"⚖️ Вес: *{last_weight} кг*  →  до цели: {left} кг")
    else:
        lines.append("⚖️ Вес не записан")

    active_days = sum(1 for r in recent if r.get("activities"))
    total_pages = sum(r.get("pages_read") or 0 for r in recent)
    lines.append(f"🏃 Активных дней: {active_days}/7")
    lines.append(f"📚 Страниц прочитано: {total_pages}")

    if recent:
        lines.append("\n*Последние записи:*")
        for r in recent[:3]:
            parts = [r["date"]]
            if r.get("weight"):
                parts.append(f"{r['weight']} кг")
            if r.get("activities"):
                parts.append(r["activities"])
            lines.append("  " + " | ".join(parts))

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анализирую неделю...")
    logs = db.get_week_summary()
    text = await week_summary(logs)
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    recent = db.get_recent_logs(3)
    last_weight = db.get_last_weight()
    reply = await get_ai_response(update.message.text, recent, last_weight)
    await update.message.reply_text(reply)


# ── Расписание ────────────────────────────────────────────────

async def job_morning(context: ContextTypes.DEFAULT_TYPE):
    last_weight = db.get_last_weight()
    recent = db.get_recent_logs(3)
    text = await morning_message(last_weight=last_weight, recent=recent)
    await context.bot.send_message(chat_id=config.MY_CHAT_ID, text=text)


async def job_evening(context: ContextTypes.DEFAULT_TYPE):
    text = await evening_message()
    await context.bot.send_message(chat_id=config.MY_CHAT_ID, text=text)


# ── Запуск ────────────────────────────────────────────────────

def main():
    db.init()

    app = Application.builder().token(config.TG_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start,  filters=ME))
    app.add_handler(CommandHandler("weight", cmd_weight, filters=ME))
    app.add_handler(CommandHandler("done",   cmd_done,   filters=ME))
    app.add_handler(CommandHandler("read",   cmd_read,   filters=ME))
    app.add_handler(CommandHandler("status", cmd_status, filters=ME))
    app.add_handler(CommandHandler("week",   cmd_week,   filters=ME))
    app.add_handler(MessageHandler(ME & filters.TEXT & ~filters.COMMAND, handle_message))

    jq = app.job_queue
    jq.run_daily(job_morning, time=dtime(config.MORNING_HOUR, 0, tzinfo=config.TZ))
    jq.run_daily(job_evening, time=dtime(config.EVENING_HOUR, 0, tzinfo=config.TZ))

    log.info(f"Бот запущен. Утро: {config.MORNING_HOUR}:00, Вечер: {config.EVENING_HOUR}:00 МСК")
    app.run_polling()


if __name__ == "__main__":
    main()
