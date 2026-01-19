import os
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Импортируем всё нужное из твоего RAG-скрипта
from rag_multi_docs_v1 import (
    Embedder,
    build_index,
    retrieve,
    answer_with_llm,
    EMBED_MODEL,
    DOC_PATHS,
    TOP_K,
)

# ----------------- Логирование -----------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ----------------- Инициализация RAG -----------------
print("Инициализация RAG (эмбеддер + индекс)...")
embedder = Embedder(EMBED_MODEL)
index = build_index(DOC_PATHS, embedder)
print(f"RAG готов. Всего чанков: {len(index.meta)}")


# ----------------- Хендлеры бота -----------------
WELCOME_TEXT = (
    "👋 Привет! Я Офис-помощник.\n\n"
    "Я помогу найти нужную информацию в внутренних документах компании "
    "(положения, регламенты и т.п.).\n\n"
    "Просто напиши вопрос, например:\n"
    "«Какие документы нужны для командировки?»\n"
    "«Как оформляется тендер?»"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ответ на команду /start."""
    await update.message.reply_text(WELCOME_TEXT)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ответ на /help."""
    await update.message.reply_text(
        "Напиши обычным текстом вопрос, а я постараюсь найти ответ в документах."
    )

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатываем любой текст как вопрос к RAG."""
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    chat_id = update.message.chat_id
    logger.info("Запрос от %s: %s", chat_id, question)

    # Быстрый ответ-заглушка (чтобы пользователь видел, что бот думает)
    await update.message.reply_chat_action("typing")

    try:
        hits = retrieve(question, index, embedder, TOP_K)
        answer, srcs_secs = answer_with_llm(question, hits)

        # Собираем строку с источниками
        src_part = ""
        if srcs_secs:
            parts = []
            for name, secs in srcs_secs:
                if secs:
                    parts.append(f"{name} (пп.: {', '.join(secs)})")
                else:
                    parts.append(name)
            src_part = "\n\nИсточник: " + "; ".join(parts)

        text = f"💬 *Ответ:*\n{answer}{src_part}"

        # MarkdownV2 требует экранирования, но ответ может быть любой.
        # Чтобы не заморачиваться — отправим обычным текстом.
        await update.message.reply_text(text)

    except Exception as e:
        logger.exception("Ошибка при обработке запроса: %s", e)
        await update.message.reply_text(
            "Произошла ошибка при поиске информации. Попробуй ещё раз позже."
        )

# ----------------- Запуск бота -----------------
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    # Любой текст, кроме команд — как вопрос к RAG
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("Бот запущен. Нажми Ctrl+C для остановки.")
    app.run_polling()

if __name__ == "__main__":
    main()