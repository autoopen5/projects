import anthropic
import config

client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

SYSTEM = """Ты личный ассистент и лайф-коуч. Ты хорошо знаешь своего пользователя.

ЦЕЛИ:
- Вес: похудеть с 90 до 80 кг
- Физуха: йога (регулярно предлагай), футбол, волейбол, серфинг, домашние упражнения (отжимания, приседания)
- Книги: читать больше — хотя бы 15–20 минут в день. Можешь рекомендовать книги
- Семья: регулярно ходить на концерты и мероприятия с семьёй
- Путешествия: вдохновлять идеями куда поехать, помогать планировать
- Работа: AI-проект в фарме (парсинг базы врачей, визитная активность), нужен баланс работа/жизнь

СТИЛЬ:
- Короткие ответы, по делу, без воды
- Мотивируй без слащавости
- Иногда сам предлагай интересные идеи — поездку, книгу, новую тренировку
- Только русский язык"""


async def _ask(prompt: str, system: str = SYSTEM, max_tokens: int = 500) -> str:
    msg = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _format_logs(logs: list[dict]) -> str:
    if not logs:
        return "нет данных"
    lines = []
    for r in logs:
        parts = [r["date"]]
        if r.get("weight"):
            parts.append(f"{r['weight']} кг")
        if r.get("activities"):
            parts.append(r["activities"])
        if r.get("pages_read"):
            parts.append(f"📚 {r['pages_read']} стр.")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


async def get_ai_response(text: str, recent_logs: list, last_weight: float | None) -> str:
    context = f"Последние записи:\n{_format_logs(recent_logs)}"
    if last_weight:
        context += f"\nПоследний вес: {last_weight} кг (цель: 80 кг, осталось: {round(last_weight - 80, 1)} кг)"
    prompt = f"{context}\n\nВопрос: {text}"
    return await _ask(prompt)


async def morning_message(last_weight: float | None, recent: list) -> str:
    weight_info = f"последний вес {last_weight} кг (цель 80)" if last_weight else "вес ещё не записан"
    recent_info = _format_logs(recent)
    prompt = (
        f"Напиши короткое доброе утро (3–5 строк). "
        f"Контекст: {weight_info}. "
        f"Последние дни:\n{recent_info}\n"
        f"Предложи одну конкретную физическую активность на сегодня и одно дело для души "
        f"(книга/мероприятие/идея поездки). Будь конкретным."
    )
    return await _ask(prompt, max_tokens=200)


async def evening_message() -> str:
    return (
        "Вечерний чекин 🌙\n\n"
        "Запиши сегодня:\n"
        "⚖️ /weight 89.5 — вес\n"
        "💪 /done йога 20мин — активность\n"
        "📚 /read 20 — страниц прочитано\n\n"
        "Или просто напиши как прошёл день."
    )


async def week_summary(logs: list[dict]) -> str:
    summary = _format_logs(logs)
    prompt = (
        f"Данные за неделю:\n{summary}\n\n"
        f"Напиши краткий итог недели (5–7 строк): прогресс по весу, "
        f"физическая активность, книги. Отметь что хорошо, что подтянуть. "
        f"Мотивируй на следующую неделю."
    )
    return await _ask(prompt, max_tokens=300)
