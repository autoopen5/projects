import pandas as pd
import requests
import asyncio
import time
import datetime

from telegram.ext import ApplicationBuilder, CommandHandler


TOKEN = "8691798405:AAGzC1Ooe90EI6J6JkeRuZ5wQGyP_3UxZh4"

FILE = "bonds.xlsx"
INTERVAL = 300

# MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{}.json"


subscribers = set()
sent_signals = set()

# -----------------------
# загрузка Excel
# -----------------------

def load_bonds():

    df = pd.read_excel(FILE)

    bonds = {}

    for _, row in df.iterrows():

        bonds[row["ISIN"]] = {
                "SellPrice": row.get("Продать не ниже, в %"),
                "Рейтинг": row.get("Рейтинг"),
                "Кол-во обл": row.get("Кол-во обл"),
                "Средняя цена": row.get("Средняя цена"),
                "ТКД": row.get("ТКД"),
                "Дата оферты": row.get("Дата оферты"),
                "Спекуляции": row.get("Спекуляции"),
            }

    return bonds

# -----------------------
# загрузка цен МОЕХ
# -----------------------

def load_moex_prices():

    url = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"

    params = {
        "iss.meta": "off",
        "marketdata.columns": "SECID,LAST"
    }

    print("Request MOEX prices")

    r = requests.get(url, params=params, timeout=10)

    data = r.json()

    prices = {}

    for row in data["marketdata"]["data"]:

        secid = row[0]
        price = row[1]

        prices[secid] = price

    return prices


# -----------------------
# собрать цены нужных бумаг
# -----------------------

def get_prices():

    bonds = load_bonds()
    moex = load_moex_prices()

    result = {}

    for isin, target in bonds.items():

        if pd.isna(target):
            continue

        price = moex.get(isin)

        result[isin] = {
            "price": price,
            "target": target
        }

    return result


# -----------------------
# команды
# -----------------------

async def start(update, context):

    chat_id = update.effective_chat.id

    subscribers.add(chat_id)

    print("New subscriber:", chat_id)

    await update.message.reply_text("Bond bot started")


async def price(update, context):

    print("/price command")

    bonds = get_prices()

    text = "Bond prices:\n\n"

    for isin, data in bonds.items():

        text += f"{isin}\n"

        text += f"price: {data['price']}\n"

        text += f"target: {data['target']}\n\n"

    await update.message.reply_text(text)


async def report(update, context):

    await send_report(context)

# -----------------------
# проверка сигналов
# -----------------------

async def monitor(context):

    print("Check prices")

    bonds = load_bonds()
    moex = load_moex_prices()

    for isin, info in bonds.items():

        target = info.get("SellPrice")
        price = moex.get(isin)

        print(isin, "price:", price, "target:", target)

        # пропускаем если нет цены продажи
        if target is None or pd.isna(target):
            continue

        if price is None:
            continue

        try:
            price = float(price)
            target = float(target)
        except:
            continue

        if price >= target and isin not in sent_signals:

            text = f"""
    SELL SIGNAL

    {isin}

    price: {price}
    target: {target}
    """

            print("SIGNAL:", isin)

            for chat_id in subscribers:
                await context.bot.send_message(chat_id=chat_id, text=text)

            sent_signals.add(isin)

        elif price < target and isin in sent_signals:

            print("Reset signal for", isin)

            sent_signals.remove(isin)

# -----------------------
# функция отчёта
# -----------------------

async def send_report(context):

    print("Generate report")

    bonds = load_bonds()
    moex = load_moex_prices()

    rows = []

    for isin, info in bonds.items():

        price = moex.get(isin)

        avg_price = info.get("Средняя цена")

        # расчет дохода
        income = None

        if price is not None and avg_price is not None:
            try:
                income = price - (avg_price / 10)
            except:
                income = None

        rows.append({
            "ISIN": isin,
            "Price": price,
            "SellPrice": info.get("SellPrice"),

            "Рейтинг": info.get("Рейтинг"),
            "Кол-во обл": info.get("Кол-во обл"),
            "Средняя цена": avg_price,
            "ТКД": info.get("ТКД"),
            "Дата оферты": info.get("Дата оферты"),
            "Спекуляции": info.get("Спекуляции"),

            "Доход": income
        })

    df = pd.DataFrame(rows)

    file = "bond_report.xlsx"

    df.to_excel(file, index=False)

    print("Report created:", file)

    for chat_id in subscribers:
        await context.bot.send_document(
            chat_id=chat_id,
            document=open(file, "rb")
        )


# -----------------------
# запуск
# -----------------------

def main():

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("report", report))

    app.job_queue.run_repeating(
        monitor,
        interval=INTERVAL,
        first=20
    )
    app.job_queue.run_daily(
    send_report,
    time=datetime.time(hour=12, minute=0)
    )

    app.job_queue.run_daily(
        send_report,
        time=datetime.time(hour=18, minute=0)
    )
    
    print("BOT STARTED")

    app.run_polling()


if __name__ == "__main__":
    main()