import pandas as pd
import requests
import asyncio
import time

from telegram.ext import ApplicationBuilder, CommandHandler

TOKEN = "8691798405:AAGzC1Ooe90EI6J6JkeRuZ5wQGyP_3UxZh4"

FILE = "bonds.xlsx"
INTERVAL = 300

# MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{}.json"


subscribers = set()


# -----------------------
# загрузка Excel
# -----------------------

def load_bonds():

    df = pd.read_excel(FILE)

    bonds = {}

    for _, row in df.iterrows():
        bonds[row["ISIN"]] = row["Продать не ниже, в %"]

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

    subscribers.add(update.effective_chat.id)

    print("Subscriber:", update.effective_chat.id)

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


# -----------------------
# проверка сигналов
# -----------------------

async def monitor(context):

    print("Check prices")

    bonds = get_prices()

    for isin, data in bonds.items():

        price = data["price"]
        target = data["target"]

        if price and price >= target:

            text = f"""
SELL SIGNAL

{isin}

price: {price}
target: {target}
"""

            print("SIGNAL:", isin)

            for chat_id in subscribers:

                await context.bot.send_message(chat_id, text)


# -----------------------
# запуск
# -----------------------

def main():

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("price", price))

    app.job_queue.run_repeating(
        monitor,
        interval=INTERVAL,
        first=20
    )

    print("BOT STARTED")

    app.run_polling()


if __name__ == "__main__":
    main()