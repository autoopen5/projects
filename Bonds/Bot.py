import pandas as pd
import requests
import asyncio
import time
import datetime
from io import BytesIO
from telegram.ext import ApplicationBuilder, CommandHandler


TOKEN = "8691798405:AAGzC1Ooe90EI6J6JkeRuZ5wQGyP_3UxZh4"

FILE = "bonds.xlsx"
INTERVAL = 300


subscribers = set()
sent_signals = set()


PUBLIC_KEY = "https://disk.yandex.ru/i/lUBDhlSS469W3Q" # твоя ссылка

_cache = {
    "df": None,
    "timestamp": 0
}

CACHE_TTL = 300  # 5 минут


def load_bonds_from_yadisk():
    # кеш (чтобы не дергать диск каждый раз)
    if _cache["df"] is not None and time.time() - _cache["timestamp"] < CACHE_TTL:
        return _cache["df"]

    try:
        # получаем download URL
        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        params = {"public_key": PUBLIC_KEY}

        response = requests.get(url, params=params, timeout=10)
        download_url = response.json()["href"]

        # скачиваем файл
        file = requests.get(download_url, timeout=10)

        # читаем Excel
        df = pd.read_excel(BytesIO(file.content))
        df = df[
        df["ISIN"].notna() & 
        (df["ISIN"] != "") & 
        (df["ISIN"] != "-")
]
        # обновляем кеш
        _cache["df"] = df
        _cache["timestamp"] = time.time()

        return df

    except Exception as e:
        print("Ошибка загрузки bonds.xlsx:", e)

        # fallback: вернуть старые данные если есть
        if _cache["df"] is not None:
            return _cache["df"]

        return pd.DataFrame()
    
# -----------------------
# загрузка Excel
# -----------------------

def load_bonds():

    # sheet_id = "1p0jGfSuoi3eX3y5LWguo8kj87_mUq68TFDkt3bRRLdg"
    
    # url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    df = load_bonds_from_yadisk()

    bonds = []

    for _, row in df.iterrows():
        if pd.isna(row["ISIN"]) or row["ISIN"] == "":
            continue

        bonds.append({
                "ISIN": row["ISIN"],
                "Название": row.get("Характеристики Вклада"),
                "SellPrice": row.get("Продать не ниже, в %"),
                "Рейтинг": row.get("Рейтинг"),
                "Депозиты Банка": row.get("Депозиты Банка"),	
                "Фио": row.get("Фио"),
                "Кол-во обл": row.get("Кол-во обл"),
                "Средняя цена": row.get("Средняя цена"),
                "ТКД": row.get("ТКД"),
                "Дата оферты": row.get("Дата оферты"),
                "Спекуляции": row.get("Спекуляции"),
            })

    return bonds

def is_ofz(x):
    return str(x).startswith("SU")


def load_full_board(board):

    all_marketdata = []
    all_securities = []

    start = 0
    step = 100

    while True:

        url = f"https://iss.moex.com/iss/engines/stock/markets/bonds/boards/{board}/securities.json"

        params = {
            "iss.meta": "off",
            "iss.only": "marketdata,securities",
            "marketdata.columns": "SECID,LAST,PREVPRICE",
            "securities.columns": "SECID,ISIN",
            "start": start
        }

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        md = data.get("marketdata", {}).get("data", [])
        sec = data.get("securities", {}).get("data", [])

        if not md:
            break

        all_marketdata.extend(md)
        all_securities.extend(sec)

        start += step

    return all_marketdata, all_securities
# -----------------------
# загрузка цен МОЕХ
# -----------------------

def load_moex_prices(bonds):

    print("Request MOEX prices")

    ofz = [str(b["ISIN"]).strip() for b in bonds if is_ofz(b["ISIN"])]
    corp = [str(b["ISIN"]).strip() for b in bonds if not is_ofz(b["ISIN"])]

    print("OFZ:", len(ofz), "CORP:", len(corp))

    prices = {}

    # ========= КОРПОРАТЫ =========
    if corp:
        try:
            md, sec = load_full_board("TQCB")

            secid_to_isin = {row[0]: row[1] for row in sec if len(row) >= 2}

            for row in md:
                if len(row) < 3:
                    continue

                secid, last, prev = row
                isin = secid_to_isin.get(secid)

                if isin in corp:
                    price = last if last is not None else prev
                    prices[isin] = price

        except Exception as e:
            print("Corp MOEX error:", e)

    # ========= ОФЗ =========
    if ofz:
        try:
            md, _ = load_full_board("TQOB")

            for row in md:
                if len(row) < 3:
                    continue

                secid, last, prev = row

                if secid in ofz:
                    price = last if last is not None else prev
                    prices[secid] = price

        except Exception as e:
            print("OFZ MOEX error:", e)

    print("FOUND:", len(prices))
    print("SAMPLE:", list(prices.keys())[:10])

    return prices


# -----------------------
# собрать цены нужных бумаг
# -----------------------

def get_prices():

    bonds = load_bonds()
    moex = load_moex_prices(bonds)

    result = []

    for bond in bonds:

        isin = bond["ISIN"]
        price = moex.get(isin)

        result.append({
            "ISIN": isin,
            "price": price,
            "target": bond.get("SellPrice")
        })

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

    for bond in bonds:

        text += f"{bond['ISIN']}\n"
        text += f"price: {bond['price']}\n"
        text += f"target: {bond['target']}\n\n"

    await update.message.reply_text(text)


async def report(update, context):

    await send_report(context)

# -----------------------
# проверка сигналов
# -----------------------

async def monitor(context):

    print("Check prices")

    bonds = load_bonds()
    moex = load_moex_prices(bonds)

    for bond in bonds:

        isin = bond["ISIN"]
        target = bond.get("SellPrice")
        name = bond.get("Название")
        price = moex.get(isin)

        print(isin, "price:", price, "target:", target)

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
name: {name}
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
    moex = load_moex_prices(bonds)

    rows = []

    for bond in bonds:

        isin = bond["ISIN"]
        price = moex.get(isin)
        avg_price = bond.get("Средняя цена")

        income = None

        if price is not None and avg_price is not None:
            try:
                income = price - (avg_price / 10)
            except:
                income = None

        rows.append({
            "Название": bond.get("Название"),
            "ISIN": isin,
            "Цена": price,
            "Продать не ниже, в %": bond.get("SellPrice"),
            "Депозиты Банка": bond.get("Депозиты Банка"),
            "Фио": bond.get("Фио"),
            "Рейтинг": bond.get("Рейтинг"),
            "Кол-во обл": bond.get("Кол-во обл"),
            "Средняя цена": avg_price,
            "ТКД": bond.get("ТКД"),
            "Дата оферты": bond.get("Дата оферты"),
            "Спекуляции": bond.get("Спекуляции"),
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
