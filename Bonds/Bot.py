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


PUBLIC_KEY = "https://disk.yandex.ru/i/2vPKVDCLTThDww" # твоя ссылка

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

# -----------------------
# загрузка цен МОЕХ
# -----------------------

def load_moex_data():

    url = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"

    params = {
        "iss.meta": "off",
        "marketdata.columns": "SECID,LAST,YIELD,YIELD_OFFER"
    }

    print("Request MOEX data")

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    result = {}

    for row in data["marketdata"]["data"]:
        secid = row[0]
        last = row[1]
        ytm = row[2]
        ytm_offer = row[3]

        # выбор правильной доходности
        if ytm_offer and ytm_offer > 0:
            final_ytm = ytm_offer
            ytm_type = "offer"
        elif ytm and ytm > 0:
            final_ytm = ytm
            ytm_type = "maturity"
        else:
            final_ytm = None
            ytm_type = None

        result[secid] = {
            "price": last,
            "ytm": final_ytm,
            "ytm_type": ytm_type
        }

    return result

# def load_moex_prices():

#     url = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"

#     params = {
#         "iss.meta": "off",
#         "marketdata.columns": "SECID,LAST"
#     }

#     print("Request MOEX prices")

#     r = requests.get(url, params=params, timeout=10)

#     data = r.json()

#     prices = {}

#     for row in data["marketdata"]["data"]:

#         secid = row[0]
#         price = row[1]

#         prices[secid] = price

#     return prices

# -----------------------
# поиск альтернатив при цене выше обозначенной
# -----------------------

RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-"
]

def rating_ok(rating):
    if not isinstance(rating, str):
        return False
    rating = rating.strip().upper()
    return rating in RATING_ORDER[:7]  # до A-

def calc_score(ytm, rating):
    penalty = 0

    if rating in ["A-"]:
        penalty += 0.3
    elif rating in ["BBB+", "BBB"]:
        penalty += 0.6

    return ytm - penalty

def find_alternatives(all_bonds, moex_data, current_ytm):

    candidates = []

    for bond in all_bonds:

        isin = bond["ISIN"]
        rating = bond.get("Рейтинг")

        moex = moex_data.get(isin)
        if not moex:
            continue

        ytm = moex.get("ytm")

        if not ytm or not rating:
            continue

        if not rating_ok(rating):
            continue

        if current_ytm and ytm <= current_ytm:
            continue

        score = calc_score(ytm, rating)

        candidates.append({
            "name": bond.get("Название"),
            "isin": isin,
            "ytm": ytm,
            "rating": rating,
            "score": score
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[:3]

# -----------------------
# собрать цены нужных бумаг
# -----------------------

def get_prices():

    bonds = load_bonds()
    moex = load_moex_prices()

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

    bonds = load_bonds()
    moex_data = load_moex_data()

    text = "Bond prices:\n\n"

    for bond in bonds:
        moex = moex_data.get(bond["ISIN"], {})

        text += f"{bond['ISIN']}\n"
        text += f"price: {moex.get('price')}\n"
        text += f"ytm: {moex.get('ytm')} ({moex.get('ytm_type')})\n"
        text += f"target: {bond.get('SellPrice')}\n\n"    

    await update.message.reply_text(text)


async def report(update, context):

    await send_report(context)

# -----------------------
# проверка сигналов
# -----------------------

async def monitor(context):

    print("Check prices")

    bonds = load_bonds()
    moex_data = load_moex_data()

    for bond in bonds:

        isin = bond["ISIN"]
        target = bond.get("SellPrice")
        name = bond.get("Название")
        rating = bond.get("Рейтинг")

        moex = moex_data.get(isin)

        if not moex:
            continue

        price = moex.get("price")
        ytm = moex.get("ytm")
        ytm_type = moex.get("ytm_type")

        print(isin, "price:", price, "target:", target, "ytm:", ytm)

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

            # ищем альтернативы
            alternatives = find_alternatives(bonds, moex_data, ytm)

            text = f"""
📈 SELL SIGNAL

{isin}
{name}

Цена: {price:.2f} (цель {target})
Рейтинг: {rating}

Доходность ({ytm_type}): {round(ytm,2) if ytm else "N/A"}%

🔄 Альтернативы:
"""

            if not alternatives:
                text += "\nНет лучших вариантов"
            else:
                for i, alt in enumerate(alternatives, 1):
                    text += f"\n{i}. {alt['name']} ({alt['isin']}) — {round(alt['ytm'],2)}% ({alt['rating']})"

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
    moex_data = load_moex_data()

    rows = []

    for bond in bonds:

        isin = bond["ISIN"]
        name = bond.get("Название")
        rating = bond.get("Рейтинг")

        moex = moex_data.get(isin, {})

        price = moex.get("price")
        ytm = moex.get("ytm")
        ytm_type = moex.get("ytm_type")

        avg_price = bond.get("Средняя цена")

        income = None

        if price is not None and avg_price is not None:
            try:
                income = price - (avg_price / 10)
            except:
                income = None

        # скоринг
        score = None
        if ytm and rating:
            try:
                score = calc_score(float(ytm), rating)
            except:
                score = None

        # поиск лучшей альтернативы
        best_alt = None
        if ytm:
            alternatives = find_alternatives(bonds, moex_data, ytm)
            if alternatives:
                best_alt = alternatives[0]

        rows.append({
            "Название": name,
            "ISIN": isin,
            "Цена": price,
            "YTM": ytm,
            "Тип YTM": ytm_type,
            "Score": score,
            "Лучшая альтернатива": best_alt["isin"] if best_alt else None,
            "YTM альтернативы": best_alt["ytm"] if best_alt else None,
            "Рейтинг": rating,
            "Продать не ниже, в %": bond.get("SellPrice"),
            "Средняя цена": avg_price,
            "Доход": income,
            "Кол-во обл": bond.get("Кол-во обл"),
            "ТКД": bond.get("ТКД"),
            "Дата оферты": bond.get("Дата оферты"),
            "Спекуляции": bond.get("Спекуляции"),
            "Фио": bond.get("Фио"),
            "Депозиты Банка": bond.get("Депозиты Банка")
        })

    df = pd.DataFrame(rows)

    # сортировка — самые интересные сверху
    df = df.sort_values(by="Score", ascending=False, na_position="last")

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
    time=datetime.time(hour=9, minute=0)
    )

    app.job_queue.run_daily(
        send_report,
        time=datetime.time(hour=15, minute=0)
    )
    
    print("BOT STARTED")

    app.run_polling()


if __name__ == "__main__":
    main()
