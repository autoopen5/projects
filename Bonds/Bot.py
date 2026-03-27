import pandas as pd
import requests
import asyncio
import time
import datetime
from io import BytesIO
from telegram.ext import ApplicationBuilder, CommandHandler
from openpyxl import load_workbook
import zipfile
import xml.etree.ElementTree as ET

TOKEN = "8691798405:AAGzC1Ooe90EI6J6JkeRuZ5wQGyP_3UxZh4"

# FILE = "bonds.xlsx"
INTERVAL = 300


subscribers = set()
sent_signals = set()
sent_buy_signals = set()

# https://disk.yandex.ru/d/ejWF4wGI3-0cww  gfgby xlsx
PUBLIC_KEY = "https://disk.yandex.ru/i/4ow9EC89R_7ADw" # твоя ссылка
# https://disk.yandex.ru/i/2vPKVDCLTThDww https://disk.yandex.ru/i/bpEO6PndD__7ZQ

_cache = {
    "df": None,
    "timestamp": 0
}

CACHE_TTL = 300  # 5 минут

def read_excel_safe(content):

    try:
        file_bytes = BytesIO(content)

        with zipfile.ZipFile(file_bytes) as z:

            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

            # 🔹 shared strings
            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                sst_xml = z.read("xl/sharedStrings.xml")
                sst_root = ET.fromstring(sst_xml)

                for si in sst_root.findall(".//a:t", ns):
                    shared_strings.append(si.text)

            # 🔹 находим первый лист автоматически
            sheet_name = [f for f in z.namelist() if "worksheets/sheet" in f][0]
            sheet_xml = z.read(sheet_name)

            root = ET.fromstring(sheet_xml)

            rows = []

            for row in root.findall(".//a:sheetData/a:row", ns):

                row_data = {}

                for cell in row.findall("a:c", ns):

                    cell_ref = cell.attrib.get("r")  # например A1
                    col = ''.join(filter(str.isalpha, cell_ref))

                    t = cell.attrib.get("t")
                    v = cell.find("a:v", ns)

                    value = None

                    if v is not None:
                        if t == "s":
                            value = shared_strings[int(v.text)]
                        else:
                            value = v.text

                    row_data[col] = value

                rows.append(row_data)

            if not rows:
                return pd.DataFrame()

            # 🔥 преобразуем в DataFrame
            df = pd.DataFrame(rows)

            # первая строка = заголовки
            df.columns = df.iloc[0]
            df = df[1:]

            # чистим названия колонок
            df.columns = [
                str(c).strip().replace("\n", " ").replace("\r", "")
                for c in df.columns
            ]

            return df.reset_index(drop=True)

    except Exception as e:
        print("❌ XML Excel parser failed:", e)
        return pd.DataFrame()

  
def load_bonds_from_yadisk():

    if _cache["df"] is not None and time.time() - _cache["timestamp"] < CACHE_TTL:
        return _cache["df"]

    try:
        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        params = {"public_key": PUBLIC_KEY}

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "href" not in data:
            print("❌ Yandex error:", data)
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        download_url = data["href"]
        file = requests.get(download_url, timeout=10)

        if not file.content.startswith(b'PK'):
            print("❌ Not XLSX")
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        df = read_excel_safe(file.content)
        print("Columns parsed:", df.columns.tolist())
        print("Rows parsed:", len(df))    
        # 🔥 оставляем только нужные колонки (игнорим NRL и прочее)
        needed_cols = [
            "ISIN",
            "Характеристики Вклада",
            "Продать не ниже, в %",
            "Купить не выше",
            "Рейтинг",
            "Депозиты Банка",	
            "Фио",
            "Кол-во обл",
            "Средняя цена",
            "Дата оферты",
            "Спекуляции"
        ]

        if df is None or df.empty:
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()
        
        df = df[[col for col in needed_cols if col in df.columns]]

        # фильтр
        df = df[
            df["ISIN"].notna() &
            (df["ISIN"] != "") &
            (df["ISIN"] != "-")
        ]
        print("After ISIN filter:", len(df))

        _cache["df"] = df
        _cache["timestamp"] = time.time()

        return df

    except Exception as e:
        print("Ошибка загрузки:", e)
        return _cache["df"] if _cache["df"] is not None else pd.DataFrame()
    
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
                "BuyPrice": row.get("Купить не выше"),
                "Рейтинг": row.get("Рейтинг"),
                "Депозиты Банка": row.get("Депозиты Банка"),	
                "Фио": row.get("Фио"),
                "Кол-во обл": row.get("Кол-во обл"),
                "Средняя цена": row.get("Средняя цена"),
                # "ТКД": row.get("ТКД"),
                "Дата оферты": row.get("Дата оферты"),
                "Спекуляции": row.get("Спекуляции"),
            })

    return bonds

# -----------------------
# загрузка цен МОЕХ
# -----------------------

def load_moex_prices():

    url = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"

    params = {
        "iss.meta": "off",
        "marketdata.columns": "SECID,LAST,YIELD"
    }

    print("Request MOEX prices")

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    prices = {}

    for row in data["marketdata"]["data"]:

        secid = row[0]
        price = row[1]
        ytm = row[2]

        prices[secid] = {
            "price": price,
            "ytm": ytm
        }

    return prices


# -----------------------
# собрать цены нужных бумаг
# -----------------------

def get_prices():

    bonds = load_bonds()
    moex = load_moex_prices()
    print("TOTAL bonds:", len(bonds))
    print("MOEX prices:", len(moex))
    result = []

    for bond in bonds:

        isin = bond["ISIN"]
        data = moex.get(isin)

        price = data["price"] if data else None
        ytm = data["ytm"] if data else None

        result.append({
            "ISIN": isin,
            "price": price,
            "target": bond.get("SellPrice"),
            "ytm": ytm
        })  
    
    missing = [b["ISIN"] for b in bonds if b["ISIN"] not in moex]
    print("Missing prices:", len(missing))
    print("Sample missing:", missing[:10])
    return result

def rating_ok(rating):
    allowed = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-"]
    return rating in allowed

def find_alternatives(current_bond, bonds, moex):

    results = []

    current_data = moex.get(current_bond["ISIN"])
    if not current_data:
        return []

    current_ytm = current_data.get("ytm")

    if not current_ytm:
        return []

    try:
        current_ytm = float(current_ytm)
    except:
        return []

    for bond in bonds:

        isin = bond["ISIN"]

        # не сравниваем саму с собой
        if isin == current_bond["ISIN"]:
            continue

        # фильтр рейтинга
        if not rating_ok(bond.get("Рейтинг")):
            continue

        data = moex.get(isin)
        if not data:
            continue

        ytm = data.get("ytm")
        price = data.get("price")

        if ytm is None or price is None:
            continue

        try:
            ytm = float(ytm)
        except:
            continue

        # 🔥 ключевая логика
        if ytm > current_ytm + 1:   # +1% лучше
            results.append({
                "ISIN": isin,
                "ytm": ytm,
                "price": price,
                "rating": bond.get("Рейтинг"),
                "name": bond.get("Название")
            })

    # сортировка по доходности
    results.sort(key=lambda x: x["ytm"], reverse=True)

    return results[:3]


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
        text += f"YTM: {bond['ytm']}\n\n"

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

    for bond in bonds:

        isin = bond["ISIN"]
        name = bond.get("Название")

        sell_target = bond.get("SellPrice")
        buy_target = bond.get("BuyPrice")

        data = moex.get(isin)

        if not data:
            continue

        price = data.get("price")
        ytm = data.get("ytm")

        print(isin, "price:", price, "ytm:", ytm,
              "sell:", sell_target, "buy:", buy_target)

        # проверка данных
        if price is None:
            continue

        try:
            price = float(price)
        except:
            continue

        try:
            if sell_target is not None:
                sell_target = float(sell_target)
        except:
            sell_target = None

        try:
            if buy_target is not None:
                buy_target = float(buy_target)
        except:
            buy_target = None

        # ---------------- SELL SIGNAL ----------------

        if sell_target is not None:

            if price >= sell_target and isin not in sent_signals:

                text = f"""
SELL SIGNAL

{isin}
name: {name}
price: {price}
target: {sell_target}
YTM: {ytm}
"""

                # 🔥 ищем альтернативы
                alts = find_alternatives(bond, bonds, moex)

                if alts:
                    text += "\n🔎 Better alternatives:\n"

                    for alt in alts:
                        text += (
                            f"\n{alt['ISIN']}"
                            f"\nYTM: {alt['ytm']}%"
                            f"\nprice: {alt['price']}"
                            f"\nrating: {alt['rating']}\n"
                        )
                else:
                    text += "\n❗ No better alternatives found\n"

                print("SELL SIGNAL:", isin)

                for chat_id in subscribers:
                    await context.bot.send_message(chat_id=chat_id, text=text)

                sent_signals.add(isin)

            elif price < sell_target and isin in sent_signals:

                print("Reset SELL signal for", isin)
                sent_signals.remove(isin)

        # ---------------- BUY SIGNAL ----------------

        if buy_target is not None:

            if price <= buy_target and isin not in sent_buy_signals:

                text = f"""
BUY SIGNAL

{isin}
name: {name}
price: {price}
buy target: {buy_target}
YTM: {ytm}
"""

                print("BUY SIGNAL:", isin)

                for chat_id in subscribers:
                    await context.bot.send_message(chat_id=chat_id, text=text)

                sent_buy_signals.add(isin)

            elif price > buy_target and isin in sent_buy_signals:

                print("Reset BUY signal for", isin)
                sent_buy_signals.remove(isin)
                
# -----------------------
# функция отчёта
# -----------------------

async def send_report(context):

    print("Generate report")

    bonds = load_bonds()
    moex = load_moex_prices()

    rows = []

    for bond in bonds:

        isin = bond["ISIN"]
        data = moex.get(isin)

        price = data["price"] if data else None
        ytm = data["ytm"] if data else None

        avg_price = bond.get("Средняя цена")

        income = None

        try:
            if price is not None and avg_price is not None:

                # 🔥 чистим avg_price
                if isinstance(avg_price, str):
                    avg_price = avg_price.replace(",", ".").strip()

                avg_price = float(avg_price)
                price = float(price)

                income = price - (avg_price / 10)

                income = round(income, 2)

        except Exception as e:
            print("Income calc error:", isin, avg_price, e)
            income = None

        rows.append({
            "Название": bond.get("Название"),
            "ISIN": isin,
            "Цена": price,
            "YTM": ytm,
            "Продать не ниже, в %": bond.get("SellPrice"),
            "Купить не выше": bond.get("BuyPrice"),
            "Депозиты Банка": bond.get("Депозиты Банка"),
            "Фио": bond.get("Фио"),
            "Рейтинг": bond.get("Рейтинг"),
            "Кол-во обл": bond.get("Кол-во обл"),
            "Средняя цена": avg_price,
            # "ТКД": bond.get("ТКД"),
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

    app.job_queue.scheduler.configure(
    job_defaults={
        "coalesce": True,
        "max_instances": 1
    }
)    
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
