
import os
import time
import math
import zipfile
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO

import pandas as pd
import requests
from telegram.ext import ApplicationBuilder, CommandHandler

# =======================
# CONFIG
# =======================

TOKEN = "7464147832:AAE7X1tcc-ca9RZgca-2UCVYGNI5adOIBLg"
PUBLIC_KEY = "https://disk.yandex.ru/i/4ow9EC89R_7ADw"

# Optional: file with market ratings for ALL bonds, format: ISIN,Rating
MARKET_RATINGS_FILE = os.getenv("MARKET_RATINGS_FILE", "market_ratings.csv")

# Bot checks
INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))

# Cache TTLs
CACHE_TTL = 300          # portfolio Excel cache
MARKET_CACHE_TTL = 900   # market data cache
REQUEST_TIMEOUT = 15

# Scheduler report times (server local time)
REPORT_TIMES = [
    datetime.time(hour=9, minute=0),
    datetime.time(hour=15, minute=0),
]

# Business logic thresholds
MIN_YTM_ADVANTAGE = 0.8         # minimum YTM advantage in p.p.
MAX_YEARS_DIFF = 1.0            # max difference vs current bond by offer/maturity years
MIN_ALT_TRADES = 5              # minimum trades for alternatives
MIN_ANY_TRADES = 1              # minimum trades for general market universe

# Only allow ratings A- and above
RATING_SCALE = {
    "AAA": 8,
    "AA+": 7,
    "AA": 6,
    "AA-": 5,
    "A+": 4,
    "A": 3,
    "A-": 2,
    "BBB+": 1,
    "BBB": 0,
}
MIN_RATING_SCORE = RATING_SCALE["A-"]

# Globals
subscribers = set()
sent_signals = set()
sent_buy_signals = set()

_cache = {
    "df": None,
    "timestamp": 0
}

_market_cache = {
    "df": None,
    "timestamp": 0
}


async def error_handler(update, context):
    print("Telegram error:", repr(context.error))
    try:
        if update and getattr(update, "effective_chat", None):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Ошибка при выполнении команды: {context.error}"[:4000]
            )
    except Exception as e:
        print("Failed to notify user about error:", e)


# =======================
# HELPERS
# =======================
def format_offer_date(value):
    if value is None or value == "":
        return None

    if isinstance(value, datetime.datetime):
        return value.date()

    if isinstance(value, datetime.date):
        return value

    try:
        s = str(value).strip().replace(",", ".")
        if s.replace(".", "", 1).isdigit():
            serial = float(s)
            if serial > 1000:  # похоже на Excel serial date
                base_date = datetime.datetime(1899, 12, 30)
                return (base_date + datetime.timedelta(days=serial)).date()
    except:
        pass

    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()
    except:
        return None
    
def safe_float(x):
    if x is None or x == "":
        return None
    try:
        if isinstance(x, str):
            x = x.replace("%", "").replace("\xa0", "").replace(" ", "").replace(",", ".").strip()
        return float(x)
    except Exception:
        return None


def normalize_text(x):
    if x is None:
        return None
    return str(x).strip()


def normalize_rating(rating):
    if rating is None:
        return None
    rating = str(rating).strip().upper()
    # sometimes agencies add outlook or extra text
    for base in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB"]:
        if rating.startswith(base):
            return base
    return rating


def rating_score(rating):
    return RATING_SCALE.get(normalize_rating(rating))


def rating_not_lower_than_a_minus(rating):
    score = rating_score(rating)
    return score is not None and score >= MIN_RATING_SCORE


def parse_date_any(x):
    if x is None or x == "" or (isinstance(x, float) and pd.isna(x)):
        return None

    if isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp)):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return None

    try:
        # Excel serial date
        if isinstance(x, (int, float)) and not pd.isna(x):
            dt = pd.to_datetime("1899-12-30") + pd.to_timedelta(float(x), unit="D")
            return dt.date()
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return None

    # first day=False (Russian style), then fallback
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(s, errors="raise", dayfirst=dayfirst).date()
        except Exception:
            continue

    return None


def years_to_event_from_dates(offer_date, mat_date):
    today = datetime.date.today()
    event_date = offer_date or mat_date
    if not event_date:
        return None

    days = (event_date - today).days
    if days < 0:
        return None

    return round(days / 365.25, 2)


def excel_date_or_none(x):
    d = parse_date_any(x)
    if d is None:
        return None
    return datetime.datetime.combine(d, datetime.time.min)


# =======================
# EXCEL PARSER
# =======================

def read_excel_safe(content):
    """
    Fallback XLSX parser through XML.
    Reads first worksheet.
    """
    try:
        file_bytes = BytesIO(content)

        with zipfile.ZipFile(file_bytes) as z:
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

            shared_strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                sst_xml = z.read("xl/sharedStrings.xml")
                sst_root = ET.fromstring(sst_xml)
                for si in sst_root.findall(".//a:t", ns):
                    shared_strings.append(si.text if si.text is not None else "")

            sheet_files = [f for f in z.namelist() if "xl/worksheets/sheet" in f and f.endswith(".xml")]
            if not sheet_files:
                return pd.DataFrame()

            sheet_xml = z.read(sheet_files[0])
            root = ET.fromstring(sheet_xml)

            rows = []
            for row in root.findall(".//a:sheetData/a:row", ns):
                row_data = {}
                for cell in row.findall("a:c", ns):
                    cell_ref = cell.attrib.get("r", "")
                    col = ''.join(filter(str.isalpha, cell_ref))
                    t = cell.attrib.get("t")
                    v = cell.find("a:v", ns)

                    value = None
                    if v is not None and v.text is not None:
                        if t == "s":
                            idx = int(v.text)
                            value = shared_strings[idx] if 0 <= idx < len(shared_strings) else None
                        else:
                            value = v.text

                    row_data[col] = value
                rows.append(row_data)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            # First row = headers
            df.columns = df.iloc[0]
            df = df[1:]

            df.columns = [
                str(c).strip().replace("\n", " ").replace("\r", "")
                for c in df.columns
            ]

            return df.reset_index(drop=True)

    except Exception as e:
        print("❌ XML Excel parser failed:", e)
        return pd.DataFrame()


# =======================
# PORTFOLIO LOAD FROM YANDEX DISK
# =======================

def load_bonds_from_yadisk():
    if _cache["df"] is not None and time.time() - _cache["timestamp"] < CACHE_TTL:
        return _cache["df"]

    try:
        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        params = {"public_key": PUBLIC_KEY}

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = response.json()

        if "href" not in data:
            print("❌ Yandex error:", data)
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        download_url = data["href"]
        file = requests.get(download_url, timeout=REQUEST_TIMEOUT)

        if not file.content.startswith(b"PK"):
            print("❌ Not XLSX")
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        df = read_excel_safe(file.content)

        if df is None or df.empty:
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        print("Columns parsed:", df.columns.tolist())
        print("Rows parsed:", len(df))

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

        df = df[[col for col in needed_cols if col in df.columns]]

        if "ISIN" not in df.columns:
            print("❌ Column ISIN not found")
            return _cache["df"] if _cache["df"] is not None else pd.DataFrame()

        df["ISIN"] = df["ISIN"].astype(str).str.strip()

        df = df[
            df["ISIN"].notna() &
            (df["ISIN"] != "") &
            (df["ISIN"] != "-") &
            (df["ISIN"].str.lower() != "none")
        ].copy()

        # normalize some columns
        if "Рейтинг" in df.columns:
            df["Рейтинг"] = df["Рейтинг"].apply(normalize_rating)

        if "Дата оферты" in df.columns:
            df["Дата оферты"] = df["Дата оферты"].apply(parse_date_any)

        print("After ISIN filter:", len(df))

        _cache["df"] = df
        _cache["timestamp"] = time.time()

        return df

    except Exception as e:
        print("Ошибка загрузки:", e)
        return _cache["df"] if _cache["df"] is not None else pd.DataFrame()


def load_bonds():
    df = load_bonds_from_yadisk()
    bonds = []

    if df is None or df.empty:
        return bonds

    for _, row in df.iterrows():
        isin = normalize_text(row.get("ISIN"))
        if not isin:
            continue

        bonds.append({
            "ISIN": isin,
            "Название": row.get("Характеристики Вклада"),
            "SellPrice": row.get("Продать не ниже, в %"),
            "BuyPrice": row.get("Купить не выше"),
            "Рейтинг": normalize_rating(row.get("Рейтинг")),
            "Депозиты Банка": row.get("Депозиты Банка"),
            "Фио": row.get("Фио"),
            "Кол-во обл": row.get("Кол-во обл"),
            "Средняя цена": row.get("Средняя цена"),
            "Дата оферты": parse_date_any(row.get("Дата оферты")),
            "Спекуляции": row.get("Спекуляции"),
        })

    return bonds


# =======================
# MARKET DATA
# =======================

def load_market_ratings(path=MARKET_RATINGS_FILE):
    """
    Expected format:
    ISIN,Rating
    RU000A...,A
    RU000A...,AA-
    """
    try:
        if not os.path.exists(path):
            print(f"⚠️ Rating file not found: {path}")
            return pd.DataFrame(columns=["ISIN", "Rating"])

        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        if "ISIN" not in df.columns or "Rating" not in df.columns:
            print("⚠️ Rating file must contain columns: ISIN, Rating")
            return pd.DataFrame(columns=["ISIN", "Rating"])

        df["ISIN"] = df["ISIN"].astype(str).str.strip()
        df["Rating"] = df["Rating"].apply(normalize_rating)
        df = df[df["ISIN"].notna() & (df["ISIN"] != "")]
        return df[["ISIN", "Rating"]].drop_duplicates(subset=["ISIN"])

    except Exception as e:
        print("Rating file load error:", e)
        return pd.DataFrame(columns=["ISIN", "Rating"])


def fetch_moex_all_bonds():
    """
    Loads all MOEX bond rows page by page and merges securities + marketdata.
    """
    start = 0
    all_rows = []

    while True:
        url = "https://iss.moex.com/iss/engines/stock/markets/bonds/securities.json"
        params = {
            "iss.meta": "off",
            "start": start,
            "securities.columns": ",".join([
                "SECID", "ISIN", "SHORTNAME", "BOARDID",
                "FACEVALUE", "FACEUNIT", "MATDATE", "OFFERDATE",
                "COUPONVALUE", "COUPONPERIOD", "STATUS", "LISTLEVEL"
            ]),
            "marketdata.columns": ",".join([
                "SECID", "LAST", "YIELD", "DURATION",
                "ACCRUEDINT", "MARKETPRICE", "VOLTODAY", "NUMTRADES"
            ])
        }

        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        sec_cols = data.get("securities", {}).get("columns", [])
        sec_data = data.get("securities", {}).get("data", [])

        md_cols = data.get("marketdata", {}).get("columns", [])
        md_data = data.get("marketdata", {}).get("data", [])

        if not sec_data:
            break

        sec_df = pd.DataFrame(sec_data, columns=sec_cols)
        md_df = pd.DataFrame(md_data, columns=md_cols)

        df = sec_df.merge(md_df, on="SECID", how="left")
        all_rows.append(df)

        if len(sec_df) < 100:
            break

        start += 100

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def build_market_universe(force_refresh=False):
    if (
        not force_refresh
        and _market_cache["df"] is not None
        and time.time() - _market_cache["timestamp"] < MARKET_CACHE_TTL
    ):
        return _market_cache["df"]

    try:
        market_df = fetch_moex_all_bonds()
        ratings_df = load_market_ratings()

        if market_df.empty:
            return _market_cache["df"] if _market_cache["df"] is not None else pd.DataFrame()

        market_df["ISIN"] = market_df["ISIN"].astype(str).str.strip()

        if not ratings_df.empty:
            market_df = market_df.merge(ratings_df, on="ISIN", how="left")
        else:
            market_df["Rating"] = None

        for col in ["LAST", "YIELD", "DURATION", "ACCRUEDINT", "MARKETPRICE", "VOLTODAY", "NUMTRADES"]:
            if col in market_df.columns:
                market_df[col] = market_df[col].apply(safe_float)

        market_df["OFFERDATE_DT"] = market_df["OFFERDATE"].apply(parse_date_any)
        market_df["MATDATE_DT"] = market_df["MATDATE"].apply(parse_date_any)
        market_df["years_to_event"] = market_df.apply(
            lambda row: years_to_event_from_dates(row.get("OFFERDATE_DT"), row.get("MATDATE_DT")),
            axis=1
        )

        market_df["Rating"] = market_df["Rating"].apply(normalize_rating)
        market_df["price"] = market_df["LAST"].combine_first(market_df["MARKETPRICE"])
        market_df["ytm"] = market_df["YIELD"]

        # basic filters
        market_df = market_df[
            market_df["ISIN"].notna() &
            (market_df["ISIN"] != "") &
            market_df["price"].notna() &
            market_df["ytm"].notna()
        ].copy()

        # Only ruble bonds if desired
        if "FACEUNIT" in market_df.columns:
            market_df = market_df[market_df["FACEUNIT"].fillna("") == "SUR"].copy()

        # minimal liquidity
        market_df = market_df[
            market_df["NUMTRADES"].fillna(0) >= MIN_ANY_TRADES
        ].copy()

        # Filter by rating A- or above
        market_df = market_df[
            market_df["Rating"].apply(rating_not_lower_than_a_minus)
        ].copy()

        # Deduplicate by ISIN: choose row with most trades and available price
        market_df["sort_trades"] = market_df["NUMTRADES"].fillna(0)
        market_df = market_df.sort_values(by=["ISIN", "sort_trades", "ytm"], ascending=[True, False, False])
        market_df = market_df.drop_duplicates(subset=["ISIN"], keep="first").copy()
        market_df.drop(columns=["sort_trades"], inplace=True, errors="ignore")

        _market_cache["df"] = market_df
        _market_cache["timestamp"] = time.time()

        print("Market universe loaded:", len(market_df))
        return market_df

    except Exception as e:
        print("Market load error:", e)
        return _market_cache["df"] if _market_cache["df"] is not None else pd.DataFrame()


def build_market_map(market_df):
    market_map = {}
    if market_df is None or market_df.empty:
        return market_map

    for _, row in market_df.iterrows():
        isin = row.get("ISIN")
        if not isin:
            continue

        market_map[isin] = {
            "price": row.get("price"),
            "ytm": row.get("ytm"),
            "rating": row.get("Rating"),
            "name": row.get("SHORTNAME"),
            "years_to_event": row.get("years_to_event"),
            "currency": row.get("FACEUNIT"),
            "num_trades": row.get("NUMTRADES"),
            "offer_date": row.get("OFFERDATE_DT"),
            "mat_date": row.get("MATDATE_DT"),
        }

    return market_map


# =======================
# ALTERNATIVE SEARCH
# =======================

def get_current_bond_profile(current_bond, market_df):
    isin = current_bond["ISIN"]
    row = market_df[market_df["ISIN"] == isin]

    if row.empty:
        return None

    row = row.iloc[0].to_dict()

    current_rating = normalize_rating(current_bond.get("Рейтинг")) or normalize_rating(row.get("Rating"))
    current_offer = parse_date_any(current_bond.get("Дата оферты")) or row.get("OFFERDATE_DT")
    current_mat = row.get("MATDATE_DT")
    current_years = years_to_event_from_dates(current_offer, current_mat)

    return {
        "ISIN": isin,
        "name": current_bond.get("Название") or row.get("SHORTNAME"),
        "rating": current_rating,
        "ytm": safe_float(row.get("ytm")),
        "price": safe_float(row.get("price")),
        "years_to_event": current_years if current_years is not None else row.get("years_to_event"),
        "currency": row.get("FACEUNIT"),
        "offer_date": current_offer,
        "mat_date": current_mat,
    }


def compare_comment(current, candidate):
    diff = round(candidate["ytm"] - current["ytm"], 2)
    current_years = current.get("years_to_event")
    cand_years = candidate.get("years_to_event")

    if current_years is not None and cand_years is not None:
        term_comment = f"term {cand_years}y vs {current_years}y"
    else:
        term_comment = "term n/a"

    return f"+{diff} p.p. YTM, {term_comment}"


def find_market_alternatives(current_bond, market_df, top_n=3):
    """
    Search alternatives across ALL market universe.
    Rules:
    - rating >= A-
    - candidate rating not worse than current bond rating
    - same currency
    - comparable term (offer/maturity)
    - YTM advantage >= MIN_YTM_ADVANTAGE
    - enough liquidity
    """
    current = get_current_bond_profile(current_bond, market_df)
    if not current:
        return []

    current_ytm = safe_float(current.get("ytm"))
    if current_ytm is None:
        return []

    current_rating = normalize_rating(current.get("rating"))
    current_rating_score = rating_score(current_rating)
    if current_rating_score is None:
        current_rating_score = MIN_RATING_SCORE

    current_years = current.get("years_to_event")
    current_currency = current.get("currency")

    candidates = market_df.copy()
    candidates = candidates[candidates["ISIN"] != current["ISIN"]]

    if current_currency:
        candidates = candidates[candidates["FACEUNIT"] == current_currency]

    candidates = candidates[
        candidates["Rating"].apply(lambda x: (rating_score(x) or -999) >= current_rating_score)
    ]

    candidates = candidates[
        candidates["NUMTRADES"].fillna(0) >= MIN_ALT_TRADES
    ]

    if current_years is not None:
        candidates = candidates[
            candidates["years_to_event"].notna() &
            (abs(candidates["years_to_event"] - current_years) <= MAX_YEARS_DIFF)
        ]

    candidates = candidates[
        candidates["ytm"].notna() &
        (candidates["ytm"] >= current_ytm + MIN_YTM_ADVANTAGE)
    ]

    candidates = candidates[
        candidates["price"].notna() &
        (candidates["price"] > 50) &
        (candidates["price"] < 130)
    ]

    if candidates.empty:
        return []

    def score_row(row):
        score = row["ytm"]

        num_trades = row.get("NUMTRADES") or 0
        score += min(math.log1p(num_trades), 3) * 0.2

        if current_years is not None and row.get("years_to_event") is not None:
            score -= abs(row["years_to_event"] - current_years) * 0.5

        return score

    candidates = candidates.copy()
    candidates["score"] = candidates.apply(score_row, axis=1)
    candidates = candidates.sort_values(by=["score", "ytm", "NUMTRADES"], ascending=[False, False, False])

    results = []
    for _, row in candidates.head(top_n).iterrows():
        candidate = {
            "ISIN": row.get("ISIN"),
            "name": row.get("SHORTNAME"),
            "ytm": round(float(row.get("ytm")), 2) if row.get("ytm") is not None else None,
            "price": round(float(row.get("price")), 2) if row.get("price") is not None else None,
            "rating": row.get("Rating"),
            "years_to_event": row.get("years_to_event"),
            "num_trades": int(row.get("NUMTRADES")) if row.get("NUMTRADES") is not None else None,
            "comment": None,
        }
        candidate["comment"] = compare_comment(current, candidate)
        results.append(candidate)

    return results


# =======================
# REPORT
# =======================

def calculate_income(current_price, avg_price):
    current_price = safe_float(current_price)
    avg_price = safe_float(avg_price)

    if current_price is None or avg_price is None:
        return None

    # NOTE:
    # Original logic used avg_price / 10.
    # If your average price in source file is already in price points like 101.25,
    # then this division is wrong and should be removed.
    # Keeping your original approach here because it is in your current bot.
    try:
        income = current_price - (avg_price / 10)
        return round(income, 2)
    except Exception:
        return None


async def send_report(context):
    print("Generate report")

    bonds = load_bonds()
    market_df = get_market_universe()

    market_map = {}
    for _, row in market_df.iterrows():
        market_map[row["ISIN"]] = {
            "price": row.get("price"),
            "ytm": row.get("ytm")
        }

    rows = []

    for bond in bonds:
        isin = bond["ISIN"]
        md = market_map.get(isin, {})

        price = md.get("price")
        ytm = md.get("ytm")

        avg_price = bond.get("Средняя цена")
        offer_date = format_offer_date(bond.get("Дата оферты"))

        income = None

        try:
            if price is not None and avg_price is not None:
                price = safe_float(price)
                avg_price = safe_float(avg_price)

                if price is not None and avg_price is not None:
                    income = round(price - (avg_price / 10), 2)
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
            "Дата оферты": offer_date,
            "Спекуляции": bond.get("Спекуляции"),
            "Доход": income
        })

    df = pd.DataFrame(rows)

    if "Дата оферты" in df.columns:
        df["Дата оферты"] = pd.to_datetime(df["Дата оферты"], errors="coerce")

    file = "bond_report.xlsx"

    with pd.ExcelWriter(file, engine="openpyxl", datetime_format="DD.MM.YYYY", date_format="DD.MM.YYYY") as writer:
        df.to_excel(writer, index=False)

        ws = writer.book.active

        for idx, col in enumerate(df.columns, start=1):
            if col == "Дата оферты":
                for row_num in range(2, len(df) + 2):
                    ws.cell(row=row_num, column=idx).number_format = "DD.MM.YYYY"
                break

    print("Report created:", file)

    for chat_id in subscribers:
        with open(file, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=f
            )


# =======================
# COMMANDS
# =======================

async def start(update, context):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    print("New subscriber:", chat_id)
    await update.message.reply_text("Bond bot started")


async def price(update, context):
    print("/price command")
    await update.message.reply_text("Загружаю цены...")

    bonds = load_bonds()
    market_df = build_market_universe()
    market_map = build_market_map(market_df)

    if not bonds:
        await update.message.reply_text("Portfolio is empty")
        return

    parts = ["Bond prices:\n"]
    for bond in bonds:
        isin = bond["ISIN"]
        data = market_map.get(isin, {})
        price_val = data.get("price")
        ytm_val = data.get("ytm")

        parts.append(
            f"{isin}\n"
            f"price: {price_val}\n"
            f"target sell: {bond.get('SellPrice')}\n"
            f"target buy: {bond.get('BuyPrice')}\n"
            f"YTM: {ytm_val}\n"
        )

    text = "\n".join(parts)
    if len(text) > 3900:
        # Split long messages
        chunk = ""
        for line in parts:
            if len(chunk) + len(line) + 1 > 3900:
                await update.message.reply_text(chunk)
                chunk = line + "\n"
            else:
                chunk += line + "\n"
        if chunk:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(text)


async def report(update, context):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    await update.message.reply_text("Формирую отчет...")
    await send_report(context, target_chat_ids=[chat_id])


# =======================
# MONITOR
# =======================

async def monitor(context):
    print("Check prices")

    bonds = load_bonds()
    if not bonds:
        print("No bonds loaded")
        return

    market_df = build_market_universe()
    if market_df is None or market_df.empty:
        print("No market data loaded")
        return

    market_map = build_market_map(market_df)

    for bond in bonds:
        isin = bond["ISIN"]
        name = bond.get("Название")

        sell_target = safe_float(bond.get("SellPrice"))
        buy_target = safe_float(bond.get("BuyPrice"))

        data = market_map.get(isin)
        if not data:
            print("No market data for:", isin)
            continue

        price = safe_float(data.get("price"))
        ytm = safe_float(data.get("ytm"))

        print(isin, "price:", price, "ytm:", ytm, "sell:", sell_target, "buy:", buy_target)

        if price is None:
            continue

        # ---------------- SELL SIGNAL ----------------
        if sell_target is not None:
            if price >= sell_target and isin not in sent_signals:
                text = (
                    f"SELL SIGNAL\n\n"
                    f"{isin}\n"
                    f"name: {name}\n"
                    f"price: {price}\n"
                    f"target: {sell_target}\n"
                    f"YTM: {ytm}\n"
                )

                alts = find_market_alternatives(bond, market_df, top_n=3)

                if alts:
                    text += "\n🔎 Better market alternatives:\n"
                    for alt in alts:
                        text += (
                            f"\n{alt['ISIN']}"
                            f"\n{alt['name']}"
                            f"\nYTM: {alt['ytm']}%"
                            f"\nprice: {alt['price']}"
                            f"\nrating: {alt['rating']}"
                            f"\nyears: {alt['years_to_event']}"
                            f"\ntrades: {alt['num_trades']}"
                            f"\nreason: {alt['comment']}\n"
                        )
                else:
                    text += "\n❗ No suitable market alternatives found\n"

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
                text = (
                    f"BUY SIGNAL\n\n"
                    f"{isin}\n"
                    f"name: {name}\n"
                    f"price: {price}\n"
                    f"buy target: {buy_target}\n"
                    f"YTM: {ytm}\n"
                )

                print("BUY SIGNAL:", isin)

                for chat_id in subscribers:
                    await context.bot.send_message(chat_id=chat_id, text=text)

                sent_buy_signals.add(isin)

            elif price > buy_target and isin in sent_buy_signals:
                print("Reset BUY signal for", isin)
                sent_buy_signals.remove(isin)


# =======================
# MAIN
# =======================

def main():
    if TOKEN == "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        raise ValueError("Set TELEGRAM_BOT_TOKEN env var or TOKEN in code")

    if PUBLIC_KEY == "PUT_YOUR_YANDEX_PUBLIC_KEY_HERE":
        raise ValueError("Set YANDEX_PUBLIC_KEY env var or PUBLIC_KEY in code")

    app = ApplicationBuilder().token(TOKEN).build()

    if app.job_queue is None:
        raise RuntimeError("JobQueue is not available. Install: python-telegram-bot[job-queue]")

    app.job_queue.scheduler.configure(
        job_defaults={
            "coalesce": True,
            "max_instances": 1
        }
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("report", report))
    app.add_error_handler(error_handler)

    app.job_queue.run_repeating(
        monitor,
        interval=INTERVAL,
        first=20
    )

    for t in REPORT_TIMES:
        app.job_queue.run_daily(send_report, time=t)

    print("BOT STARTED")
    app.run_polling()


if __name__ == "__main__":
    main()
