"""
searcher.py — поиск сайта МО.

Стратегия:
1. Яндекс Cloud Search API — официальный, без капчи, 10к запросов/день
2. 2GIS Catalog API — если есть сайт в карточке
3. Перебор типовых URL (латиница)
"""
import re
import time
import base64
import logging
import requests
import xml.etree.ElementTree as ET

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from urllib.parse import urlparse
from config import (
    FETCH_TIMEOUT_S, FETCH_DELAY_S,
    DGIS_API_KEY,
    YANDEX_SEARCH_KEY, YANDEX_SEARCH_FOLDER,
)

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

BLACKLIST_DOMAINS = {
    "prodoctorov.ru", "napopravku.ru", "docdoc.ru", "zoon.ru",
    "yell.ru", "2gis.ru", "google.com", "yandex.ru", "yandex.com",
    "gosuslugi.ru", "mos.ru", "rosminzdrav.ru", "egisz.ru",
    "wikipedia.org", "vk.com", "ok.ru", "hh.ru", "avito.ru",
    "t.me", "telegram.me", "rutube.ru", "youtube.com",
    "bus.gov.ru", "zakupki.gov.ru", "nalog.ru", "rusprofile.ru",
    "list-org.com", "kartoteka.ru", "sbis.ru", "kontur.ru",
}

MO_SITE_SIGNALS = [
    r"врач", r"расписание", r"запись", r"поликлиник",
    r"больниц", r"медицин", r"специалист", r"прием",
    r"стационар", r"консультация",
]

CITY_SLUGS = {
    "Москва": "moscow", "Санкт-Петербург": "spb",
    "Новосибирск": "nsk", "Екатеринбург": "ekb",
    "Казань": "kazan", "Нижний Новгород": "nnov",
    "Челябинск": "chel", "Самара": "samara",
    "Уфа": "ufa", "Ростов-на-Дону": "rostov",
    "Ульяновск": "ulyanovsk", "Омск": "omsk",
    "Красноярск": "krsk", "Воронеж": "voronezh",
    "Пермь": "perm", "Волгоград": "volgograd",
    "Краснодар": "krasnodar", "Саратов": "saratov",
    "Тюмень": "tyumen", "Тольятти": "tlt",
    "Ижевск": "izhevsk", "Барнаул": "barnaul",
    "Иркутск": "irkutsk", "Хабаровск": "khabarovsk",
    "Ярославль": "yaroslavl", "Владивосток": "vlad",
    "Томск": "tomsk", "Оренбург": "orenburg",
    "Кемерово": "kemerovo", "Рязань": "ryazan",
    "Астрахань": "astrakhan", "Пенза": "penza",
    "Тула": "tula", "Киров": "kirov",
}


def _is_blacklisted(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(bl in domain for bl in BLACKLIST_DOMAINS)
    except Exception:
        return True


def _looks_like_mo_site(html: str) -> bool:
    text = html.lower()
    return sum(1 for s in MO_SITE_SIGNALS if re.search(s, text)) >= 2


def _validate_url(url: str) -> str | None:
    if not url or _is_blacklisted(url):
        return None
    try:
        head = requests.head(
            url, headers=HEADERS, timeout=6,
            allow_redirects=True, verify=False,
        )
        final_url = head.url
        if _is_blacklisted(final_url):
            return None
        get = requests.get(
            final_url, headers=HEADERS,
            timeout=FETCH_TIMEOUT_S, verify=False,
        )
        if _looks_like_mo_site(get.text):
            return final_url
    except Exception:
        pass
    return None


def _normalize_mo_name(name: str) -> str:
    prefixes = [
        r"ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ (УЧРЕЖДЕНИЕ|НАУЧНОЕ УЧРЕЖДЕНИЕ|ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ)",
        r"(КРАЕВОЕ|ОБЛАСТНОЕ|ГОРОДСКОЕ|РАЙОННОЕ) ГОСУДАРСТВЕННОЕ (БЮДЖЕТНОЕ|КАЗЁННОЕ|КАЗЕННОЕ|АВТОНОМНОЕ) УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"ГОСУДАРСТВЕННОЕ (БЮДЖЕТНОЕ|КАЗЁННОЕ|КАЗЕННОЕ|АВТОНОМНОЕ) УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"ГОСУДАРСТВЕННОЕ УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"МУНИЦИПАЛЬНОЕ (БЮДЖЕТНОЕ|КАЗЁННОЕ|АВТОНОМНОЕ) УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"МУНИЦИПАЛЬНОЕ УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"БЮДЖЕТНОЕ УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"АВТОНОМНОЕ УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ",
        r"(КРАЕВОЕ|ОБЛАСТНОЕ) ГОСУДАРСТВЕННОЕ (БЮДЖЕТНОЕ|АВТОНОМНОЕ|КАЗЕННОЕ) УЧРЕЖДЕНИЕ",
    ]
    result = name.strip()
    for p in prefixes:
        result = re.sub(p, "", result, flags=re.IGNORECASE).strip()
    result = result.strip('"\'«»').strip()
    return re.sub(r"\s+", " ", result) or name


# ── Уровень 1: Яндекс Cloud Search API ───────────────────────

def _yandex_search(query: str) -> list[str]:
    """POST → base64 XML → список URL."""
    if not YANDEX_SEARCH_KEY or YANDEX_SEARCH_KEY == "ВАША_API_КЛЮЧ":
        return []
    try:
        time.sleep(1.0)
        resp = requests.post(
            "https://searchapi.api.cloud.yandex.net/v2/web/search",
            headers={
                "Authorization": f"Api-Key {YANDEX_SEARCH_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "query": {
                    "searchType": "SEARCH_TYPE_RU",
                    "queryText":  query,
                },
                "folderId": YANDEX_SEARCH_FOLDER,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning(f"Яндекс HTTP {resp.status_code}")
            return []

        raw_b64 = resp.json().get("rawData", "")
        if not raw_b64:
            return []

        root = ET.fromstring(base64.b64decode(raw_b64).decode("utf-8"))
        return [doc.findtext("url") for doc in root.iter("doc") if doc.findtext("url")]

    except Exception as e:
        log.warning(f"Яндекс ошибка: {e}")
        return []


# ── Уровень 2: 2GIS Catalog API ──────────────────────────────

def _dgis_search(mo_name: str, city: str) -> str | None:
    if not DGIS_API_KEY or DGIS_API_KEY == "ВАШ_КЛЮЧ_2GIS":
        return None
    try:
        time.sleep(0.5)
        resp = requests.get(
            "https://catalog.api.2gis.com/3.0/items",
            params={
                "key":       DGIS_API_KEY,
                "q":         f"{mo_name} {city}",
                "fields":    "items.contact_groups",
                "page_size": 5,
                "locale":    "ru_RU",
            },
            timeout=10,
        )
        resp.raise_for_status()
        for item in resp.json().get("result", {}).get("items", []):
            url = item.get("url", "").strip()
            if url and not _is_blacklisted(url):
                return url if url.startswith("http") else "https://" + url
            for group in item.get("contact_groups", []):
                for c in group.get("contacts", []):
                    if c.get("type") == "website":
                        val = c.get("value", "").strip()
                        if val and not _is_blacklisted(val):
                            return val if val.startswith("http") else "https://" + val
    except Exception as e:
        log.warning(f"2GIS ошибка: {e}")
    return None


# ── Уровень 3: перебор типовых URL ───────────────────────────

def _guess_urls(short_name: str, city: str, region: str) -> list[str]:
    num_match = re.search(r"№\s*(\d+)", short_name)
    num = num_match.group(1) if num_match else ""

    abbr_map = {
        "городская поликлиника":         f"gp{num}",
        "детская поликлиника":            f"dp{num}",
        "городская больница":             f"gb{num}",
        "городская клиническая больница": f"gkb{num}",
        "областная клиническая больница": "okb",
        "областная больница":             f"ob{num}",
        "детская больница":               f"db{num}",
        "центральная районная больница":  "crb",
        "психиатрическая больница":       f"pb{num}",
        "онкологический диспансер":       "onko",
        "кожно-венерологический":         "kvd",
        "противотуберкулёзный":           "ptd",
        "медико-санитарная часть":        f"msch{num}",
    }

    name_lower = short_name.lower()
    prefix = next((abbr for kw, abbr in abbr_map.items() if kw in name_lower), "")
    city_slug = CITY_SLUGS.get(city) or CITY_SLUGS.get(region, "")

    if not prefix or not city_slug:
        return []

    return [
        f"https://{prefix}.{city_slug}.ru",
        f"https://{prefix}{city_slug}.ru",
        f"https://{prefix}-{city_slug}.ru",
    ]


# ── Главная функция ────────────────────────────────────────────

def find_mo_site(
    mo_id: str, mo_name: str, inn: str, ogrn: str,
    city: str, region: str = "",
) -> str | None:
    short_name = _normalize_mo_name(mo_name)
    query = f"{short_name} {city} официальный сайт"
    log.debug(f"[{mo_id}] Ищем: '{short_name}' / {city}")

    # Уровень 1: Яндекс
    for url in _yandex_search(query):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ Яндекс: {validated}")
            return validated

    # Уровень 2: 2GIS
    site = _dgis_search(short_name, city)
    if site:
        validated = _validate_url(site)
        result = validated or site
        log.info(f"[{mo_id}] ✓ 2GIS: {result}")
        return result

    # Уровень 3: URL перебор
    for url in _guess_urls(short_name, city, region):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ Перебор: {validated}")
            return validated

    log.info(f"[{mo_id}] ✗ Не найден: {short_name}")
    return None
