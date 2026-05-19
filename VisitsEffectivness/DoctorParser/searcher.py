"""
searcher.py — поиск сайта МО.

Стратегия:
1. 2GIS Catalog API — официальный, без капчи, 25к запросов/день
2. Перебор типовых URL (латиница, паттерны гос. сайтов)
"""
import re
import time
import logging
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config import FETCH_TIMEOUT_S, FETCH_DELAY_S, DGIS_API_KEY

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
    "mos.ru", "rosminzdrav.ru", "egisz.ru", "gosuslugi.ru",
    "wikipedia.org", "vk.com", "ok.ru", "hh.ru", "avito.ru",
    "bus.gov.ru", "zakupki.gov.ru", "nalog.ru", "rusprofile.ru",
    "list-org.com", "kartoteka.ru", "sbis.ru", "kontur.ru",
}

MO_SITE_SIGNALS = [
    r"врач", r"расписание", r"запись", r"поликлиник",
    r"больниц", r"медицин", r"специалист", r"прием",
    r"стационар", r"консультация",
]

# Латинские slug-и городов для URL-перебора
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
    "Махачкала": "makhachkala", "Томск": "tomsk",
    "Оренбург": "orenburg", "Кемерово": "kemerovo",
    "Рязань": "ryazan", "Астрахань": "astrakhan",
    "Пенза": "penza", "Липецк": "lipetsk",
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
    """Проверяет что URL доступен и похож на сайт МО."""
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
    """Убирает казённые префиксы типа 'ГБУЗ', оставляет суть."""
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


# ── Уровень 1: 2GIS Catalog API ──────────────────────────────

DGIS_URL = "https://catalog.api.2gis.com/3.0/items"

def _dgis_search(mo_name: str, city: str) -> str | None:
    """
    Ищет МО в 2GIS и возвращает URL сайта из карточки организации.
    Самый надёжный источник — данные актуальные, API официальный.
    """
    if not DGIS_API_KEY or DGIS_API_KEY == "ВАШ_КЛЮЧ_2GIS":
        log.debug("2GIS API ключ не задан")
        return None

    try:
        time.sleep(0.5)
        params = {
            "key":       DGIS_API_KEY,
            "q":         f"{mo_name} {city}",
            "fields":    "items.contact_groups",
            "page_size": 5,
            "locale":    "ru_RU",
        }
        resp = requests.get(DGIS_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("result", {}).get("items", [])

        for item in items:
            # Прямой URL в карточке
            url = item.get("url", "").strip()
            if url and not _is_blacklisted(url):
                return url if url.startswith("http") else "https://" + url

            # URL в контактах
            for group in item.get("contact_groups", []):
                for c in group.get("contacts", []):
                    if c.get("type") == "website":
                        val = c.get("value", "").strip()
                        if val and not _is_blacklisted(val):
                            return val if val.startswith("http") else "https://" + val

    except Exception as e:
        log.warning(f"2GIS API ошибка: {e}")

    return None


# ── Уровень 2: перебор типовых URL (латиница) ─────────────────

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

    # Ищем slug города (латиница)
    city_slug = CITY_SLUGS.get(city) or CITY_SLUGS.get(region, "")
    if not city_slug:
        # Простая транслитерация если города нет в словаре
        tr = str.maketrans("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                           "abvgdeyozhziyklmnoprstufhtsчshschyeyuya")
        city_slug = city.lower().translate(tr)[:10]

    if not prefix or not city_slug:
        return []

    return [
        f"https://{prefix}.{city_slug}.ru",
        f"https://{prefix}{city_slug}.ru",
        f"https://{prefix}-{city_slug}.ru",
        f"http://{prefix}.{city_slug}.ru",
    ]


# ── Главная функция ────────────────────────────────────────────

def find_mo_site(
    mo_id: str, mo_name: str, inn: str, ogrn: str,
    city: str, region: str = "",
) -> str | None:
    short_name = _normalize_mo_name(mo_name)
    log.debug(f"[{mo_id}] Ищем: '{short_name}' / {city}")

    # Уровень 1: 2GIS API
    site = _dgis_search(short_name, city)
    if site:
        validated = _validate_url(site)
        result = validated or site  # доверяем 2GIS даже без валидации контента
        log.info(f"[{mo_id}] ✓ 2GIS: {result}")
        return result

    # Уровень 2: перебор типовых URL
    for url in _guess_urls(short_name, city, region):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ Перебор URL: {validated}")
            return validated

    log.info(f"[{mo_id}] ✗ Не найден: {short_name}")
    return None
