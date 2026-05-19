"""
searcher.py — поиск сайта МО.

Стратегия (4 уровня):
1. DuckDuckGo (duckduckgo-search) — основной, без капчи и блокировок
2. googlesearch-python — запасной, Google без капчи
3. bus.gov.ru API — государственный реестр учреждений с официальными сайтами
4. Перебор типовых URL по названию МО + региону
"""
import re
import time
import logging
import requests

# Корпоративный SSL-прокси — отключаем верификацию сертификата
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config import SEARCH_DELAY_S, FETCH_TIMEOUT_S

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
    "reformagkh.ru", "gisgkh.ru", "duckduckgo.com",
}

MO_SITE_SIGNALS = [
    r"врач", r"расписание", r"запись", r"поликлиник",
    r"больниц", r"медицин", r"специалист", r"прием",
    r"стационар", r"консультация",
]


def _is_blacklisted(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(bl in domain for bl in BLACKLIST_DOMAINS)
    except Exception:
        return True


def _looks_like_mo_site(html: str) -> bool:
    text = html.lower()
    return sum(1 for s in MO_SITE_SIGNALS if re.search(s, text)) >= 2


def _normalize_mo_name(name: str) -> str:
    """Убирает казённые префиксы, оставляет суть."""
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
    result = re.sub(r"\s+", " ", result)
    return result if result else name


def _validate_url(url: str) -> str | None:
    """Проверяет что URL доступен и похож на сайт МО."""
    if _is_blacklisted(url):
        return None
    try:
        head = requests.head(
            url, headers=HEADERS, timeout=6, allow_redirects=True, verify=False
        )
        final_url = head.url
        if _is_blacklisted(final_url):
            return None
        get = requests.get(
            final_url, headers=HEADERS, timeout=FETCH_TIMEOUT_S, verify=False
        )
        if _looks_like_mo_site(get.text):
            return final_url
    except Exception:
        pass
    return None


# ── Уровень 1: DuckDuckGo (основной) ─────────────────────────

def _ddg_search(query: str) -> list[str]:
    """
    DuckDuckGo через duckduckgo-search — не блокирует, не требует ключа.
    pip install duckduckgo-search
    """
    try:
        from duckduckgo_search import DDGS
        time.sleep(SEARCH_DELAY_S)
        results = DDGS().text(query, max_results=5, region="ru-ru")
        return [r["href"] for r in results if r.get("href")]
    except ImportError:
        log.debug("duckduckgo-search не установлен")
        return []
    except Exception as e:
        log.warning(f"DuckDuckGo ошибка: {e}")
        return []


# ── Уровень 2: googlesearch-python (запасной) ─────────────────

def _google_search(query: str) -> list[str]:
    try:
        from googlesearch import search
        time.sleep(SEARCH_DELAY_S)
        results = list(search(query, num_results=5, lang="ru", sleep_interval=1))
        return results
    except ImportError:
        log.debug("googlesearch-python не установлен")
        return []
    except Exception as e:
        log.warning(f"googlesearch ошибка: {e}")
        return []


# ── Уровень 3: bus.gov.ru API ─────────────────────────────────

BUS_GOV_URL = "https://bus.gov.ru/pub/agency/search.json"

def _bus_gov_search(mo_name: str, inn: str, ogrn: str) -> str | None:
    """
    Поиск через bus.gov.ru — реестр государственных и муниципальных
    учреждений РФ. Содержит официальные сайты МО. Открытый API.
    """
    time.sleep(1.0)

    for query_type, query_val in [("inn", inn), ("ogrn", ogrn), ("name", mo_name[:60])]:
        if not query_val:
            continue
        try:
            params = {
                "searchString": query_val,
                "page":         0,
                "size":         5,
                "oktmoCode":    "",
            }
            resp = requests.get(
                BUS_GOV_URL,
                params=params,
                headers=HEADERS,
                timeout=10,
                verify=False,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            agencies = data.get("agencies") or data.get("data") or []
            for ag in agencies:
                site = ag.get("site") or ag.get("siteUrl") or ag.get("webSite") or ""
                site = site.strip()
                if site and not site.startswith("http"):
                    site = "https://" + site
                if site and not _is_blacklisted(site):
                    log.debug(f"bus.gov.ru нашёл: {site}")
                    return site
        except Exception as e:
            log.debug(f"bus.gov.ru ошибка ({query_type}): {e}")

    return None


# ── Уровень 4: перебор типовых URL ───────────────────────────

def _guess_urls(short_name: str, region: str, city: str) -> list[str]:
    candidates = []

    num_match = re.search(r"№\s*(\d+)", short_name)
    num = num_match.group(1) if num_match else ""

    abbr_map = {
        "городская поликлиника":          f"gp{num}",
        "детская поликлиника":             f"dp{num}",
        "городская больница":              f"gb{num}",
        "городская клиническая больница":  f"gkb{num}",
        "областная клиническая больница":  "okb",
        "областная больница":              f"ob{num}",
        "детская больница":                f"db{num}",
        "центральная районная больница":   "crb",
        "психиатрическая больница":        f"pb{num}",
        "онкологический диспансер":        "onko",
        "кожно-венерологический":          "kvd",
        "противотуберкулёзный":            "ptd",
        "станция скорой":                  "ssmp",
        "медико-санитарная часть":         f"msch{num}",
    }

    name_lower = short_name.lower()
    prefix = ""
    for keyword, abbr in abbr_map.items():
        if keyword in name_lower:
            prefix = abbr
            break

    city_clean = re.sub(r"\s+", "", (city or region or "").lower())[:12]

    tld_variants = [
        f"https://{prefix}.{city_clean}.ru",
        f"https://{prefix}{city_clean}.ru",
        f"http://{prefix}.{city_clean}.ru",
    ] if prefix and city_clean else []

    words = re.findall(r"[а-яё]+", name_lower)
    if len(words) >= 2:
        slug = words[0][:4] + words[1][:4]
        tld_variants += [
            f"https://{slug}.{city_clean}.ru",
            f"https://{slug}{num}.{city_clean}.ru",
        ]

    return [u for u in tld_variants if len(u) > 12]


# ── Главная функция ────────────────────────────────────────────

def find_mo_site(
    mo_id: str, mo_name: str, inn: str, ogrn: str,
    city: str, region: str = "",
) -> str | None:
    short_name = _normalize_mo_name(mo_name)
    query = f"{short_name} {city} официальный сайт".strip()
    log.debug(f"[{mo_id}] Ищем: '{short_name}'")

    # Уровень 1: DuckDuckGo
    for url in _ddg_search(query):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ DuckDuckGo: {validated}")
            return validated

    # Уровень 2: Google (запасной)
    for url in _google_search(query):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ Google: {validated}")
            return validated

    # Уровень 3: bus.gov.ru
    site = _bus_gov_search(short_name, inn, ogrn)
    if site:
        validated = _validate_url(site)
        if validated:
            log.info(f"[{mo_id}] ✓ bus.gov.ru: {validated}")
            return validated
        # URL из гос. реестра — доверяем даже без валидации контента
        log.info(f"[{mo_id}] ✓ bus.gov.ru (без валидации): {site}")
        return site

    # Уровень 4: перебор типовых URL
    for url in _guess_urls(short_name, region, city):
        validated = _validate_url(url)
        if validated:
            log.info(f"[{mo_id}] ✓ Перебор: {validated}")
            return validated

    log.info(f"[{mo_id}] ✗ Не найден: {short_name}")
    return None
