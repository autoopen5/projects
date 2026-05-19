"""
debug_search.py — диагностика поиска сайтов МО.
Запуск: python debug_search.py

Показывает что реально возвращают поисковики и почему URL не проходят валидацию.
"""
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TEST_QUERIES = [
    "Городская Поликлиника № 3 Ульяновск официальный сайт",
    "Ульяновская Областная Клиническая Больница официальный сайт",
    "Алтайский Краевой Онкологический Диспансер официальный сайт",
]


def test_duckduckgo(query: str):
    print(f"\n{'='*60}")
    print(f"ЗАПРОС: {query}")
    print(f"{'='*60}")
    try:
        from duckduckgo_search import DDGS
        print("DDG: ждём 6 сек (rate-limit)...")
        time.sleep(6)
        results = DDGS().text(query, max_results=5, region="ru-ru", backend="lite")
        print(f"DuckDuckGo: найдено {len(results)} результатов")
        for r in results:
            print(f"  → {r.get('href')}  |  {r.get('title', '')[:60]}")
    except ImportError:
        print("duckduckgo-search не установлен: pip install duckduckgo-search")
    except Exception as e:
        print(f"ОШИБКА DuckDuckGo: {e}")


def test_yandex(query: str):
    print(f"\n-- Яндекс для: {query[:50]}")
    try:
        resp = requests.get(
            "https://yandex.ru/search/",
            params={"text": query, "lr": "225"},
            headers=HEADERS,
            timeout=15,
        )
        print(f"HTTP статус: {resp.status_code}, размер: {len(resp.text)} байт")

        soup = BeautifulSoup(resp.text, "html.parser")
        selectors = [
            "a.organic__url", "a[data-log-node]",
            "h2.organic__title a", ".serp-item a", ".organic a",
        ]
        found_any = False
        for sel in selectors:
            links = soup.select(sel)
            external = [
                a.get("href", "") for a in links
                if a.get("href", "").startswith("http")
                and "yandex" not in a.get("href", "")
            ]
            if external:
                print(f"  {sel}: {len(external)} ссылок → {external[0]}")
                found_any = True
        if not found_any:
            print("  Ни один селектор не нашёл ссылок (капча или смена вёрстки)")

    except Exception as e:
        print(f"ОШИБКА Яндекс: {e}")


def test_bus_gov(query: str):
    print(f"\n-- bus.gov.ru для: {query[:50]}")
    try:
        resp = requests.get(
            "https://bus.gov.ru/pub/agency/search.json",
            params={"searchString": query[:60], "page": 0, "size": 3},
            headers={"User-Agent": "MO-Research/1.0", "Accept": "application/json"},
            timeout=4,
            verify=False,
        )
        print(f"HTTP статус: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            agencies = data.get("agencies") or data.get("data") or []
            print(f"Найдено учреждений: {len(agencies)}")
            for ag in agencies[:3]:
                site = ag.get("site") or ag.get("siteUrl") or ag.get("webSite") or "—"
                print(f"  {ag.get('fullName', '')[:50]} → {site}")
    except Exception as e:
        print(f"ОШИБКА bus.gov.ru: {e}")


def test_google(query: str):
    print(f"\n-- Google для: {query[:60]}")
    try:
        from googlesearch import search
        time.sleep(2)
        results = list(search(query, num_results=5, lang="ru", sleep_interval=2))
        print(f"Google: найдено {len(results)} результатов")
        for url in results:
            print(f"  → {url}")
    except ImportError:
        print("googlesearch-python не установлен")
    except Exception as e:
        print(f"ОШИБКА Google: {e}")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()

    for q in TEST_QUERIES[:2]:
        # Google — основной
        test_google(q)
        # DDG — запасной
        test_duckduckgo(q)
        # bus.gov.ru
        test_bus_gov(q.split()[0] + " " + q.split()[1])
        time.sleep(3)

    print("\n\nДиагностика завершена.")
