"""
debug_search.py — диагностика поиска сайтов МО.
Запуск: python debug_search.py
"""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from bs4 import BeautifulSoup

import urllib3
urllib3.disable_warnings()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

TEST_QUERIES = [
    "Городская Поликлиника № 3 Ульяновск официальный сайт",
    "Ульяновская Областная Клиническая Больница официальный сайт",
]


def test_google(query: str):
    print(f"\n-- Google для: {query[:60]}")
    print("   запрос...", end=" ", flush=True)

    def _do_search():
        from googlesearch import search
        return list(search(query, num_results=5, lang="ru", sleep_interval=1, timeout=10))

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do_search)
            results = future.result(timeout=20)
        print(f"найдено {len(results)}")
        for url in results:
            print(f"  → {url}")
    except FuturesTimeout:
        print("ТАЙМАУТ (>20 сек) — Google блокирует или нет сети")
    except ImportError:
        print("googlesearch-python не установлен")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_yandex(query: str):
    print(f"\n-- Яндекс для: {query[:60]}")
    print("   запрос...", end=" ", flush=True)
    try:
        resp = requests.get(
            "https://yandex.ru/search/",
            params={"text": query, "lr": "225"},
            headers=HEADERS,
            timeout=10,
        )
        print(f"HTTP {resp.status_code}, {len(resp.text)} байт")
        soup = BeautifulSoup(resp.text, "html.parser")
        urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "yandex" not in href:
                urls.append(href)
        if urls:
            print(f"  Найдено ссылок: {len(urls)}")
            for u in urls[:5]:
                print(f"  → {u}")
        else:
            print("  Ссылок нет — скорее всего капча")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_bus_gov(query: str):
    print(f"\n-- bus.gov.ru для: {query[:50]}")
    print("   запрос...", end=" ", flush=True)
    try:
        resp = requests.get(
            "https://bus.gov.ru/pub/agency/search.json",
            params={"searchString": query[:60], "page": 0, "size": 3},
            headers={"User-Agent": "MO-Research/1.0", "Accept": "application/json"},
            timeout=5,
            verify=False,
        )
        print(f"HTTP {resp.status_code}, тело: {repr(resp.text[:100])}")
        if resp.status_code == 200 and resp.text.strip():
            data = resp.json()
            agencies = data.get("agencies") or data.get("data") or []
            print(f"  Учреждений: {len(agencies)}")
            for ag in agencies[:3]:
                site = ag.get("site") or ag.get("siteUrl") or ag.get("webSite") or "—"
                print(f"  {ag.get('fullName', '')[:50]} → {site}")
    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    q = TEST_QUERIES[0]
    print(f"Тестируем: {q}\n")

    test_google(q)
    time.sleep(3)
    test_yandex(q)
    time.sleep(2)
    test_bus_gov("Городская Поликлиника 3 Ульяновск")

    print("\n\nДиагностика завершена.")
