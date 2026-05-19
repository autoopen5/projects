"""
debug_search.py — диагностика поиска сайтов МО.
Запуск: python debug_search.py
"""
import socket
import requests
import time
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TEST_MO = [
    ("Городская Поликлиника № 3", "Ульяновск"),
    ("Областная Клиническая Больница", "Ульяновск"),
]


def test_yandex(name: str, city: str):
    query = f"{name} {city} официальный сайт"
    print(f"\n{'='*60}")
    print(f"Яндекс: {query}")
    print(f"{'='*60}")
    try:
        resp = requests.get(
            "https://yandex.ru/search/",
            params={"text": query, "lr": "225"},
            headers=HEADERS,
            timeout=10,
        )
        print(f"HTTP {resp.status_code}, {len(resp.text)} байт")

        soup = BeautifulSoup(resp.text, "html.parser")

        # Собираем все внешние ссылки из результатов
        found = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (href.startswith("http")
                    and "yandex" not in href
                    and "ya.ru" not in href):
                found.append(href)

        if found:
            print(f"Найдено внешних ссылок: {len(found)}")
            for url in found[:8]:
                print(f"  → {url}")
        else:
            # Показываем кусок HTML — понять дали ли капчу
            print("Внешних ссылок нет. Первые 500 символов HTML:")
            print(resp.text[:500])

    except requests.Timeout:
        print("ТАЙМАУТ (10 сек) — Яндекс не отвечает")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_bus_gov(name: str, city: str):
    query = f"{name} {city}"
    print(f"\n-- bus.gov.ru: {query}")
    try:
        resp = requests.get(
            "https://bus.gov.ru/pub/agency/search.json",
            params={"searchString": name, "page": 0, "size": 5},
            headers={"User-Agent": "MO-Research/1.0", "Accept": "application/json"},
            timeout=5,
            verify=False,
        )
        print(f"HTTP {resp.status_code}, тело: {repr(resp.text[:200])}")
        if resp.status_code == 200 and resp.text.strip():
            data = resp.json()
            agencies = data.get("agencies") or data.get("data") or []
            print(f"Учреждений: {len(agencies)}")
            for ag in agencies[:3]:
                site = ag.get("site") or ag.get("siteUrl") or ag.get("webSite") or "—"
                print(f"  {ag.get('fullName', '')[:60]} → {site}")
    except requests.Timeout:
        print("ТАЙМАУТ")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_url_guess(name: str, city: str):
    """Проверяем угадывание URL без поисковиков."""
    import re
    print(f"\n-- Перебор URL: {name} {city}")
    num = re.search(r"№\s*(\d+)", name)
    num = num.group(1) if num else ""
    city_c = re.sub(r"\s+", "", city.lower())

    candidates = []
    name_l = name.lower()
    if "поликлиник" in name_l:
        candidates += [f"https://gp{num}.{city_c}.ru", f"https://gp{num}{city_c}.ru"]
    if "областная клинич" in name_l:
        candidates += [f"https://okb.{city_c}.ru", f"https://oblbolnica.{city_c}.ru"]

    for url in candidates:
        try:
            r = requests.head(url, headers=HEADERS, timeout=4, allow_redirects=True, verify=False)
            print(f"  {url} → HTTP {r.status_code}")
        except Exception as e:
            print(f"  {url} → {type(e).__name__}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_yandex(name, city)
        time.sleep(2)
        test_bus_gov(name, city)
        test_url_guess(name, city)
        print()

    print("\nДиагностика завершена.")
