"""
debug_search.py — диагностика поиска сайтов МО через Яндекс Cloud Search API.
Запуск: python debug_search.py
"""
import requests
import urllib3
urllib3.disable_warnings()

from config import YANDEX_SEARCH_KEY, YANDEX_SEARCH_FOLDER

YANDEX_SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"

TEST_MO = [
    ("Городская Поликлиника № 3", "Ульяновск"),
    ("Областная Клиническая Больница", "Ульяновск"),
    ("Городская Поликлиника № 1", "Москва"),
]


def test_yandex_cloud(mo_name: str, city: str):
    query = f"{mo_name} {city} официальный сайт"
    print(f"\n{'='*60}")
    print(f"Яндекс Cloud Search: {query}")
    print(f"{'='*60}")

    if YANDEX_SEARCH_KEY in ("ВАША_API_КЛЮЧ", ""):
        print("КЛЮЧ НЕ ЗАДАН — вставь YANDEX_SEARCH_KEY в config.py")
        return

    headers = {
        "Authorization": f"Api-Key {YANDEX_SEARCH_KEY}",
        "Content-Type":  "application/json",
    }

    # Пробуем POST с JSON телом (v2 API)
    body = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText":  query,
        },
        "folderId": YANDEX_SEARCH_FOLDER,
    }

    endpoints = [
        ("POST", "https://searchapi.api.cloud.yandex.net/v2/web/search"),
        ("GET",  "https://searchapi.api.cloud.yandex.net/v2/web/search"),
        ("POST", "https://searchapi.api.cloud.yandex.net/v1/search"),
    ]

    for method, url in endpoints:
        try:
            if method == "POST":
                resp = requests.post(url, headers=headers, json=body, timeout=10)
            else:
                resp = requests.get(url, headers=headers,
                                    params={"query": query, "folderId": YANDEX_SEARCH_FOLDER},
                                    timeout=10)
            print(f"\n{method} {url}")
            print(f"HTTP {resp.status_code}")
            print(f"Ответ: {resp.text[:400]}")
            if resp.status_code == 200:
                break
        except Exception as e:
            print(f"{method} {url} → ОШИБКА: {e}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_yandex_cloud(name, city)

    print("\nДиагностика завершена.")
