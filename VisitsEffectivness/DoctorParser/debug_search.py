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
        "Accept": "application/json",
    }
    params = {
        "query":    query,
        "folderId": YANDEX_SEARCH_FOLDER,
        "lang":     "ru",
        "region":   "ru",
        "limit":    5,
    }

    try:
        resp = requests.get(YANDEX_SEARCH_URL, headers=headers, params=params, timeout=10)
        print(f"HTTP {resp.status_code}")

        if resp.status_code != 200:
            print(f"Ошибка: {resp.text[:400]}")
            return

        data = resp.json()
        # Структура ответа Yandex Cloud Search API v2
        results = (
            data.get("web", {}).get("searchResults", {}).get("cluster", [])
            or data.get("results", [])
            or []
        )
        print(f"Результатов: {len(results)}")
        for r in results[:5]:
            # Разные варианты структуры ответа
            url   = r.get("url") or r.get("link") or "—"
            title = r.get("title") or r.get("headline") or "—"
            print(f"  → {url}")
            print(f"     {title[:70]}")

        if not results:
            # Показываем сырой ответ чтобы понять структуру
            import json
            print("Сырой ответ (первые 800 символов):")
            print(json.dumps(data, ensure_ascii=False)[:800])

    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_yandex_cloud(name, city)

    print("\nДиагностика завершена.")
