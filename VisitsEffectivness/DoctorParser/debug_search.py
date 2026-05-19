"""
debug_search.py — диагностика поиска сайтов МО через 2GIS.
Запуск: python debug_search.py
"""
import requests
import urllib3
urllib3.disable_warnings()

from config import DGIS_API_KEY

DGIS_URL = "https://catalog.api.2gis.com/3.0/items"

TEST_MO = [
    ("Городская Поликлиника № 3", "Ульяновск"),
    ("Областная Клиническая Больница", "Ульяновск"),
    ("Городская Поликлиника № 1", "Москва"),
]


def test_dgis(mo_name: str, city: str):
    print(f"\n{'='*60}")
    print(f"2GIS: {mo_name} / {city}")
    print(f"{'='*60}")

    if not DGIS_API_KEY or DGIS_API_KEY == "ВАШ_КЛЮЧ_2GIS":
        print("КЛЮЧ НЕ ЗАДАН — вставь DGIS_API_KEY в config.py")
        return

    try:
        params = {
            "key":       DGIS_API_KEY,
            "q":         f"{mo_name} {city}",
            "fields":    "items.contact_groups,items.links,items.org",
            "page_size": 3,
            "locale":    "ru_RU",
        }
        resp = requests.get(DGIS_URL, params=params, timeout=10)
        print(f"HTTP {resp.status_code}")

        if resp.status_code != 200:
            print(f"Тело ответа: {resp.text[:300]}")
            return

        data = resp.json()
        items = data.get("result", {}).get("items", [])
        print(f"Найдено организаций: {len(items)}")

        # Показываем первую организацию полностью — смотрим структуру
        if items:
            import json
            first = items[0]
            print(f"\n  Первая организация — все поля:")
            print(f"  name: {first.get('name')}")
            print(f"  url:  {first.get('url')}")
            print(f"  contact_groups RAW: {json.dumps(first.get('contact_groups', []), ensure_ascii=False)[:500]}")
            print(f"  links RAW: {json.dumps(first.get('links', []), ensure_ascii=False)[:300]}")
            print(f"  org RAW:   {json.dumps(first.get('org', {}), ensure_ascii=False)[:300]}")

    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_dgis(name, city)

    print("\nДиагностика завершена.")
