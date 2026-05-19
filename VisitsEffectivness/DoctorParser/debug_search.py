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
            "fields":    "items.contact_groups",
            "page_size": 5,
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

        for item in items:
            name = item.get("name", "—")
            url  = item.get("url", "")

            # Сайт из контактов
            website = ""
            for group in item.get("contact_groups", []):
                for c in group.get("contacts", []):
                    if c.get("type") == "website":
                        website = c.get("value", "")

            site = url or website or "—"
            print(f"  [{item.get('id','')}] {name[:55]}")
            print(f"         сайт: {site}")

    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_dgis(name, city)

    print("\nДиагностика завершена.")
