"""
debug_search.py — диагностика поиска сайтов МО.
Тестирует: 2GIS detail endpoint + Яндекс XML API
"""
import requests
import json
import urllib3
urllib3.disable_warnings()

from config import DGIS_API_KEY

DGIS_URL        = "https://catalog.api.2gis.com/3.0/items"
DGIS_DETAIL_URL = "https://catalog.api.2gis.com/3.0/items/byid"


def test_dgis_detail(org_id: str, label: str = ""):
    """Пробуем получить детали по org_id — там может быть сайт."""
    print(f"\n-- 2GIS detail: {label or org_id}")
    params = {
        "key":    DGIS_API_KEY,
        "id":     org_id,
        "fields": "items.contact_groups,items.links,items.org,items.reviews",
        "locale": "ru_RU",
    }
    try:
        resp = requests.get(DGIS_DETAIL_URL, params=params, timeout=10)
        print(f"HTTP {resp.status_code}")
        data = resp.json()
        items = data.get("result", {}).get("items", [])
        if items:
            item = items[0]
            print(f"name: {item.get('name')}")
            print(f"url:  {item.get('url')}")
            print(f"contact_groups: {json.dumps(item.get('contact_groups', []), ensure_ascii=False)[:400]}")
        else:
            print("Пусто")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_dgis_search_verbose(mo_name: str, city: str):
    """Поиск с максимальным набором полей."""
    print(f"\n{'='*60}")
    print(f"2GIS verbose: {mo_name} / {city}")
    params = {
        "key":       DGIS_API_KEY,
        "q":         f"{mo_name} {city}",
        "fields":    "items.contact_groups,items.links,items.org,items.schedule",
        "page_size": 2,
        "locale":    "ru_RU",
        "type":      "branch",
    }
    try:
        resp = requests.get(DGIS_URL, params=params, timeout=10)
        print(f"HTTP {resp.status_code}")
        data = resp.json()
        items = data.get("result", {}).get("items", [])
        print(f"Организаций: {len(items)}")
        for item in items:
            print(f"\n  {item.get('name')} [{item.get('id')}]")
            # Все ключи верхнего уровня
            print(f"  Ключи: {list(item.keys())}")
            print(f"  url: {item.get('url')}")
            cg = item.get("contact_groups", [])
            if cg:
                print(f"  contact_groups: {json.dumps(cg, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"ОШИБКА: {e}")


def test_yandex_xml(query: str, yandex_user: str, yandex_key: str):
    """
    Яндекс XML API — официальный поиск без капчи.
    Регистрация: https://xml.yandex.ru/
    """
    print(f"\n{'='*60}")
    print(f"Яндекс XML: {query}")
    url = f"https://xmlsearch.yandex.ru/xmlsearch"
    params = {
        "user": yandex_user,
        "key":  yandex_key,
        "query": query,
        "lr":  "73",   # регион Ульяновск
        "l10n": "ru",
        "page": "0",
        "groupby": "attr=d.mode=deep.groups-on-page=5.docs-in-group=1",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        print(f"HTTP {resp.status_code}, {len(resp.text)} байт")
        print(f"Начало ответа: {resp.text[:500]}")
    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    # Тест 1: детальный запрос по ID (из предыдущего теста Москва)
    test_dgis_detail("4504136499333114", "Городская поликлиника №1 Москва (org_id)")
    test_dgis_detail("7741347838558866", "Городская поликлиника №3 Ульяновск (branch_id)")

    # Тест 2: поиск с type=branch
    test_dgis_search_verbose("Городская Поликлиника № 3", "Ульяновск")

    # Тест 3: Яндекс XML API (раскомментировать после регистрации на xml.yandex.ru)
    # test_yandex_xml(
    #     "Городская Поликлиника № 3 Ульяновск официальный сайт",
    #     yandex_user="ВАШ_ЛОГИН",
    #     yandex_key="ВАШ_КЛЮЧ",
    # )

    print("\nДиагностика завершена.")
