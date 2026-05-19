"""
debug_search.py — диагностика поиска сайтов МО через Яндекс Cloud Search API.
Запуск: python debug_search.py
"""
import requests
import base64
import xml.etree.ElementTree as ET
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

    url = "https://searchapi.api.cloud.yandex.net/v2/web/search"
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        print(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"Ошибка: {resp.text[:300]}")
            return

        # rawData — base64-encoded XML
        raw_b64 = resp.json().get("rawData", "")
        xml_bytes = base64.b64decode(raw_b64)
        xml_text  = xml_bytes.decode("utf-8")

        root = ET.fromstring(xml_text)
        ns   = {"ya": ""}  # Яндекс XML без namespace

        urls = [doc.findtext("url") for doc in root.iter("doc") if doc.findtext("url")]
        titles = [doc.findtext("title") for doc in root.iter("doc")]

        print(f"Найдено результатов: {len(urls)}")
        for u, t in zip(urls[:6], titles):
            print(f"  → {u}")
            print(f"     {(t or '')[:70]}")

    except Exception as e:
        print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    for name, city in TEST_MO:
        test_yandex_cloud(name, city)

    print("\nДиагностика завершена.")
