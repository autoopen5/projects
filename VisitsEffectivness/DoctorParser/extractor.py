"""
extractor.py — загрузка страницы врачей + LLM-экстракция через Yandex GPT.
"""
import re
import json
import time
import logging
import requests

# Корпоративный SSL-прокси — отключаем верификацию сертификата
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
from config import (
    YANDEX_GPT_KEY, YANDEX_SEARCH_FOLDER,
    FETCH_TIMEOUT_S, FETCH_DELAY_S, MAX_RETRIES,
)

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip, deflate",
}

DOCTOR_PAGE_HINTS = [
    "/meditsinskie-rabotniki", "/meditsinskie-rabotniki/",
    "/vrachi", "/vrachi/", "/doctors", "/doctors/",
    "/specialists", "/personnel", "/medpersonel",
    "/about/doctors", "/o-nas/vrachi", "/uslugi/vrachi",
    "/about/staff", "/sotrudniki", "/about/specialists",
    "/our-doctors", "/nashi-vrachi", "/medrabotniki",
]

DOCTOR_PAGE_LINK_RE = re.compile(
    r"(медицинск[а-я]* работник|врач|доктор|специалист|персонал|сотрудник|медицинский состав)",
    re.IGNORECASE
)

PAGINATION_RE = re.compile(r'[?&]page=(\d+)|/page/(\d+)', re.IGNORECASE)


def _fetch(url: str, retries: int = MAX_RETRIES) -> str | None:
    """HTTP GET с ретраями."""
    for attempt in range(retries):
        try:
            time.sleep(FETCH_DELAY_S)
            resp = requests.get(
                url, headers=HEADERS,
                timeout=FETCH_TIMEOUT_S,
                allow_redirects=True,
                verify=False,
            )
            if resp.status_code == 200:
                ct = resp.headers.get("Content-Type", "")
                if not any(t in ct for t in ("text", "html", "xml")):
                    return None
                resp.encoding = resp.apparent_encoding or "utf-8"
                text = resp.text
                # Отбрасываем бинарный/сжатый контент (brotli без декодирования и т.п.)
                sample = text[:300]
                non_print = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
                if non_print > 20:
                    return None
                return text
            log.debug(f"HTTP {resp.status_code} для {url}")
        except Exception as e:
            log.debug(f"Попытка {attempt+1}/{retries}: {e}")
    return None


def _find_doctors_page(base_url: str, html: str) -> str | None:
    """
    Ищет ссылку на страницу врачей на главной странице МО.
    1. Проверяем известные суффиксы URL
    2. Ищем в ссылках с нужными текстами
    """
    base = base_url.rstrip("/")

    # 1. Пробуем типовые пути
    for hint in DOCTOR_PAGE_HINTS:
        candidate = base + hint
        try:
            r = requests.head(
                candidate, headers=HEADERS,
                timeout=5, allow_redirects=True, verify=False,
            )
            if r.status_code == 200:
                log.debug(f"Страница врачей по пути: {candidate}")
                return candidate
        except Exception:
            continue

    # 2. Ищем в HTML главной
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if DOCTOR_PAGE_LINK_RE.search(text):
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return base + href

    return None


def _clean_html(html: str, max_chars: int = 30_000) -> str:
    """Очищает HTML: убирает script/style/nav, оставляет текст."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "noscript", "iframe", "svg"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


EXTRACT_PROMPT = """Ты извлекаешь структурированные данные о врачах с сайта медицинской организации.

Ниже — текст страницы. Найди всех врачей и верни JSON-массив.
Для каждого врача:
- "name": ФИО (обязательно, строка)
- "specialty": специальность/должность (строка, если есть)
- "category": квалификационная категория (строка, если есть — "высшая", "первая", "вторая" или "")
- "education": образование/вуз (строка, если есть)
- "experience": стаж или опыт (строка, если есть)

Правила:
1. Если врачей нет — верни пустой массив []
2. Не добавляй поля которых нет на странице, оставь ""
3. Верни ТОЛЬКО валидный JSON-массив, без пояснений, без markdown

Текст страницы:
{page_text}
"""


def extract_doctors_llm(page_text: str) -> list[dict]:
    """
    Отправляет текст страницы в YandexGPT, получает JSON со списком врачей.
    """
    prompt = EXTRACT_PROMPT.format(page_text=page_text)

    try:
        resp = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers={
                "Authorization": f"Api-Key {YANDEX_GPT_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "modelUri": f"gpt://{YANDEX_SEARCH_FOLDER}/yandexgpt-lite/latest",
                "completionOptions": {
                    "stream":      False,
                    "temperature": 0.1,
                    "maxTokens":   "4096",
                },
                "messages": [{"role": "user", "text": prompt}],
            },
            timeout=60,
            verify=False,
        )
        if resp.status_code != 200:
            log.warning(f"LLM error {resp.status_code}: {resp.text[:200]}")
            return []

        raw = resp.json()["result"]["alternatives"][0]["message"]["text"].strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        doctors = json.loads(match.group())
        result = []
        for d in doctors:
            if not d.get("name"):
                continue
            result.append({
                "doctor_name": str(d.get("name", "")).strip(),
                "specialty":   str(d.get("specialty", "")).strip(),
                "category":    str(d.get("category", "")).strip(),
                "education":   str(d.get("education", "")).strip(),
                "experience":  str(d.get("experience", "")).strip(),
            })
        return result

    except Exception as e:
        log.warning(f"LLM extraction error: {e}")
        return []


def _get_next_page_url(current_url: str, html: str) -> str | None:
    """Ищет ссылку на следующую страницу пагинации."""
    soup = BeautifulSoup(html, "html.parser") if html else None
    if not soup:
        return None
    # Ищем ссылку с текстом "следующая", "вперёд", "→", ">" или rel="next"
    next_re = re.compile(r'(следующ|вперёд|далее|next|\→|»|>)', re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        if a.get("rel") == ["next"] or next_re.search(a.get_text(strip=True)):
            href = a["href"]
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                from urllib.parse import urlparse
                base = urlparse(current_url)
                return f"{base.scheme}://{base.netloc}{href}"
    return None


def parse_doctors_from_site(mo_id: str, site_url: str) -> list[dict]:
    """
    Полный цикл: сайт МО → страница врачей → текст (с пагинацией) → LLM → список врачей.
    """
    main_html = _fetch(site_url)
    if not main_html:
        log.warning(f"[{mo_id}] Главная не загрузилась: {site_url}")
        return []

    doctor_page_url = _find_doctors_page(site_url, main_html)

    if doctor_page_url and doctor_page_url != site_url:
        page_html = _fetch(doctor_page_url)
    else:
        page_html = main_html
        doctor_page_url = site_url

    if not page_html:
        log.warning(f"[{mo_id}] Страница врачей не загрузилась")
        return []

    # Собираем текст со всех страниц пагинации (максимум 10 страниц)
    all_doctors: list[dict] = []
    current_url  = doctor_page_url
    current_html = page_html
    seen_urls    = {current_url}

    for page_num in range(1, 11):
        page_text = _clean_html(current_html)
        if len(page_text) < 100:
            break

        doctors = extract_doctors_llm(page_text)
        all_doctors.extend(doctors)
        log.debug(f"[{mo_id}] Страница {page_num}: {len(doctors)} врачей")

        next_url = _get_next_page_url(current_url, current_html)
        if not next_url or next_url in seen_urls:
            break
        seen_urls.add(next_url)
        current_html = _fetch(next_url)
        if not current_html:
            break
        current_url = next_url

    # Дедупликация по имени
    seen_names = set()
    result = []
    for d in all_doctors:
        if d["doctor_name"] not in seen_names:
            seen_names.add(d["doctor_name"])
            result.append(d)

    log.info(f"[{mo_id}] Извлечено врачей: {len(result)}")
    return result
