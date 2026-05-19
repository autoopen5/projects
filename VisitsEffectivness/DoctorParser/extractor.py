"""
extractor.py — загрузка страницы врачей + LLM-экстракция через Anthropic API.
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
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
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
    "/vrachi", "/doctors", "/specialists", "/personnel",
    "/about/doctors", "/o-nas/vrachi", "/uslugi/vrachi",
    "/medpersonel", "/about/staff", "/sotrudniki",
    "/about/specialists", "/our-doctors",
]

DOCTOR_PAGE_LINK_RE = re.compile(
    r"(врач|доктор|специалист|персонал|сотрудник|медицинский состав)",
    re.IGNORECASE
)


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


def _clean_html(html: str, max_chars: int = 12_000) -> str:
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
    Отправляет текст страницы в Claude, получает JSON со списком врачей.
    """
    prompt = EXTRACT_PROMPT.format(page_text=page_text)

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      ANTHROPIC_MODEL,
                "max_tokens": 4096,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60,
            verify=False,
        )
        if resp.status_code != 200:
            log.warning(f"LLM error {resp.status_code}: {resp.text[:200]}")
            return []

        raw = resp.json()["content"][0]["text"].strip()
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


def parse_doctors_from_site(mo_id: str, site_url: str) -> list[dict]:
    """
    Полный цикл: сайт МО → страница врачей → текст → LLM → список врачей.
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

    page_text = _clean_html(page_html)
    if len(page_text) < 100:
        log.debug(f"[{mo_id}] Страница пустая после очистки")
        return []

    doctors = extract_doctors_llm(page_text)
    log.info(f"[{mo_id}] Извлечено врачей: {len(doctors)}")
    return doctors
