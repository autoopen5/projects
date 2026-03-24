#!/usr/bin/env python3
"""
Telegram bot MVP for 3D print lead intake + quote + deposit/status flow.
No third-party deps: uses Telegram Bot HTTP API via urllib.
"""

from __future__ import annotations

import json
import os
import re
import time
import csv
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
ORDERS_PATH = DATA_DIR / "orders.json"
SESSIONS_PATH = DATA_DIR / "sessions.json"
OFFSET_PATH = DATA_DIR / "offset.txt"
ORDERS_EXCEL_PATH = DATA_DIR / "orders_excel.csv"

MATERIAL_RATES = {
    "PLA": 6.0,
    "PETG": 8.0,
    "TPU": 12.0,
}

POSTPROC_FEES = {
    "none": 0.0,
    "basic": 150.0,
    "advanced": 300.0,
}

DEADLINE_FEES = {
    "standard": 0.0,
    "rush_24h": 500.0,
}

DELIVERY_FEES = {
    "pickup": 0.0,
    "cdek": 350.0,
    "yandex": 450.0,
}

STATUS_FLOW = [
    "REQUEST_RECEIVED",
    "REVIEWING_DETAILS",
    "WAITING_DEPOSIT",
    "DEPOSIT_UNDER_REVIEW",
    "PRINTING",
    "POST_PROCESSING",
    "READY_FOR_PICKUP",
    "SHIPPED",
    "CANCELLED",
]

STATUS_LABELS_RU = {
    "REQUEST_RECEIVED": "Заявка получена",
    "REVIEWING_DETAILS": "Уточнение деталей",
    "WAITING_DEPOSIT": "Ожидаем предоплату",
    "DEPOSIT_UNDER_REVIEW": "Предоплата на проверке",
    "PRINTING": "Печатаем",
    "POST_PROCESSING": "Постобработка",
    "READY_FOR_PICKUP": "Готово к выдаче",
    "SHIPPED": "Отправлено",
    "CANCELLED": "Отменено",
}

INTAKE_FIELDS = [
    "product_type",
    "example",
    "dimensions_mm",
    "material",
    "color",
    "deadline",
    "quantity",
    "postproc",
    "contact",
    "delivery_mode",
]


@dataclass
class Config:
    token: str
    admin_chat_id: str | None
    payment_instructions: str
    tripo_api_key: str | None
    tripo_api_url: str
    tripo_model_version: str
    tripo_poll_attempts: int
    printer_dispatch_webhook_url: str | None
    printer_dispatch_token: str | None
    auto_dispatch_on_approval: bool
    openai_api_key: str | None
    openai_model: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def save_orders_excel_csv(path: Path, orders: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "order_id",
        "created_at",
        "updated_at",
        "source",
        "customer_user_id",
        "customer_chat_id",
        "contact",
        "product_type",
        "example",
        "dimensions_mm",
        "material",
        "color",
        "quantity",
        "deadline",
        "postproc",
        "delivery_mode",
        "estimate_material_g",
        "estimate_print_hours",
        "quote_total_rub",
        "deposit_required_rub",
        "deposit_status",
        "status",
        "model_approval_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for order_id in sorted(orders.keys()):
            order = orders[order_id]
            dims = order.get("dimensions_mm", [])
            writer.writerow(
                {
                    "order_id": order_id,
                    "created_at": order.get("created_at", ""),
                    "updated_at": order.get("updated_at", ""),
                    "source": order.get("source", ""),
                    "customer_user_id": order.get("customer", {}).get("user_id", ""),
                    "customer_chat_id": order.get("customer", {}).get("chat_id", ""),
                    "contact": order.get("customer", {}).get("contact", ""),
                    "product_type": order.get("product_type", ""),
                    "example": order.get("example", ""),
                    "dimensions_mm": "x".join(str(x) for x in dims) if isinstance(dims, list) else "",
                    "material": order.get("material", ""),
                    "color": order.get("color", ""),
                    "quantity": order.get("quantity", ""),
                    "deadline": order.get("deadline", ""),
                    "postproc": order.get("postproc", ""),
                    "delivery_mode": order.get("delivery_mode", ""),
                    "estimate_material_g": order.get("estimate", {}).get("material_g", ""),
                    "estimate_print_hours": order.get("estimate", {}).get("print_hours", ""),
                    "quote_total_rub": order.get("quote_total_rub", ""),
                    "deposit_required_rub": order.get("deposit_required_rub", ""),
                    "deposit_status": order.get("deposit_status", ""),
                    "status": order.get("status", ""),
                    "model_approval_status": order.get("model_approval", {}).get("status", ""),
                }
            )


def load_offset() -> int:
    if not OFFSET_PATH.exists():
        return 0
    raw = OFFSET_PATH.read_text(encoding="utf-8").strip()
    return int(raw) if raw else 0


def save_offset(offset: int) -> None:
    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(str(offset), encoding="utf-8")


def load_config() -> Config:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    admin_chat_id = os.getenv("ADMIN_CHAT_ID", "").strip() or None
    payment_instructions = os.getenv(
        "PAYMENT_INSTRUCTIONS",
        "Внесите предоплату 50% по согласованным реквизитам и отправьте скриншот оплаты в этот чат.",
    ).strip()
    tripo_api_key = os.getenv("TRIPO_API_KEY", "").strip() or None
    tripo_api_url = os.getenv("TRIPO_API_URL", "https://api.tripo3d.ai/v2/openapi/task").strip()
    tripo_model_version = os.getenv("TRIPO_MODEL_VERSION", "auto").strip() or "auto"
    tripo_poll_attempts_raw = os.getenv("TRIPO_POLL_ATTEMPTS", "8").strip()
    tripo_poll_attempts = int(tripo_poll_attempts_raw) if tripo_poll_attempts_raw.isdigit() else 8
    printer_dispatch_webhook_url = os.getenv("PRINTER_DISPATCH_WEBHOOK_URL", "").strip() or None
    printer_dispatch_token = os.getenv("PRINTER_DISPATCH_TOKEN", "").strip() or None
    auto_dispatch_on_approval = os.getenv("AUTO_DISPATCH_ON_APPROVAL", "").strip().lower() in {"1", "true", "yes", "on"}
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    return Config(
        token=token,
        admin_chat_id=admin_chat_id,
        payment_instructions=payment_instructions,
        tripo_api_key=tripo_api_key,
        tripo_api_url=tripo_api_url,
        tripo_model_version=tripo_model_version,
        tripo_poll_attempts=max(1, min(tripo_poll_attempts, 30)),
        printer_dispatch_webhook_url=printer_dispatch_webhook_url,
        printer_dispatch_token=printer_dispatch_token,
        auto_dispatch_on_approval=auto_dispatch_on_approval,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )


class TgClient:
    def __init__(self, token: str) -> None:
        self.base = f"https://api.telegram.org/bot{token}"

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base}/{method}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API error: {parsed}")
        return parsed["result"]

    def get_updates(self, offset: int, timeout: int = 25) -> list[dict[str, Any]]:
        result = self.call("getUpdates", {"offset": offset, "timeout": timeout})
        return result

    def send_message(self, chat_id: int | str, text: str) -> None:
        self.call("sendMessage", {"chat_id": chat_id, "text": text})


class ModelConstructor:
    def __init__(self, cfg: Config) -> None:
        self.tripo_api_key = cfg.tripo_api_key
        self.tripo_api_url = cfg.tripo_api_url
        self.tripo_model_version = cfg.tripo_model_version
        self.tripo_poll_attempts = cfg.tripo_poll_attempts
        self.openai_api_key = cfg.openai_api_key
        self.openai_model = cfg.openai_model

    def build(self, order: dict[str, Any], revision_note: str | None = None) -> dict[str, Any]:
        if self.tripo_api_key:
            try:
                text = self.build_with_tripo(order, revision_note)
                return {"mode": "tripo", "text": text, "generated_at": now_iso()}
            except urllib.error.HTTPError as e:
                return {"mode": "tripo_error", "text": self.describe_tripo_http_error(e), "generated_at": now_iso()}
            except Exception:  # noqa: BLE001
                pass
        if self.openai_api_key:
            try:
                text = self.build_with_openai(order, revision_note)
                return {"mode": "openai", "text": text, "generated_at": now_iso()}
            except Exception:  # noqa: BLE001
                pass
        return {"mode": "heuristic", "text": self.build_heuristic(order, revision_note), "generated_at": now_iso()}

    def build_with_tripo(self, order: dict[str, Any], revision_note: str | None) -> str:
        prompt = self.build_tripo_prompt(order, revision_note)
        payload: dict[str, Any] = {
            "type": "text_to_model",
            "prompt": prompt,
        }
        if self.tripo_model_version != "auto":
            payload["model_version"] = self.tripo_model_version

        create_req = urllib.request.Request(
            url=self.tripo_api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.tripo_api_key}",
                "x-api-key": self.tripo_api_key or "",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(create_req, timeout=35) as resp:
            created = json.loads(resp.read().decode("utf-8"))

        task_id = self.extract_task_id(created)
        model_url = self.extract_model_url(created)
        if model_url:
            return self.format_tripo_result(task_id, model_url)

        if not task_id:
            raise RuntimeError(f"Tripo create response missing task_id: {created}")

        task_url = f"{self.tripo_api_url.rstrip('/')}/{task_id}"
        last_status = "SUBMITTED"
        for _ in range(self.tripo_poll_attempts):
            time.sleep(2)
            poll_req = urllib.request.Request(
                url=task_url,
                method="GET",
                headers={
                    "Authorization": f"Bearer {self.tripo_api_key}",
                    "x-api-key": self.tripo_api_key or "",
                },
            )
            with urllib.request.urlopen(poll_req, timeout=35) as resp:
                polled = json.loads(resp.read().decode("utf-8"))
            last_status = self.extract_status(polled)
            model_url = self.extract_model_url(polled)
            if model_url:
                return self.format_tripo_result(task_id, model_url)
            if last_status in {"FAILED", "ERROR", "CANCELLED"}:
                raise RuntimeError(f"Tripo task {task_id} failed with status={last_status}")

        return (
            "Tripo task submitted.\n"
            f"- task_id: {task_id}\n"
            f"- status: {last_status}\n"
            "Model is still processing. Please check task status a bit later."
        )

    def build_tripo_prompt(self, order: dict[str, Any], revision_note: str | None) -> str:
        dims = order["dimensions_mm"]
        revision = f" Revision: {revision_note}." if revision_note else ""
        return (
            f"Create a printable 3D model for: {order['product_type']}. "
            f"Target dimensions around {dims[0]}x{dims[1]}x{dims[2]} mm. "
            f"Material intent: {order['material']}, color preference: {order['color']}. "
            f"Quantity: {order['quantity']}, post-processing: {order['postproc']}. "
            "Prioritize manufacturable geometry and avoid impossible overhangs."
            f"{revision}"
        )

    def format_tripo_result(self, task_id: str | None, model_url: str) -> str:
        task_line = f"- task_id: {task_id}\n" if task_id else ""
        return (
            "Tripo model draft is ready for review.\n"
            f"{task_line}"
            f"- model_url: {model_url}\n\n"
            "Please review this model link. If approved, send: СОГЛАСОВАНО"
        )

    def extract_task_id(self, payload: Any) -> str | None:
        candidates: list[str] = []
        if isinstance(payload, dict):
            for key in ("task_id", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("task_id", "id"):
                    value = data.get(key)
                    if isinstance(value, str) and value:
                        candidates.append(value)
        return candidates[0] if candidates else None

    def extract_status(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("status", "task_status", "state"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value.upper()
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("status", "task_status", "state"):
                    value = data.get(key)
                    if isinstance(value, str):
                        return value.upper()
        return "UNKNOWN"

    def extract_model_url(self, payload: Any) -> str | None:
        if isinstance(payload, str):
            if payload.startswith("http://") or payload.startswith("https://"):
                return payload
            return None
        if isinstance(payload, list):
            for item in payload:
                found = self.extract_model_url(item)
                if found:
                    return found
            return None
        if isinstance(payload, dict):
            for key in ("model_url", "download_url", "glb", "obj", "fbx", "stl", "url"):
                value = payload.get(key)
                found = self.extract_model_url(value)
                if found:
                    return found
            for value in payload.values():
                found = self.extract_model_url(value)
                if found:
                    return found
        return None

    def describe_tripo_http_error(self, exc: urllib.error.HTTPError) -> str:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        if exc.code == 403 and "enough credit" in body.lower():
            return (
                "Tripo API is configured, but there is not enough credit on the account.\n"
                "Please top up Tripo credits, then retry the order."
            )
        return f"Tripo API error: HTTP {exc.code}. Response: {body[:300]}"

    def build_with_openai(self, order: dict[str, Any], revision_note: str | None) -> str:
        prompt_data = {
            "product_type": order["product_type"],
            "example": order["example"],
            "dimensions_mm": order["dimensions_mm"],
            "material": order["material"],
            "color": order["color"],
            "quantity": order["quantity"],
            "deadline": order["deadline"],
            "postproc": order["postproc"],
            "delivery_mode": order["delivery_mode"],
            "revision_note": revision_note or "",
        }
        payload = {
            "model": self.openai_model,
            "temperature": 0.2,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Ты инженер-конструктор 3D печати. Сделай краткий, практичный "
                                "черновик модели для согласования с клиентом на русском языке. "
                                "Формат: 1) Концепт модели 2) Ключевые параметры 3) Риски/ограничения "
                                "4) Что подтвердить у клиента. Не придумывай невозможное."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(prompt_data, ensure_ascii=False)}],
                },
            ],
        }
        req = urllib.request.Request(
            url="https://api.openai.com/v1/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        text = parsed.get("output_text", "").strip()
        if text:
            return text

        chunks: list[str] = []
        for item in parsed.get("output", []):
            for content in item.get("content", []):
                txt = content.get("text", "")
                if txt:
                    chunks.append(txt)
        joined = "\n".join(chunks).strip()
        if not joined:
            raise RuntimeError("OpenAI response is empty")
        return joined

    def build_heuristic(self, order: dict[str, Any], revision_note: str | None) -> str:
        dims = order["dimensions_mm"]
        material = order["material"]
        deadline = order["deadline"]
        qty = order["quantity"]
        strength_hint = "повышенную толщину стенок 2.0-2.4 мм" if material in {"PETG", "TPU"} else "стандартную толщину стенок 1.6-2.0 мм"
        infill_hint = "30-40%" if material in {"PETG", "TPU"} else "20-30%"
        deadline_hint = "приоритет на скорость печати и минимизацию поддержек" if deadline == "rush_24h" else "баланс качества поверхности и времени печати"
        revision = f"\nУчет правок клиента: {revision_note}" if revision_note else ""
        return (
            "1) Концепт модели\n"
            f"- Изделие: {order['product_type']} на базе описания клиента.\n"
            f"- Габариты ориентировочно: {dims[0]}x{dims[1]}x{dims[2]} мм, партия {qty} шт.\n"
            f"- Материал/цвет: {material}, {order['color']}.\n\n"
            "2) Ключевые параметры печати\n"
            f"- Рекомендуем {strength_hint} и заполнение {infill_hint}.\n"
            f"- Приоритет: {deadline_hint}.\n"
            f"- Постобработка: {order['postproc']}.\n\n"
            "3) Риски и ограничения\n"
            "- Нужна проверка посадочных размеров и допусков перед финальной печатью.\n"
            "- Возможна корректировка геометрии для снижения поддержек и расхода материала.\n\n"
            "4) Что подтвердить у клиента\n"
            "- Точные критичные размеры и допуски.\n"
            "- Где нужна усиленная зона/ребра жесткости.\n"
            "- Приоритет: внешний вид или механическая прочность."
            f"{revision}"
        )


class Bot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tg = TgClient(cfg.token)
        self.constructor = ModelConstructor(cfg)
        self.orders: dict[str, dict[str, Any]] = load_json(ORDERS_PATH, {})
        self.sessions: dict[str, dict[str, Any]] = load_json(SESSIONS_PATH, {})
        self.offset = load_offset()

    def persist(self) -> None:
        save_json(ORDERS_PATH, self.orders)
        save_orders_excel_csv(ORDERS_EXCEL_PATH, self.orders)
        save_json(SESSIONS_PATH, self.sessions)
        save_offset(self.offset)

    def run_forever(self) -> None:
        print("Bot started. Polling updates...")
        while True:
            try:
                updates = self.tg.get_updates(self.offset, timeout=25)
                for upd in updates:
                    self.offset = max(self.offset, upd["update_id"] + 1)
                    self.handle_update(upd)
                self.persist()
            except urllib.error.URLError as e:
                print(f"Network error: {e}")
                time.sleep(2)
            except Exception as e:  # noqa: BLE001
                print(f"Runtime error: {e}")
                time.sleep(2)

    def handle_update(self, upd: dict[str, Any]) -> None:
        msg = upd.get("message")
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        user_id = str(msg["from"]["id"])
        text = (msg.get("text") or "").strip()
        photo = msg.get("photo")

        if text.startswith("/start"):
            source = self.extract_source(text)
            self.start_intake(user_id, chat_id, source)
            return
        if text.startswith("/help"):
            self.send_help(chat_id)
            return
        if text.startswith("/status"):
            self.handle_status_query(chat_id, text)
            return
        if text.startswith("/approve"):
            self.handle_approve(chat_id, text)
            return
        if text.startswith("/dispatch"):
            self.handle_dispatch(user_id, chat_id, text)
            return
        if text.startswith("/setstatus"):
            self.handle_set_status(user_id, chat_id, text)
            return

        session = self.sessions.get(user_id)
        if photo and session and session.get("step") == "await_deposit_proof":
            self.handle_deposit_proof(user_id, chat_id)
            return

        if not session:
            self.tg.send_message(
                chat_id,
                "Чтобы начать, отправьте /start. Список команд: /help.",
            )
            return

        self.handle_intake_step(user_id, chat_id, text)

    def extract_source(self, text: str) -> str:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "direct"
        return re.sub(r"[^a-zA-Z0-9_-]", "", parts[1].strip())[:32] or "direct"

    def start_intake(self, user_id: str, chat_id: int, source: str) -> None:
        self.sessions[user_id] = {
            "step": "product_type",
            "source": source,
            "answers": {},
            "chat_id": chat_id,
        }
        self.tg.send_message(
            chat_id,
            "Привет. Рассчитаю заказ на 3D-печать за пару минут.\n"
            "Шаг 1/10: Что нужно напечатать? (например: кронштейн, держатель, подарок)",
        )

    def send_help(self, chat_id: int) -> None:
        self.tg.send_message(
            chat_id,
            "Команды:\n"
            "/start [source] - начать расчет\n"
            "/status <ORDER_ID> - проверить статус заказа\n"
            "/approve <ORDER_ID> - согласовать модель\n"
            "/help - показать помощь\n"
            "Для администратора: /setstatus <ORDER_ID> <STATUS> [комментарий], /dispatch <ORDER_ID>",
        )

    def handle_status_query(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self.tg.send_message(chat_id, "Формат: /status <ORDER_ID>")
            return
        order_id = parts[1].strip().upper()
        order = self.orders.get(order_id)
        if not order:
            self.tg.send_message(chat_id, f"Заказ {order_id} не найден.")
            return
        status_ru = STATUS_LABELS_RU.get(order["status"], order["status"])
        deposit_status_ru = {"PENDING": "Ожидается", "UNDER_REVIEW": "На проверке", "CONFIRMED": "Подтверждена"}.get(
            order["deposit_status"], order["deposit_status"]
        )
        model_approval_ru = {"PENDING": "На согласовании", "APPROVED": "Согласовано", "REVISION_REQUESTED": "Нужны правки"}.get(
            order.get("model_approval", {}).get("status", "PENDING"),
            order.get("model_approval", {}).get("status", "PENDING"),
        )
        self.tg.send_message(
            chat_id,
            f"Заказ {order_id}\n"
            f"Статус: {status_ru}\n"
            f"Итого: {order['quote_total_rub']} RUB\n"
            f"Нужна предоплата: {order['deposit_required_rub']} RUB\n"
            f"Статус предоплаты: {deposit_status_ru}\n"
            f"Согласование модели: {model_approval_ru}",
        )

    def handle_approve(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self.tg.send_message(chat_id, "Формат: /approve <ORDER_ID>")
            return
        order_id = parts[1].strip().upper()
        order = self.orders.get(order_id)
        if not order:
            self.tg.send_message(chat_id, f"Заказ {order_id} не найден.")
            return
        if str(chat_id) != str(order["customer"]["chat_id"]):
            self.tg.send_message(chat_id, "Подтвердить модель может только клиент этого заказа.")
            return
        self.approve_model(chat_id, order)

    def handle_set_status(self, user_id: str, chat_id: int, text: str) -> None:
        if self.cfg.admin_chat_id and str(chat_id) != self.cfg.admin_chat_id:
            self.tg.send_message(chat_id, "Команда администратора недоступна в этом чате.")
            return
        parts = text.split(maxsplit=3)
        if len(parts) < 3:
            self.tg.send_message(chat_id, "Формат: /setstatus <ORDER_ID> <STATUS> [комментарий]")
            return
        order_id = parts[1].upper()
        new_status = parts[2].upper()
        note = parts[3] if len(parts) >= 4 else ""
        order = self.orders.get(order_id)
        if not order:
            self.tg.send_message(chat_id, f"Заказ {order_id} не найден.")
            return
        if new_status not in STATUS_FLOW:
            self.tg.send_message(chat_id, f"Недопустимый статус. Разрешено: {', '.join(STATUS_FLOW)}")
            return
        order["status"] = new_status
        order["updated_at"] = now_iso()
        customer_chat_id = order["customer"]["chat_id"]
        customer_msg = f"Статус заказа {order_id} обновлен: {STATUS_LABELS_RU.get(new_status, new_status)}."
        if note:
            customer_msg += f" Комментарий: {note}"
        self.tg.send_message(customer_chat_id, customer_msg)
        self.tg.send_message(chat_id, f"Заказ {order_id} обновлен: {new_status}.")

    def handle_dispatch(self, user_id: str, chat_id: int, text: str) -> None:
        if self.cfg.admin_chat_id and str(chat_id) != self.cfg.admin_chat_id:
            self.tg.send_message(chat_id, "Команда администратора недоступна в этом чате.")
            return
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            self.tg.send_message(chat_id, "Формат: /dispatch <ORDER_ID>")
            return
        order_id = parts[1].strip().upper()
        order = self.orders.get(order_id)
        if not order:
            self.tg.send_message(chat_id, f"Заказ {order_id} не найден.")
            return
        ok, message = self.dispatch_to_printer(order)
        self.tg.send_message(chat_id, message)
        if ok:
            customer_chat_id = order["customer"]["chat_id"]
            self.tg.send_message(customer_chat_id, f"Заказ {order_id} отправлен на печать.")

    def handle_deposit_proof(self, user_id: str, chat_id: int) -> None:
        session = self.sessions[user_id]
        order_id = session.get("order_id")
        if not order_id or order_id not in self.orders:
            self.tg.send_message(chat_id, "Активный заказ не найден. Отправьте /start.")
            session["step"] = "idle"
            return
        order = self.orders[order_id]
        order["deposit_status"] = "UNDER_REVIEW"
        order["status"] = "DEPOSIT_UNDER_REVIEW"
        order["updated_at"] = now_iso()
        self.tg.send_message(chat_id, f"Скриншот оплаты по {order_id} получен. Скоро подтвердим.")
        if self.cfg.admin_chat_id:
            self.tg.send_message(
                self.cfg.admin_chat_id,
                f"Получено подтверждение оплаты по {order_id}. Проверьте и обновите статус после подтверждения.",
            )
        session["step"] = "idle"

    def handle_intake_step(self, user_id: str, chat_id: int, text: str) -> None:
        session = self.sessions[user_id]
        step = session.get("step")
        answers = session.setdefault("answers", {})

        if step == "product_type":
            answers["product_type"] = text
            session["step"] = "example"
            self.tg.send_message(chat_id, "Шаг 2/10: Пришлите ссылку-пример или кратко опишите желаемый результат.")
            return

        if step == "example":
            answers["example"] = text
            session["step"] = "dimensions_mm"
            self.tg.send_message(chat_id, "Шаг 3/10: Размеры в мм в формате ДxШxВ (пример: 120x45x20).")
            return

        if step == "dimensions_mm":
            dims = self.parse_dimensions(text)
            if not dims:
                self.tg.send_message(chat_id, "Неверный формат. Используйте ДxШxВ в мм, например: 120x45x20.")
                return
            answers["dimensions_mm"] = dims
            session["step"] = "material"
            self.tg.send_message(chat_id, "Шаг 4/10: Материал? Ответьте: PLA, PETG или TPU.")
            return

        if step == "material":
            mat = text.upper()
            if mat not in MATERIAL_RATES:
                self.tg.send_message(chat_id, "Материал должен быть одним из: PLA, PETG, TPU.")
                return
            answers["material"] = mat
            session["step"] = "color"
            self.tg.send_message(chat_id, "Шаг 5/10: Нужный цвет?")
            return

        if step == "color":
            answers["color"] = text
            session["step"] = "deadline"
            self.tg.send_message(chat_id, "Шаг 6/10: Срок? Ответьте: standard или rush_24h.")
            return

        if step == "deadline":
            dl = text.lower()
            if dl not in DEADLINE_FEES:
                self.tg.send_message(chat_id, "Срок должен быть: standard или rush_24h.")
                return
            answers["deadline"] = dl
            session["step"] = "quantity"
            self.tg.send_message(chat_id, "Шаг 7/10: Количество (целое число).")
            return

        if step == "quantity":
            if not text.isdigit() or int(text) <= 0:
                self.tg.send_message(chat_id, "Количество должно быть положительным целым числом.")
                return
            answers["quantity"] = int(text)
            session["step"] = "postproc"
            self.tg.send_message(chat_id, "Шаг 8/10: Постобработка? Ответьте: none, basic или advanced.")
            return

        if step == "postproc":
            pp = text.lower()
            if pp not in POSTPROC_FEES:
                self.tg.send_message(chat_id, "Постобработка должна быть: none, basic или advanced.")
                return
            answers["postproc"] = pp
            session["step"] = "contact"
            self.tg.send_message(chat_id, "Шаг 9/10: Контактный телефон или @username.")
            return

        if step == "contact":
            answers["contact"] = text
            session["step"] = "delivery_mode"
            self.tg.send_message(chat_id, "Шаг 10/10: Доставка? Ответьте: pickup, cdek или yandex.")
            return

        if step == "delivery_mode":
            mode = text.lower()
            if mode not in DELIVERY_FEES:
                self.tg.send_message(chat_id, "Доставка должна быть: pickup, cdek или yandex.")
                return
            answers["delivery_mode"] = mode
            order = self.create_order(user_id, chat_id, session)
            proposal = self.generate_model_proposal(order["order_id"])
            proposal_mode = proposal.get("mode", "unknown")
            session["step"] = "await_model_approval"
            session["order_id"] = order["order_id"]
            self.tg.send_message(
                chat_id,
                f"Расчет готов по заказу {order['order_id']}:\n"
                f"Итого: {order['quote_total_rub']} RUB\n"
                f"Предоплата (50%): {order['deposit_required_rub']} RUB\n"
                f"Статус: {STATUS_LABELS_RU.get(order['status'], order['status'])}\n"
                f"Источник конструктора: {proposal_mode}\n\n"
                "Черновик модели для согласования:\n\n"
                f"{proposal['text']}\n\n"
                "Если согласны, отправьте: СОГЛАСОВАНО или /approve "
                f"{order['order_id']}\n"
                "Если нужны правки, напишите одним сообщением, что изменить.",
            )
            if self.cfg.admin_chat_id:
                self.tg.send_message(
                    self.cfg.admin_chat_id,
                    f"Новый заказ {order['order_id']} от пользователя {user_id}. "
                    f"Сумма {order['quote_total_rub']} RUB, предоплата {order['deposit_required_rub']} RUB. "
                    f"Источник конструктора: {proposal_mode}. Модель отправлена клиенту на согласование.",
                )
            return

        if step == "await_model_approval":
            order_id = session.get("order_id")
            if not order_id or order_id not in self.orders:
                self.tg.send_message(chat_id, "Активный заказ не найден. Отправьте /start.")
                session["step"] = "idle"
                return
            order = self.orders[order_id]
            normalized = text.strip().lower()
            if normalized in {"согласовано", "approve", "ok", "ок"}:
                self.approve_model(chat_id, order)
                session["step"] = "await_deposit_proof"
                return
            proposal = self.generate_model_proposal(order_id, revision_note=text)
            order["model_approval"] = {"status": "REVISION_REQUESTED", "updated_at": now_iso(), "note": text}
            order["updated_at"] = now_iso()
            self.tg.send_message(
                chat_id,
                "Обновил черновик модели с учетом комментария:\n\n"
                f"(Источник конструктора: {proposal.get('mode', 'unknown')})\n\n"
                f"{proposal['text']}\n\n"
                f"Если согласны, отправьте СОГЛАСОВАНО или /approve {order_id}.",
            )
            if self.cfg.admin_chat_id:
                self.tg.send_message(
                    self.cfg.admin_chat_id,
                    f"Клиент запросил правки по {order_id}: {text}",
                )
            return

        self.tg.send_message(chat_id, "Состояние сессии некорректно. Отправьте /start для перезапуска.")

    def create_order(self, user_id: str, chat_id: int, session: dict[str, Any]) -> dict[str, Any]:
        answers = session["answers"]
        dims = answers["dimensions_mm"]
        quantity = answers["quantity"]
        material = answers["material"]
        deadline = answers["deadline"]
        postproc = answers["postproc"]
        delivery_mode = answers["delivery_mode"]

        material_g, print_hours = self.estimate_print(dims, quantity)
        setup_fee = 150.0
        machine_rate = 250.0
        material_cost = material_g * MATERIAL_RATES[material]
        machine_cost = print_hours * machine_rate
        total = setup_fee + material_cost + machine_cost + POSTPROC_FEES[postproc] + DEADLINE_FEES[deadline] + DELIVERY_FEES[delivery_mode]
        total = max(total, 700.0)
        deposit = round(total * 0.5, 2)

        order_id = self.new_order_id()
        order = {
            "order_id": order_id,
            "source": session.get("source", "direct"),
            "customer": {
                "user_id": user_id,
                "chat_id": chat_id,
                "contact": answers["contact"],
            },
            "product_type": answers["product_type"],
            "example": answers["example"],
            "dimensions_mm": dims,
            "material": material,
            "color": answers["color"],
            "quantity": quantity,
            "deadline": deadline,
            "postproc": postproc,
            "delivery_mode": delivery_mode,
            "estimate": {
                "material_g": round(material_g, 2),
                "print_hours": round(print_hours, 2),
            },
            "quote_total_rub": round(total, 2),
            "deposit_required_rub": deposit,
            "deposit_status": "PENDING",
            "status": "REVIEWING_DETAILS",
            "model_proposal": {},
            "model_approval": {"status": "PENDING", "updated_at": now_iso(), "note": ""},
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        self.orders[order_id] = order
        return order

    def generate_model_proposal(self, order_id: str, revision_note: str | None = None) -> dict[str, Any]:
        order = self.orders[order_id]
        proposal = self.constructor.build(order, revision_note)
        order["model_proposal"] = proposal
        order["updated_at"] = now_iso()
        return proposal

    def approve_model(self, chat_id: int, order: dict[str, Any]) -> None:
        order["model_approval"] = {"status": "APPROVED", "updated_at": now_iso(), "note": ""}
        order["status"] = "WAITING_DEPOSIT"
        order["updated_at"] = now_iso()
        self.tg.send_message(
            chat_id,
            "Модель согласована.\n\n"
            f"Инструкция по оплате:\n{self.cfg.payment_instructions}\n\n"
            "После оплаты отправьте скриншот в этот чат.",
        )
        if self.cfg.auto_dispatch_on_approval and self.cfg.admin_chat_id:
            ok, message = self.dispatch_to_printer(order)
            self.tg.send_message(self.cfg.admin_chat_id, f"Автодиспетчер: {message}")
            if ok:
                self.tg.send_message(chat_id, f"Заказ {order['order_id']} отправлен на печать автоматически.")

    def dispatch_to_printer(self, order: dict[str, Any]) -> tuple[bool, str]:
        if not self.cfg.printer_dispatch_webhook_url:
            return False, "PRINTER_DISPATCH_WEBHOOK_URL не настроен."
        model_url = self.extract_model_url_for_dispatch(order)
        if not model_url:
            return False, "Не найден model_url для отправки на принтер."
        payload = {
            "order_id": order["order_id"],
            "customer_user_id": order["customer"]["user_id"],
            "model_url": model_url,
            "material": order["material"],
            "color": order["color"],
            "quantity": order["quantity"],
            "postproc": order["postproc"],
            "deadline": order["deadline"],
            "delivery_mode": order["delivery_mode"],
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.printer_dispatch_token:
            headers["Authorization"] = f"Bearer {self.cfg.printer_dispatch_token}"
        req = urllib.request.Request(
            url=self.cfg.printer_dispatch_webhook_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = resp.read().decode("utf-8", "replace")
                order["print_dispatch"] = {
                    "status": "SUBMITTED",
                    "at": now_iso(),
                    "response_status": resp.status,
                    "response_body": body[:500],
                }
            order["status"] = "PRINTING"
            order["updated_at"] = now_iso()
            return True, f"Заказ {order['order_id']} отправлен в диспетчер печати."
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            order["print_dispatch"] = {"status": "FAILED", "at": now_iso(), "error": f"HTTP {e.code}: {detail[:300]}"}
            order["updated_at"] = now_iso()
            return False, f"Ошибка отправки на принтер: HTTP {e.code}."
        except Exception as e:  # noqa: BLE001
            order["print_dispatch"] = {"status": "FAILED", "at": now_iso(), "error": str(e)}
            order["updated_at"] = now_iso()
            return False, f"Ошибка отправки на принтер: {e}"

    def extract_model_url_for_dispatch(self, order: dict[str, Any]) -> str | None:
        proposal = order.get("model_proposal", {})
        if isinstance(proposal, dict):
            direct_url = proposal.get("model_url")
            if isinstance(direct_url, str) and direct_url.startswith(("http://", "https://")):
                return direct_url
            text = proposal.get("text", "")
        else:
            text = ""
        if not isinstance(text, str):
            return None
        m = re.search(r"model_url:\s*(https?://\S+)", text)
        if m:
            return m.group(1)
        m = re.search(r"(https?://\S+)", text)
        if m:
            return m.group(1)
        return None

    def new_order_id(self) -> str:
        if not self.orders:
            return "ORD-0001"
        nums = [int(k.split("-")[1]) for k in self.orders.keys() if re.match(r"^ORD-\d{4}$", k)]
        next_num = (max(nums) + 1) if nums else 1
        return f"ORD-{next_num:04d}"

    def parse_dimensions(self, text: str) -> list[int] | None:
        m = re.match(r"^\s*(\d{1,4})\s*[xX*]\s*(\d{1,4})\s*[xX*]\s*(\d{1,4})\s*$", text)
        if not m:
            return None
        dims = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        if any(d <= 0 for d in dims):
            return None
        return dims

    def estimate_print(self, dims_mm: list[int], quantity: int) -> tuple[float, float]:
        l, w, h = dims_mm
        volume_mm3 = l * w * h
        effective_volume_cm3 = (volume_mm3 / 1000.0) * 0.2
        material_g_single = effective_volume_cm3 * 1.24
        print_hours_single = max(0.5, effective_volume_cm3 / 12.0)
        return material_g_single * quantity, print_hours_single * quantity


def main() -> None:
    load_local_env(Path(".env"))
    cfg = load_config()
    bot = Bot(cfg)
    bot.run_forever()


if __name__ == "__main__":
    main()
