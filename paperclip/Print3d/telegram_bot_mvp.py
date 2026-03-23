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
    openai_api_key: str | None
    openai_model: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    return Config(
        token=token,
        admin_chat_id=admin_chat_id,
        payment_instructions=payment_instructions,
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
        self.openai_api_key = cfg.openai_api_key
        self.openai_model = cfg.openai_model

    def build(self, order: dict[str, Any], revision_note: str | None = None) -> dict[str, Any]:
        if self.openai_api_key:
            try:
                text = self.build_with_openai(order, revision_note)
                return {"mode": "openai", "text": text, "generated_at": now_iso()}
            except Exception:  # noqa: BLE001
                pass
        return {"mode": "heuristic", "text": self.build_heuristic(order, revision_note), "generated_at": now_iso()}

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
            "/help - показать помощь\n"
            "Для администратора: /setstatus <ORDER_ID> <STATUS> [комментарий]",
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
            session["step"] = "await_model_approval"
            session["order_id"] = order["order_id"]
            self.tg.send_message(
                chat_id,
                f"Расчет готов по заказу {order['order_id']}:\n"
                f"Итого: {order['quote_total_rub']} RUB\n"
                f"Предоплата (50%): {order['deposit_required_rub']} RUB\n"
                f"Статус: {STATUS_LABELS_RU.get(order['status'], order['status'])}\n\n"
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
                    "Модель отправлена клиенту на согласование.",
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
    cfg = load_config()
    bot = Bot(cfg)
    bot.run_forever()


if __name__ == "__main__":
    main()
