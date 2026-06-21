# Домашняя ситиферма: план развития

> Старт: минвата + полив по расписанию + свет по расписанию  
> Финиш: умная ферма с компьютерным зрением и Telegram-управлением

---

## Что есть сейчас (база)

- Субстрат: минеральная вата
- Полив: по расписанию (cron/таймер)
- Освещение: по расписанию
- Управление: ручное или скрипт на VPS/Raspberry Pi

---

## Этапы развития

### Этап 1 — Стабилизация (сейчас, 1-2 недели)
**Цель**: убедиться что текущая автоматика надёжна

- [ ] Задокументировать текущее расписание полива (когда, сколько, как долго)
- [ ] Задокументировать расписание света (часов в сутки, спектр)
- [ ] Понять что растёт (культуры, стадии)
- [ ] Есть ли уведомления если что-то пошло не так?

**Итог**: знаем что у нас есть и что работает

---

### Этап 2 — Сенсоры (2-4 недели, ~3000-5000 руб)
**Цель**: видеть что происходит внутри, не заходя физически

#### Что добавить:
| Датчик | Что измеряет | Цена |
|--------|--------------|------|
| DHT22 | Температура + влажность воздуха | 200 руб |
| DS18B20 | Температура питательного раствора | 150 руб |
| pH-метр (аналог) | pH раствора | 500-1500 руб |
| EC-метр | Концентрация питательных веществ | 800-2000 руб |
| Датчик потока воды | Контроль помпы | 300 руб |

#### Платформа:
- **Raspberry Pi Zero 2W** (1500 руб) — уже Linux, Python, Wi-Fi
- Или **ESP32** (300 руб) — дешевле, но меньше возможностей

#### Код (Python, Raspberry Pi):
```python
import time
import json
from datetime import datetime

# Пример сбора данных с DHT22
import Adafruit_DHT

SENSOR = Adafruit_DHT.DHT22
PIN = 4  # GPIO pin

def read_climate():
    humidity, temperature = Adafruit_DHT.read_retry(SENSOR, PIN)
    return {
        "timestamp": datetime.now().isoformat(),
        "temp_air": round(temperature, 1),
        "humidity": round(humidity, 1)
    }

def log_to_file(data):
    with open("/home/pi/farm_log.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

while True:
    data = read_climate()
    log_to_file(data)
    print(data)
    time.sleep(300)  # каждые 5 минут
```

**Результат**: данные в файл → можно строить графики

---

### Этап 3 — Telegram-бот управления (параллельно с этапом 2)
**Цель**: видеть статус и управлять с телефона

#### Команды бота:
- `/status` — текущие показатели (температура, влажность, pH)
- `/water now` — запустить полив немедленно
- `/light on/off` — управление светом
- `/photo` — сделать снимок с камеры
- `/log today` — статистика за день

#### Структура:
```
farm-bot/
├── bot.py          # Telegram bot (aiogram)
├── sensors.py      # Чтение с датчиков
├── relay.py        # Управление реле (насос, свет)
├── camera.py       # Работа с камерой
└── config.py       # Токен, пины GPIO
```

```python
# bot.py — скелет
from aiogram import Bot, Dispatcher, types
import asyncio

TOKEN = "YOUR_TOKEN"
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(commands=["status"])
async def status(message: types.Message):
    from sensors import read_all
    data = read_all()
    await message.answer(
        f"🌡 Воздух: {data['temp_air']}°C, {data['humidity']}%\n"
        f"💧 Раствор: {data['temp_water']}°C\n"
        f"⚗️ pH: {data['ph']}\n"
        f"🔌 EC: {data['ec']} mS/cm"
    )

@dp.message(commands=["photo"])
async def photo(message: types.Message):
    from camera import capture
    img_path = capture()
    with open(img_path, "rb") as f:
        await message.answer_photo(f)

async def main():
    await dp.start_polling(bot)

asyncio.run(main())
```

---

### Этап 4 — Камера и мониторинг (4-6 недель, ~2000-4000 руб)
**Цель**: видеть растения в реальном времени

#### Железо:
- **Raspberry Pi Camera Module 3** (1500 руб) — лучшее качество
- Или обычная USB-веб-камера (500 руб) — проще
- Позиция: сверху (overhead) для равномерного охвата

#### Базовый мониторинг (без AI):
```python
# camera.py
import subprocess
from datetime import datetime
import os

PHOTO_DIR = "/home/pi/farm_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

def capture():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{PHOTO_DIR}/{timestamp}.jpg"
    subprocess.run(["libcamera-jpeg", "-o", path, "--width", "1920", "--height", "1080"])
    return path

def schedule_timelapse():
    # Снимок каждые 2 часа — таймлапс роста
    import schedule
    schedule.every(2).hours.do(capture)
```

**Профит**: таймлапс роста растений — видишь как растут за неделю одним видео

---

### Этап 5 — Компьютерное зрение (2-3 месяца после камеры)
**Цель**: автоматически обнаруживать проблемы

#### Что детектировать:
| Проблема | Как выглядит | Критичность |
|----------|--------------|-------------|
| Пожелтение листьев | Изменение цвета | Высокая |
| Увядание | Опущенные листья | Высокая |
| Плесень/гниль | Белые/серые пятна | Критическая |
| Хорошее состояние | Тёмно-зелёный, упругий | Норма |

#### Подход 1 — Готовая модель (быстро):
```python
# Используем Claude API для анализа фото
import anthropic
import base64

client = anthropic.Anthropic()

def analyze_plant(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """Анализируй растение на фото гидропонной фермы.
                    Ответь кратко:
                    1. Общее состояние (хорошее/требует внимания/критическое)
                    2. Что видишь (цвет листьев, тургор, признаки болезней)
                    3. Что делать (если проблема)
                    Ответ в 3-5 предложениях."""
                }
            ]
        }]
    )
    return response.content[0].text

def daily_check():
    from camera import capture
    img = capture()
    analysis = analyze_plant(img)
    
    # Отправить в Telegram если проблема
    if "критическое" in analysis.lower() or "требует внимания" in analysis.lower():
        send_alert(f"⚠️ Ферма требует внимания!\n\n{analysis}")
    
    return analysis
```

#### Подход 2 — Обучить свою модель (долго, но точнее):
- Собрать 200+ фото с метками (здоров/болен/стресс)
- Обучить YOLOv8 или EfficientNet
- Время: 2-3 месяца сбора данных + 1 неделя обучения

**Рекомендация**: начать с Claude API (Подход 1) — работает сразу

---

### Этап 6 — Умная адаптация (финал)
**Цель**: ферма сама корректирует условия

- Анализ данных сенсоров → автокоррекция расписания полива
- Детекция стресса → SMS/Telegram + предложение действий
- История роста → предсказание урожая
- Dashboard в Grafana (красивые графики)

---

## Стек технологий

| Уровень | Технология | Зачем |
|---------|-----------|-------|
| Железо | Raspberry Pi 4 (или Zero 2W) | Мозг фермы |
| Реле | 4-канальное реле 5V | Управление помпой и светом |
| Язык | Python 3.11 | Всё на одном языке |
| Бот | aiogram 3.x | Telegram-управление |
| CV | Claude API / OpenCV | Анализ фото |
| БД | SQLite или InfluxDB | Хранение метрик |
| Графики | Grafana + InfluxDB | Визуализация |
| Деплой | systemd сервисы | Автозапуск |

---

## Бюджет поэтапно

| Этап | Что купить | Стоимость |
|------|-----------|-----------|
| 2 | Датчики (DHT22, DS18B20) | ~1000 руб |
| 2 | Raspberry Pi Zero 2W | ~1500 руб |
| 3 | — (только код) | 0 |
| 4 | Камера Pi Camera 3 | ~1500 руб |
| 5 | Claude API | ~200 руб/мес |
| 6 | — (только код) | 0 |
| **Итого** | | **~4500 руб** |

---

## Следующий шаг прямо сейчас

1. Ответить: на чём сейчас работает автоматика? (Raspberry Pi? Arduino? ESP32? просто розетки с таймером?)
2. Что растёшь сейчас?
3. Исходя из этого — пишем код под конкретное железо

---

*Обновлено: июнь 2026*
