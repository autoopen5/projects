"""
Mock-коллектор с реалистичными данными фармкомпании.
Используется для разработки и демо. Заменяется реальными коннекторами.
"""
import random
from datetime import date, datetime, timedelta
from typing import Any


BRANDS = [
    "Арбидол", "Анаферон", "Циклоферон", "Кагоцел",
    "Гриппферон", "Цитовир", "Ингавирин", "Генферон",
    "Виферон", "Амиксин"
]

REGIONS = [
    "Москва", "СПб", "Екатеринбург", "Новосибирск",
    "Казань", "Ростов-на-Дону", "Самара", "Краснодар"
]


def _rnd(base: float, noise_pct: float = 0.1) -> float:
    return base * (1 + random.uniform(-noise_pct, noise_pct))


class MockFinanceData:
    def get_revenue(self) -> dict:
        today = date.today()
        month_day = today.day
        month_days = 30
        plan_monthly = 450_000_000  # 450 млн руб в месяц

        # Проращиваем план на текущий день месяца
        plan_to_date = plan_monthly * (month_day / month_days)
        # Иногда создаём проблемную ситуацию
        scenario = random.choice(["ok", "ok", "ok", "underperform", "critical"])
        if scenario == "ok":
            fact = _rnd(plan_to_date, 0.05)
        elif scenario == "underperform":
            fact = plan_to_date * random.uniform(0.78, 0.88)
        else:
            fact = plan_to_date * random.uniform(0.55, 0.70)

        return {
            "plan_monthly": plan_monthly,
            "plan_to_date": round(plan_to_date),
            "fact_to_date": round(fact),
            "deviation_pct": round((fact - plan_to_date) / plan_to_date * 100, 1),
            "period": today.strftime("%Y-%m"),
        }

    def get_cashflow(self) -> dict:
        scenarios = [
            {"balance": 85_000_000, "outflow_daily": 4_200_000},
            {"balance": 35_000_000, "outflow_daily": 4_500_000},  # tight
            {"balance": 18_000_000, "outflow_daily": 4_800_000},  # danger
        ]
        weights = [0.6, 0.25, 0.15]
        s = random.choices(scenarios, weights=weights)[0]
        days_left = round(s["balance"] / s["outflow_daily"])
        return {
            "current_balance": s["balance"],
            "daily_outflow": s["outflow_daily"],
            "days_covered": days_left,
            "incoming_expected_7d": round(_rnd(s["outflow_daily"] * 6, 0.3)),
        }

    def get_receivables(self) -> dict:
        total_revenue_monthly = 450_000_000
        overdue = round(_rnd(total_revenue_monthly * 0.08, 0.4))
        return {
            "total_receivables": round(_rnd(total_revenue_monthly * 0.18, 0.15)),
            "overdue_30d": overdue,
            "overdue_60d": round(overdue * 0.4),
            "top_debtors": [
                {"name": "Аптечная сеть Ригла", "amount": round(_rnd(8_000_000, 0.3)), "days": random.randint(25, 75)},
                {"name": "Сбер Еаптека", "amount": round(_rnd(5_500_000, 0.3)), "days": random.randint(20, 50)},
                {"name": "Апрель", "amount": round(_rnd(3_200_000, 0.3)), "days": random.randint(15, 45)},
            ]
        }

    def get_brand_margins(self) -> list:
        result = []
        for brand in BRANDS:
            margin = random.uniform(18, 55)
            result.append({
                "brand": brand,
                "margin_pct": round(margin, 1),
                "revenue_monthly": round(_rnd(45_000_000, 0.4)),
            })
        return result


class MockSalesData:
    def get_sales_plan(self) -> list:
        result = []
        for region in REGIONS:
            plan = round(_rnd(55_000_000, 0.2))
            deviation = random.uniform(-0.30, 0.10)
            fact = round(plan * (1 + deviation))
            result.append({
                "region": region,
                "plan": plan,
                "fact": fact,
                "deviation_pct": round(deviation * 100, 1),
            })
        return result

    def get_brand_sales(self) -> list:
        result = []
        for brand in BRANDS:
            plan = round(_rnd(45_000_000, 0.3))
            # Сезонность: некоторые бренды просели
            if brand in ["Арбидол", "Кагоцел"] and random.random() > 0.6:
                deviation = random.uniform(-0.35, -0.15)
            else:
                deviation = random.uniform(-0.15, 0.05)
            fact = round(plan * (1 + deviation))
            result.append({
                "brand": brand,
                "plan": plan,
                "fact": fact,
                "deviation_pct": round(deviation * 100, 1),
            })
        return result

    def get_crm_stalled_deals(self) -> list:
        deals = []
        for _ in range(random.randint(2, 8)):
            deals.append({
                "deal_id": f"DEAL-{random.randint(1000, 9999)}",
                "client": random.choice(["Протек", "Пульс", "Катрен", "РОСТА", "ФармЗнак"]),
                "amount": round(_rnd(2_500_000, 0.5)),
                "stage": random.choice(["Переговоры", "Коммерческое предложение", "Согласование договора"]),
                "days_no_activity": random.randint(5, 21),
                "manager": random.choice(["Иванов А.", "Петрова К.", "Сидоров М.", "Козлова Н."]),
            })
        return deals

    def get_top_client_dynamics(self) -> list:
        clients = []
        for name in ["Протек", "Пульс", "Катрен", "РОСТА", "Сбер Еаптека"]:
            prev = round(_rnd(25_000_000, 0.2))
            change = random.uniform(-0.45, 0.10)
            curr = round(prev * (1 + change))
            clients.append({
                "client": name,
                "prev_month": prev,
                "curr_month": curr,
                "change_pct": round(change * 100, 1),
            })
        return clients


class MockProductionData:
    def get_production_plan(self) -> list:
        result = []
        for brand in random.sample(BRANDS, 5):
            plan = random.randint(50_000, 500_000)  # упаковок
            deviation = random.uniform(-0.15, 0.05)
            fact = round(plan * (1 + deviation))
            result.append({
                "brand": brand,
                "plan_units": plan,
                "fact_units": fact,
                "deviation_pct": round(deviation * 100, 1),
            })
        return result

    def get_raw_materials(self) -> list:
        materials = []
        items = [
            ("Умифеновир API", "Арбидол"),
            ("Антителаносного анаферона", "Анаферон"),
            ("Циклоферон субст.", "Циклоферон"),
            ("Кагоцел субст.", "Кагоцел"),
            ("Интерферон α-2b", "Гриппферон"),
            ("Вспомогательные в-ва", None),
        ]
        for name, brand in items:
            days_left = random.choice([5, 8, 12, 18, 25, 40, 60])
            materials.append({
                "material": name,
                "brand": brand,
                "stock_days": days_left,
                "stock_kg": round(days_left * _rnd(50, 0.2)),
            })
        return materials

    def get_expiry_batches(self) -> list:
        batches = []
        for _ in range(random.randint(0, 4)):
            brand = random.choice(BRANDS)
            days_to_expiry = random.randint(30, 100)
            batches.append({
                "batch_id": f"B{random.randint(10000, 99999)}",
                "brand": brand,
                "units": random.randint(5_000, 50_000),
                "expiry_date": (date.today() + timedelta(days=days_to_expiry)).isoformat(),
                "days_to_expiry": days_to_expiry,
                "warehouse": random.choice(["Склад-1 Обнинск", "Склад-2 Москва"]),
            })
        return batches


class MockHRData:
    def get_headcount(self) -> dict:
        return {
            "total": 1240,
            "resigned_this_month": random.randint(8, 25),
            "hired_this_month": random.randint(5, 20),
            "open_vacancies": random.randint(15, 45),
        }

    def get_kpi_underperformers(self) -> list:
        underperformers = []
        managers = [
            ("Коммерческий директор", "Смирнов В.В."),
            ("Директор по производству", "Кузнецов И.А."),
            ("HR директор", "Орлова М.С."),
            ("Директор по логистике", "Федоров А.П."),
            ("Региональный директор МСК", "Белова Т.Н."),
        ]
        for role, name in random.sample(managers, random.randint(1, 3)):
            kpi_pct = random.randint(60, 95)
            if kpi_pct < 80:
                underperformers.append({
                    "name": name,
                    "role": role,
                    "kpi_pct": kpi_pct,
                    "period": date.today().strftime("%B %Y"),
                })
        return underperformers

    def get_long_open_vacancies(self) -> list:
        vacancies = []
        positions = [
            "Медицинский представитель (МСК)",
            "Продакт-менеджер Арбидол",
            "Аналитик данных",
            "Региональный менеджер СПб",
        ]
        for pos in random.sample(positions, random.randint(1, 3)):
            days = random.randint(15, 60)
            vacancies.append({
                "position": pos,
                "days_open": days,
                "candidates_in_funnel": random.randint(0, 5),
            })
        return vacancies


class MockSeasonalData:
    def get_incidence_index(self) -> dict:
        today = date.today()
        month = today.month
        # Сезон: октябрь-март
        if month in [11, 12, 1, 2]:
            base_index = random.uniform(140, 200)
        elif month in [10, 3]:
            base_index = random.uniform(90, 140)
        else:
            base_index = random.uniform(40, 90)

        return {
            "current_index": round(base_index, 1),
            "prev_week_index": round(base_index * random.uniform(0.85, 1.15), 1),
            "season_active": base_index > 120,
            "trend": "растёт" if base_index > 120 else "снижается",
            "regions_hot": random.sample(REGIONS, random.randint(2, 5)) if base_index > 100 else [],
        }

    def get_stock_readiness(self) -> list:
        today = date.today()
        month = today.month
        # Перед сезоном
        pre_season = month in [8, 9, 10]
        result = []
        for brand in BRANDS[:5]:  # топ противовирусные
            stock_days = random.randint(20, 90) if pre_season else random.randint(30, 120)
            result.append({
                "brand": brand,
                "stock_days": stock_days,
                "ready_for_season": stock_days >= 45,
            })
        return result
