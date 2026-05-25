from typing import List
from signals.core.signal import Signal, Domain, Priority
from signals.collectors.mock import MockProductionData
import uuid


class ProductionAnalyzer:
    def __init__(self, config: dict, data_source=None):
        self.cfg = config["production"]
        self.data = data_source or MockProductionData()

    def analyze(self) -> List[Signal]:
        signals = []
        signals.extend(self._check_production_plan())
        signals.extend(self._check_raw_materials())
        signals.extend(self._check_expiry())
        return signals

    def _check_production_plan(self) -> List[Signal]:
        signals = []
        plan_data = self.data.get_production_plan()
        threshold = -self.cfg["plan_deviation_pct"]

        for item in plan_data:
            if item["deviation_pct"] <= threshold:
                priority = Priority.HIGH if item["deviation_pct"] <= threshold * 2 else Priority.MEDIUM
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.PRODUCTION,
                    priority=priority,
                    title=f"Производственный план под угрозой: {item['brand']}",
                    body=f"Факт: {item['fact_units']:,} уп. | План: {item['plan_units']:,} уп.",
                    value=item["fact_units"],
                    threshold=item["plan_units"],
                    deviation_pct=item["deviation_pct"],
                    brand=item["brand"],
                    action="Уточнить причины у директора по производству: сырьё, оборудование, персонал",
                    data=item,
                ))
        return signals

    def _check_raw_materials(self) -> List[Signal]:
        signals = []
        materials = self.data.get_raw_materials()

        for m in materials:
            days = m["stock_days"]
            if days <= self.cfg["raw_stock_critical_days"]:
                priority = Priority.CRITICAL
                action = f"СРОЧНО: разместить экстренный заказ {m['material']}"
            elif days <= self.cfg["raw_stock_min_days"]:
                priority = Priority.HIGH
                action = f"Ускорить закупку {m['material']}, проверить статус поставки"
            else:
                continue

            brand_str = f" (для {m['brand']})" if m["brand"] else ""
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.PRODUCTION,
                priority=priority,
                title=f"Критичный остаток сырья{brand_str}",
                body=f"{m['material']}: запас на {days} дней ({m['stock_kg']} кг)",
                value=days,
                threshold=self.cfg["raw_stock_min_days"],
                brand=m.get("brand"),
                action=action,
                data=m,
            ))
        return signals

    def _check_expiry(self) -> List[Signal]:
        signals = []
        batches = self.data.get_expiry_batches()

        for b in batches:
            if b["days_to_expiry"] <= self.cfg["expiry_warning_days"]:
                priority = Priority.HIGH if b["days_to_expiry"] <= 60 else Priority.MEDIUM
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.PRODUCTION,
                    priority=priority,
                    title=f"Истекает срок годности партии {b['brand']}",
                    body=f"Партия {b['batch_id']}: {b['units']:,} уп. | Срок: {b['expiry_date']} ({b['days_to_expiry']} дн.) | {b['warehouse']}",
                    value=b["days_to_expiry"],
                    threshold=self.cfg["expiry_warning_days"],
                    brand=b["brand"],
                    action="Ускорить отгрузку или рассмотреть уценку/утилизацию",
                    data=b,
                ))
        return signals
