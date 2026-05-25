from typing import List
from signals.core.signal import Signal, Domain, Priority
from signals.collectors.mock import MockFinanceData
import uuid


class FinanceAnalyzer:
    def __init__(self, config: dict, data_source=None):
        self.cfg = config["finance"]
        self.data = data_source or MockFinanceData()

    def analyze(self) -> List[Signal]:
        signals = []
        signals.extend(self._check_revenue())
        signals.extend(self._check_cashflow())
        signals.extend(self._check_receivables())
        signals.extend(self._check_margins())
        return signals

    def _check_revenue(self) -> List[Signal]:
        signals = []
        rev = self.data.get_revenue()
        dev = rev["deviation_pct"]
        critical_threshold = -self.cfg["revenue_critical_pct"]
        high_threshold = -self.cfg["revenue_drop_pct"]

        if dev <= critical_threshold:
            priority = Priority.CRITICAL
            action = "Срочное совещание: анализ причин провала плана"
        elif dev <= high_threshold:
            priority = Priority.HIGH
            action = "Запросить объяснения коммерческого директора, проверить воронку"
        else:
            return signals

        fact_m = rev["fact_to_date"] / 1_000_000
        plan_m = rev["plan_to_date"] / 1_000_000
        signals.append(Signal(
            id=str(uuid.uuid4()),
            domain=Domain.FINANCE,
            priority=priority,
            title="Отставание выручки от плана",
            body=f"Факт: {fact_m:.1f} млн ₽ | План: {plan_m:.1f} млн ₽",
            value=rev["fact_to_date"],
            threshold=rev["plan_to_date"],
            deviation_pct=dev,
            action=action,
            data=rev,
        ))
        return signals

    def _check_cashflow(self) -> List[Signal]:
        signals = []
        cf = self.data.get_cashflow()
        days = cf["days_covered"]

        if days <= self.cfg["cashflow_critical_days"]:
            priority = Priority.CRITICAL
            action = f"СРОЧНО: организовать поступление средств. Остаток закроется через {days} дн."
        elif days <= self.cfg["cashflow_danger_days"]:
            priority = Priority.HIGH
            action = f"Ускорить сбор дебиторки, проверить входящие платежи на неделю"
        else:
            return signals

        balance_m = cf["current_balance"] / 1_000_000
        signals.append(Signal(
            id=str(uuid.uuid4()),
            domain=Domain.FINANCE,
            priority=priority,
            title="Риск кассового разрыва",
            body=f"Баланс: {balance_m:.1f} млн ₽ | Хватит на {days} дней при текущем расходе",
            value=days,
            threshold=self.cfg["cashflow_danger_days"],
            action=action,
            data=cf,
        ))
        return signals

    def _check_receivables(self) -> List[Signal]:
        signals = []
        rec = self.data.get_receivables()
        revenue_monthly = 450_000_000
        overdue_pct = rec["overdue_30d"] / revenue_monthly * 100

        if overdue_pct >= self.cfg["receivables_critical_pct"]:
            priority = Priority.HIGH
            overdue_m = rec["overdue_30d"] / 1_000_000
            top = rec["top_debtors"][0]
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.FINANCE,
                priority=priority,
                title="Высокая просроченная дебиторка",
                body=f"Просрочено >30 дн: {overdue_m:.1f} млн ₽ ({overdue_pct:.1f}% от выручки)\nКрупнейший должник: {top['name']} — {top['amount']/1_000_000:.1f} млн ₽ ({top['days']} дн.)",
                value=rec["overdue_30d"],
                threshold=revenue_monthly * self.cfg["receivables_critical_pct"] / 100,
                deviation_pct=overdue_pct,
                action="Включить в повестку встречи с коммерческим директором",
                data=rec,
            ))
        return signals

    def _check_margins(self) -> List[Signal]:
        signals = []
        margins = self.data.get_brand_margins()
        low_margin = [b for b in margins if b["margin_pct"] < self.cfg["margin_min_pct"]]

        for brand_data in low_margin:
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.FINANCE,
                priority=Priority.MEDIUM,
                title=f"Низкая маржа: {brand_data['brand']}",
                body=f"Маржа {brand_data['margin_pct']}% при минимуме {self.cfg['margin_min_pct']}%",
                value=brand_data["margin_pct"],
                threshold=self.cfg["margin_min_pct"],
                deviation_pct=brand_data["margin_pct"] - self.cfg["margin_min_pct"],
                brand=brand_data["brand"],
                action="Проверить ценообразование и структуру себестоимости",
                data=brand_data,
            ))
        return signals
