from typing import List
from signals.core.signal import Signal, Domain, Priority
from signals.collectors.mock import MockSalesData
import uuid


class SalesAnalyzer:
    def __init__(self, config: dict, data_source=None):
        self.cfg = config["sales"]
        self.data = data_source or MockSalesData()

    def analyze(self) -> List[Signal]:
        signals = []
        signals.extend(self._check_brand_sales())
        signals.extend(self._check_regional_sales())
        signals.extend(self._check_stalled_deals())
        signals.extend(self._check_top_clients())
        return signals

    def _check_brand_sales(self) -> List[Signal]:
        signals = []
        brand_sales = self.data.get_brand_sales()
        threshold = -self.cfg["plan_underperformance_pct"]

        for item in brand_sales:
            if item["deviation_pct"] <= threshold:
                priority = Priority.CRITICAL if item["deviation_pct"] <= threshold * 2 else Priority.HIGH
                fact_m = item["fact"] / 1_000_000
                plan_m = item["plan"] / 1_000_000
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.SALES,
                    priority=priority,
                    title=f"План продаж не выполняется: {item['brand']}",
                    body=f"Факт: {fact_m:.1f} млн ₽ | План: {plan_m:.1f} млн ₽",
                    value=item["fact"],
                    threshold=item["plan"],
                    deviation_pct=item["deviation_pct"],
                    brand=item["brand"],
                    action="Проверить промо-активности, наличие на полке, работу КАМ по бренду",
                    data=item,
                ))
        return signals

    def _check_regional_sales(self) -> List[Signal]:
        signals = []
        regions = self.data.get_sales_plan()
        threshold = -self.cfg["plan_underperformance_pct"]

        problematic = [r for r in regions if r["deviation_pct"] <= threshold]
        if len(problematic) >= 3:
            names = ", ".join(r["region"] for r in problematic[:3])
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.SALES,
                priority=Priority.HIGH,
                title=f"Отставание плана в {len(problematic)} регионах",
                body=f"Регионы: {names}" + (" и др." if len(problematic) > 3 else ""),
                value=len(problematic),
                threshold=2,
                action="Провести ревью с региональными директорами",
                data={"regions": problematic},
            ))
        elif problematic:
            for r in problematic:
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.SALES,
                    priority=Priority.MEDIUM,
                    title=f"Отставание плана: {r['region']}",
                    body=f"Факт: {r['fact']/1_000_000:.1f} млн | План: {r['plan']/1_000_000:.1f} млн",
                    value=r["fact"],
                    threshold=r["plan"],
                    deviation_pct=r["deviation_pct"],
                    region=r["region"],
                    action="Запросить объяснительную у регионального менеджера",
                    data=r,
                ))
        return signals

    def _check_stalled_deals(self) -> List[Signal]:
        signals = []
        deals = self.data.get_crm_stalled_deals()
        stuck = [d for d in deals if d["days_no_activity"] >= self.cfg["crm_deal_stuck_days"]]

        if not stuck:
            return signals

        total_amount = sum(d["amount"] for d in stuck) / 1_000_000
        critical = [d for d in stuck if d["days_no_activity"] >= self.cfg["crm_deal_stuck_days"] * 2]

        priority = Priority.HIGH if critical else Priority.MEDIUM
        details = "\n".join(
            f"• {d['client']} — {d['amount']/1_000_000:.1f} млн ₽ ({d['days_no_activity']} дн. без движения, {d['manager']})"
            for d in stuck[:4]
        )
        signals.append(Signal(
            id=str(uuid.uuid4()),
            domain=Domain.SALES,
            priority=priority,
            title=f"Зависшие сделки в CRM: {len(stuck)} шт.",
            body=f"Сумма под угрозой: {total_amount:.1f} млн ₽\n{details}",
            value=len(stuck),
            threshold=self.cfg["crm_deal_stuck_days"],
            action="Разобрать на ближайшем sales-стендапе",
            data={"deals": stuck},
        ))
        return signals

    def _check_top_clients(self) -> List[Signal]:
        signals = []
        clients = self.data.get_top_client_dynamics()
        threshold = -self.cfg["top_client_drop_pct"]

        for c in clients:
            if c["change_pct"] <= threshold:
                prev_m = c["prev_month"] / 1_000_000
                curr_m = c["curr_month"] / 1_000_000
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.SALES,
                    priority=Priority.HIGH,
                    title=f"Резкое снижение закупок: {c['client']}",
                    body=f"Пред. месяц: {prev_m:.1f} млн ₽ → Текущий: {curr_m:.1f} млн ₽",
                    value=c["curr_month"],
                    threshold=c["prev_month"],
                    deviation_pct=c["change_pct"],
                    action=f"Выяснить причину у КАМ, проверить удовлетворённость клиента",
                    data=c,
                ))
        return signals
