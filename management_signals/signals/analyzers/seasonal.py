from datetime import date
from typing import List
from signals.core.signal import Signal, Domain, Priority
from signals.collectors.mock import MockSeasonalData
import uuid


class SeasonalAnalyzer:
    def __init__(self, config: dict, data_source=None):
        self.cfg = config["seasonal"]
        self.data = data_source or MockSeasonalData()

    def analyze(self) -> List[Signal]:
        signals = []
        signals.extend(self._check_incidence())
        signals.extend(self._check_stock_readiness())
        return signals

    def _check_incidence(self) -> List[Signal]:
        signals = []
        inc = self.data.get_incidence_index()
        threshold = self.cfg["season_start_threshold"]
        idx = inc["current_index"]
        prev = inc["prev_week_index"]
        week_change = (idx - prev) / prev * 100 if prev else 0

        if inc["season_active"]:
            priority = Priority.HIGH if week_change > 15 else Priority.INFO
            hot_regions = ", ".join(inc["regions_hot"][:4]) if inc["regions_hot"] else "данные уточняются"
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.SEASONAL,
                priority=priority,
                title="Эпидсезон активен — высокий индекс заболеваемости",
                body=f"Индекс: {idx} (порог {threshold}) | Неделя: {'+' if week_change > 0 else ''}{week_change:.1f}%\nГорячие регионы: {hot_regions}",
                value=idx,
                threshold=threshold,
                action="Проверить наличие товара в ключевых регионах, усилить промо",
                data=inc,
            ))
        elif month := date.today().month:
            # Предупреждение перед сезоном
            if month in [8, 9, 10] and idx > threshold * 0.7:
                signals.append(Signal(
                    id=str(uuid.uuid4()),
                    domain=Domain.SEASONAL,
                    priority=Priority.MEDIUM,
                    title="Приближается эпидсезон",
                    body=f"Индекс заболеваемости: {idx} и {inc['trend']}. Порог сезона: {threshold}",
                    value=idx,
                    threshold=threshold,
                    action="Убедиться в готовности склада, согласовать промо-план на сезон",
                    data=inc,
                ))
        return signals

    def _check_stock_readiness(self) -> List[Signal]:
        signals = []
        month = date.today().month
        if month not in [8, 9, 10]:
            return signals

        stocks = self.data.get_stock_readiness()
        not_ready = [s for s in stocks if not s["ready_for_season"]]

        if not_ready:
            details = "\n".join(
                f"• {s['brand']}: {s['stock_days']} дн. (нужно ≥{self.cfg['pre_season_stock_days']} дн.)"
                for s in not_ready
            )
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.SEASONAL,
                priority=Priority.HIGH,
                title=f"Недостаточный запас к сезону: {len(not_ready)} брендов",
                body=f"Нужен запас ≥{self.cfg['pre_season_stock_days']} дней до начала сезона:\n{details}",
                value=len(not_ready),
                threshold=0,
                action="Ускорить производство / закупку, проверить готовность склада",
                data={"brands": not_ready},
            ))
        return signals
