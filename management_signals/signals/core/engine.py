from typing import List
from signals.core.signal import Signal, Priority, Domain
from signals.analyzers.finance import FinanceAnalyzer
from signals.analyzers.sales import SalesAnalyzer
from signals.analyzers.production import ProductionAnalyzer
from signals.analyzers.hr import HRAnalyzer
from signals.analyzers.seasonal import SeasonalAnalyzer


PRIORITY_ORDER = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 1,
    Priority.MEDIUM: 2,
    Priority.INFO: 3,
}


class SignalEngine:
    def __init__(self, config: dict):
        self.config = config
        self.analyzers = [
            FinanceAnalyzer(config),
            SalesAnalyzer(config),
            ProductionAnalyzer(config),
            HRAnalyzer(config),
            SeasonalAnalyzer(config),
        ]

    def run_all(self) -> List[Signal]:
        signals = []
        for analyzer in self.analyzers:
            try:
                signals.extend(analyzer.analyze())
            except Exception as e:
                print(f"[ERROR] {analyzer.__class__.__name__}: {e}")
        return sorted(signals, key=lambda s: PRIORITY_ORDER[s.priority])

    def run_critical_only(self) -> List[Signal]:
        all_signals = self.run_all()
        return [s for s in all_signals if s.priority == Priority.CRITICAL]

    def get_summary(self, signals: List[Signal]) -> dict:
        by_priority = {}
        by_domain = {}
        for s in signals:
            by_priority[s.priority.value] = by_priority.get(s.priority.value, 0) + 1
            by_domain[s.domain.value] = by_domain.get(s.domain.value, 0) + 1
        return {
            "total": len(signals),
            "by_priority": by_priority,
            "by_domain": by_domain,
            "critical_count": by_priority.get("critical", 0),
            "high_count": by_priority.get("high", 0),
        }
