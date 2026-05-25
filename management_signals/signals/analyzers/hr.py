from typing import List
from signals.core.signal import Signal, Domain, Priority
from signals.collectors.mock import MockHRData
import uuid


class HRAnalyzer:
    def __init__(self, config: dict, data_source=None):
        self.cfg = config["hr"]
        self.data = data_source or MockHRData()

    def analyze(self) -> List[Signal]:
        signals = []
        signals.extend(self._check_turnover())
        signals.extend(self._check_kpi())
        signals.extend(self._check_vacancies())
        return signals

    def _check_turnover(self) -> List[Signal]:
        signals = []
        hc = self.data.get_headcount()
        resigned = hc["resigned_this_month"]
        total = hc["total"]
        turnover_pct = resigned / total * 100

        if turnover_pct >= self.cfg["turnover_monthly_pct"]:
            priority = Priority.HIGH if turnover_pct >= self.cfg["turnover_monthly_pct"] * 1.5 else Priority.MEDIUM
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.HR,
                priority=priority,
                title="Повышенная текучесть персонала",
                body=f"Уволились в этом месяце: {resigned} чел. ({turnover_pct:.1f}% от штата {total})\nПринято: {hc['hired_this_month']} чел.",
                value=turnover_pct,
                threshold=self.cfg["turnover_monthly_pct"],
                deviation_pct=turnover_pct - self.cfg["turnover_monthly_pct"],
                action="Провести exit-интервью, проверить причины. Запросить HR-анализ по подразделениям",
                data=hc,
            ))
        return signals

    def _check_kpi(self) -> List[Signal]:
        signals = []
        underperformers = self.data.get_kpi_underperformers()

        for u in underperformers:
            priority = Priority.HIGH if u["kpi_pct"] < 70 else Priority.MEDIUM
            signals.append(Signal(
                id=str(uuid.uuid4()),
                domain=Domain.HR,
                priority=priority,
                title=f"KPI не выполняется: {u['role']}",
                body=f"{u['name']} — {u['kpi_pct']}% от плана за {u['period']}",
                value=u["kpi_pct"],
                threshold=100,
                deviation_pct=u["kpi_pct"] - 100,
                action=f"Запросить план корректирующих мероприятий от {u['name']}",
                data=u,
            ))
        return signals

    def _check_vacancies(self) -> List[Signal]:
        signals = []
        vacancies = self.data.get_long_open_vacancies()

        long_open = [v for v in vacancies if v["days_open"] >= self.cfg["vacancy_open_days"]]
        if not long_open:
            return signals

        details = "\n".join(
            f"• {v['position']}: {v['days_open']} дн., кандидатов в воронке: {v['candidates_in_funnel']}"
            for v in long_open
        )
        signals.append(Signal(
            id=str(uuid.uuid4()),
            domain=Domain.HR,
            priority=Priority.MEDIUM,
            title=f"Долго открытые вакансии: {len(long_open)} шт.",
            body=details,
            value=len(long_open),
            threshold=1,
            action="Обсудить с HR: нужна ли смена требований, агентство или повышение зарплатной вилки",
            data={"vacancies": long_open},
        ))
        return signals
