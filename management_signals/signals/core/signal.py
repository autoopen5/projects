from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Priority(Enum):
    CRITICAL = "critical"   # немедленно
    HIGH = "high"           # в течение часа
    MEDIUM = "medium"       # в дайджесте
    INFO = "info"           # справочно


class Domain(Enum):
    FINANCE = "finance"
    SALES = "sales"
    PRODUCTION = "production"
    HR = "hr"
    SEASONAL = "seasonal"


@dataclass
class Signal:
    id: str
    domain: Domain
    priority: Priority
    title: str
    body: str
    value: Any
    threshold: Any
    deviation_pct: Optional[float] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    action: Optional[str] = None  # рекомендуемое действие
    created_at: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)

    def to_telegram(self) -> str:
        icons = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠",
            Priority.MEDIUM: "🟡",
            Priority.INFO: "🔵",
        }
        icon = icons[self.priority]
        lines = [f"{icon} *{self.title}*", self.body]
        if self.deviation_pct is not None:
            sign = "+" if self.deviation_pct > 0 else ""
            lines.append(f"Отклонение: {sign}{self.deviation_pct:.1f}%")
        if self.brand:
            lines.append(f"Бренд: {self.brand}")
        if self.region:
            lines.append(f"Регион: {self.region}")
        if self.action:
            lines.append(f"💡 {self.action}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "priority": self.priority.value,
            "title": self.title,
            "body": self.body,
            "value": str(self.value),
            "threshold": str(self.threshold),
            "deviation_pct": self.deviation_pct,
            "brand": self.brand,
            "region": self.region,
            "action": self.action,
            "created_at": self.created_at.isoformat(),
        }
