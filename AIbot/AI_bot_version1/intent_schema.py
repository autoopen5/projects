# intent_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any

class SortSpec(BaseModel):
    field: str
    dir: Literal["asc", "desc"] = "desc"

class FilterSpec(BaseModel):
    field: str
    op: Literal["=", "!=", ">", "<", ">=", "<=", "between", "in", "contains", "ilike"]
    value: Any  # str | number | [str,str] for between | list

class Intent(BaseModel):
    measures: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    time_grain: Optional[Literal["day","week","month","quarter","year"]] = None
    sort_by: List[SortSpec] = Field(default_factory=list)
    top_n: Optional[int] = None
    explain: Optional[str] = None
    confidence: float = 0.0