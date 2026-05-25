from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from signals.core.engine import SignalEngine
from signals.core.signal import Domain, Priority
from config.settings import settings

app = FastAPI(title="Management Signals API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = settings.config
_engine = SignalEngine(config)


@app.get("/signals")
def get_signals(
    domain: Optional[str] = Query(None, description="finance|sales|production|hr|seasonal"),
    priority: Optional[str] = Query(None, description="critical|high|medium|info"),
    limit: int = Query(50, le=200),
):
    signals = _engine.run_all()

    if domain:
        try:
            d = Domain(domain)
            signals = [s for s in signals if s.domain == d]
        except ValueError:
            pass

    if priority:
        try:
            p = Priority(priority)
            signals = [s for s in signals if s.priority == p]
        except ValueError:
            pass

    return {
        "signals": [s.to_dict() for s in signals[:limit]],
        "total": len(signals),
        "summary": _engine.get_summary(signals),
    }


@app.get("/signals/critical")
def get_critical():
    signals = _engine.run_critical_only()
    return {"signals": [s.to_dict() for s in signals], "total": len(signals)}


@app.get("/signals/summary")
def get_summary():
    signals = _engine.run_all()
    return _engine.get_summary(signals)


@app.get("/health")
def health():
    return {"status": "ok"}
