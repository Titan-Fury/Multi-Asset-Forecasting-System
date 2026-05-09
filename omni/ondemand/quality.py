"""Per-prediction provenance / data-quality tracking.

The Provenance object captures which sources contributed to a single
on-demand prediction so the UI can show a confidence pill and the user
can judge whether to trust the result.

`data_quality_score` rubric (0-100):
    + 30 if prices have 2+ sources agreeing >99%
    + 15 if prices have only 1 source (yfinance)
    + 20 if fundamentals have a real source (not all-NaN)
    + 15 if news_count >= 5
    +  5 per news source up to +15
    + 20 if macro is fresh (<24h)
    + 15 if regime label is fresh (<24h)

Capped at 100. A score below 50 should display a warning in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Provenance:
    prices_sources: list[str] = field(default_factory=list)
    prices_agreement_pct: float | None = None  # None if only one source
    fundamentals_sources: list[str] = field(default_factory=list)
    news_sources: list[str] = field(default_factory=list)
    news_count: int = 0
    macro_fresh: bool = False
    regime_fresh: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def data_quality_score(self) -> int:
        s = 0
        if len(self.prices_sources) >= 2 and (self.prices_agreement_pct or 0) > 0.99:
            s += 30
        elif self.prices_sources:
            s += 15
        if self.fundamentals_sources:
            s += 20
        if self.news_count >= 5:
            s += 15
        s += min(15, 5 * len(self.news_sources))
        if self.macro_fresh:
            s += 20
        if self.regime_fresh:
            s += 15
        return min(100, s)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> dict:
        return {
            "data_quality_score": self.data_quality_score,
            "prices": self.prices_sources,
            "prices_agreement_pct": self.prices_agreement_pct,
            "fundamentals": self.fundamentals_sources,
            "news_sources": self.news_sources,
            "news_count": self.news_count,
            "macro_fresh": self.macro_fresh,
            "regime_fresh": self.regime_fresh,
            "warnings": self.warnings,
        }
