"""TradingView Lightweight-Charts wrapper.

Renders a professional-grade candle + volume chart inside a Streamlit
component iframe. ESM build pulled from unpkg; ~45 KB; WebGL-smooth.
"""
from __future__ import annotations

import json
import uuid

import pandas as pd

from omni.dashboard.charts import render_html

_LWC_URL = "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"


def _to_lwc_rows(df: pd.DataFrame) -> tuple[list, list]:
    """Convert a daily OHLCV frame to lightweight-charts payload."""
    if df is None or df.empty:
        return [], []
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    candles = []
    volumes = []
    for _, r in d.iterrows():
        ts = int(pd.Timestamp(r["date"]).timestamp())
        candles.append({
            "time": ts,
            "open": float(r.get("open", r.get("adj_close", 0)) or 0),
            "high": float(r.get("high", r.get("adj_close", 0)) or 0),
            "low":  float(r.get("low",  r.get("adj_close", 0)) or 0),
            "close": float(r.get("adj_close", r.get("close", 0)) or 0),
        })
        vol = float(r.get("volume", 0) or 0)
        is_up = candles[-1]["close"] >= candles[-1]["open"]
        volumes.append({
            "time": ts,
            "value": vol,
            "color": "rgba(34, 197, 94, 0.45)" if is_up else "rgba(239, 68, 68, 0.45)",
        })
    return candles, volumes


def render_lwchart(df: pd.DataFrame, *, height: int = 380, title: str = "") -> None:
    """Mount a Lightweight-Charts candle+volume chart in the current container."""
    candles, volumes = _to_lwc_rows(df)
    if not candles:
        import streamlit as st
        st.info("No price history available for chart.")
        return

    div_id = f"lwc-{uuid.uuid4().hex[:10]}"
    payload = json.dumps({"candles": candles, "volumes": volumes})

    html = f"""
<div id="{div_id}" style="width:100%;height:{height}px;
     border-radius:14px;border:1px solid rgba(148,163,184,0.20);
     background:rgba(15,23,42,0.35);overflow:hidden;position:relative;">
  <div style="position:absolute;top:10px;left:14px;font:600 12px/1.2 'Inter',sans-serif;
       color:#cbd5e1;letter-spacing:0.04em;text-transform:uppercase;z-index:2;">
    {title}
  </div>
</div>
<script src="{_LWC_URL}"></script>
<script>
(function() {{
  const data = {payload};
  const target = document.getElementById("{div_id}");
  const chart = LightweightCharts.createChart(target, {{
    layout: {{
      background: {{ type: 'solid', color: 'transparent' }},
      textColor: '#cbd5e1',
      fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      fontSize: 11,
    }},
    grid: {{
      vertLines: {{ color: 'rgba(148,163,184,0.10)' }},
      horzLines: {{ color: 'rgba(148,163,184,0.10)' }},
    }},
    rightPriceScale: {{ borderColor: 'rgba(148,163,184,0.25)' }},
    timeScale:       {{ borderColor: 'rgba(148,163,184,0.25)', timeVisible: true }},
    crosshair:       {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const candle = chart.addCandlestickSeries({{
    upColor:  '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e',   wickDownColor: '#ef4444',
  }});
  candle.setData(data.candles);
  const vol = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: '',
    scaleMargins: {{ top: 0.78, bottom: 0 }},
  }});
  vol.setData(data.volumes);

  // Resize handler
  new ResizeObserver(() => chart.applyOptions({{ width: target.clientWidth }}))
    .observe(target);
  chart.timeScale().fitContent();
}})();
</script>
"""
    render_html(html, height=height)
