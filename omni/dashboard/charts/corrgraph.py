"""D3 force-directed correlation graph.

Nodes = tickers (+ optionally benchmarks).
Edges = |correlation_60d| above a configurable threshold; edge thickness ∝ |corr|.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from omni.dashboard.charts import render_html


def _build_edges(price_panel: pd.DataFrame, threshold: float = 0.4, max_edges: int = 250) -> tuple[list, list]:
    """Given a wide returns frame [date x ticker], return (nodes, links) for D3."""
    if price_panel is None or price_panel.empty or price_panel.shape[1] < 2:
        return [], []
    rets = price_panel.pct_change().dropna(how="all")
    rets = rets.tail(60)
    corr = rets.corr().fillna(0)
    nodes = [{"id": t, "group": 1} for t in corr.columns]
    links = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            c = float(corr.iloc[i, j])
            if abs(c) >= threshold and not np.isnan(c):
                links.append({"source": a, "target": b, "value": round(c, 3)})
    links.sort(key=lambda e: -abs(e["value"]))
    return nodes, links[:max_edges]


def render_corrgraph(price_panel: pd.DataFrame, *, height: int = 460, threshold: float = 0.4,
                    title: str = "Correlation Graph") -> None:
    """Mount a D3 force-directed graph from a wide price panel."""
    nodes, links = _build_edges(price_panel, threshold=threshold)
    if not nodes:
        import streamlit as st
        st.info("Not enough overlapping price history to draw a correlation graph.")
        return

    div_id = f"cg-{uuid.uuid4().hex[:10]}"
    payload = json.dumps({"nodes": nodes, "links": links})

    html = f"""
<div id="{div_id}" style="position:relative;width:100%;height:{height}px;
     border-radius:14px;border:1px solid rgba(148,163,184,0.20);
     background:rgba(15,23,42,0.35);overflow:hidden;">
  <div style="position:absolute;top:10px;left:14px;font:600 12px/1.2 'Inter',sans-serif;
       color:#cbd5e1;letter-spacing:0.04em;text-transform:uppercase;z-index:2;">
    {title} · {len(nodes)} nodes · {len(links)} edges (|ρ| ≥ {threshold:.2f})
  </div>
</div>
<script src="https://unpkg.com/d3@7.9.0/dist/d3.min.js"></script>
<script>
(function() {{
  const graph = {payload};
  const root = document.getElementById('{div_id}');
  const W = root.clientWidth, H = root.clientHeight;
  const svg = d3.select(root).append('svg').attr('width', W).attr('height', H);

  const sim = d3.forceSimulation(graph.nodes)
    .force('link', d3.forceLink(graph.links).id(d=>d.id).distance(d => 100 - 60*Math.abs(d.value)))
    .force('charge', d3.forceManyBody().strength(-110))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide(18));

  const link = svg.append('g')
    .selectAll('line').data(graph.links).enter().append('line')
    .attr('stroke', d => d.value >= 0 ? 'rgba(34,197,94,0.45)' : 'rgba(239,68,68,0.45)')
    .attr('stroke-width', d => 0.5 + 3 * Math.abs(d.value));

  const node = svg.append('g')
    .selectAll('g').data(graph.nodes).enter().append('g')
    .call(d3.drag()
      .on('start', (e,d) => {{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on('drag',  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
      .on('end',   (e,d) => {{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));

  node.append('circle').attr('r', 6)
    .attr('fill', '#06b6d4')
    .attr('stroke', 'rgba(255,255,255,0.55)')
    .attr('stroke-width', 1.2);

  node.append('text').text(d=>d.id)
    .attr('x', 9).attr('y', 4)
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('font-size', 10).attr('fill', '#cbd5e1');

  sim.on('tick', () => {{
    link.attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
        .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
    node.attr('transform', d=>`translate(${{d.x}},${{d.y}})`);
  }});
}})();
</script>
"""
    render_html(html, height=height)
