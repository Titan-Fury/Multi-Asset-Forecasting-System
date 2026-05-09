"""Three.js 3D Signal Cosmos.

Renders an interactive 3D scene where each ticker is a sphere:
  radius   ∝ conviction
  colour   = green (long) / red (short) / grey (flat)
  cluster  = grouped by sector / asset_class
  orbit    = vol ∝ angular velocity
  hover    = ticker label + key metrics (raycaster picking)

ESM Three.js (r156) is loaded from unpkg via importmap.
"""
from __future__ import annotations

import json
import uuid
from typing import Iterable

import streamlit.components.v1 as components


def _normalise_card(c: dict) -> dict:
    return {
        "ticker": str(c.get("ticker", "?"))[:10],
        "asset_class": str(c.get("asset_class", "us")),
        "direction": int(c.get("direction", 0)),
        "conviction": float(c.get("conviction", 0.0)),
        "blended_p": float(c.get("blended_p", 0.5)),
        "regime": str(c.get("regime", "chop")),
        "vol_5d": float(c.get("recent_vol_5d_pct", 5.0)),
        "move_5d": float(c.get("move_1w_pct", 0.0)),
        "rubric": float(c.get("rubric_total", 50.0)),
    }


def render_cosmos(cards: Iterable[dict], *, height: int = 540, title: str = "Signal Cosmos") -> None:
    """Mount the 3D cosmos. `cards` is an iterable of PredictionCard dicts."""
    payload = [_normalise_card(c) for c in (cards or []) if c]
    if not payload:
        import streamlit as st
        st.info("No predictions yet — run a few tickers to populate the cosmos.")
        return

    div_id = f"cosmos-{uuid.uuid4().hex[:10]}"
    data_json = json.dumps(payload)

    html = f"""
<div id="{div_id}" style="position:relative;width:100%;height:{height}px;
     border-radius:14px;border:1px solid rgba(148,163,184,0.20);
     background:radial-gradient(ellipse at 50% 30%, rgba(6,182,212,0.10) 0%,
                                                   rgba(15,23,42,0.95) 70%);
     overflow:hidden;">
  <div style="position:absolute;top:10px;left:14px;font:600 12px/1.2 'Inter',sans-serif;
       color:#cbd5e1;letter-spacing:0.04em;text-transform:uppercase;z-index:2;">
    {title} · {len(payload)} tickers
  </div>
  <div id="{div_id}-tip" style="position:absolute;pointer-events:none;
       padding:6px 10px;border-radius:8px;background:rgba(15,23,42,0.92);
       border:1px solid rgba(6,182,212,0.45);font:500 11px/1.4 'JetBrains Mono',monospace;
       color:#e2e8f0;display:none;z-index:3;backdrop-filter:blur(4px);"></div>
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.156.1/build/three.module.js" }} }}
</script>
<script type="module">
import * as THREE from 'three';

const data = {data_json};
const root = document.getElementById('{div_id}');
const tip  = document.getElementById('{div_id}-tip');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, root.clientWidth/root.clientHeight, 0.1, 1000);
camera.position.set(0, 8, 32);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(root.clientWidth, root.clientHeight);
renderer.setClearColor(0x000000, 0);
root.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const key = new THREE.PointLight(0x06b6d4, 1.4, 100);
key.position.set(20, 20, 20); scene.add(key);
const rim = new THREE.PointLight(0x8b5cf6, 0.9, 100);
rim.position.set(-20, -10, 20); scene.add(rim);

// Cluster centres by asset_class
const clusters = {{ us: [-12, 0, 0], india: [12, 0, 0], crypto: [0, 8, -8], fx: [0, -8, 8] }};
const dirColor = {{ "1": 0x22c55e, "0": 0x94a3b8, "-1": 0xef4444 }};

const meshes = [];
data.forEach((d, i) => {{
  const r = Math.max(0.35, Math.min(2.2, d.conviction / 35));
  const geo = new THREE.SphereGeometry(r, 32, 32);
  const colour = dirColor[String(d.direction)] || 0x94a3b8;
  const mat = new THREE.MeshStandardMaterial({{
    color: colour, emissive: colour, emissiveIntensity: 0.30,
    metalness: 0.35, roughness: 0.45,
  }});
  const m = new THREE.Mesh(geo, mat);

  // Position on a Fibonacci-shell offset from the cluster centre
  const c = clusters[d.asset_class] || [0,0,0];
  const phi = Math.acos(1 - 2*((i + 0.5)/data.length));
  const theta = Math.PI * (1 + Math.sqrt(5)) * i;
  const radius = 5 + (d.rubric / 30);
  m.position.set(
    c[0] + radius * Math.sin(phi) * Math.cos(theta),
    c[1] + radius * Math.sin(phi) * Math.sin(theta),
    c[2] + radius * Math.cos(phi)
  );
  m.userData = d;
  m.userData.orbitSpeed = 0.00015 + Math.min(0.001, (d.vol_5d || 5)/12000);
  m.userData.cx = c[0]; m.userData.cy = c[1]; m.userData.cz = c[2];
  m.userData.theta0 = theta;
  m.userData.baseY = m.position.y;
  scene.add(m); meshes.push(m);
}});

// Subtle starfield backdrop
const stars = new THREE.BufferGeometry();
const starV = new Float32Array(1500);
for (let i = 0; i < 1500; i++) starV[i] = (Math.random() - 0.5) * 200;
stars.setAttribute('position', new THREE.BufferAttribute(starV, 3));
scene.add(new THREE.Points(stars, new THREE.PointsMaterial({{ color: 0xffffff, size: 0.05, opacity: 0.35, transparent: true }})));

// Raycaster picking
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hovered = null;
renderer.domElement.addEventListener('mousemove', (e) => {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(meshes);
  if (hits.length) {{
    const d = hits[0].object.userData;
    tip.innerHTML = `<b style="color:#06b6d4">${{d.ticker}}</b> · ${{d.asset_class}}<br>` +
                    `direction: ${{d.direction>0?'LONG':d.direction<0?'SHORT':'FLAT'}} · conv ${{d.conviction.toFixed(0)}}<br>` +
                    `5d move: ${{d.move_5d.toFixed(2)}}% · rubric ${{d.rubric.toFixed(0)}} · regime ${{d.regime}}`;
    tip.style.display = 'block';
    tip.style.left = (e.clientX - rect.left + 16) + 'px';
    tip.style.top  = (e.clientY - rect.top + 16) + 'px';
    hovered = hits[0].object;
  }} else {{
    tip.style.display = 'none'; hovered = null;
  }}
}});
renderer.domElement.addEventListener('mouseleave', () => {{ tip.style.display='none'; hovered=null; }});

// Drag to rotate
let dragging = false, lastX = 0, lastY = 0, yaw = 0, pitch = 0;
renderer.domElement.addEventListener('mousedown', (e) => {{ dragging = true; lastX=e.clientX; lastY=e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
renderer.domElement.addEventListener('mousemove', (e) => {{
  if (!dragging) return;
  yaw   += (e.clientX - lastX) * 0.005;
  pitch += (e.clientY - lastY) * 0.005;
  pitch = Math.max(-1.2, Math.min(1.2, pitch));
  lastX = e.clientX; lastY = e.clientY;
}});

function animate(t) {{
  const paused = (hovered !== null);
  meshes.forEach(m => {{
    const d = m.userData;
    if (!paused) {{
      const angle = d.theta0 + t * d.orbitSpeed;
      const r = m.position.distanceTo(new THREE.Vector3(d.cx, d.cy, d.cz));
      m.position.x = d.cx + r * Math.cos(angle);
      m.position.z = d.cz + r * Math.sin(angle);
    }}
    if (m === hovered) m.scale.setScalar(1.4);
    else                m.scale.setScalar(1.0);
  }});
  // Only respond to user drag; no auto-spin.
  scene.rotation.y = yaw;
  scene.rotation.x = pitch;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}}
requestAnimationFrame(animate);

new ResizeObserver(() => {{
  renderer.setSize(root.clientWidth, root.clientHeight);
  camera.aspect = root.clientWidth / root.clientHeight;
  camera.updateProjectionMatrix();
}}).observe(root);
</script>
"""
    components.html(html, height=height + 20, scrolling=False)
