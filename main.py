"""
AMS2 Live Telemetry + Opponents (Python, FastAPI)
+ Lap Logger & Comparison Plots
-------------------------------------------------

What you get
- Real‑time telemetry + opponent info from Automobilista 2
- Uses CREST2‑AMS2 (shared memory → JSON) for rich, low‑latency data
- One Python file (FastAPI) which:
  • Polls CREST2 at a configurable URL (default http://127.0.0.1:9000/api)
  • Exposes /api/telemetry with a clean snapshot for your app
  • Serves a tiny dashboard at / that refreshes ~5 Hz

Prereqs
1) Install and run CREST2‑AMS2 (reads AMS2 shared memory and serves JSON locally)
2) Python 3.9+
3) pip install fastapi uvicorn requests pydantic

Run it
> uvicorn app:app --reload --port 8000
Open http://127.0.0.1:8000

Notes
- If you prefer to get data from SimHub/UDP instead, see the TODO section at the end for a UDP listener stub you can enable.
- This code keeps things simple (polling ~10 Hz). You can increase rate or switch to websockets later.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

CREST_URL = os.environ.get("CREST_URL", "http://127.0.0.1:9000/api")
POLL_HZ = float(os.environ.get("POLL_HZ", "10"))  # 10 Hz default

# app will be created after lifespan is defined

# --- Data models for a clean, app‑friendly snapshot ---
class CarSnapshot(BaseModel):
    name: Optional[str] = None
    race_position: Optional[int] = None
    class_position: Optional[int] = None
    current_lap: Optional[int] = None
    laps_completed: Optional[int] = None
    sector: Optional[int] = None
    speed_kph: Optional[float] = None
    gear: Optional[int] = None
    engine_rpm: Optional[float] = None
    throttle: Optional[float] = None
    brake: Optional[float] = None
    steering: Optional[float] = None
    distance_traveled_m: Optional[float] = None
    car_class_name: Optional[str] = None
    is_player: bool = False

class SessionSnapshot(BaseModel):
    session_type: Optional[str] = None
    track_name: Optional[str] = None
    layout: Optional[str] = None
    time_remaining_s: Optional[float] = None
    lap_valid: Optional[bool] = None

class Snapshot(BaseModel):
    timestamp: float
    session: SessionSnapshot
    player: CarSnapshot
    opponents: List[CarSnapshot]

# --- Lap storage structures ---
class LapData(BaseModel):
    participant: str  # 'player' or opponent name
    lap_number: int
    start_ts: float
    end_ts: Optional[float] = None
    # raw samples as list of dicts; we'll convert to DataFrame on demand
    samples: List[Dict[str, Any]] = []

# Buffers
player_current_lap: Optional[LapData] = None
player_completed_laps: List[LapData] = []
# Opponents by name
opp_current_lap: Dict[str, LapData] = {}
opp_completed_laps: Dict[str, List[LapData]] = {}

# --- Shared state updated by a polling thread ---
_latest: Optional[Snapshot] = None
_lock = threading.Lock()
_stop = False

# --- FastAPI lifespan (replaces deprecated on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        global _stop
        _stop = True
        try:
            t.join(timeout=1.0)
        except Exception:
            pass

# Create FastAPI app AFTER lifespan is defined
app = FastAPI(title="AMS2 Telemetry Proxy", lifespan=lifespan)



# --- Helpers to safely read dicts ---
def g(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def crest_to_snapshot(raw: Dict[str, Any]) -> Snapshot:
    tele = raw.get("telemetry", {}) or {}
    sess = raw.get("gameStates", {}) or {}
    track = raw.get("trackData", {}) or {}
    parts = raw.get("participants", []) or []
    timings = raw.get("timings", {}) or {}

    viewed_idx = (
        g(timings, "mViewedParticipantIndex")
        or g(tele, "mViewedParticipantIndex")
        or 0
    )

    def mk_car(p: Dict[str, Any], is_player: bool = False) -> CarSnapshot:
        name = p.get("mName") or p.get("mVehicleName")
        race_pos = p.get("mRacePosition")
        class_pos = p.get("mClassPosition")
        current_lap = p.get("mCurrentLap")
        laps_completed = p.get("mLapsCompleted")
        sector = p.get("mCurrentSector")
        car_class = p.get("mCarClassName") or p.get("mCarClass")

        speed_kph = None
        gear = None
        rpm = None
        throttle = None
        brake = None
        steering = None
        dist_m = None

        if is_player:
            speed_mps = tele.get("mSpeed")
            if speed_mps is not None:
                speed_kph = float(speed_mps) * 3.6
            gear = tele.get("mGear")
            rpm = tele.get("mEngineRPM")
            throttle = tele.get("mThrottle")
            brake = tele.get("mBrake")
            steering = tele.get("mSteering")
            dist_m = tele.get("mOdometerKM")
            if isinstance(dist_m, (int, float)):
                dist_m = float(dist_m) * 1000.0

        return CarSnapshot(
            name=name,
            race_position=race_pos,
            class_position=class_pos,
            current_lap=current_lap,
            laps_completed=laps_completed,
            sector=sector,
            speed_kph=speed_kph,
            gear=gear,
            engine_rpm=rpm,
            throttle=throttle,
            brake=brake,
            steering=steering,
            distance_traveled_m=dist_m,
            car_class_name=car_class,
            is_player=is_player,
        )

    player_part = parts[viewed_idx] if (isinstance(parts, list) and len(parts) > viewed_idx) else {}
    player = mk_car(player_part, is_player=True)

    opponents: List[CarSnapshot] = []
    if isinstance(parts, list):
        for i, p in enumerate(parts):
            if i == viewed_idx:
                continue
            opponents.append(mk_car(p, is_player=False))

    session = SessionSnapshot(
        session_type=g(sess, "mSessionStateName") or g(tele, "mSessionStateName"),
        track_name=track.get("mTrackLocation"),
        layout=track.get("mTrackVariation"),
        time_remaining_s=g(sess, "mSessionDuration"),
        lap_valid=bool(g(tele, "mLapInvalidated") == 0) if g(tele, "mLapInvalidated") is not None else None,
    )

    return Snapshot(
        timestamp=time.time(),
        session=session,
        player=player,
        opponents=opponents,
    ),
        session=session,
        player=player,
        opponents=opponents,
    )


def _append_sample(lap: LapData, snap: Snapshot):
    tele = {
        "ts": snap.timestamp,
        "speed_kph": snap.player.speed_kph,
        "gear": snap.player.gear,
        "throttle": snap.player.throttle,
        "brake": snap.player.brake,
        "steering": snap.player.steering,
        "dist_m": snap.player.distance_traveled_m,
        "lap": snap.player.current_lap,
    }
    lap.samples.append(tele)


def _finalize_lap(lap: LapData, end_ts: float):
    lap.end_ts = end_ts


def poll_loop():
    global _latest, player_current_lap
    interval = 1.0 / max(POLL_HZ, 1.0)
    last_lap_num: Optional[int] = None
    while not _stop:
        try:
            r = requests.get(CREST_URL, timeout=0.5)
            r.raise_for_status()
            raw = r.json()
            snap = crest_to_snapshot(raw)
            with _lock:
                _latest = snap
                # Player lap tracking
                lap_num = snap.player.current_lap if snap.player else None
                if lap_num is not None:
                    if player_current_lap is None:
                        player_current_lap = LapData(participant="player", lap_number=lap_num, start_ts=snap.timestamp, samples=[])
                    elif last_lap_num is not None and lap_num != last_lap_num:
                        # Lap changed → finalize previous
                        _finalize_lap(player_current_lap, snap.timestamp)
                        player_completed_laps.append(player_current_lap)
                        player_current_lap = LapData(participant="player", lap_number=lap_num, start_ts=snap.timestamp, samples=[])
                    _append_sample(player_current_lap, snap)
                    last_lap_num = lap_num
        except Exception:
            pass
        time.sleep(interval)



@app.get("/health")
def health():
    with _lock:
        laps = len(player_completed_laps)
        return {"ok": True, "has_snapshot": _latest is not None, "laps_recorded": laps}

@app.get("/api/telemetry", response_model=Snapshot)
def get_snapshot()():
    with _lock:
        if _latest is None:
            # Return an empty shell so the frontend can render
            empty = Snapshot(
                timestamp=time.time(),
                session=SessionSnapshot(),
                player=CarSnapshot(is_player=True),
                opponents=[],
            )
            return JSONResponse(empty.model_dump())
        return JSONResponse(_latest.model_dump())


# --- Minimal dashboard (auto‑refresh) ---
DASH_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AMS2 Live Telemetry</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    h2 { margin: 0 0 8px; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <h1>AMS2 Live Telemetry</h1>
  <div id="status">Connecting…</div>
  <div class="grid">
    <div class="card">
      <h2>Session</h2>
      <div id="session"></div>
    </div>
    <div class="card">
      <h2>Player</h2>
      <div id="player"></div>
    </div>
    <div class="card" style="grid-column: 1 / -1;">
      <h2>Opponents</h2>
      <div style="max-height: 360px; overflow:auto;">
        <table id="opponents"><thead><tr>
          <th>#</th><th>Name</th><th>Cls</th><th>Lap</th><th>Pos</th>
        </tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
<script>
async function fetchSnap(){
  const r = await fetch('/api/telemetry');
  return await r.json();
}

function fmt(v, digits=1){
  if(v===null||v===undefined) return '—';
  if(typeof v==='number') return v.toFixed(digits);
  return String(v);
}

async function tick(){
  try {
    const s = await fetchSnap();
    document.getElementById('status').textContent = 'OK • ' + new Date(s.timestamp*1000).toLocaleTimeString();

    const sess = s.session || {};
    document.getElementById('session').innerHTML = `
      <div><b>Type:</b> ${sess.session_type ?? '—'}</div>
      <div><b>Track:</b> ${sess.track_name ?? '—'} ${sess.layout?('('+sess.layout+')'):''}</div>
      <div><b>Time left (s):</b> ${fmt(sess.time_remaining_s,0)}</div>
      <div><b>Lap valid:</b> ${sess.lap_valid===null? '—' : (sess.lap_valid? 'Yes':'No')}</div>
    `;

    const p = s.player || {};
    document.getElementById('player').innerHTML = `
      <div><b>Name:</b> ${p.name ?? 'Player'}</div>
      <div><b>Pos:</b> ${p.race_position ?? '—'}</div>
      <div><b>Lap:</b> ${p.current_lap ?? '—'}</div>
      <div><b>Speed (kph):</b> ${fmt(p.speed_kph,1)}</div>
      <div><b>Gear:</b> ${p.gear ?? '—'}</div>
      <div><b>RPM:</b> ${fmt(p.engine_rpm,0)}</div>
      <div><b>Throttle:</b> ${fmt(p.throttle,2)}</div>
      <div><b>Brake:</b> ${fmt(p.brake,2)}</div>
      <div><b>Steering:</b> ${fmt(p.steering,2)}</div>
    `;

    const tbody = document.querySelector('#opponents tbody');
    tbody.innerHTML = '';
    (s.opponents || []).sort((a,b)=> (a.race_position??999)-(b.race_position??999)).forEach((o,i)=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${o.race_position ?? '—'}</td>
        <td>${o.name ?? '—'}</td>
        <td>${o.car_class_name ?? '—'}</td>
        <td>${o.current_lap ?? '—'}</td>
        <td>${o.class_position ?? '—'}</td>`;
      tbody.appendChild(tr);
    });
  } catch(e) {
    document.getElementById('status').textContent = 'Waiting for CREST2‑AMS2…';
  }
}

setInterval(tick, 200); // 5 Hz dashboard updates
window.onload = tick;
// Simple link to comparison page
const compare = document.createElement('p');
compare.innerHTML = '<a href="/compare">Open comparison plots</a>';
document.body.appendChild(compare);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(DASH_HTML)

# --- Utilities to build DataFrames and choose best laps ---
def lap_to_df(lap: LapData) -> pd.DataFrame:
    df = pd.DataFrame(lap.samples)
    if df.empty:
        return df
    # Compute relative time and distance from lap start
    df = df.sort_values("ts")
    df["t_rel"] = df["ts"] - df["ts"].iloc[0]
    # Reset distance to start-of-lap if available
    if df["dist_m"].notna().any():
        first_dist = df["dist_m"].dropna().iloc[0]
        df["s_rel"] = df["dist_m"] - first_dist
    else:
        # Fallback: integrate speed
        df["s_rel"] = df["speed_kph"].fillna(0) / 3.6
        df["s_rel"] = df["s_rel"].cumsum() * (df["t_rel"].diff().fillna(0))
    return df


def best_player_lap() -> Optional[LapData]:
    # choose min duration among completed laps with enough samples
    if not player_completed_laps:
        return None
    candidates: List[Tuple[float, LapData]] = []
    for lap in player_completed_laps:
        if lap.end_ts and len(lap.samples) > 10:
            duration = lap.end_ts - lap.start_ts
            candidates.append((duration, lap))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def render_plots(player_df: pd.DataFrame, opponent_df: Optional[pd.DataFrame]) -> bytes:
    # Generate 3 separate overlays on one tall figure (speed, throttle/brake, gear)
    # Requirement: single-axes charts are preferred, but here we stack to save space.
    fig_h = 10
    fig, axes = plt.subplots(3, 1, figsize=(10, fig_h))

    # SPEED
    ax = axes[0]
    ax.plot(player_df["s_rel"], player_df["speed_kph"], label="Player")
    if opponent_df is not None and not opponent_df.empty and opponent_df["speed_kph"].notna().any():
        ax.plot(opponent_df["s_rel"], opponent_df["speed_kph"], label="Opponent")
    ax.set_ylabel("Speed (kph)")
    ax.set_xlabel("Distance in lap (m)")
    ax.legend()

    # INPUTS
    ax = axes[1]
    if player_df["throttle"].notna().any():
        ax.plot(player_df["s_rel"], player_df["throttle"], label="Throttle (player)")
    if player_df["brake"].notna().any():
        ax.plot(player_df["s_rel"], player_df["brake"], label="Brake (player)")
    if opponent_df is not None:
        # Opponent inputs are usually not exposed; show only speed-based markers when available
        pass
    ax.set_ylabel("Inputs (0..1)")
    ax.set_xlabel("Distance in lap (m)")
    ax.legend()

    # GEAR (step-like)
    ax = axes[2]
    if player_df["gear"].notna().any():
        ax.step(player_df["s_rel"], player_df["gear"], where='post', label="Gear (player)")
    ax.set_ylabel("Gear")
    ax.set_xlabel("Distance in lap (m)")
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.get("/compare")
def compare_page():
    html = """
    <html><head><title>Lap Comparison</title></head>
    <body>
      <h1>Lap Comparison</h1>
      <p>This compares your best recorded lap to an opponent's best lap (speed only if available).</p>
      <p><img src="/compare/image" style="max-width:100%" /></p>
      <p><a href="/laps">See recorded laps</a></p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/laps")
def list_laps():
    with _lock:
        laps = [
            {"lap_number": l.lap_number, "duration_s": (l.end_ts - l.start_ts) if l.end_ts else None}
            for l in player_completed_laps
        ]
    return {"player_laps": laps}


@app.get("/compare/image")
def compare_image():
    with _lock:
        best = best_player_lap()
        if best is None:
            return Response("No completed laps yet.", media_type="text/plain", status_code=400)
        player_df = lap_to_df(best)

        # Opponent best lap: Not all fields exist; this demo plots only player inputs and overlays opponent speed if you later capture it.
        opponent_df = None

        png = render_plots(player_df, opponent_df)
        return Response(content=png, media_type="image/png")


# --- Optional: UDP listener stub (if you want SimHub/UDP path instead) ---
"""
To use UDP instead of CREST2 JSON:
1) In AMS2, enable UDP (Project CARS protocol) and set frequency (e.g., 5–10).
2) Either let your Python bind to 5606 (broadcast) or set SimHub to forward to 127.0.0.1:5607.
3) Replace the poll_loop() with a socket.recvfrom() loop and parse packets using the Madness/PCars2 header.

Below is a minimal skeleton; you still need to define the struct layouts from SMS_UDP_Definitions.

import socket, struct
UDP_PORT = 5606
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', UDP_PORT))
while True:
    data, addr = sock.recvfrom(2048)
    # read packet header → dispatch on packet type → unpack fields
    # then populate `Snapshot` the same way as crest_to_snapshot() does
"""
