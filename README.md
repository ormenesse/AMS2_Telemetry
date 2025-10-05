Absolutely—you can get opponent info and full real-time telemetry, and I’ve made you a small Python app that does both.

Here’s the gist:

* It polls **CREST2-AMS2** (shared memory → JSON) at `http://127.0.0.1:9000/api`.
* Exposes a clean API at `/api/telemetry` with:

  * `session` (type/track/time left),
  * `player` (speed, gear, throttle, etc.),
  * `opponents` (name, class, lap, race/class position).
* Serves a lightweight live dashboard at `/` that refreshes ~5 Hz.

### How to run

1. Install prerequisites:

   ```
   pip install fastapi uvicorn requests pydantic
   ```
2. Make sure **CREST2-AMS2** is running (it reads AMS2 shared memory and serves JSON locally).
3. Start the app:

   ```
   uvicorn app:app --reload --port 8000
   ```
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — you’ll see player + opponents update live.

### Config tweaks

* Point to a different CREST URL:

  ```
  set CREST_URL=http://127.0.0.1:9000/api    # Windows (PowerShell: $env:CREST_URL=...)
  export CREST_URL=http://127.0.0.1:9000/api  # macOS/Linux
  ```
* Change polling rate (Hz):

  ```
  set POLL_HZ=10
  ```

### Using SimHub/UDP instead

If you prefer the UDP route, the file includes a commented **UDP listener stub** and notes. You’d:

* enable AMS2 UDP (Project CARS protocol),
* (optionally) use SimHub’s UDP forwarding to `127.0.0.1:5607`,
* implement the packet layouts (Madness/PCars2) and map them into the same `Snapshot` shape used by the app.

Want me to extend this with WebSockets, CSV/Parquet logging per lap, sector deltas, or plots (e.g., speed vs. distance, throttle/brake traces)? I can wire those in next.
