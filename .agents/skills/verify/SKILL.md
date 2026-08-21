---
name: verify
description: Build, launch, and drive the Choreo app (FastAPI backend + Vite/React frontend) to verify changes at the browser surface.
---

# Verify Choreo

## Launch

Backend (FastAPI, port 8000), from repo root:

```
venv/Scripts/python -m uvicorn server:app --app-dir src/api
```

Frontend (Vite dev server, port 5173, proxies /api to the backend):

```
cd frontend && npm run dev
```

Sanity probes: `curl http://127.0.0.1:8000/api/tracks` and `curl http://localhost:5173/`.

## Drive

Surface is the browser at http://localhost:5173. Headless driving that works here:
puppeteer-core (install in the scratchpad, not the repo) pointed at system Chrome
`C:\Program Files\Google\Chrome\Application\chrome.exe` with
`--autoplay-policy=no-user-gesture-required --mute-audio` so audio playback runs
without a user gesture.

Flow worth driving: click a track button (match by filename text) → wavesurfer
waveform renders → play/pause → time readout advances against wall clock →
click the waveform to seek.

## Gotchas

- Track analysis is cached under `cache/stems/`; a track without a cache entry
  takes minutes to analyze on first select. Probe
  `GET /api/tracks/<id>/analysis` with a generous timeout first.
- `cache/` is derived and safe to delete; `data/grids/<md5>.json` is USER DATA
  (hand-tapped grids) and cannot be regenerated from the audio. A driver that
  PUTs or DELETEs `/manual-grid` destroys real work — drive tap/grid flows
  against `test_beat` (32s, disposable), never the user's real tracks, and
  snapshot `data/grids/` first if you must touch it.
- Wavesurfer v7 renders into shadow DOM: the host is the div wavesurfer creates
  inside its container; query `host.shadowRoot` for `canvas`,
  `[part~="wrapper"]`, `[part~="progress"]`, `[part~="cursor"]`. Progress-width /
  wrapper-width gives the playhead position fraction.
- Asserting on REGIONS (grid ticks, onset markers, loop band) has two traps that
  both fail by finding nothing, which reads as a passing "all clear":
  1. A zero-length region gets `part="marker <id>"`, NOT `part="region <id>"`
     (`Region.setPart`). Every beat/onset/subdivision tick is zero-length, so
     `[part~="region"]` matches only the 8-count shading. Read the container's
     children instead: `host.shadowRoot.querySelector('[part="regions-container"]')`.
  2. RegionsPlugin VIRTUALISES (`virtualAppend`) — a region is in the DOM only
     while inside the visible scroll window, and it paints across several passes
     (create -> zoom -> re-append on the `zoom` event). Wait for a non-zero child
     count before measuring, and assert the count so an empty read can't pass.
- The Timeline's waveform container is the only unclassed `div` under `.timeline`
  (the banner, minimap, tap panel, and control rows all carry classes), so
  `.timeline > div` is NOT a stable handle for it. Find the div whose
  `firstElementChild` has a `shadowRoot`.
- The app runs in React StrictMode in dev — watch the console for double-mount
  errors when touching wavesurfer lifecycle code.
- When switching tracks in a driver script, don't wait on "canvas + overlay
  exists" alone — the OLD track's timeline still satisfies that and the wait
  resolves against stale DOM. Wait for the new track's summary line
  (`section p` contains the filename) first.
- beat_this can emit a final beat at exactly `duration` (test_beat does);
  grid code deliberately skips beats at/past duration, so expected tick
  counts are `beats.filter(t => t < duration).length`.
