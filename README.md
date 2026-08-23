# Choreo

A tool for learning choreography from a song or a dance video. Import a
track, get a beat grid you tap out by ear, then slow it down, loop a
phrase, and isolate a single instrument while you drill it.

## Features

- **Library**: import audio or video files, rename or delete a track, and
  see each track's status at a glance (ready, needs counting, or missing its
  source file).
- **Waveform timeline**: a zoomed scrolling view plus a full-track minimap,
  with drum and bass onset markers you can toggle on or off.
- **Tap-to-count**: the beat grid, downbeats, and 8-counts come from tapping
  along with the music, not from automatic beat detection. Redo the count at
  any time (behind a confirmation, since it discards the current grid).
- **Hear**: isolate vocals, drums, bass, or the instrumental mix while you
  practice, powered by Demucs source separation.
- **A/B loop**: drag two handles to loop a section, with optional snapping
  to the beat, a 4-count, or an 8-count.
- **Playback speed**: slow a track down or speed it up without leaving the
  timeline.
- **Video support**: a track imported as video gets a second screen with the
  same transport, loop, and speed controls synced to the picture.
- **Settings**: accent color, dark mode, default snap mode, and how far the
  count display subdivides below a whole count.

## Tech stack

**Backend**: Python and FastAPI, with Demucs for stem separation,
[beat_this](https://github.com/CPJKU/beat_this) for beat detection,
librosa for onset detection and duration, and MoviePy for video controls.

**Frontend**: React 19 and TypeScript, built with Vite. The waveform is
rendered with wavesurfer.js. Tests run on Vitest, linting on Oxlint.

## Setup

Backend (from the repo root):

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/api/server.py          # serves http://127.0.0.1:8000
```

Frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev                       # serves http://localhost:5173
```

The frontend dev server proxies `/api` requests to the backend, so both
need to be running. Drop audio or video files into `tracks/`, or just use
the Import button once the app is open. The first time a track is opened,
Demucs runs a full separation pass on CPU, which takes a few minutes; every
later open reuses the cached result.

## Data layout

- `tracks/`: your source audio and video files. A track's id is its
  filename without the extension.
- `cache/stems/<md5>/`: derived data (separated stems, analysis.json).
  Reproducible from the source file, safe to delete at any time.
- `data/grids/<md5>.json`: a track's tapped beat grid. This cannot be
  recomputed from the audio, so it lives outside the cache tree.
- `data/library/<md5>.json`: one manifest per song, recording what it was
  called and how long it is, so a song is never lost even if its stems are
  evicted or its source file goes missing.

Both grids and manifests are keyed by the audio's MD5 hash rather than its
filename, so renaming a file (in the OS, or through the app's own Rename)
never loses its grid or its place in the library.

## API

All routes are served under `/api`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/tracks` | List every file in `tracks/`. |
| GET | `/tracks/{id}/analysis` | Full analysis for one track (runs it on a cache miss). |
| GET | `/tracks/{id}/analysis/progress` | Live Demucs separation progress. |
| GET | `/tracks/{id}/audio` | The original audio file. |
| GET | `/tracks/{id}/video` | The original video file, if this track is one. |
| GET | `/tracks/{id}/stems/{name}` | One separated stem. |
| GET/PUT/PATCH/DELETE | `/tracks/{id}/manual-grid` | Read, save, toggle, or discard the tapped grid. |
| GET | `/library` | Every song the app has ever known, with its current state. |
| PATCH | `/library/{md5}/rename` | Rename a song's source file. |
| DELETE | `/library/{md5}` | Permanently remove a song and everything derived from it. |
| POST | `/import` | Upload a new file into the library. |

## Testing

Backend:

```bash
venv/bin/python -m unittest discover -s tests
```

Frontend (from `frontend/`):

```bash
npm run build   # type-checks, then builds
npm run lint    # oxlint
npm test        # vitest
```
