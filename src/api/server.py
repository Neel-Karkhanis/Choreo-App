"""FastAPI serving layer for the Choreo App.

Serves precomputed track analysis as JSON and streams the original audio
and separated stems. Tracks are audio files dropped into the repo-root
tracks/ folder; a track's id is its filename without the extension.

Run (from anywhere):
    venv/Scripts/python src/api/server.py
or:
    venv/Scripts/python -m uvicorn server:app --app-dir src/api
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The analysis modules in src/ import each other as top-level modules, and
# source_separation writes its stem cache to the relative path cache/stems/
# — both assume the process runs from the repo root.
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

if sys.platform == "win32":
    # torchaudio 2.9+ saves audio through torchcodec, whose native library
    # links against FFmpeg shared DLLs (avcodec etc.). Python 3.8+ resolves
    # DLL dependencies only from add_dll_directory() entries, never PATH, so
    # register every PATH dir that ships the FFmpeg runtime. Needs an FFmpeg
    # "shared" build on PATH (e.g. winget install Gyan.FFmpeg.Shared).
    import glob

    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        if _dir and glob.glob(os.path.join(_dir, "avcodec-*.dll")):
            os.add_dll_directory(_dir)

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

TRACKS_DIR = REPO_ROOT / "tracks"
CACHE_DIR = REPO_ROOT / "cache" / "stems"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
STEM_NAMES = ("vocals", "drums", "bass", "other", "instrumental")

# Bump when the analysis JSON shape changes; cached analysis.json files with
# a different version are discarded and recomputed (stems stay cached).
ANALYSIS_SCHEMA_VERSION = 4

app = FastAPI(title="Choreo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_hash_cache = {}


def _file_hash(path):
    """MD5 of the file's contents, memoized on (path, mtime, size)."""
    import source_separation

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = source_separation._hash_file(str(path))
    return _hash_cache[key]


def _track_file(track_id):
    """Resolve a track id (filename stem) to its audio file in tracks/."""
    if "/" in track_id or "\\" in track_id or track_id in (".", ".."):
        raise HTTPException(status_code=404, detail="Track not found")
    for ext in AUDIO_EXTENSIONS:
        candidate = TRACKS_DIR / (track_id + ext)
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="Track not found")


def _beat_index(beats, timestamp, label):
    """Index of an exact timestamp in the beats array.

    beat_this's minimal postprocessor snaps each downbeat onto a beat time
    (an exact float copy), so downbeats — and eight-count starts derived from
    them — are guaranteed to appear in beats verbatim. A missing match means
    that upstream guarantee broke; fail loudly rather than snapping to the
    nearest beat and hiding it.
    """
    matches = np.where(beats == timestamp)[0]
    if len(matches) == 0:
        raise RuntimeError(
            f"{label} at {timestamp}s has no exact match in the beat array; "
            "beat_this may no longer guarantee downbeats are a subset of beats"
        )
    return int(matches[0])


def _onset_payload(onsets):
    """Serialize detect_onsets output as parallel t/strength lists."""
    return {
        "t": [round(float(t), 3) for t in onsets["t"]],
        "strength": [round(float(s), 4) for s in onsets["strength"]],
    }


def _analyze(audio_file):
    """Run the full analysis pipeline for one track.

    Slow on a cache miss (Demucs + beat_this on CPU); the caller persists
    the result to analysis.json so this runs at most once per file hash.

    In the returned dict, `downbeats` and `eight_counts` are integer indices
    into `beats`, not timestamps: the frontend reads beats[i] for the time,
    so markers always coincide with a beat exactly. The last eight-count
    index may start a partial group (fewer than 8 beats before the track
    ends); its extent is up to the next index, or the end of beats.

    `onsets.drums` and `onsets.bass` are independent of the beat grid: each
    carries parallel lists `t` (timestamps in seconds, not indices, not
    guaranteed to land on a beat) and `strength` (onset-envelope value at
    that onset). Strength is plumbed by schema v4 for future use and is not
    consumed by the frontend yet.
    """
    # Lazy imports: beat_detection loads the beat_this model at import time,
    # so the server boots fast and pays that cost on the first analysis.
    import beat_detection
    import onset_detection
    import source_separation

    stems = source_separation.separate(str(audio_file))
    source_separation.get_instrumental(stems)

    beats, downbeats = beat_detection.detect_beats(str(audio_file))
    groups = beat_detection.eight_count_grouping(beats, downbeats)

    drum_onsets = onset_detection.detect_onsets(source_separation.get_drums(stems), "drums")
    bass_onsets = onset_detection.detect_onsets(source_separation.get_bass(stems), "bass")

    if len(beats) >= 2:
        tempo = round(60.0 / float(np.median(np.diff(beats))), 1)
    else:
        tempo = None

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "duration": round(librosa.get_duration(path=str(audio_file)), 3),
        "tempo": tempo,
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [_beat_index(beats, d, "downbeat") for d in downbeats],
        "eight_counts": [
            _beat_index(beats, group[0][0], "eight-count start") for group in groups
        ],
        "onsets": {
            "drums": _onset_payload(drum_onsets),
            "bass": _onset_payload(bass_onsets),
        },
    }


@app.get("/api/tracks")
def list_tracks():
    TRACKS_DIR.mkdir(exist_ok=True)
    tracks = [
        {"id": f.stem, "filename": f.name}
        for f in sorted(TRACKS_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return {"tracks": tracks}


@app.get("/api/tracks/{track_id}/analysis")
def get_analysis(track_id):
    audio_file = _track_file(track_id)
    cache_dir = CACHE_DIR / _file_hash(audio_file)
    cache_path = cache_dir / "analysis.json"

    analysis = None
    if cache_path.exists():
        analysis = json.loads(cache_path.read_text())
        if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            analysis = None

    if analysis is None:
        analysis = _analyze(audio_file)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(analysis))

    # URLs are built from the current track id rather than cached, so a
    # renamed file (same content hash) still gets working links.
    return {
        "id": track_id,
        "filename": audio_file.name,
        **analysis,
        "audio_url": f"/api/tracks/{track_id}/audio",
        "stems": [
            {"name": name, "url": f"/api/tracks/{track_id}/stems/{name}"}
            for name in STEM_NAMES
            if (cache_dir / (name + ".wav")).exists()
        ],
    }


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id):
    return FileResponse(_track_file(track_id))


@app.get("/api/tracks/{track_id}/stems/{stem_name}")
def get_stem(track_id, stem_name):
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Unknown stem")
    audio_file = _track_file(track_id)
    stem_file = CACHE_DIR / _file_hash(audio_file) / (stem_name + ".wav")
    if not stem_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Stem not generated yet — request the track's analysis first",
        )
    return FileResponse(stem_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
