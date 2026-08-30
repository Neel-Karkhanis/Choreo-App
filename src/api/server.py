"""FastAPI serving layer for the Choreo App.

Serves precomputed track analysis as JSON and streams the original audio
and separated stems. Every account gets its own tracks/, cache/stems/,
data/grids/, and data/library/ subtree, named by the account's numeric id
(see the ACCOUNTS AND DATA ISOLATION note below) — a track's id within that
subtree is its filename without the extension.

Run (from anywhere):
    venv/Scripts/python src/api/server.py
or:
    venv/Scripts/python -m uvicorn server:app --app-dir src/api

Either way requires SESSION_SECRET to be set (see .env.example) — auth.py
fails fast at import time without it.
"""

import json
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The analysis modules in src/ import each other as top-level modules, and
# source_separation writes its stem cache to a path handed to it explicitly
# by this file — both assume the process runs from the repo root.
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

# Loaded before any env-driven constant below is read, so a local .env (see
# .env.example) can supply SESSION_SECRET, CORS_ALLOWED_ORIGINS, etc. In
# Docker, env vars come from the container/compose environment directly and
# this is a no-op (no .env file is shipped in the image).
from dotenv import load_dotenv

load_dotenv()

import asyncio

import librosa
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator

# db/auth/tos/jobs/analysis live in src/api/ alongside this file, which is
# on sys.path whenever this module is (either via uvicorn's --app-dir or
# PYTHONPATH in the Docker image) — bare imports, same pattern as the
# analysis modules under src/.
import auth
import db
import jobs
from analysis import ANALYSIS_SCHEMA_VERSION

TRACKS_DIR = REPO_ROOT / "tracks"

# Two trees, split by LIFETIME rather than by key.
#
# CACHE_DIR is derived data: stems and analysis.json are reproducible from the
# audio by spending CPU. Deleting it is a normal operation — it runs ~145MB per
# song — and must stay safe to do at any moment.
#
# GRIDS_DIR is user data: a tapped grid is a human sitting there counting a song
# out by ear, and NOTHING can recompute it from the audio. It is deliberately a
# SIBLING of cache/, never a child, so that no cleanup routine aimed at the
# cache — `rm -rf cache/`, or any eviction policy added later — can reach it.
#
# Both stay keyed by the audio's MD5: that survives a rename and cannot follow a
# filename onto different audio.
#
# ACCOUNTS AND DATA ISOLATION: every one of these four trees, plus DB_PATH
# below, gets a `<user_id>/` segment inserted directly under its root (e.g.
# TRACKS_DIR / str(user.id) / ...). Every helper that resolves a path takes
# user_id as its first argument, sourced only from the authenticated
# session (auth.get_current_user via the `user` dependency every route
# below takes) — never from a client-supplied value — so a request
# literally cannot address another account's files, and content-hash
# dedup is per-account by construction rather than by an extra check.
CACHE_DIR = REPO_ROOT / "cache" / "stems"
GRIDS_DIR = REPO_ROOT / "data" / "grids"

# Library manifests are the app's record that a song EXISTS at all — the only
# thing that can still name an md5 after its stems have been evicted and its
# source file has moved. That makes them durable data, so they sit next to
# grids/ under data/, never under cache/.
LIBRARY_DIR = REPO_ROOT / "data" / "library"

DB_PATH = REPO_ROOT / "data" / "users.db"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
STEM_NAMES = ("vocals", "drums", "bass", "other", "instrumental")

# The four Demucs outputs. "instrumental" is derived from these and "original"
# is served from tracks/, so these four are what "stems are present" means.
REQUIRED_STEMS = ("vocals", "drums", "bass", "other")

# Per-account guardrails against one account filling the disk — an abuse/cost
# backstop, not a billing system. Counted against ORIGINAL source bytes only;
# derived cache (stems) is regenerable but runs several times larger than the
# source (observed ~8.7x on a real personal library), so an operator sizing a
# server's disk needs to budget for that multiplier on top of this cap, not
# assume the cap bounds total disk use.
CHOREO_MAX_STORAGE_BYTES = int(os.environ.get("CHOREO_MAX_STORAGE_BYTES", 2 * 1024**3))
CHOREO_MAX_TRACKS = int(os.environ.get("CHOREO_MAX_TRACKS", 60))

# Whether the session cookie gets the Secure flag (HTTPS-only). Off by
# default so local HTTP dev (no TLS) still works; set true in production,
# where Caddy always terminates real TLS in front of the app.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app):
    """Boot work. Resolves the callees at call time, so each one can stay down
    with the code it belongs to rather than being hoisted up here.
    """
    db.init_db(DB_PATH)
    yield


app = FastAPI(title="Choreo API", lifespan=lifespan)

_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_hash_cache = {}


def _file_hash(path):
    """MD5 of the file's contents, memoized on (path, mtime, size).

    Takes no user_id: the path handed in is already fully resolved (inside
    a specific account's subtree, when relevant), and different accounts'
    files never share a path even when their bytes are identical, so the
    memo key never collides across accounts.
    """
    import source_separation

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = source_separation._hash_file(str(path))
    return _hash_cache[key]


def _track_file(user_id, track_id):
    """Resolve a track id (filename stem) to its source media file in this
    account's tracks/ subtree.

    The source is audio or video, whichever extension is actually on disk.
    Callers that need a librosa-loadable audio path (analysis, the "original"
    stem) go through _resolve_analysis_audio, which is the one place that
    cares which kind this is.
    """
    if "/" in track_id or "\\" in track_id or track_id in (".", ".."):
        raise HTTPException(status_code=404, detail="Track not found")
    user_tracks_dir = TRACKS_DIR / str(user_id)
    for ext in MEDIA_EXTENSIONS:
        candidate = user_tracks_dir / (track_id + ext)
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="Track not found")


def _extracted_audio_path(user_id, md5):
    return CACHE_DIR / str(user_id) / md5 / "source_audio.wav"


def _resolve_analysis_audio(user_id, source_file, md5):
    """A librosa-loadable audio path for this track's source file.

    Audio sources are used directly. A video container isn't guaranteed
    decodable by librosa/soundfile, so its audio track is extracted once via
    moviepy and cached next to the stems — regenerable derived data, keyed by
    the same md5 the stems use, extracted at most once per hash.
    """
    if source_file.suffix.lower() in AUDIO_EXTENSIONS:
        return source_file

    wav_path = _extracted_audio_path(user_id, md5)
    if not wav_path.exists():
        import video_controls

        clip = video_controls.load_video(str(source_file))
        try:
            if clip.audio is None:
                raise HTTPException(status_code=400, detail="Video has no audio track")
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            clip.audio.write_audiofile(str(wav_path), fps=44100, codec="pcm_s16le", logger=None)
        finally:
            clip.close()
    return wav_path


@app.get("/api/tracks")
def list_tracks(user: db.User = Depends(auth.get_current_user)):
    tracks_dir = TRACKS_DIR / str(user.id)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    tracks = [
        {"id": f.stem, "filename": f.name}
        for f in sorted(tracks_dir.iterdir())
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return {"tracks": tracks}


@app.get("/api/tracks/{track_id}/analysis")
async def get_analysis(track_id, user: db.User = Depends(auth.get_current_user)):
    source_file = _track_file(user.id, track_id)
    md5 = _file_hash(source_file)
    cache_dir = CACHE_DIR / str(user.id) / md5
    cache_path = cache_dir / "analysis.json"

    analysis_data = None
    if cache_path.exists():
        analysis_data = json.loads(cache_path.read_text())
        if analysis_data.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            analysis_data = None

    if analysis_data is None:
        audio_file = _resolve_analysis_audio(user.id, source_file, md5)
        job = jobs.enqueue_or_attach(user.id, md5, audio_file, CACHE_DIR / str(user.id))
        # Poll for the worker's own on-disk result (see jobs.py's module
        # docstring for why this reads a file rather than RQ's result
        # channel), not a busy-loop: each iteration sleeps first, so a
        # cache hit racing a just-finished job still yields once before
        # returning. Ceiling matches the job_timeout jobs.py enqueues with.
        for _ in range(30 * 60 * 2):
            status = job.get_status(refresh=True)
            if cache_path.exists():
                analysis_data = json.loads(cache_path.read_text())
                break
            if status == "failed":
                raise HTTPException(status_code=500, detail="Analysis failed")
            await asyncio.sleep(0.5)
        else:
            raise HTTPException(
                status_code=504,
                detail="Analysis is taking unusually long; try again shortly",
            )

    # Analysis is the moment the app first knows a song's duration, so it is
    # also the moment its library manifest can be written in full.
    _write_manifest(user.id, source_file, md5, analysis_data.get("duration"))

    is_video = source_file.suffix.lower() in VIDEO_EXTENSIONS
    # URLs are built from the current track id rather than cached, so a
    # renamed file (same content hash) still gets working links.
    return {
        "id": track_id,
        "filename": source_file.name,
        **analysis_data,
        "audio_url": f"/api/tracks/{track_id}/audio",
        "video_url": f"/api/tracks/{track_id}/video" if is_video else None,
        "stems": [
            {"name": name, "url": f"/api/tracks/{track_id}/stems/{name}"}
            for name in STEM_NAMES
            if (cache_dir / (name + ".wav")).exists()
        ],
    }


@app.get("/api/tracks/{track_id}/analysis/progress")
def get_analysis_progress(track_id, user: db.User = Depends(auth.get_current_user)):
    """Demucs' own live separation progress for this track, if it is running.

    Meant to be polled (Song.tsx does, every ~400ms) alongside the /analysis
    request above while that request is in flight, to drive a loading ring
    off Demucs' real per-chunk progress rather than a simulated animation.
    {"active": false} covers every case where there's nothing to show: a
    cached track's near-instant reload never starts a Demucs run at all, and
    a finished or not-yet-started run both read the same way. That is also
    why this is its own endpoint rather than a field on /analysis: /analysis
    doesn't return until the whole pipeline is done, so nothing polling it
    could ever observe a run in progress.
    """
    import source_separation

    source_file = _track_file(user.id, track_id)
    md5 = _file_hash(source_file)
    state = source_separation.get_progress(f"{user.id}:{md5}")
    if state is None:
        return {"active": False}
    return {"active": True, "done": state["done"], "total": state["total"]}


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id, user: db.User = Depends(auth.get_current_user)):
    """The decodable audio the StemEngine's "original" buffer plays.

    For an audio source this is the file itself. For a video source it is the
    extracted track from _resolve_analysis_audio — the video container is
    served separately, from /video, for the <video> element.
    """
    source_file = _track_file(user.id, track_id)
    md5 = _file_hash(source_file)
    return FileResponse(_resolve_analysis_audio(user.id, source_file, md5))


@app.get("/api/tracks/{track_id}/video")
def get_video(track_id, user: db.User = Depends(auth.get_current_user)):
    source_file = _track_file(user.id, track_id)
    if source_file.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Track has no video")
    return FileResponse(source_file)


@app.get("/api/tracks/{track_id}/stems/{stem_name}")
def get_stem(track_id, stem_name, user: db.User = Depends(auth.get_current_user)):
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Unknown stem")
    audio_file = _track_file(user.id, track_id)
    stem_file = CACHE_DIR / str(user.id) / _file_hash(audio_file) / (stem_name + ".wav")
    if not stem_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Stem not generated yet — request the track's analysis first",
        )
    return FileResponse(stem_file)


class ManualGrid(BaseModel):
    """A user-tapped beat grid, in the analysis payload's own grid shape.

    Schema v4 exactly: `downbeats` and `eight_counts` are integer INDICES into
    `beats`, not timestamps. The frontend fits tempo/phase from the taps and
    derives all three arrays; the server validates the shape and stores it
    verbatim. Nothing here re-derives or "corrects" the grid — a tapped grid and
    a beat_this grid are the same object to every consumer, which is what lets
    the rest of the app stay ignorant of where a grid came from.
    """

    beats: list[float]
    downbeats: list[int]
    eight_counts: list[int]

    @model_validator(mode="after")
    def _check_shape(self):
        if len(self.beats) < 2:
            raise ValueError("beats must hold at least two entries")
        if self.beats[0] < 0:
            raise ValueError("beats must not be negative")
        if any(a >= b for a, b in zip(self.beats, self.beats[1:])):
            raise ValueError("beats must be strictly ascending")
        for name, indices in (
            ("downbeats", self.downbeats),
            ("eight_counts", self.eight_counts),
        ):
            if any(i < 0 or i >= len(self.beats) for i in indices):
                raise ValueError(f"{name} must be indices into beats")
        return self


def _manual_grid_path(user_id, audio_file):
    """Where a track's tapped grid lives: data/grids/<user_id>/<md5>.json.

    Keyed by the audio's MD5, like the cache — but stored OUTSIDE the cache
    tree, because this is the one artifact in the app that cannot be
    regenerated. See the GRIDS_DIR comment above.
    """
    return GRIDS_DIR / str(user_id) / f"{_file_hash(audio_file)}.json"


def _migrate_manual_grids(user_id):
    """Lift one account's tapped grids out of the old in-cache location.

    Grids used to be written to cache/stems/<md5>/manual_grid.json, where an
    `rm -rf cache/` would have destroyed them. Same MD5 key, new home, so the
    move is a pure relocation — no reinterpretation of the data.

    Not called from lifespan(): a brand-new account can never have the
    legacy shape (import_song/put_manual_grid always write the new layout),
    so there is nothing to lift for anyone but a migrated pre-existing
    account. scripts/migrate_owner_data.py calls this once, directly, for
    exactly that account.

    Idempotent: once the cache tree holds no manual_grid.json, the glob is empty
    and this is a no-op. If a grid somehow exists in BOTH places, data/ wins —
    it is the only one the app writes to now, so the cache copy is a stale
    leftover — and the leftover is dropped.

    Returns the number of files relocated.
    """
    cache_dir = CACHE_DIR / str(user_id)
    grids_dir = GRIDS_DIR / str(user_id)
    if not cache_dir.exists():
        return 0
    moved = 0
    for legacy in sorted(cache_dir.glob("*/manual_grid.json")):
        target = grids_dir / f"{legacy.parent.name}.json"
        grids_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            legacy.unlink()
            continue
        # Same filesystem (both under REPO_ROOT), so this is an atomic rename:
        # the grid is never momentarily absent from both trees.
        legacy.replace(target)
        moved += 1
    if moved:
        print(f"[grids] migrated {moved} tapped grid(s) for user {user_id} -> {grids_dir}")
    return moved


@app.get("/api/tracks/{track_id}/manual-grid")
def get_manual_grid(track_id, user: db.User = Depends(auth.get_current_user)):
    """The track's tapped grid, or null if it has never been tapped.

    Null rather than 404: having no manual grid is the normal state (auto is the
    default), not an error.
    """
    path = _manual_grid_path(user.id, _track_file(user.id, track_id))
    if not path.exists():
        return {"manual_grid": None}
    record = json.loads(path.read_text())
    # Grids written before `active` existed have no such key; their mere
    # presence on disk used to mean "this is the live grid", so that is the
    # default we read them back as.
    record.setdefault("active", True)
    return {"manual_grid": record}


@app.put("/api/tracks/{track_id}/manual-grid")
def put_manual_grid(
    track_id, payload: ManualGrid, user: db.User = Depends(auth.get_current_user)
):
    """Replace the track's tapped grid.

    Whole-set write: a grid is one object, and the frontend always has the
    complete fit in hand, so there is nothing to merge. A freshly accepted tap
    always comes back active — see ManualGridStore.save.
    """
    path = _manual_grid_path(user.id, _track_file(user.id, track_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = payload.model_dump()
    record["active"] = True
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


class ManualGridActive(BaseModel):
    active: bool


@app.patch("/api/tracks/{track_id}/manual-grid/active")
def set_manual_grid_active(
    track_id, payload: ManualGridActive, user: db.User = Depends(auth.get_current_user)
):
    """Switch between the tapped grid and auto detection without touching the
    tap data — the reversible counterpart to DELETE below.

    "Use auto detection" sets active=False; switching back sets it True. Either
    way beats/downbeats/eight_counts on disk never move, so the most recently
    accepted tap is always one flip away, no retapping needed.
    """
    path = _manual_grid_path(user.id, _track_file(user.id, track_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="no tapped grid for this track")
    record = json.loads(path.read_text())
    record["active"] = payload.active
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


@app.delete("/api/tracks/{track_id}/manual-grid")
def delete_manual_grid(track_id, user: db.User = Depends(auth.get_current_user)):
    """Permanently drop the tapped grid. Idempotent.

    The ONLY route in the app that destroys user data, and it is unrecoverable:
    nothing can re-derive a tapped grid from the audio. Deliberately NOT the
    "use auto detection" path anymore — that is set_manual_grid_active, which
    keeps the taps on disk. This exists for actually discarding them.
    """
    _manual_grid_path(user.id, _track_file(user.id, track_id)).unlink(missing_ok=True)
    return {"manual_grid": None}


# ---------------------------------------------------------------------------
# Library
#
# A manifest is one small JSON file per song, keyed by the audio's md5, holding
# the facts that survive everything else: what the file was called, how long it
# is, and whether it is audio or video. Stems get evicted and analyses get
# invalidated by a schema bump; the manifest is what still knows the song was
# ever here, which is what lets an evicted entry be shown honestly instead of
# vanishing from the library.
# ---------------------------------------------------------------------------


def _media_kind(filename):
    """"audio" or "video", from the file extension alone.

    Extension is the whole test on purpose: this decides which SCREENS a song
    offers, and that decision has to be available from a manifest with no file
    on disk behind it. A manifest may carry a media_kind that disagrees with
    the current extension — see _write_manifest — and the manifest wins.
    """
    return "video" if Path(filename).suffix.lower() in VIDEO_EXTENSIONS else "audio"


def _manifest_path(user_id, md5):
    return LIBRARY_DIR / str(user_id) / f"{md5}.json"


def _read_manifest(user_id, md5):
    path = _manifest_path(user_id, md5)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_manifest(user_id, audio_file, md5, duration=None):
    """Create or refresh a song's manifest. Idempotent.

    Filename and duration are refreshed from the live file, so a rename is
    picked up. media_kind is NOT: an existing manifest's media_kind is
    preserved verbatim, so a song deliberately marked "video" stays video
    across every later re-analysis. That is the only field a human is expected
    to edit by hand, and silently reverting a hand edit on the next cache miss
    would make the edit useless.
    """
    existing = _read_manifest(user_id, md5) or {}
    record = {
        **existing,
        "md5": md5,
        "track_id": audio_file.stem,
        "filename": audio_file.name,
        "media_kind": existing.get("media_kind") or _media_kind(audio_file.name),
    }
    if duration is not None:
        record["duration"] = duration
    record.setdefault("duration", None)

    _manifest_path(user_id, md5).parent.mkdir(parents=True, exist_ok=True)
    _manifest_path(user_id, md5).write_text(json.dumps(record))
    return record


def _tracks_by_md5(user_id):
    """md5 -> the file in this account's tracks/ holding that source (audio or video).

    Hashes every track on each call; tracks/<user_id>/ holds a handful of
    files and _file_hash memoizes on (path, mtime, size), so repeat calls
    are cheap. Hashing rather than trusting the manifest's filename is what
    makes a renamed file keep its grid, its stems, and its library entry.
    """
    user_tracks_dir = TRACKS_DIR / str(user_id)
    if not user_tracks_dir.exists():
        return {}
    index = {}
    for f in sorted(user_tracks_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
            index.setdefault(_file_hash(f), f)
    return index


def _stems_present(user_id, md5):
    cache_dir = CACHE_DIR / str(user_id) / md5
    return all((cache_dir / f"{name}.wav").exists() for name in REQUIRED_STEMS)


def _library_entry(user_id, md5, record, tracks):
    """One library row, with its state derived from what is actually on disk.

    THE THREE STATES, and why the boundary sits where it does:

      ready         — the source file is here and the song has a tapped grid.
      needs_tap     — the source file is here and it does not. Opens straight
                      into the tap state; this is the normal state of a song
                      that has just been imported.
      stems_evicted — the source file is GONE from tracks/. Cannot be opened
                      at all: the engine plays the original, and no amount of
                      cached derived data substitutes for it. Recovering it
                      means re-importing the same file, which restores the same
                      md5 and so picks the grid back up untouched.

    Missing stems are deliberately NOT what makes an entry unopenable. With
    the source in hand the app regenerates them, so that is a slow open, not a
    dead one — `stems_present` is reported separately so the UI can warn about
    the wait without demoting the row to a state it is not in.

    grid_present is reported on every row, including evicted ones, because an
    evicted row's whole message is "your tapped grid is still here".
    """
    grid_present = (GRIDS_DIR / str(user_id) / f"{md5}.json").exists()
    source = tracks.get(md5)

    if source is None:
        state = "stems_evicted"
    elif grid_present:
        state = "ready"
    else:
        state = "needs_tap"

    return {
        "md5": md5,
        # Follow the live file when we have it; fall back to what the manifest
        # last saw, which for an evicted entry is the only name left.
        "id": source.stem if source else record.get("track_id"),
        "filename": source.name if source else record.get("filename"),
        "duration": record.get("duration"),
        "media_kind": record.get("media_kind") or _media_kind(record.get("filename") or ""),
        "state": state,
        "grid_present": grid_present,
        "stems_present": _stems_present(user_id, md5),
        "original_present": source is not None,
    }


def _backfill_library(user_id):
    """Write manifests for one account's songs that predate manifests. Runs
    on every /api/library request for that account (see list_library) —
    idempotent and cheap (a handful of files per account), the same as it
    always was; the only change is which account's subtree it scans.

    Two sources of truth about what already exists: a cache dir with an
    analysis.json, and a grid file. Their union is every md5 this account has
    ever known.

    An md5 with neither a name nor a grid is skipped — a stray cache dir whose
    source is gone and which was never tapped has nothing to show and nothing
    to lose, so listing it would be noise, not honesty. An md5 with a GRID is
    always registered even when nothing can name it, because that grid is
    unrecoverable user data and must not disappear from the library.

    Returns the number of manifests written.
    """
    tracks = _tracks_by_md5(user_id)
    cache_dir = CACHE_DIR / str(user_id)
    grids_dir = GRIDS_DIR / str(user_id)

    known = set()
    if cache_dir.exists():
        known |= {d.name for d in cache_dir.iterdir() if (d / "analysis.json").exists()}
    if grids_dir.exists():
        known |= {f.stem for f in grids_dir.glob("*.json")}

    written = 0
    for md5 in sorted(known):
        if _manifest_path(user_id, md5).exists():
            continue
        source = tracks.get(md5)
        if source is not None:
            duration = None
            analysis_path = cache_dir / md5 / "analysis.json"
            if analysis_path.exists():
                try:
                    duration = json.loads(analysis_path.read_text()).get("duration")
                except json.JSONDecodeError:
                    pass
            _write_manifest(user_id, source, md5, duration)
            written += 1
        elif (grids_dir / f"{md5}.json").exists():
            _manifest_path(user_id, md5).parent.mkdir(parents=True, exist_ok=True)
            _manifest_path(user_id, md5).write_text(
                json.dumps(
                    {
                        "md5": md5,
                        "track_id": None,
                        "filename": None,
                        "duration": None,
                        "media_kind": "audio",
                    }
                )
            )
            written += 1

    if written:
        print(f"[library] backfilled {written} manifest(s) for user {user_id} -> {LIBRARY_DIR / str(user_id)}")
    return written


@app.get("/api/library")
def list_library(user: db.User = Depends(auth.get_current_user)):
    """Every song this account has ever known, with its current state.

    Entries are never filtered out by state. A song whose source file has gone
    missing still appears, still says whether its grid survived, and still
    tells the user which file to re-import to get it back.
    """
    _backfill_library(user.id)
    tracks = _tracks_by_md5(user.id)

    songs = []
    library_dir = LIBRARY_DIR / str(user.id)
    if library_dir.exists():
        for path in sorted(library_dir.glob("*.json")):
            record = _read_manifest(user.id, path.stem)
            if record is None:
                continue
            songs.append(_library_entry(user.id, path.stem, record, tracks))

    # Openable first, then by name, so the dead rows sink to the bottom
    # without being hidden.
    order = {"ready": 0, "needs_tap": 1, "stems_evicted": 2}
    songs.sort(key=lambda s: (order[s["state"]], (s["filename"] or "").lower()))
    return {"songs": songs}


class RenameTrack(BaseModel):
    name: str


@app.patch("/api/library/{md5}/rename")
def rename_track(md5, payload: RenameTrack, user: db.User = Depends(auth.get_current_user)):
    """Rename a song's source file on disk, keeping its extension.

    Keyed by md5, not track_id: the whole point of md5-keying (see the
    module docstring above _tracks_by_md5) is that a rename must not fork
    the library entry, its grid, or its cache into a second row — this is
    the one endpoint that deliberately changes the name that key maps to,
    so it has to take the key itself, not the name being replaced.

    Only the STEM is user-editable; the extension always follows the file
    actually on disk, so a rename can never turn audio into a wrong-typed
    video (or vice versa) by accident. Requires the source file to still be
    here — an evicted entry has nothing to rename until its file comes back.
    """
    source = _tracks_by_md5(user.id).get(md5)
    if source is None:
        raise HTTPException(status_code=404, detail="Track not found")

    new_stem = payload.name.strip()
    if not new_stem:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if "/" in new_stem or "\\" in new_stem or new_stem in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid name")

    target = source.with_name(new_stem + source.suffix)
    if target != source:
        if target.exists():
            raise HTTPException(status_code=409, detail="A track with that name already exists")
        source.rename(target)

    record = _write_manifest(user.id, target, md5)
    return {"md5": md5, "id": target.stem, "filename": record["filename"]}


@app.delete("/api/library/{md5}")
def delete_track(md5, user: db.User = Depends(auth.get_current_user)):
    """Permanently remove a song: source file, cached stems, tapped grid, and
    the library manifest itself. Idempotent — safe to call on a row missing
    any subset of these already (an evicted entry has no source file left,
    for instance).

    Unlike DELETE .../manual-grid, this also takes the SOURCE audio with
    it — there is no recovery path afterward, not even by re-importing the
    same bytes, since the manifest that would have reconnected them is gone
    too. The frontend gates this behind a confirmation for exactly that
    reason.
    """
    source = _tracks_by_md5(user.id).get(md5)
    if source is not None:
        source.unlink()
    stems_dir = CACHE_DIR / str(user.id) / md5
    if stems_dir.exists():
        shutil.rmtree(stems_dir)
    (GRIDS_DIR / str(user.id) / f"{md5}.json").unlink(missing_ok=True)
    _manifest_path(user.id, md5).unlink(missing_ok=True)
    return {"deleted": md5}


def _check_storage_quota(user_id, incoming_bytes):
    """Reject an import that would push this account over its cap.

    Only called right before bytes actually land on disk (see import_song) —
    never on the "identical file already here" early-return path, which
    doesn't grow usage at all.
    """
    user_tracks_dir = TRACKS_DIR / str(user_id)
    existing = list(user_tracks_dir.iterdir()) if user_tracks_dir.exists() else []
    existing_files = [f for f in existing if f.is_file()]
    existing_bytes = sum(f.stat().st_size for f in existing_files)
    existing_count = sum(1 for f in existing_files if f.suffix.lower() in MEDIA_EXTENSIONS)

    if existing_bytes + incoming_bytes > CHOREO_MAX_STORAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Storage quota exceeded — delete a track or contact the operator",
        )
    if existing_count >= CHOREO_MAX_TRACKS:
        raise HTTPException(
            status_code=413,
            detail="Track count limit reached — delete a track or contact the operator",
        )


@app.post("/api/import")
async def import_song(
    file: UploadFile = File(...), user: db.User = Depends(auth.get_current_user)
):
    """Take an uploaded file into this account's tracks/ and register it in
    the library.

    Re-importing a file the app already has is the documented recovery path for
    an evicted entry, so it must be safe and it must be idempotent: identical
    bytes resolve to the same md5, which means the same grid, the same cache,
    and the same manifest. Nothing on disk is overwritten by that case.
    """
    # Path().name strips any directory component a client may have sent —
    # browsers send a bare name, but this endpoint must not trust that.
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="No filename")

    suffix = Path(name).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or name}'",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    user_tracks_dir = TRACKS_DIR / str(user.id)
    user_tracks_dir.mkdir(parents=True, exist_ok=True)
    target = user_tracks_dir / name
    if target.exists():
        import hashlib

        if hashlib.md5(data).hexdigest() == _file_hash(target):
            # Same song, already here. Say so instead of writing a duplicate.
            md5 = _file_hash(target)
            _write_manifest(user.id, target, md5)
            return {"id": target.stem, "filename": target.name, "md5": md5, "created": False}
        # Same name, different audio: keep both rather than clobbering one.
        stem, n = Path(name).stem, 2
        while target.exists():
            target = user_tracks_dir / f"{stem} ({n}){suffix}"
            n += 1

    _check_storage_quota(user.id, len(data))

    target.write_bytes(data)
    md5 = _file_hash(target)
    _write_manifest(user.id, target, md5)
    return {"id": target.stem, "filename": target.name, "md5": md5, "created": True}


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


class SignupRequest(BaseModel):
    email: str
    password: str
    tos_accepted: bool


class LoginRequest(BaseModel):
    email: str
    password: str


def _set_session_cookie(response: Response, user_id: int):
    response.set_cookie(
        key=auth.SESSION_COOKIE_NAME,
        value=auth.make_session_cookie(user_id),
        max_age=auth.SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
    )


@app.post("/api/auth/signup")
def signup(payload: SignupRequest, response: Response):
    """Create an account and log straight in. No forgot-password flow yet —
    there's no email-sending infra in this app — so this is deliberately the
    only way in besides /login.
    """
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not payload.tos_accepted:
        raise HTTPException(status_code=400, detail="You must accept the Terms of Use")
    if db.get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="An account with that email already exists")

    user = db.create_user(email, auth.hash_password(payload.password))
    _set_session_cookie(response, user.id)
    return {"id": user.id, "email": user.email}


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response):
    user = db.get_user_by_email(payload.email)
    if user is None or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    _set_session_cookie(response, user.id)
    return {"id": user.id, "email": user.email}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(auth.SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: db.User = Depends(auth.get_current_user)):
    return {"id": user.id, "email": user.email}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
