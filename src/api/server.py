"""FastAPI serving layer for the Choreo App.

Serves precomputed track analysis as JSON and streams the original audio
and separated stems. There are no devices: every request is scoped to an
anonymous device by a UUIDv4 owner_id (see identity.py), and every device
gets its own tracks/, cache/stems/, data/grids/, and data/library/ subtree,
named by that id (see the DEVICES AND DATA ISOLATION note below) — a
track's id within that subtree is its filename without the extension.

Run (from anywhere):
    venv/Scripts/python src/api/server.py
or:
    venv/Scripts/python -m uvicorn server:app --app-dir src/api

Either way requires SESSION_SECRET to be set (see .env.example) —
identity.py fails fast at import time without it.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Literal

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
from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# identity/jobs/analysis live in src/api/ alongside this file, which is on
# sys.path whenever this module is (either via uvicorn's --app-dir or
# PYTHONPATH in the Docker image) — bare imports, same pattern as the
# analysis modules under src/.
import identity
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
# DEVICES AND DATA ISOLATION: every one of these three trees gets an
# `<owner_id>/` segment inserted directly under its root (e.g.
# TRACKS_DIR / owner_id / ...), where owner_id is a UUIDv4 string identifying
# an anonymous device, never an account. Every helper that resolves a path
# takes owner_id as its first argument, sourced only from
# identity.get_owner_id (the `owner_id` dependency every route below takes)
# — never from a client-supplied value read any other way — so a request
# literally cannot address another device's files, and content-hash dedup is
# per-device by construction rather than by an extra check. identity.py's
# validation is what makes owner_id safe to use as a path segment at all: see
# its module docstring for why that's a canonical-UUID round-trip check and
# not a regex.
CACHE_DIR = REPO_ROOT / "cache" / "stems"
GRIDS_DIR = REPO_ROOT / "data" / "grids"

# Library manifests are the app's record that a song EXISTS at all — the only
# thing that can still name an md5 after its stems have been evicted and its
# source file has moved. That makes them durable data, so they sit next to
# grids/ under data/, never under cache/.
LIBRARY_DIR = REPO_ROOT / "data" / "library"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
STEM_NAMES = ("vocals", "drums", "bass", "other", "instrumental")

# A practical bound, not a byte-exact filesystem limit (most Linux
# filesystems cap one path component at 255 BYTES, not characters, and a
# multi-byte-heavy UTF-8 name could in principle still exceed that under
# this character count) — its job is to turn an absurd client-supplied
# name into a clean 400 in import_song/rename_track below, rather than an
# unhandled OSError (ENAMETOOLONG) surfacing as a 500. 200 leaves headroom
# for import_song's own " (2)" collision suffix plus the extension.
MAX_FILENAME_LENGTH = 200

# The four Demucs outputs. "instrumental" is derived from these and "original"
# is served from tracks/, so these four are what "stems are present" means.
REQUIRED_STEMS = ("vocals", "drums", "bass", "other")

# Per-device guardrails against one device filling the disk — an abuse/cost
# backstop, not a billing system. Counted against ORIGINAL source bytes only;
# derived cache (stems) is regenerable but runs several times larger than the
# source (observed ~8.7x on a real personal library), so an operator sizing a
# server's disk needs to budget for that multiplier on top of this cap, not
# assume the cap bounds total disk use.
CHOREO_MAX_STORAGE_BYTES = int(os.environ.get("CHOREO_MAX_STORAGE_BYTES", 2 * 1024**3))
CHOREO_MAX_TRACKS = int(os.environ.get("CHOREO_MAX_TRACKS", 60))

# Hard cap on any single upload, checked while READING the body (see
# _read_upload_within_limit), not just against Content-Length — a header is
# a claim, not a guarantee, and this app has no accounts behind an upload to
# hold accountable if it lied. Default is generous enough for a real video
# file (this app imports video, not just audio) while still bounding how
# much of a single request one anonymous, unauthenticated caller can make
# the server buffer and hash.
CHOREO_MAX_UPLOAD_BYTES = int(os.environ.get("CHOREO_MAX_UPLOAD_BYTES", 500 * 1024**2))

# Whether the device-id cookie gets the Secure flag (HTTPS-only). Off by
# default so local HTTP dev (no TLS) still works; set true in production,
# where Caddy always terminates real TLS in front of the app.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"

# Per-IP, not per-device: owner_id is free to mint (POST /api/device, no
# accounts behind it — see identity.py's BEARER CREDENTIAL note), so a
# per-device limit would cost an abuser nothing to route around. Keyed off
# request.client.host, which is only ever the REAL client address if
# something upstream both sets X-Forwarded-For and uvicorn is told to trust
# it — see the Dockerfile's uvicorn CMD (--proxy-headers
# --forwarded-allow-ips) and the Caddyfile's reverse_proxy, which sets that
# header by default. Safe to trust unconditionally here specifically
# because docker-compose.yml never publishes the backend service's own
# port — every request through this process's socket has already gone
# through Caddy, there is no direct path that could spoof the header.
#
# Rate strings are slowapi/limits syntax ("N/minute"); both configurable so
# an operator can tune without a code change. The two @limiter.limit(...)
# decorators below read these through a lambda, not the bare module-level
# name — slowapi accepts either, but a bare name is captured once at
# decoration time (import time) and can never change after; a lambda is
# re-evaluated on every request, which is also what lets tests drive a
# specific limit down for one test without needing to re-decorate the route.
CHOREO_IMPORT_RATE_LIMIT = os.environ.get("CHOREO_IMPORT_RATE_LIMIT", "10/minute")
CHOREO_ANALYSIS_RATE_LIMIT = os.environ.get("CHOREO_ANALYSIS_RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Choreo API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

    Takes no owner_id: the path handed in is already fully resolved (inside
    a specific device's subtree, when relevant), and different devices'
    files never share a path even when their bytes are identical, so the
    memo key never collides across devices.
    """
    import source_separation

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = source_separation._hash_file(str(path))
    return _hash_cache[key]


_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _validate_md5(md5: str) -> str:
    """The entire defense against a malformed `{md5}` path parameter
    reaching a filesystem operation — several routes below build a path
    directly from this value (e.g. delete_track's
    `CACHE_DIR / owner_id / md5`, with no ".json" suffix to absorb a bare
    ".." the way _manifest_path's `f"{md5}.json"` incidentally does), so
    this has to run before any of them touch disk.

    Real MD5 hex digests are always exactly 32 lowercase hex characters —
    hashlib.md5().hexdigest()'s only possible output shape — so anything
    else is rejected outright: the same "malformed input is a 400, never
    partially trusted" stance identity.py takes for owner_id, and for the
    same reason (this is client-supplied on every route below, not always
    server-recomputed).
    """
    if not _MD5_PATTERN.fullmatch(md5):
        raise HTTPException(status_code=400, detail="Malformed md5")
    return md5


def _track_file(owner_id, track_id):
    """Resolve a track id (filename stem) to its source media file in this
    device's tracks/ subtree.

    The source is audio or video, whichever extension is actually on disk.
    Callers that need a librosa-loadable audio path (analysis, the "original"
    stem) go through _resolve_analysis_audio, which is the one place that
    cares which kind this is.
    """
    if "/" in track_id or "\\" in track_id or track_id in (".", ".."):
        raise HTTPException(status_code=404, detail="Track not found")
    owner_tracks_dir = TRACKS_DIR / owner_id
    for ext in MEDIA_EXTENSIONS:
        candidate = owner_tracks_dir / (track_id + ext)
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="Track not found")


def _extracted_audio_path(owner_id, md5):
    return CACHE_DIR / owner_id / md5 / "source_audio.wav"


def _resolve_analysis_audio(owner_id, source_file, md5):
    """A librosa-loadable audio path for this track's source file.

    Audio sources are used directly. A video container isn't guaranteed
    decodable by librosa/soundfile, so its audio track is extracted once via
    moviepy and cached next to the stems — regenerable derived data, keyed by
    the same md5 the stems use, extracted at most once per hash.
    """
    if source_file.suffix.lower() in AUDIO_EXTENSIONS:
        return source_file

    wav_path = _extracted_audio_path(owner_id, md5)
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
def list_tracks(owner_id: str = Depends(identity.get_owner_id)):
    tracks_dir = TRACKS_DIR / owner_id
    tracks_dir.mkdir(parents=True, exist_ok=True)
    tracks = [
        {"id": f.stem, "filename": f.name}
        for f in sorted(tracks_dir.iterdir())
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return {"tracks": tracks}


@app.get("/api/tracks/{track_id}/analysis")
@limiter.limit(lambda: CHOREO_ANALYSIS_RATE_LIMIT)
async def get_analysis(request: Request, track_id, owner_id: str = Depends(identity.get_owner_id)):
    # `request` is required by slowapi's decorator above (it reads the
    # client address off it) — not otherwise used, same as every other
    # route here, which read identity purely from the `owner_id` dependency.
    source_file = _track_file(owner_id, track_id)
    md5 = _file_hash(source_file)
    cache_dir = CACHE_DIR / owner_id / md5
    cache_path = cache_dir / "analysis.json"

    analysis_data = None
    if cache_path.exists():
        analysis_data = json.loads(cache_path.read_text())
        if analysis_data.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            analysis_data = None

    if analysis_data is None:
        audio_file = _resolve_analysis_audio(owner_id, source_file, md5)
        job = jobs.enqueue_or_attach(owner_id, md5, audio_file, CACHE_DIR / owner_id)
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
    _write_manifest(owner_id, source_file, md5, analysis_data.get("duration"))

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
def get_analysis_progress(track_id, owner_id: str = Depends(identity.get_owner_id)):
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

    source_file = _track_file(owner_id, track_id)
    md5 = _file_hash(source_file)
    # Hashed through identity.owner_key, not owner_id itself — this is the
    # same key analysis.py's analyze() publishes progress under; see its
    # own comment for why raw owner_id must never be the literal key.
    state = source_separation.get_progress(f"{identity.owner_key(owner_id)}:{md5}")
    if state is None:
        return {"active": False}
    return {"active": True, "done": state["done"], "total": state["total"]}


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id, owner_id: str = Depends(identity.get_owner_id)):
    """The decodable audio the StemEngine's "original" buffer plays.

    For an audio source this is the file itself. For a video source it is the
    extracted track from _resolve_analysis_audio — the video container is
    served separately, from /video, for the <video> element.
    """
    source_file = _track_file(owner_id, track_id)
    md5 = _file_hash(source_file)
    return FileResponse(_resolve_analysis_audio(owner_id, source_file, md5))


@app.get("/api/tracks/{track_id}/video")
def get_video(track_id, owner_id: str = Depends(identity.get_owner_id)):
    source_file = _track_file(owner_id, track_id)
    if source_file.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Track has no video")
    return FileResponse(source_file)


STEM_OPUS_BITRATE = "128k"


def _ensure_stem_opus(wav_path: Path, opus_path: Path) -> None:
    """Encode a stem's WAV working copy to Opus for delivery, once.

    Stems come out of Demucs as WAV — lossless, and far too large to hand a
    client or cache on-device (observed 100-180MB per song across all four
    stems in this repo's own fixtures). The WAV stays the server-side
    working copy that nothing else reads from directly; opus_path is the
    one clients ever see, encoded lazily on first request and cached
    alongside the WAV from then on — the same lazy-cache-once shape as
    _resolve_analysis_audio's video-audio extraction above, not something
    the analysis job itself needs to wait on.
    """
    if opus_path.exists():
        return
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-c:a",
            "libopus",
            "-b:a",
            STEM_OPUS_BITRATE,
            "-vn",
            str(opus_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        opus_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Stem encoding failed: {result.stderr.decode(errors='replace')[-500:]}",
        )


@app.get("/api/tracks/{track_id}/stems/{stem_name}")
def get_stem(track_id, stem_name, owner_id: str = Depends(identity.get_owner_id)):
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Unknown stem")
    audio_file = _track_file(owner_id, track_id)
    stem_dir = CACHE_DIR / owner_id / _file_hash(audio_file)
    wav_path = stem_dir / (stem_name + ".wav")
    if not wav_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Stem not generated yet — request the track's analysis first",
        )
    opus_path = stem_dir / (stem_name + ".opus")
    _ensure_stem_opus(wav_path, opus_path)
    return FileResponse(opus_path, media_type="audio/ogg")


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


def _manual_grid_path(owner_id, audio_file):
    """Where a track's tapped grid lives: data/grids/<owner_id>/<md5>.json.

    Keyed by the audio's MD5, like the cache — but stored OUTSIDE the cache
    tree, because this is the one artifact in the app that cannot be
    regenerated. See the GRIDS_DIR comment above.
    """
    return GRIDS_DIR / owner_id / f"{_file_hash(audio_file)}.json"


def _migrate_manual_grids(owner_id):
    """Lift one device's tapped grids out of the old in-cache location.

    Grids used to be written to cache/stems/<md5>/manual_grid.json, where an
    `rm -rf cache/` would have destroyed them. Same MD5 key, new home, so the
    move is a pure relocation — no reinterpretation of the data.

    Not called at startup: a brand-new device can never have the legacy
    shape (import_song/put_manual_grid always write the new layout), so
    there is nothing to lift for anyone but a device whose data predates
    this layout. scripts/reassign_owner.py calls this once, directly, for
    exactly that device, as part of reassigning pre-existing data to a
    device id.

    Idempotent: once the cache tree holds no manual_grid.json, the glob is empty
    and this is a no-op. If a grid somehow exists in BOTH places, data/ wins —
    it is the only one the app writes to now, so the cache copy is a stale
    leftover — and the leftover is dropped.

    Returns the number of files relocated.
    """
    cache_dir = CACHE_DIR / owner_id
    grids_dir = GRIDS_DIR / owner_id
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
        # identity.owner_key(owner_id), never owner_id itself — see that
        # function's docstring for why a device's actual id must never be
        # written to a log line, and the path is omitted for the same
        # reason (it has owner_id baked into it as a directory name).
        print(f"[grids] migrated {moved} tapped grid(s) for device {identity.owner_key(owner_id)}")
    return moved


@app.get("/api/tracks/{track_id}/manual-grid")
def get_manual_grid(track_id, owner_id: str = Depends(identity.get_owner_id)):
    """The track's tapped grid, or null if it has never been tapped.

    Null rather than 404: having no manual grid is the normal state (auto is the
    default), not an error.
    """
    path = _manual_grid_path(owner_id, _track_file(owner_id, track_id))
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
    track_id, payload: ManualGrid, owner_id: str = Depends(identity.get_owner_id)
):
    """Replace the track's tapped grid.

    Whole-set write: a grid is one object, and the frontend always has the
    complete fit in hand, so there is nothing to merge. A freshly accepted tap
    always comes back active — see ManualGridStore.save.
    """
    path = _manual_grid_path(owner_id, _track_file(owner_id, track_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = payload.model_dump()
    record["active"] = True
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


class ManualGridActive(BaseModel):
    active: bool


@app.patch("/api/tracks/{track_id}/manual-grid/active")
def set_manual_grid_active(
    track_id, payload: ManualGridActive, owner_id: str = Depends(identity.get_owner_id)
):
    """Switch between the tapped grid and auto detection without touching the
    tap data — the reversible counterpart to DELETE below.

    "Use auto detection" sets active=False; switching back sets it True. Either
    way beats/downbeats/eight_counts on disk never move, so the most recently
    accepted tap is always one flip away, no retapping needed.
    """
    path = _manual_grid_path(owner_id, _track_file(owner_id, track_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="no tapped grid for this track")
    record = json.loads(path.read_text())
    record["active"] = payload.active
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


@app.delete("/api/tracks/{track_id}/manual-grid")
def delete_manual_grid(track_id, owner_id: str = Depends(identity.get_owner_id)):
    """Permanently drop the tapped grid. Idempotent.

    The ONLY route in the app that destroys user data, and it is unrecoverable:
    nothing can re-derive a tapped grid from the audio. Deliberately NOT the
    "use auto detection" path anymore — that is set_manual_grid_active, which
    keeps the taps on disk. This exists for actually discarding them.
    """
    _manual_grid_path(owner_id, _track_file(owner_id, track_id)).unlink(missing_ok=True)
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


def _manifest_path(owner_id, md5):
    return LIBRARY_DIR / owner_id / f"{md5}.json"


def _read_manifest(owner_id, md5):
    path = _manifest_path(owner_id, md5)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_manifest_fields(owner_id, md5, filename, media_kind, duration=None):
    """Create or refresh a manifest from explicit field values rather than a
    live file on disk — the core _write_manifest delegates to this once it
    has read those values off a real Path; import_project (Phase 4) calls it
    directly, since a freshly-imported project has a filename to RECORD but
    no source bytes to read one from (see server.py's export/import note).

    Idempotent, same as _write_manifest. media_kind is preserved from any
    existing manifest, same reasoning as _write_manifest: it is the one
    field a human is expected to hand-edit, and reverting that edit on the
    next write would make the edit useless.
    """
    existing = _read_manifest(owner_id, md5) or {}
    record = {
        **existing,
        "md5": md5,
        "track_id": Path(filename).stem if filename else existing.get("track_id"),
        "filename": filename if filename is not None else existing.get("filename"),
        "media_kind": existing.get("media_kind") or media_kind,
    }
    if duration is not None:
        record["duration"] = duration
    record.setdefault("duration", None)

    _manifest_path(owner_id, md5).parent.mkdir(parents=True, exist_ok=True)
    _manifest_path(owner_id, md5).write_text(json.dumps(record))
    return record


def _write_manifest(owner_id, audio_file, md5, duration=None):
    """Create or refresh a song's manifest from a live file. Idempotent.

    Filename and duration are refreshed from the live file, so a rename is
    picked up.
    """
    return _write_manifest_fields(
        owner_id, md5, audio_file.name, _media_kind(audio_file.name), duration
    )


def _tracks_by_md5(owner_id):
    """md5 -> the file in this device's tracks/ holding that source (audio or video).

    Hashes every track on each call; tracks/<owner_id>/ holds a handful of
    files and _file_hash memoizes on (path, mtime, size), so repeat calls
    are cheap. Hashing rather than trusting the manifest's filename is what
    makes a renamed file keep its grid, its stems, and its library entry.
    """
    owner_tracks_dir = TRACKS_DIR / owner_id
    if not owner_tracks_dir.exists():
        return {}
    index = {}
    for f in sorted(owner_tracks_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
            index.setdefault(_file_hash(f), f)
    return index


def _stems_present(owner_id, md5):
    cache_dir = CACHE_DIR / owner_id / md5
    return all((cache_dir / f"{name}.wav").exists() for name in REQUIRED_STEMS)


def _library_entry(owner_id, md5, record, tracks):
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
    grid_present = (GRIDS_DIR / owner_id / f"{md5}.json").exists()
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
        "stems_present": _stems_present(owner_id, md5),
        "original_present": source is not None,
    }


def _backfill_library(owner_id):
    """Write manifests for one device's songs that predate manifests. Runs
    on every /api/library request for that device (see list_library) —
    idempotent and cheap (a handful of files per device), the same as it
    always was; the only change is which device's subtree it scans.

    Two sources of truth about what already exists: a cache dir with an
    analysis.json, and a grid file. Their union is every md5 this device has
    ever known.

    An md5 with neither a name nor a grid is skipped — a stray cache dir whose
    source is gone and which was never tapped has nothing to show and nothing
    to lose, so listing it would be noise, not honesty. An md5 with a GRID is
    always registered even when nothing can name it, because that grid is
    unrecoverable user data and must not disappear from the library.

    Returns the number of manifests written.
    """
    tracks = _tracks_by_md5(owner_id)
    cache_dir = CACHE_DIR / owner_id
    grids_dir = GRIDS_DIR / owner_id

    known = set()
    if cache_dir.exists():
        known |= {d.name for d in cache_dir.iterdir() if (d / "analysis.json").exists()}
    if grids_dir.exists():
        known |= {f.stem for f in grids_dir.glob("*.json")}

    written = 0
    for md5 in sorted(known):
        if _manifest_path(owner_id, md5).exists():
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
            _write_manifest(owner_id, source, md5, duration)
            written += 1
        elif (grids_dir / f"{md5}.json").exists():
            _manifest_path(owner_id, md5).parent.mkdir(parents=True, exist_ok=True)
            _manifest_path(owner_id, md5).write_text(
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
        # See the matching comment in _migrate_manual_grids above: hashed
        # key and no path, because owner_id itself must never reach a log.
        print(f"[library] backfilled {written} manifest(s) for device {identity.owner_key(owner_id)}")
    return written


@app.get("/api/library")
def list_library(owner_id: str = Depends(identity.get_owner_id)):
    """Every song this device has ever known, with its current state.

    Entries are never filtered out by state. A song whose source file has gone
    missing still appears, still says whether its grid survived, and still
    tells the user which file to re-import to get it back.
    """
    _backfill_library(owner_id)
    tracks = _tracks_by_md5(owner_id)

    songs = []
    library_dir = LIBRARY_DIR / owner_id
    if library_dir.exists():
        for path in sorted(library_dir.glob("*.json")):
            record = _read_manifest(owner_id, path.stem)
            if record is None:
                continue
            songs.append(_library_entry(owner_id, path.stem, record, tracks))

    # Openable first, then by name, so the dead rows sink to the bottom
    # without being hidden.
    order = {"ready": 0, "needs_tap": 1, "stems_evicted": 2}
    songs.sort(key=lambda s: (order[s["state"]], (s["filename"] or "").lower()))
    return {"songs": songs}


class RenameTrack(BaseModel):
    name: str


@app.patch("/api/library/{md5}/rename")
def rename_track(md5, payload: RenameTrack, owner_id: str = Depends(identity.get_owner_id)):
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
    md5 = _validate_md5(md5)
    source = _tracks_by_md5(owner_id).get(md5)
    if source is None:
        raise HTTPException(status_code=404, detail="Track not found")

    new_stem = payload.name.strip()
    if not new_stem:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if "/" in new_stem or "\\" in new_stem or new_stem in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid name")
    if new_stem.startswith("."):
        raise HTTPException(status_code=400, detail="Name must not start with a dot")
    if len(new_stem) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"Name is too long (max {MAX_FILENAME_LENGTH} characters)"
        )

    target = source.with_name(new_stem + source.suffix)
    if target != source:
        if target.exists():
            raise HTTPException(status_code=409, detail="A track with that name already exists")
        source.rename(target)

    record = _write_manifest(owner_id, target, md5)
    return {"md5": md5, "id": target.stem, "filename": record["filename"]}


@app.delete("/api/library/{md5}")
def delete_track(md5, owner_id: str = Depends(identity.get_owner_id)):
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
    md5 = _validate_md5(md5)
    source = _tracks_by_md5(owner_id).get(md5)
    if source is not None:
        source.unlink()
    stems_dir = CACHE_DIR / owner_id / md5
    if stems_dir.exists():
        shutil.rmtree(stems_dir)
    (GRIDS_DIR / owner_id / f"{md5}.json").unlink(missing_ok=True)
    _manifest_path(owner_id, md5).unlink(missing_ok=True)
    return {"deleted": md5}


# ---------------------------------------------------------------------------
# Project export / import (Phase 4)
#
# There is no account, so there is no password-reset-style recovery — this
# is the mitigation: a device can save its own project state to a file it
# controls, and load it back in (on the same device after data loss, or on
# a different one entirely). Deliberately narrow in scope:
#
#   - No audio, ever, of any kind — not stems (see Phase 3's stem cache;
#     it's already local-only and evictable, no reason for an export to
#     duplicate that) and not the source track either. Only the source
#     FILENAME is recorded, so re-importing that same file the ordinary way
#     (POST /api/import) is the documented way to reconnect the audio —
#     identical bytes hash to the same md5 they always did, which is what
#     lets a re-import pick this project's grid back up untouched.
#   - No owner_id, ever — see identity.py's BEARER CREDENTIAL note. It is
#     never read by import_project below, and it is not present in what
#     GET .../export (the frontend-side JSON builder, since this data is
#     already all available client-side with no new endpoint needed for
#     the export direction itself) writes into the file.
#   - Keyed by md5, not track_id: import has no source file to hash yet,
#     so it cannot resolve a track_id the way every other manual-grid route
#     does (see _track_file). Reads and writes below go straight to the
#     same md5-keyed paths those routes eventually resolve to, just
#     without needing a live file to get there.
# ---------------------------------------------------------------------------


@app.get("/api/library/{md5}/manual-grid")
def get_manual_grid_by_md5(md5, owner_id: str = Depends(identity.get_owner_id)):
    """The tapped grid for one song, read directly by md5 — export's
    counterpart to GET .../tracks/{track_id}/manual-grid, which needs a
    live track_id (and therefore a source file still on disk) to resolve.
    Export needs neither: works the same for a ready row, a needs_tap row
    (returns manual_grid: null, same as the track_id route would), or a
    stems_evicted row with no source file at all — the grid was always
    keyed by md5 on disk regardless of which of those states the row is in.
    """
    md5 = _validate_md5(md5)
    path = GRIDS_DIR / owner_id / f"{md5}.json"
    if not path.exists():
        return {"manual_grid": None}
    record = json.loads(path.read_text())
    record.setdefault("active", True)
    return {"manual_grid": record}


class ProjectTrack(BaseModel):
    md5: str
    filename: str | None = None
    media_kind: Literal["audio", "video"] = "audio"
    duration: float | None = None


class ProjectImport(BaseModel):
    # Checked against ANALYSIS_SCHEMA_VERSION — the same version number
    # that already governs the grid shape (downbeats/eight_counts as
    # indices into beats), not a separate export-format version to keep in
    # sync by hand. An export file IS a schema v4 grid plus enough track
    # metadata to re-anchor it; there is nothing else versioned about it.
    schema_version: int
    track: ProjectTrack
    manual_grid: ManualGrid | None = None


@app.post("/api/library/import")
def import_project(payload: ProjectImport, owner_id: str = Depends(identity.get_owner_id)):
    """Restore a project from an exported file: recreates the library
    manifest and, if present, the tapped grid — both keyed by
    payload.track.md5, both independent of whether that track's audio has
    been re-imported yet (see the module note above).

    Whole-set write, same as PUT .../manual-grid: an imported grid always
    lands active and REPLACES whatever grid (if any) already exists for
    this md5 on this device. The frontend confirms before calling this for
    exactly that reason — there is no merge, and there is no undo beyond
    whatever the device's own IndexedDB mirror (Phase 3) still remembers
    from before the import.
    """
    if payload.schema_version != ANALYSIS_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported project schema_version {payload.schema_version} "
                f"(this server understands {ANALYSIS_SCHEMA_VERSION})"
            ),
        )
    md5 = _validate_md5(payload.track.md5)
    filename = Path(payload.track.filename).name if payload.track.filename else None

    _write_manifest_fields(owner_id, md5, filename, payload.track.media_kind, payload.track.duration)

    if payload.manual_grid is not None:
        path = GRIDS_DIR / owner_id / f"{md5}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = payload.manual_grid.model_dump()
        record["active"] = True
        path.write_text(json.dumps(record))

    return {"md5": md5, "manual_grid_imported": payload.manual_grid is not None}


def _check_storage_quota(owner_id, incoming_bytes):
    """Reject an import that would push this device over its cap.

    Only called right before bytes actually land on disk (see import_song) —
    never on the "identical file already here" early-return path, which
    doesn't grow usage at all.
    """
    owner_tracks_dir = TRACKS_DIR / owner_id
    existing = list(owner_tracks_dir.iterdir()) if owner_tracks_dir.exists() else []
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


async def _read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Reads an UploadFile in chunks, aborting the moment the running total
    would exceed max_bytes — never trusts Content-Length alone (it's a
    client-supplied claim, and this app has no account behind an upload to
    hold accountable if it lied), and never buffers more than one chunk
    past the limit before rejecting.
    """
    chunk_size = 1024 * 1024
    chunks = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {max_bytes} byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/import")
@limiter.limit(lambda: CHOREO_IMPORT_RATE_LIMIT)
async def import_song(
    request: Request, file: UploadFile = File(...), owner_id: str = Depends(identity.get_owner_id)
):
    """Take an uploaded file into this device's tracks/ and register it in
    the library.

    Re-importing a file the app already has is the documented recovery path for
    an evicted entry, so it must be safe and it must be idempotent: identical
    bytes resolve to the same md5, which means the same grid, the same cache,
    and the same manifest. Nothing on disk is overwritten by that case.
    """
    # Path(...).name strips any directory component a client may have sent
    # — browsers send a bare name, but this endpoint must not trust that.
    # It only strips '/', though: backslash is a legal filename character
    # on this server's own (POSIX) filesystem, not a separator, so it
    # would survive .name untouched — checked explicitly below instead of
    # relying on which OS happens to be running (this same module also
    # runs in local Windows dev; see the win32 DLL-path block above).
    name = Path(file.filename or "").name
    if not name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if name.startswith("."):
        raise HTTPException(status_code=400, detail="Filename must not start with a dot")
    if len(name) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"Filename is too long (max {MAX_FILENAME_LENGTH} characters)"
        )

    suffix = Path(name).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or name}'",
        )

    data = await _read_upload_within_limit(file, CHOREO_MAX_UPLOAD_BYTES)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    owner_tracks_dir = TRACKS_DIR / owner_id
    owner_tracks_dir.mkdir(parents=True, exist_ok=True)
    target = owner_tracks_dir / name
    if target.exists():
        import hashlib

        if hashlib.md5(data).hexdigest() == _file_hash(target):
            # Same song, already here. Say so instead of writing a duplicate.
            md5 = _file_hash(target)
            _write_manifest(owner_id, target, md5)
            return {"id": target.stem, "filename": target.name, "md5": md5, "created": False}
        # Same name, different audio: keep both rather than clobbering one.
        stem, n = Path(name).stem, 2
        while target.exists():
            target = owner_tracks_dir / f"{stem} ({n}){suffix}"
            n += 1

    _check_storage_quota(owner_id, len(data))

    target.write_bytes(data)
    md5 = _file_hash(target)
    _write_manifest(owner_id, target, md5)
    return {"id": target.stem, "filename": target.name, "md5": md5, "created": True}


# ---------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------


class DeviceRequest(BaseModel):
    # The frontend's own IndexedDB-mirrored id, sent back in when the cookie
    # has been evicted (Safari ITP) but the mirror survived — see
    # identity.issue_device for why that re-cookies the SAME id instead of
    # minting a new one. Omitted entirely on a real first visit.
    device_id: str | None = None


@app.post("/api/device")
def register_device(
    payload: DeviceRequest,
    response: Response,
    device_cookie: str | None = Cookie(default=None, alias=identity.DEVICE_COOKIE_NAME),
):
    """Bootstrap, recover, or reaffirm this browser's device id.

    Called once by the frontend on startup, before any other /api/* call —
    every time, not just on a real first visit, which is what makes this
    the one place reconciliation between the cookie and the IndexedDB
    mirror can actually happen (see identity.issue_device: cookie wins
    when both are present and differ, since only the server can ever read
    the HttpOnly cookie's real value). Three outcomes: an already-valid
    `device_cookie` is reaffirmed as-is; failing that, a `device_id` in
    the body is adopted; failing that, a fresh id is minted. This is the
    only /api/* route that does not depend on identity.get_owner_id — it
    is what hands out the id that dependency later requires.
    """
    device_id = identity.issue_device(response, COOKIE_SECURE, payload.device_id, device_cookie)
    return {"device_id": device_id}


# ---------------------------------------------------------------------------
# NOTE FOR PHASE 4 (export/import), not yet implemented — leaving this here
# rather than only in identity.py's docstring, since this is where a
# GET /api/export-style route will actually get added.
#
# owner_id is a bearer credential (see identity.py's BEARER CREDENTIAL note):
# presenting it is sufficient to claim a device's data, with no other check.
# The export file MUST NOT include owner_id anywhere in its payload — a
# leaked or casually-shared export (email attachment, cloud drive, a
# support ticket) would otherwise hand the recipient the same access the
# original device has, permanently, with no way to revoke it short of the
# device abandoning that id entirely. Everything else in the exported
# project state is fine to include verbatim; this one field is not.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
