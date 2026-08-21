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
import subprocess
import sys
from contextlib import asynccontextmanager
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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator

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

# The four Demucs outputs. "instrumental" is derived from these and "original"
# is served from tracks/, so these four are what "stems are present" means.
REQUIRED_STEMS = ("vocals", "drums", "bass", "other")

# Bump when the analysis JSON shape changes; cached analysis.json files with
# a different version are discarded and recomputed (stems stay cached).
ANALYSIS_SCHEMA_VERSION = 4

@asynccontextmanager
async def lifespan(app):
    """Boot work. Resolves the callees at call time, so each one can stay down
    with the code it belongs to rather than being hoisted up here.
    """
    _migrate_manual_grids()
    _backfill_library()
    yield


app = FastAPI(title="Choreo API", lifespan=lifespan)

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
    """Resolve a track id (filename stem) to its source media file in tracks/.

    The source is audio or video, whichever extension is actually on disk.
    Callers that need a librosa-loadable audio path (analysis, the "original"
    stem) go through _resolve_analysis_audio, which is the one place that
    cares which kind this is.
    """
    if "/" in track_id or "\\" in track_id or track_id in (".", ".."):
        raise HTTPException(status_code=404, detail="Track not found")
    for ext in MEDIA_EXTENSIONS:
        candidate = TRACKS_DIR / (track_id + ext)
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="Track not found")


def _extracted_audio_path(md5):
    return CACHE_DIR / md5 / "source_audio.wav"


def _resolve_analysis_audio(source_file, md5):
    """A librosa-loadable audio path for this track's source file.

    Audio sources are used directly. A video container isn't guaranteed
    decodable by librosa/soundfile, so its audio track is extracted once via
    moviepy and cached next to the stems — regenerable derived data, keyed by
    the same md5 the stems use, extracted at most once per hash.
    """
    if source_file.suffix.lower() in AUDIO_EXTENSIONS:
        return source_file

    wav_path = _extracted_audio_path(md5)
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


def _relocate_stems(audio_file, stems, md5):
    """Move source_separation's output into this track's own md5-keyed dir.

    source_separation.separate() keys its cache purely on a hash of the audio
    file it was actually handed. For an audio track that file IS the source,
    so the hash already equals `md5` and there is nothing to do. For a video
    track it is the extracted audio (_resolve_analysis_audio) — different
    bytes, a different hash — so its stems land in a sibling directory that
    nothing else in the app knows to look in. Relocating them here keeps
    every downstream lookup (get_stem, stems_present, this analysis payload's
    own stems listing) keyed on the one identity the rest of the app uses.
    """
    if _file_hash(audio_file) == md5:
        return
    produced_dir = Path(stems["drums"]).parent
    target_dir = CACHE_DIR / md5
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in produced_dir.iterdir():
        f.replace(target_dir / f.name)
    produced_dir.rmdir()


def _analyze(audio_file, md5):
    """Run the full analysis pipeline for one track.

    Slow on a cache miss (Demucs + beat_this on CPU); the caller persists
    the result to analysis.json so this runs at most once per file hash.
    `md5` is the TRACK's identity (see _relocate_stems) — not necessarily the
    hash of `audio_file` itself, which for a video track is an extracted,
    regenerable copy of its audio.

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

    # progress_key=md5: audio_file is the extracted wav for a video track,
    # not the source itself, so it hashes to something other than md5 (see
    # _relocate_stems). Tracking progress under md5 explicitly is what lets
    # get_analysis_progress below — which only ever knows the track's md5 —
    # find this run.
    stems = source_separation.separate(str(audio_file), progress_key=md5)
    source_separation.get_instrumental(stems)

    beats, downbeats = beat_detection.detect_beats(str(audio_file))
    groups = beat_detection.eight_count_grouping(beats, downbeats)

    drum_onsets = onset_detection.detect_onsets(source_separation.get_drums(stems), "drums")
    bass_onsets = onset_detection.detect_onsets(source_separation.get_bass(stems), "bass")

    _relocate_stems(audio_file, stems, md5)

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
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return {"tracks": tracks}


@app.get("/api/tracks/{track_id}/analysis")
def get_analysis(track_id):
    source_file = _track_file(track_id)
    md5 = _file_hash(source_file)
    cache_dir = CACHE_DIR / md5
    cache_path = cache_dir / "analysis.json"

    analysis = None
    if cache_path.exists():
        analysis = json.loads(cache_path.read_text())
        if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            analysis = None

    if analysis is None:
        audio_file = _resolve_analysis_audio(source_file, md5)
        analysis = _analyze(audio_file, md5)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(analysis))

    # Analysis is the moment the app first knows a song's duration, so it is
    # also the moment its library manifest can be written in full.
    _write_manifest(source_file, md5, analysis.get("duration"))

    is_video = source_file.suffix.lower() in VIDEO_EXTENSIONS
    # URLs are built from the current track id rather than cached, so a
    # renamed file (same content hash) still gets working links.
    return {
        "id": track_id,
        "filename": source_file.name,
        **analysis,
        "audio_url": f"/api/tracks/{track_id}/audio",
        "video_url": f"/api/tracks/{track_id}/video" if is_video else None,
        "stems": [
            {"name": name, "url": f"/api/tracks/{track_id}/stems/{name}"}
            for name in STEM_NAMES
            if (cache_dir / (name + ".wav")).exists()
        ],
    }


@app.get("/api/tracks/{track_id}/analysis/progress")
def get_analysis_progress(track_id):
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

    source_file = _track_file(track_id)
    md5 = _file_hash(source_file)
    state = source_separation.get_progress(md5)
    if state is None:
        return {"active": False}
    return {"active": True, "done": state["done"], "total": state["total"]}


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id):
    """The decodable audio the StemEngine's "original" buffer plays.

    For an audio source this is the file itself. For a video source it is the
    extracted track from _resolve_analysis_audio — the video container is
    served separately, from /video, for the <video> element.
    """
    source_file = _track_file(track_id)
    md5 = _file_hash(source_file)
    return FileResponse(_resolve_analysis_audio(source_file, md5))


@app.get("/api/tracks/{track_id}/video")
def get_video(track_id):
    source_file = _track_file(track_id)
    if source_file.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Track has no video")
    return FileResponse(source_file)


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


def _manual_grid_path(audio_file):
    """Where a track's tapped grid lives: data/grids/<md5>.json.

    Keyed by the audio's MD5, like the cache — but stored OUTSIDE the cache
    tree, because this is the one artifact in the app that cannot be
    regenerated. See the GRIDS_DIR comment above.
    """
    return GRIDS_DIR / f"{_file_hash(audio_file)}.json"


def _migrate_manual_grids():
    """Lift tapped grids out of the old in-cache location. Runs on every boot.

    Grids used to be written to cache/stems/<md5>/manual_grid.json, where an
    `rm -rf cache/` would have destroyed them. Same MD5 key, new home, so the
    move is a pure relocation — no reinterpretation of the data.

    Idempotent: once the cache tree holds no manual_grid.json, the glob is empty
    and this is a no-op. If a grid somehow exists in BOTH places, data/ wins —
    it is the only one the app writes to now, so the cache copy is a stale
    leftover — and the leftover is dropped.

    Returns the number of files relocated.
    """
    if not CACHE_DIR.exists():
        return 0
    moved = 0
    for legacy in sorted(CACHE_DIR.glob("*/manual_grid.json")):
        target = GRIDS_DIR / f"{legacy.parent.name}.json"
        GRIDS_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            legacy.unlink()
            continue
        # Same filesystem (both under REPO_ROOT), so this is an atomic rename:
        # the grid is never momentarily absent from both trees.
        legacy.replace(target)
        moved += 1
    if moved:
        print(f"[grids] migrated {moved} tapped grid(s) out of the cache -> {GRIDS_DIR}")
    return moved


@app.get("/api/tracks/{track_id}/manual-grid")
def get_manual_grid(track_id):
    """The track's tapped grid, or null if it has never been tapped.

    Null rather than 404: having no manual grid is the normal state (auto is the
    default), not an error.
    """
    path = _manual_grid_path(_track_file(track_id))
    if not path.exists():
        return {"manual_grid": None}
    record = json.loads(path.read_text())
    # Grids written before `active` existed have no such key; their mere
    # presence on disk used to mean "this is the live grid", so that is the
    # default we read them back as.
    record.setdefault("active", True)
    return {"manual_grid": record}


@app.put("/api/tracks/{track_id}/manual-grid")
def put_manual_grid(track_id, payload: ManualGrid):
    """Replace the track's tapped grid.

    Whole-set write: a grid is one object, and the frontend always has the
    complete fit in hand, so there is nothing to merge. A freshly accepted tap
    always comes back active — see ManualGridStore.save.
    """
    path = _manual_grid_path(_track_file(track_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = payload.model_dump()
    record["active"] = True
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


class ManualGridActive(BaseModel):
    active: bool


@app.patch("/api/tracks/{track_id}/manual-grid/active")
def set_manual_grid_active(track_id, payload: ManualGridActive):
    """Switch between the tapped grid and auto detection without touching the
    tap data — the reversible counterpart to DELETE below.

    "Use auto detection" sets active=False; switching back sets it True. Either
    way beats/downbeats/eight_counts on disk never move, so the most recently
    accepted tap is always one flip away, no retapping needed.
    """
    path = _manual_grid_path(_track_file(track_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="no tapped grid for this track")
    record = json.loads(path.read_text())
    record["active"] = payload.active
    path.write_text(json.dumps(record))
    return {"manual_grid": record}


@app.delete("/api/tracks/{track_id}/manual-grid")
def delete_manual_grid(track_id):
    """Permanently drop the tapped grid. Idempotent.

    The ONLY route in the app that destroys user data, and it is unrecoverable:
    nothing can re-derive a tapped grid from the audio. Deliberately NOT the
    "use auto detection" path anymore — that is set_manual_grid_active, which
    keeps the taps on disk. This exists for actually discarding them.
    """
    _manual_grid_path(_track_file(track_id)).unlink(missing_ok=True)
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


def _manifest_path(md5):
    return LIBRARY_DIR / f"{md5}.json"


def _read_manifest(md5):
    path = _manifest_path(md5)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_manifest(audio_file, md5, duration=None):
    """Create or refresh a song's manifest. Idempotent.

    Filename and duration are refreshed from the live file, so a rename is
    picked up. media_kind is NOT: an existing manifest's media_kind is
    preserved verbatim, so a song deliberately marked "video" stays video
    across every later re-analysis. That is the only field a human is expected
    to edit by hand, and silently reverting a hand edit on the next cache miss
    would make the edit useless.
    """
    existing = _read_manifest(md5) or {}
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

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    _manifest_path(md5).write_text(json.dumps(record))
    return record


def _tracks_by_md5():
    """md5 -> the file in tracks/ holding that source (audio or video).

    Hashes every track on each call; tracks/ holds a handful of files and
    _file_hash memoizes on (path, mtime, size), so repeat calls are cheap.
    Hashing rather than trusting the manifest's filename is what makes a
    renamed file keep its grid, its stems, and its library entry.
    """
    if not TRACKS_DIR.exists():
        return {}
    index = {}
    for f in sorted(TRACKS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
            index.setdefault(_file_hash(f), f)
    return index


def _stems_present(md5):
    cache_dir = CACHE_DIR / md5
    return all((cache_dir / f"{name}.wav").exists() for name in REQUIRED_STEMS)


def _library_entry(md5, record, tracks):
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
    grid_present = (GRIDS_DIR / f"{md5}.json").exists()
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
        "stems_present": _stems_present(md5),
        "original_present": source is not None,
    }


def _backfill_library():
    """Write manifests for songs that predate manifests. Runs on every boot.

    Two sources of truth about what already exists: a cache dir with an
    analysis.json, and a grid file. Their union is every md5 the app has ever
    known.

    An md5 with neither a name nor a grid is skipped — a stray cache dir whose
    source is gone and which was never tapped has nothing to show and nothing
    to lose, so listing it would be noise, not honesty. An md5 with a GRID is
    always registered even when nothing can name it, because that grid is
    unrecoverable user data and must not disappear from the library.

    Returns the number of manifests written.
    """
    tracks = _tracks_by_md5()
    known = set()
    if CACHE_DIR.exists():
        known |= {d.name for d in CACHE_DIR.iterdir() if (d / "analysis.json").exists()}
    if GRIDS_DIR.exists():
        known |= {f.stem for f in GRIDS_DIR.glob("*.json")}

    written = 0
    for md5 in sorted(known):
        if _manifest_path(md5).exists():
            continue
        source = tracks.get(md5)
        if source is not None:
            duration = None
            analysis_path = CACHE_DIR / md5 / "analysis.json"
            if analysis_path.exists():
                try:
                    duration = json.loads(analysis_path.read_text()).get("duration")
                except json.JSONDecodeError:
                    pass
            _write_manifest(source, md5, duration)
            written += 1
        elif (GRIDS_DIR / f"{md5}.json").exists():
            LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
            _manifest_path(md5).write_text(
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
        print(f"[library] backfilled {written} manifest(s) -> {LIBRARY_DIR}")
    return written


@app.get("/api/library")
def list_library():
    """Every song the app has ever known, with its current state.

    Entries are never filtered out by state. A song whose source file has gone
    missing still appears, still says whether its grid survived, and still
    tells the user which file to re-import to get it back.
    """
    _backfill_library()
    tracks = _tracks_by_md5()

    songs = []
    if LIBRARY_DIR.exists():
        for path in sorted(LIBRARY_DIR.glob("*.json")):
            record = _read_manifest(path.stem)
            if record is None:
                continue
            songs.append(_library_entry(path.stem, record, tracks))

    # Openable first, then by name, so the dead rows sink to the bottom
    # without being hidden.
    order = {"ready": 0, "needs_tap": 1, "stems_evicted": 2}
    songs.sort(key=lambda s: (order[s["state"]], (s["filename"] or "").lower()))
    return {"songs": songs}


@app.post("/api/import")
async def import_song(file: UploadFile = File(...)):
    """Take an uploaded file into tracks/ and register it in the library.

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

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    target = TRACKS_DIR / name
    if target.exists():
        import hashlib

        if hashlib.md5(data).hexdigest() == _file_hash(target):
            # Same song, already here. Say so instead of writing a duplicate.
            md5 = _file_hash(target)
            _write_manifest(target, md5)
            return {"id": target.stem, "filename": target.name, "md5": md5, "created": False}
        # Same name, different audio: keep both rather than clobbering one.
        stem, n = Path(name).stem, 2
        while target.exists():
            target = TRACKS_DIR / f"{stem} ({n}){suffix}"
            n += 1

    target.write_bytes(data)
    md5 = _file_hash(target)
    _write_manifest(target, md5)
    return {"id": target.stem, "filename": target.name, "md5": md5, "created": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
