"""The analysis pipeline: Demucs separation, beat detection, onset detection.

Factored out of server.py so the API process (which only ever takes the
cache-hit fast path or enqueues a job) and the worker process (which
actually runs the slow pipeline — see jobs.py) share one implementation
instead of two copies that could drift apart.

Every function here takes fully-resolved paths, never a bare track id or a
module-level TRACKS_DIR/CACHE_DIR constant: the worker that calls analyze()
is a separate Python process from the API (docker-compose's `worker`
service), started independently, and has no reason to share the API
process's path constants or its notion of the current device.
"""

from pathlib import Path

import librosa
import numpy as np

import identity

# Bump when the analysis JSON shape changes; cached analysis.json files with
# a different version are discarded and recomputed (stems stay cached).
# Imported by server.py as the single source of truth, rather than each
# process defining its own copy that could drift out of sync.
ANALYSIS_SCHEMA_VERSION = 4


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


def relocate_stems(audio_file_hash, stems, md5, cache_root):
    """Move source_separation's output into this track's own md5-keyed dir.

    source_separation.separate() keys its cache purely on a hash of the audio
    file it was actually handed. For an audio track that file IS the source,
    so the hash already equals `md5` and there is nothing to do. For a video
    track it is the extracted audio — different bytes, a different hash — so
    its stems land in a sibling directory that nothing else in the app knows
    to look in. Relocating them here keeps every downstream lookup (get_stem,
    stems_present, this analysis payload's own stems listing) keyed on the
    one identity the rest of the app uses.
    """
    if audio_file_hash == md5:
        return
    produced_dir = Path(stems["drums"]).parent
    target_dir = Path(cache_root) / md5
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in produced_dir.iterdir():
        f.replace(target_dir / f.name)
    produced_dir.rmdir()


def analyze(owner_id, md5, audio_file, cache_root):
    """Run the full analysis pipeline for one track. Slow on a cache miss
    (Demucs + beat_this on CPU) — this is the job body a worker executes
    (see jobs.run_analysis_job); the caller persists the result to
    analysis.json so this runs at most once per (device, file hash).

    `audio_file` is a librosa-loadable path, already resolved by the caller
    (server.py's _resolve_analysis_audio). `md5` is the TRACK's identity
    (see relocate_stems) — not necessarily the hash of `audio_file` itself,
    which for a video track is an extracted, regenerable copy of its audio.
    `cache_root` is this device's own cache/stems/<owner_id>/ directory,
    passed explicitly since the worker process has no shared global for it.

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
    # so a process that imports this module (e.g. the API, on a cache hit)
    # doesn't pay that cost unless it actually runs an analysis.
    import beat_detection
    import onset_detection
    import source_separation

    audio_file = Path(audio_file)
    cache_root = Path(cache_root)

    # progress_key hashes owner_id through identity.owner_key rather than
    # embedding it verbatim — this key ends up in progress_store (Redis)
    # and, via _ProgressTrackingTqdm, in whatever logging wraps this call;
    # owner_id itself must never appear in either (see identity.owner_key's
    # own docstring). audio_file is the extracted wav for a video track,
    # not the source itself, so it hashes to something other than md5 (see
    # relocate_stems) — tracking progress under this explicit key is what
    # lets the API's /analysis/progress poll, which only ever knows the
    # track's (owner_id, md5), find this run.
    progress_key = f"{identity.owner_key(owner_id)}:{md5}"
    stems = source_separation.separate(str(audio_file), progress_key=progress_key, cache_root=cache_root)
    source_separation.get_instrumental(stems)

    beats, downbeats = beat_detection.detect_beats(str(audio_file))
    groups = beat_detection.eight_count_grouping(beats, downbeats)

    drum_onsets = onset_detection.detect_onsets(source_separation.get_drums(stems), "drums")
    bass_onsets = onset_detection.detect_onsets(source_separation.get_bass(stems), "bass")

    audio_hash = source_separation._hash_file(str(audio_file))
    relocate_stems(audio_hash, stems, md5, cache_root)

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
