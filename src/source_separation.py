import contextlib
import os
import threading
import demucs.apply
import demucs.separate
import tqdm
from pydub import AudioSegment
import hashlib
from pathlib import Path
import shutil

EXPECTED_STEMS = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")

# Live Demucs separation progress, keyed by whatever identity the caller of
# separate() asked to track it under (see progress_key on separate()).
# Populated only while a separate() call is actually running Demucs — a
# cache hit never touches this dict at all. Read by get_progress(), which the
# API layer polls so the frontend's loading ring can track Demucs' own
# per-chunk progress instead of faking one.
_progress = {}
_progress_lock = threading.Lock()


def get_progress(key):
    """The live Demucs separation progress for `key`, or None.

    None means separation under this key either hasn't started, already
    finished, or was skipped entirely because the stems were already cached
    — in every one of those cases there is nothing to report. While active,
    returns {"done": int, "total": int} counting the same per-chunk futures
    Demucs' own tqdm bar counts (see _ProgressTrackingTqdm below), not an
    estimate derived some other way.
    """
    with _progress_lock:
        state = _progress.get(key)
        return dict(state) if state is not None else None


_tracking_key = threading.local()


class _ProgressTrackingTqdm(tqdm.tqdm):
    """tqdm.tqdm, plus publishing every update() through get_progress().

    demucs.apply.apply_model() draws its own CLI progress bar by wrapping its
    list of per-chunk futures in exactly one call: tqdm.tqdm(futures, ...)
    (demucs/apply.py), and tqdm's own __iter__ funnels every step of that
    through self.update(). Subclassing (rather than reimplementing) means
    that hook is the ONLY behavior change — Demucs' console bar still draws
    exactly as before, and any other tqdm.tqdm call demucs.separate.main()
    happens to make on this same thread during the swap (e.g. a first-run
    model-weights download, which isn't iterable-shaped like the chunk list
    is) still works, since it only ever reaches update()/__init__(), both of
    which fall through to the real implementation.

    Swapped in for demucs.apply's tqdm.tqdm only for the duration of one
    separate() call — see _tracking_demucs_progress — so /analysis/progress
    reports the identical chunk-completion signal Demucs' own bar reads from,
    not a separately guessed animation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_key = getattr(_tracking_key, "key", None)

    def update(self, n=1):
        result = super().update(n)
        if self._progress_key is not None and self.total:
            with _progress_lock:
                _progress[self._progress_key] = {"done": self.n, "total": self.total}
        return result


@contextlib.contextmanager
def _tracking_demucs_progress(key):
    """Report `key`'s Demucs progress via get_progress() for one separate() call.

    Patches the real tqdm.tqdm that demucs.apply imported (demucs.apply.main
    hardcodes progress=True, so this constructor always gets called on a
    separation run) and restores it afterward, whether the run succeeded or
    raised. The entry under `key` is popped in the same finally so a finished
    or failed run never leaves a stale percentage for a later poll to find.
    """
    _tracking_key.key = key
    original_tqdm = demucs.apply.tqdm.tqdm
    demucs.apply.tqdm.tqdm = _ProgressTrackingTqdm
    try:
        yield
    finally:
        demucs.apply.tqdm.tqdm = original_tqdm
        _tracking_key.key = None
        with _progress_lock:
            _progress.pop(key, None)

# Root of the per-file stem cache, relative to the process CWD (the repo root
# in every real entry point). A single module-level binding — not an inline
# literal — so the test suite can point it at a temp directory and never
# touch the real cache the API server serves stems from.
CACHE_ROOT = Path("cache/stems")

def _hash_file(path):
    """Compute the MD5 hash of a file's contents.

    Used as a stable cache key so the same audio file maps to the same
    cache directory across runs, regardless of its filename.

    Args:
        path: Path to the file to hash.

    Returns:
        The hex-encoded MD5 digest of the file's contents.
    """
    hash_object = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_object.update(chunk)

    return hash_object.hexdigest()

def _is_cached(file_hash):
    """Check whether a complete set of cached stems exists for a file hash.

    A cache entry is only considered valid if every stem in EXPECTED_STEMS
    is present, so a partial/interrupted run does not falsely signal a hit.

    Args:
        file_hash: The MD5 hash returned by _hash_file().

    Returns:
        True if CACHE_ROOT/<file_hash>/ contains all expected stems, else False.
    """
    cache_dir = CACHE_ROOT / file_hash
    if not cache_dir.exists():
        return False

    for stem in EXPECTED_STEMS:
        if not (cache_dir / stem).exists():
            return False

    return True

def separate(audio_path, progress_key=None):
    """Run Demucs 4-stem separation on an audio file, with caching.

    The audio file is hashed and stems are written to cache/stems/<hash>/.
    If a complete cached set already exists for this file, Demucs is skipped
    and the cached paths are returned directly.

    Args:
        audio_path: Path to the audio file.
        progress_key: Identity to publish live progress under while Demucs
            runs (read back via get_progress()). Defaults to audio_path's own
            content hash, but a caller whose own identity for this track is a
            DIFFERENT hash — e.g. the API server tracking a video track by
            the original file's md5, when audio_path here is its extracted,
            separately-hashed audio — should pass that identity explicitly so
            a progress poll keyed on it actually finds this run. Ignored on a
            cache hit: there is no Demucs run to report progress for.

    Returns:
        A dictionary mapping stem names ("vocals", "drums", "bass", "other")
        to their .wav file paths inside the per-file cache directory.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
        FileNotFoundError: If audio_path does not exist, or if Demucs
            failed to produce one of the expected stems.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    file_hash = _hash_file(audio_path)
    cache_dir = CACHE_ROOT / file_hash

    if _is_cached(file_hash):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stems[stem] = str(cache_dir / f"{stem}.wav")

        return stems

    cache_dir.mkdir(parents=True, exist_ok=True)

    with _tracking_demucs_progress(progress_key if progress_key is not None else file_hash):
        demucs.separate.main([
            "-n", "htdemucs",
            "-o", str(cache_dir),
            audio_path
        ])

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(cache_dir, "htdemucs", track_name)

    stems = {}
    for stem in ["vocals", "drums", "bass", "other"]:
        path = os.path.join(stem_dir, f"{stem}.wav")
        if os.path.exists(path):
            new_path = cache_dir / f"{stem}.wav"
            shutil.move(path, new_path)
            stems[stem] = str(new_path)
        else:
            raise FileNotFoundError(f"Expected stem '{stem}' not found at {path}")

    (cache_dir / "htdemucs" / track_name).rmdir()
    (cache_dir / "htdemucs").rmdir()

    return stems

def get_stem(stems, name):
    """Get the path to a specific stem.

    Args:
        stems: Dictionary returned by separate().
        name: Stem name (vocals, drums, bass, other).

    Returns:
        The file path to the requested stem.

    Raises:
        KeyError: If the stem name is not found.
    """
    if name not in stems:
        raise KeyError(f"Stem '{name}' not found. Available: {list(stems.keys())}")
    return stems[name]

def get_vocals(stems):
    return get_stem(stems, "vocals")

def get_drums(stems):
    return get_stem(stems, "drums")

def get_bass(stems):
    return get_stem(stems, "bass")

def get_other(stems):
    return get_stem(stems, "other")

def get_instrumental(stems):
    """Combine drums, bass, and other into an instrumental track.

    The output is written to instrumental.wav alongside the cached stems.
    If that file already exists, the existing path is returned without
    re-running the overlay.

    Args:
        stems: Dictionary returned by separate().

    Returns:
        The path to instrumental.wav inside the per-file cache directory.
    """
    cache_dir = Path(stems["drums"]).parent
    output_path = cache_dir / "instrumental.wav"

    if output_path.exists():
        return str(output_path)

    drums = AudioSegment.from_wav(get_drums(stems))
    bass = AudioSegment.from_wav(get_bass(stems))
    other = AudioSegment.from_wav(get_other(stems))

    instrumental = drums.overlay(bass).overlay(other)
    instrumental.export(str(output_path), format="wav")

    return str(output_path)
