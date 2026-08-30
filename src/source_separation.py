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
from uuid import uuid4

import progress_store

EXPECTED_STEMS = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")


def get_progress(key):
    """The live Demucs separation progress for `key`, or None.

    None means separation under this key either hasn't started, already
    finished, or was skipped entirely because the stems were already cached
    — in every one of those cases there is nothing to report. While active,
    returns {"done": int, "total": int} counting the same per-chunk futures
    Demucs' own tqdm bar counts (see _ProgressTrackingTqdm below), not an
    estimate derived some other way.
    """
    return progress_store.get_progress(key)


_tracking_key = threading.local()


class _ProgressTrackingTqdm(tqdm.tqdm):
    """tqdm.tqdm, plus publishing every update() through progress_store.

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
            progress_store.set_progress(self._progress_key, self.n, self.total)
        return result


@contextlib.contextmanager
def _tracking_demucs_progress(key):
    """Report `key`'s Demucs progress via progress_store for one separate() call.

    Patches the real tqdm.tqdm that demucs.apply imported (demucs.apply.main
    hardcodes progress=True, so this constructor always gets called on a
    separation run) and restores it afterward, whether the run succeeded or
    raised. The entry under `key` is cleared in the same finally so a
    finished or failed run never leaves a stale percentage for a later poll
    to find.
    """
    _tracking_key.key = key
    original_tqdm = demucs.apply.tqdm.tqdm
    demucs.apply.tqdm.tqdm = _ProgressTrackingTqdm
    try:
        yield
    finally:
        demucs.apply.tqdm.tqdm = original_tqdm
        _tracking_key.key = None
        progress_store.clear_progress(key)

# Root of the per-file stem cache, relative to the process CWD (the repo root
# in every real entry point), used whenever a caller doesn't pass its own
# cache_root to separate() (see below) — every real caller in the multi-user
# app does, since stems live under a per-user subtree; this default exists so
# the test suite can point it at a temp directory and never touch the real
# cache the API server serves stems from.
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

def _stems_complete(cache_dir):
    """Whether cache_dir holds a complete set of the expected stems.

    A cache entry is only considered valid if every stem in EXPECTED_STEMS
    is present, so a partial/interrupted run does not falsely signal a hit.
    """
    if not cache_dir.exists():
        return False

    for stem in EXPECTED_STEMS:
        if not (cache_dir / stem).exists():
            return False

    return True

def _is_cached(file_hash):
    """Check whether a complete set of cached stems exists for a file hash,
    under the module-level CACHE_ROOT.

    Args:
        file_hash: The MD5 hash returned by _hash_file().

    Returns:
        True if CACHE_ROOT/<file_hash>/ contains all expected stems, else False.
    """
    return _stems_complete(CACHE_ROOT / file_hash)

def separate(audio_path, progress_key=None, cache_root=None):
    """Run Demucs 4-stem separation on an audio file, with caching.

    The audio file is hashed and stems are written to
    <cache_root>/<hash>/. If a complete cached set already exists for this
    file, Demucs is skipped and the cached paths are returned directly.

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
        cache_root: Directory stems are cached under, as <cache_root>/<hash>/.
            Defaults to the module-level CACHE_ROOT. The multi-user API
            passes each user's own cache/stems/<user_id>/ directory here, so
            two users uploading identical audio each get their own Demucs
            run and their own stem cache rather than silently sharing one.

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

    root = Path(cache_root) if cache_root is not None else CACHE_ROOT
    file_hash = _hash_file(audio_path)
    cache_dir = root / file_hash

    if _stems_complete(cache_dir):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stems[stem] = str(cache_dir / f"{stem}.wav")

        return stems

    # Demucs writes into a uniquely-named STAGING directory, never cache_dir
    # itself, until the whole run has succeeded. Two requests racing on the
    # same uncached file (the API has no per-track lock at this layer — see
    # jobs.py for the higher-level dedup that makes this rare in practice,
    # not impossible) used to both write into, and shutil.move out of, the
    # same cache_dir; the loser's move would raise FileNotFoundError once the
    # winner had already relocated the files out from under it. Isolating
    # each run in its own staging directory turns that race into harmless
    # redundant computation at worst — see the _stems_complete recheck below.
    root.mkdir(parents=True, exist_ok=True)
    staging_dir = root / f"{file_hash}.tmp-{uuid4().hex}"
    staging_dir.mkdir(parents=True)

    try:
        with _tracking_demucs_progress(progress_key if progress_key is not None else file_hash):
            demucs.separate.main([
                "-n", "htdemucs",
                "-d", "cpu",
                "-o", str(staging_dir),
                audio_path
            ])

        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(staging_dir, "htdemucs", track_name)

        staged_stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            path = os.path.join(stem_dir, f"{stem}.wav")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected stem '{stem}' not found at {path}")
            new_path = staging_dir / f"{stem}.wav"
            shutil.move(path, new_path)
            staged_stems[stem] = new_path

        (staging_dir / "htdemucs" / track_name).rmdir()
        (staging_dir / "htdemucs").rmdir()
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    if _stems_complete(cache_dir):
        # Another caller finished first while we were computing — discard
        # our redundant work rather than clobbering the winner's result.
        shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        staging_dir.replace(cache_dir)  # atomic rename, same filesystem

    return {stem: str(cache_dir / f"{stem}.wav") for stem in ["vocals", "drums", "bass", "other"]}

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
