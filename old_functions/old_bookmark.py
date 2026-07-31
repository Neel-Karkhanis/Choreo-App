from pathlib import Path
import json
import uuid

VALID_SNAP_MODES = ("none", "beat", "4-count", "8-count")


def _bookmark_path(audio_path):
    """Build the sidecar bookmarks file path for an audio file.

    Bookmarks for ``song.mp3`` are stored next to the audio at
    ``song.bookmarks.json``.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A ``pathlib.Path`` pointing to the matching ``.bookmarks.json`` file.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")

    p = Path(audio_path)
    return p.with_name(p.stem + ".bookmarks.json")

def load_bookmarks(audio_path):
    """Load the bookmarks dictionary for an audio file.

    If no bookmarks file exists yet for this audio file, an empty dict
    is returned rather than raising.

    Args:
        audio_path: Path to the audio file whose bookmarks should be loaded.

    Returns:
        A dict mapping bookmark IDs to bookmark records. Empty if the
        sidecar file does not exist.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")

    path = _bookmark_path(audio_path)
    if not path.exists():
        return {}

    with open(path, "r") as f:
        bookmarks = json.load(f)

    return bookmarks

def save_bookmarks(bookmarks, audio_path):
    """Write a bookmarks dictionary to the sidecar file for an audio file.

    Overwrites the existing file if one is present.

    Args:
        bookmarks: Dict mapping bookmark IDs to bookmark records.
        audio_path: Path to the audio file the bookmarks belong to.

    Returns:
        None.

    Raises:
        TypeError: If bookmarks is not a dict, or audio_path is not a string.
        ValueError: If audio_path is empty.
    """
    if not isinstance(bookmarks, dict):
        raise TypeError("bookmarks must be a dict")
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")

    path = _bookmark_path(audio_path)

    with open(path, "w") as f:
        json.dump(bookmarks, f, indent=2)

    return None

def snap_timestamp(t, sliced_beats):
    """Snap a timestamp back to the nearest preceding anchor.

    Returns the largest value in ``sliced_beats`` that is less than or equal
    to ``t``. If ``t`` precedes the first anchor, ``t`` is returned unchanged.
    If ``t`` is past the last anchor, the last anchor is returned.

    Args:
        t: The timestamp in seconds to snap.
        sliced_beats: A non-empty sequence of anchor timestamps in seconds,
            sorted in ascending order.

    Returns:
        The snapped timestamp in seconds.

    Raises:
        TypeError: If t is not numeric.
        ValueError: If sliced_beats is empty.
    """
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise TypeError("t must be numeric")
    if len(sliced_beats) == 0:
        raise ValueError("sliced_beats must not be empty")

    if t < sliced_beats[0]:
        return t

    for i in range(len(sliced_beats)):
        if t < sliced_beats[i]:
            return sliced_beats[i-1]

    return sliced_beats[-1]

def get_count_anchors(beats, count):
    """Return every Nth beat to use as snap anchors for a count grouping.

    For ``count=4`` this yields the start of each 4-count; for ``count=8``
    the start of each 8-count.

    Args:
        beats: Sequence of beat timestamps in seconds.
        count: Positive integer step size between anchors.

    Returns:
        A sliced view of ``beats`` containing every ``count``-th entry,
        starting at index 0.

    Raises:
        TypeError: If count is not an int.
        ValueError: If count is not positive.
    """
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an int")
    if count < 1:
        raise ValueError("count must be a positive integer")

    return beats[::count]

def add_bookmark(t, beats, snap_mode, audio_path, bookmarks, label=""):
    """Create a new bookmark and persist the updated bookmarks dict.

    The supplied timestamp is snapped according to ``snap_mode``, a fresh
    UUID is assigned, and the bookmark is inserted into ``bookmarks`` and
    written to the sidecar file via :func:`save_bookmarks`.

    Args:
        t: Timestamp in seconds where the bookmark should land.
        beats: Sequence of beat timestamps in seconds, used by snap modes
            other than ``"none"``.
        snap_mode: One of ``"none"``, ``"beat"``, ``"4-count"``, ``"8-count"``.
        audio_path: Path to the audio file the bookmark belongs to.
        bookmarks: Dict mapping bookmark IDs to bookmark records. Mutated
            in place.
        label: Optional human-readable label for the bookmark.

    Returns:
        The string UUID assigned to the new bookmark.

    Raises:
        TypeError: If t is not numeric, audio_path is not a string,
            bookmarks is not a dict, or label is not a string.
        ValueError: If audio_path is empty or snap_mode is not one of
            the supported values.
    """
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise TypeError("t must be numeric")
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if not isinstance(bookmarks, dict):
        raise TypeError("bookmarks must be a dict")
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if snap_mode not in VALID_SNAP_MODES:
        raise ValueError(f"Unknown snap_mode: {snap_mode!r}")

    if snap_mode == "none":
        timestamp = t
    elif snap_mode == "beat":
        timestamp = snap_timestamp(t, beats)
    elif snap_mode == "4-count":
        timestamp = snap_timestamp(t, get_count_anchors(beats, 4))
    elif snap_mode == "8-count":
        timestamp = snap_timestamp(t, get_count_anchors(beats, 8))

    bookmark_id = str(uuid.uuid4())

    bookmarks[bookmark_id] = {
        "id": bookmark_id,
        "timestamp": timestamp,
        "label": label,
        "snap_mode": snap_mode,
    }

    save_bookmarks(bookmarks, audio_path)

    return bookmark_id

def delete_bookmark(bookmark_id, bookmarks, audio_path):
    """Remove a bookmark by ID and persist the change if it existed.

    Silently does nothing if ``bookmark_id`` is not present.

    Args:
        bookmark_id: The UUID string of the bookmark to remove.
        bookmarks: Dict mapping bookmark IDs to bookmark records. Mutated
            in place.
        audio_path: Path to the audio file the bookmarks belong to.

    Returns:
        None.

    Raises:
        TypeError: If bookmark_id is not a string, bookmarks is not a dict,
            or audio_path is not a string.
        ValueError: If audio_path is empty.
    """
    if not isinstance(bookmark_id, str):
        raise TypeError("bookmark_id must be a string")
    if not isinstance(bookmarks, dict):
        raise TypeError("bookmarks must be a dict")
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")

    if bookmarks.pop(bookmark_id, None) is not None:
        save_bookmarks(bookmarks, audio_path)

def update_bookmark_label(bookmark_id, new_label, bookmarks, audio_path):
    """Change the label on an existing bookmark and persist the change.

    If the new label matches the current label, the sidecar file is not
    rewritten.

    Args:
        bookmark_id: The UUID string of the bookmark to update.
        new_label: The replacement label.
        bookmarks: Dict mapping bookmark IDs to bookmark records. Mutated
            in place.
        audio_path: Path to the audio file the bookmarks belong to.

    Returns:
        None.

    Raises:
        TypeError: If bookmark_id is not a string, new_label is not a string,
            bookmarks is not a dict, or audio_path is not a string.
        ValueError: If audio_path is empty.
        KeyError: If bookmark_id is not present in bookmarks.
    """
    if not isinstance(bookmark_id, str):
        raise TypeError("bookmark_id must be a string")
    if not isinstance(new_label, str):
        raise TypeError("new_label must be a string")
    if not isinstance(bookmarks, dict):
        raise TypeError("bookmarks must be a dict")
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")

    if bookmark_id not in bookmarks:
        raise KeyError("Bookmark doesn't exist!")
    if new_label != bookmarks[bookmark_id]["label"]:
        bookmarks[bookmark_id]["label"] = new_label
        save_bookmarks(bookmarks, audio_path)

def list_bookmarks(bookmarks):
    """Return bookmarks as a list of (id, record) pairs sorted by timestamp.

    Args:
        bookmarks: Dict mapping bookmark IDs to bookmark records.

    Returns:
        A list of ``(bookmark_id, record)`` tuples ordered by the record's
        ``"timestamp"`` field, ascending.

    Raises:
        TypeError: If bookmarks is not a dict.
    """
    if not isinstance(bookmarks, dict):
        raise TypeError("bookmarks must be a dict")

    return sorted(bookmarks.items(), key=lambda x: x[1]["timestamp"])
