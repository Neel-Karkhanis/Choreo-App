from pathlib import Path
import json
import uuid

def _bookmark_path(audio_path):
    p = Path(audio_path
)
    return p.with_name(p.stem + ".bookmarks.json")

def load_bookmarks(audio_path):
    path = _bookmark_path(audio_path
)
    if not path.exists():
        return {}
    
    with open(path, "r") as f:
        bookmarks = json.load(f)

    return bookmarks

def save_bookmarks(bookmarks, audio_path):
    path = _bookmark_path(audio_path
)

    with open(path, "w") as f:
        json.dump(bookmarks, f, indent=2)

    return None

def snap_timestamp(t, sliced_beats):
    if t < sliced_beats[0]:
        return t

    for i in range(len(sliced_beats)):
        if t < sliced_beats[i]:
            return sliced_beats[i-1]
    
    return sliced_beats[-1]

def get_count_anchors(beats, count):
    return beats[::count]

def add_bookmark(t, beats, snap_mode, audio_path, bookmarks, label=""):
    if snap_mode == "none":
        timestamp = t
    elif snap_mode == "beat":
        timestamp = snap_timestamp(t, beats)
    elif snap_mode == "4-count":
        timestamp = snap_timestamp(t, get_count_anchors(beats, 4))
    elif snap_mode == "8-count":
        timestamp = snap_timestamp(t, get_count_anchors(beats, 8))
    else:
        raise ValueError(f"Unknown snap_mode: {snap_mode!r}")

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
    bookmarks.pop(bookmark_id, None)
    if bookmarks.pop(bookmark_id, None) is not None:
        save_bookmarks(bookmarks, audio_path)

def update_bookmark_label(bookmark_id, new_label, bookmarks, audio_path):
    if bookmark_id not in bookmarks:
        raise KeyError("Bookmark doesn't exist!")
    if new_label != bookmarks[bookmark_id]["label"]:
        bookmarks[bookmark_id]["label"] = new_label
        save_bookmarks(bookmarks, audio_path)

def list_bookmarks(bookmarks):
    return sorted(bookmarks.items(), key=lambda x: x[1]["timestamp"])


beats = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
bookmarks = {}

id1 = add_bookmark(11.5, beats, "8-count", "test.mp3", bookmarks, "testing label different")
id2 = add_bookmark(13.2, beats, "4-count", "test.mp3", bookmarks, "testing label same")
id3 = add_bookmark(14.2, beats, "none", "test.mp3", bookmarks, "testing label same")
id2 = add_bookmark(15, beats, "beat", "test.mp3", bookmarks, "testing label same")


print(list_bookmarks({}))
print(list_bookmarks(bookmarks))