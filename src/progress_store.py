"""Redis-backed replacement for source_separation's old in-process progress
dict.

Analysis used to run inline in the API request's own thread, so a plain
module-global dict was enough to pass Demucs' live progress from that
thread back to a polling request in the same process. Now that analysis
runs in a separate worker process (see src/api/jobs.py), a poll from the
API process needs somewhere cross-process to read that progress from —
this is that somewhere. Same {"done", "total"} shape and the same
set-while-running / clear-when-done lifecycle as the dict it replaces, so
source_separation.py's _ProgressTrackingTqdm only needed a call-site swap,
not a redesign.

Progress reporting is a nice-to-have (it drives a loading ring, nothing
correctness-critical), so every call here swallows a Redis outage rather
than letting a flaky progress channel take down an actual analysis run —
worst case the ring stops moving until Redis comes back.
"""

import json
import os

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_TTL_SECONDS = 60 * 60  # a crashed run's stale entry expires on its own

_redis = redis.from_url(REDIS_URL)


def _key(key):
    return f"progress:{key}"


def set_progress(key, done, total):
    try:
        _redis.set(_key(key), json.dumps({"done": done, "total": total}), ex=_TTL_SECONDS)
    except redis.RedisError:
        pass


def get_progress(key):
    try:
        raw = _redis.get(_key(key))
    except redis.RedisError:
        return None
    return json.loads(raw) if raw is not None else None


def clear_progress(key):
    try:
        _redis.delete(_key(key))
    except redis.RedisError:
        pass
