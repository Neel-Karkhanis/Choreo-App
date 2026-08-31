"""The analysis job queue: one RQ queue, one dedup lock, one job function.

Demucs+beat_this analysis takes minutes on a cache miss — far too slow to
run inline in an HTTP request once more than one person can hit the API at
once. This module is the only place that talks to RQ; server.py enqueues
through it, and the worker process (docker-compose's `worker` service,
`rq worker analysis`) executes run_analysis_job by importing this same
module off the shared PYTHONPATH.

The job's real, durable output is the analysis.json (and stem .wav files)
it writes to the shared cache volume — NOT RQ's own result-serialization
mechanism, which this module deliberately never reads back. That keeps
server.py's polling loop (see get_analysis) simple and version-proof: it
waits for a file to exist, not for an RQ client API to agree on how a
Python object round-trips through Redis.
"""

import os

import redis
from rq import Queue

import analysis
import identity

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

redis_conn = redis.from_url(REDIS_URL)
queue = Queue("analysis", connection=redis_conn)

_LOCK_TTL_SECONDS = 60 * 60  # generous ceiling; a stuck claim self-clears here
_ACTIVE_STATUSES = ("queued", "started", "deferred", "scheduled")


def _job_id(owner_id, md5):
    # Hashed through identity.owner_key rather than embedding owner_id
    # verbatim: this becomes the RQ job id, and RQ's default worker logging
    # prints job ids on every start/finish — owner_id itself must never
    # appear there (see identity.owner_key's own docstring).
    #
    # RQ validates job ids against [A-Za-z0-9_-]+ (rq.job.validate_job_id) —
    # no colons, unlike the Redis keys elsewhere in this module and in
    # progress_store.py, where ':' is the normal, allowed separator.
    return f"{identity.owner_key(owner_id)}-{md5}"


def _lock_key(owner_id, md5):
    return f"lock:analyze:{identity.owner_key(owner_id)}:{md5}"


def run_analysis_job(owner_id, md5, audio_file, cache_root):
    """The job body a worker executes.

    Runs the real pipeline (analysis.analyze), writes its result to
    <cache_root>/<md5>/analysis.json itself — this is what server.py's poll
    loop is actually waiting to see appear on the shared volume — then
    releases the dedup lock so a later legitimate re-analysis (e.g. after a
    schema bump) isn't blocked by a stale claim.
    """
    import json
    from pathlib import Path

    try:
        result = analysis.analyze(owner_id, md5, audio_file, cache_root)
        cache_dir = Path(cache_root) / md5
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "analysis.json").write_text(json.dumps(result))
        return result
    finally:
        redis_conn.delete(_lock_key(owner_id, md5))


def enqueue_or_attach(owner_id, md5, audio_file, cache_root):
    """Start analysis for (owner_id, md5) unless it is already running.

    Atomically claims a per-track lock with SET NX before enqueuing, so two
    concurrent requests for the same uncached track only ever start one
    Demucs run — whichever request loses the race just attaches to (polls)
    the job the winner started, rather than double-enqueuing. Returns the
    Job either way.
    """
    job_id = _job_id(owner_id, md5)
    existing = queue.fetch_job(job_id)
    if existing is not None and existing.get_status(refresh=True) in _ACTIVE_STATUSES:
        return existing

    claimed = redis_conn.set(_lock_key(owner_id, md5), "1", nx=True, ex=_LOCK_TTL_SECONDS)
    if not claimed:
        # Someone else won the race between our fetch_job above and here —
        # their enqueue should already be visible; attach to it.
        existing = queue.fetch_job(job_id)
        if existing is not None:
            return existing
        # Lock held but no job object found (e.g. a leftover claim from a
        # process that crashed between claiming and enqueueing) — fall
        # through and enqueue fresh; the lock's TTL bounds how long a truly
        # stuck claim can block real work.

    return queue.enqueue(
        run_analysis_job,
        owner_id,
        md5,
        str(audio_file),
        str(cache_root),
        job_id=job_id,
        job_timeout="30m",
        result_ttl=300,
        failure_ttl=600,
    )


def get_job(owner_id, md5):
    return queue.fetch_job(_job_id(owner_id, md5))
