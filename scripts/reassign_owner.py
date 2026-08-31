#!/usr/bin/env python
"""One-time migration: reassign this install's pre-existing single-tenant
data (everything sitting directly at the root of tracks/, cache/stems/,
data/grids/, and data/library/, from before this app went account-less and
device-scoped) onto a real device UUID.

Usage (from the repo root):
    venv/Scripts/python scripts/reassign_owner.py --owner-id <uuid4>

The UUID is yours to choose — mint one with `python -c "import uuid;
print(uuid.uuid4())"`, or use the id your browser already has (readable
from IndexedDB via devtools once you've opened the app once, or from the
device_id field the /api/device response returns). Whichever id you pass
becomes the one your browser needs to present — via the choreo_device
cookie or the X-Choreo-Device-Id header — to see this data again.

Idempotent — safe to re-run. Each tree is moved entry-by-entry into
<tree>/<owner_id>/; an entry already at its destination (a second run, or
a run resumed after a partial failure) is left alone rather than
clobbered. Only entries sitting directly at a tree's root are moved — an
entry that's already inside some other <owner_id>/ subdirectory (i.e. an
install that already went through a device split) is left exactly where
it is.

Does not touch, rename, or delete anything if --owner-id is malformed —
validated the same way the server itself validates a device id (see
identity.py), since this value becomes a raw path segment same as any
request's.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "api"))

import identity  # noqa: E402  (path setup above must run first)
import server  # noqa: E402  (importing this also loads .env; harmless for a one-shot script)


def _move_into(base_dir: Path, owner_id: str) -> int:
    """Move every entry sitting directly at base_dir's root into
    base_dir/<owner_id>/.

    Skips the destination directory itself (so a second run over an
    already-migrated tree is a no-op) and skips any individual entry that
    already exists at the destination (so resuming a partially-completed
    run never overwrites what a prior run already placed there).
    """
    if not base_dir.exists():
        return 0
    target = base_dir / owner_id
    moved = 0
    for entry in list(base_dir.iterdir()):
        if entry == target:
            continue
        dest = target / entry.name
        if dest.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        entry.replace(dest)
        moved += 1
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", required=True, help="Destination device UUIDv4")
    args = parser.parse_args()

    owner_id = args.owner_id
    if not identity._is_canonical_uuid4(owner_id):
        raise SystemExit(
            f"{owner_id!r} is not a canonical UUIDv4. Generate one with "
            "python -c 'import uuid; print(uuid.uuid4())'. Nothing was moved."
        )

    moved_tracks = _move_into(server.TRACKS_DIR, owner_id)
    moved_cache = _move_into(server.CACHE_DIR, owner_id)
    moved_grids = _move_into(server.GRIDS_DIR, owner_id)
    moved_library = _move_into(server.LIBRARY_DIR, owner_id)
    print(
        f"Moved {moved_tracks} track(s), {moved_cache} cache dir(s), "
        f"{moved_grids} grid(s), {moved_library} manifest(s) under owner {owner_id}."
    )

    # Lifts grids from the OLDER cache/stems/<md5>/manual_grid.json shape
    # (pre-dating data/grids/ itself) — a separate, older migration than the
    # one this script performs, but still possible for a long-lived install.
    grids_migrated = server._migrate_manual_grids(owner_id)
    if grids_migrated:
        print(f"Also lifted {grids_migrated} legacy in-cache grid(s).")

    backfilled = server._backfill_library(owner_id)
    if backfilled:
        print(f"Backfilled {backfilled} manifest(s) for songs that predate manifests.")

    print(
        "\nDone. Verify locally before touching a real server — start the app "
        f"and set the choreo_device cookie (or X-Choreo-Device-Id header) to "
        f"{owner_id!r}; confirm the library looks exactly like it did before "
        "this ran."
    )


if __name__ == "__main__":
    main()
