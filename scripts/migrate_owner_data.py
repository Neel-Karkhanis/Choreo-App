#!/usr/bin/env python
"""One-time migration: lift the pre-existing single-user tracks/cache/grids/
library trees into the multi-user layout, as the very first account.

Usage (from the repo root):
    venv/Scripts/python scripts/migrate_owner_data.py --email you@example.com --password "..."

Idempotent — safe to re-run. Each tree is moved entry-by-entry into
<tree>/<user_id>/; an entry already at its destination (a second run, or a
run resumed after a partial failure) is left alone rather than clobbered.
Run this locally FIRST and verify the result (log in as the migrated
account, confirm the library looks exactly like it did before) — only
then copy the now-restructured tracks/, cache/, data/ to a VPS's Docker
volumes. See the plan doc's Phase 11.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "api"))

import auth  # noqa: E402  (path setup above must run first)
import db  # noqa: E402
import server  # noqa: E402  (importing this also loads .env and builds the FastAPI app object; harmless for a one-shot script)


def _move_into(base_dir: Path, user_id: int) -> int:
    """Move every entry directly under base_dir into base_dir/<user_id>/.

    Skips the destination directory itself (so a second run over an
    already-migrated tree is a no-op) and skips any individual entry that
    already exists at the destination (so resuming a partially-completed
    run never overwrites what a prior run already placed there).
    """
    if not base_dir.exists():
        return 0
    target = base_dir / str(user_id)
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
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    if len(args.password) < 8:
        raise SystemExit(
            "Password must be at least 8 characters (matches /api/auth/signup's own rule)."
        )

    db.init_db(server.DB_PATH)

    existing = db.get_user_by_email(args.email)
    if existing is not None:
        print(f"Account {args.email!r} already exists (id={existing.id}); reusing it.")
        user_id = existing.id
    else:
        user = db.create_user(args.email, auth.hash_password(args.password))
        user_id = user.id
        print(f"Created account {args.email!r} (id={user_id}).")

    moved_tracks = _move_into(server.TRACKS_DIR, user_id)
    moved_cache = _move_into(server.CACHE_DIR, user_id)
    moved_grids = _move_into(server.GRIDS_DIR, user_id)
    moved_library = _move_into(server.LIBRARY_DIR, user_id)
    print(
        f"Moved {moved_tracks} track(s), {moved_cache} cache dir(s), "
        f"{moved_grids} grid(s), {moved_library} manifest(s) under user {user_id}."
    )

    # Lifts grids from the OLDER cache/stems/<md5>/manual_grid.json shape
    # (pre-dating data/grids/ itself) — a separate, older migration than the
    # one this script performs, but still possible for a long-lived install.
    grids_migrated = server._migrate_manual_grids(user_id)
    if grids_migrated:
        print(f"Also lifted {grids_migrated} legacy in-cache grid(s).")

    backfilled = server._backfill_library(user_id)
    if backfilled:
        print(f"Backfilled {backfilled} manifest(s) for songs that predate manifests.")

    print(
        "\nDone. Verify locally before touching a real server — start the app "
        f"(venv or docker compose) and log in as {args.email!r}; confirm the "
        "library looks exactly like it did before this ran."
    )


if __name__ == "__main__":
    main()
