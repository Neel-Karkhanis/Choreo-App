import { useCallback, useEffect, useRef, useState } from 'react'
import type { GridData } from './Timeline'

// Bookmarks: user-authored time markers, persisted to the backend's per-track
// JSON sidecar. The record shape mirrors src/bookmark.py's model exactly.
export interface Bookmark {
  id: string
  timestamp: number
  label: string
  snap_mode: SnapMode
}

export type SnapMode = 'none' | 'beat' | '4-count' | '8-count'

// Mirrors bookmark.VALID_SNAP_MODES; the backend rejects anything else (422).
export const SNAP_MODES: SnapMode[] = ['none', 'beat', '4-count', '8-count']

// The mode a fresh bookmark snaps with until the user picks another.
export const DEFAULT_SNAP_MODE: SnapMode = 'beat'

// Same-origin in dev: Vite proxies /api to the FastAPI backend, so app
// requests never cross an origin. (The backend also sends CORS headers for
// the Vite dev origin, so pointing this at http://127.0.0.1:8000 directly
// works too.)
export const API_BASE = '/api'

const bookmarksUrl = (trackId: string) =>
  `${API_BASE}/tracks/${encodeURIComponent(trackId)}/bookmarks`

// Snap anchors per mode, from data the browser already holds (schema v4) —
// snapping is local and instant, never a round-trip.
//
// 4-count is derived from the EIGHT-COUNT grid (every 4th beat from each
// eight_counts[k], i.e. counts 1 and 5 of each rendered block) rather than
// from downbeats[]. In real data beat_this's downbeats are not reliably every
// 4 beats — in "6 God" their index gaps run 1,2,3,4,5,6 — so they drift out of
// agreement with eight_counts, which IS a clean every-8 chunking. Anchoring on
// the eight-count grid keeps a 4-count bookmark on a count-1 or count-5 of a
// block the user can actually see on the timeline.
function snapAnchors(mode: SnapMode, grid: GridData): number[] {
  const { beats, eightCountIndices } = grid
  if (mode === 'beat') return beats
  if (mode === '8-count') return eightCountIndices.map((i) => beats[i])
  if (mode === '4-count') {
    const anchors: number[] = []
    eightCountIndices.forEach((start, k) => {
      const end = eightCountIndices[k + 1] ?? beats.length
      for (let i = start; i < end; i += 4) anchors.push(beats[i])
    })
    return anchors
  }
  return []
}

// Which way a time is allowed to move when it snaps.
//
// 'nearest' — closest anchor either side. What bookmarks use: a bookmark is a
//   point, so the honest answer is whichever grid line is actually closest.
// 'floor'   — largest anchor <= time. 'ceil' — smallest anchor >= time. These
//   exist for the LOOP, whose endpoints are a span, not a point: flooring A and
//   ceiling B makes the loop enclose whole musical units and guarantees it never
//   clips inside the range the user actually selected.
export type SnapDirection = 'nearest' | 'floor' | 'ceil'

// Anchors are compared with a small tolerance so snapping is idempotent: a time
// that already sits exactly on a boundary must floor and ceil to itself, not
// jump a whole unit because of float noise (beats come from JSON at 3dp).
const ANCHOR_EPSILON = 1e-6

// Snap a time to an anchor of the mode's grid. Defaults to NEAREST — do not
// change that default: bookmark placement depends on it (src/bookmark.py's
// snap_timestamp floors, but that function is unused; the backend stores
// whatever we send, verbatim).
//
// "none" returns the raw time in every direction, so a none-mode bookmark — and
// a none-mode loop — deliberately lands BETWEEN grid lines. Every mode clamps to
// [0, duration]: with nothing to floor onto, A falls back to 0; with nothing to
// ceil onto, B rides up to the end of the track.
export function snapTime(
  time: number,
  mode: SnapMode,
  grid: GridData | undefined,
  duration: number,
  direction: SnapDirection = 'nearest',
): number {
  const clamp = (t: number) => Math.min(Math.max(t, 0), duration)
  const clamped = clamp(time)
  if (mode === 'none' || !grid) return clamped
  const anchors = snapAnchors(mode, grid) // ascending
  if (anchors.length === 0) return clamped

  if (direction === 'floor') {
    let best: number | null = null
    for (const anchor of anchors) {
      if (anchor > clamped + ANCHOR_EPSILON) break
      best = anchor
    }
    return best === null ? 0 : clamp(best) // nothing at or before it -> track start
  }

  if (direction === 'ceil') {
    for (const anchor of anchors) {
      if (anchor >= clamped - ANCHOR_EPSILON) return clamp(anchor)
    }
    return duration // nothing at or after it -> track end
  }

  let best = anchors[0]
  for (const anchor of anchors) {
    if (Math.abs(anchor - clamped) < Math.abs(best - clamped)) best = anchor
  }
  return clamp(best)
}

async function fetchBookmarks(trackId: string): Promise<Bookmark[]> {
  const res = await fetch(bookmarksUrl(trackId))
  if (!res.ok) throw new Error(`GET bookmarks -> HTTP ${res.status}`)
  const data: { bookmarks: Bookmark[] } = await res.json()
  return data.bookmarks
}

// Whole-set write: PUT replaces every bookmark for the track. Matches the
// sidecar model (one JSON per track) and sidesteps read-modify-write races.
async function putBookmarks(trackId: string, bookmarks: Bookmark[]): Promise<Bookmark[]> {
  const res = await fetch(bookmarksUrl(trackId), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bookmarks }),
  })
  if (!res.ok) throw new Error(`PUT bookmarks -> HTTP ${res.status}`)
  const data: { bookmarks: Bookmark[] } = await res.json()
  return data.bookmarks
}

const byTime = (a: Bookmark, b: Bookmark) => a.timestamp - b.timestamp

export interface BookmarkStore {
  bookmarks: Bookmark[]
  error: string | null
  // Returns the new bookmark's id so the caller can select it right away —
  // creating one opens its info panel, no second click needed.
  create: (timestamp: number, snapMode: SnapMode) => string
  remove: (id: string) => void
  shift: (id: string, timestamp: number, snapMode: SnapMode) => void
  relabel: (id: string, label: string) => void
}

// Optimistic write-through store. Every mutation updates React state first,
// then PUTs the full set; a failed write rolls the local state back to the
// exact list that was live before the change and surfaces the error, so the
// UI never silently diverges from what persisted.
//
// StrictMode safety: all writes happen in event handlers, never in effects —
// a double-invoked effect cannot double-write. The only effect here is the
// mount load, which is idempotent (GET, then replace state) and guarded by a
// stale flag so a slow response for a previous track can't overwrite a newer
// one's bookmarks.
export function useBookmarks(trackId: string | undefined): BookmarkStore {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([])
  const [error, setError] = useState<string | null>(null)
  // The live list, readable synchronously. Mutations derive from this rather
  // than from the render's `bookmarks` closure: two mutations can fire in one
  // tick (typing a label, then clicking a button — blur commits before the
  // click handler runs), and the second would otherwise compute from a stale
  // list and silently undo the first. Not a state updater, so it stays clear
  // of StrictMode's double-invoke of impure updaters.
  const live = useRef<Bookmark[]>([])
  const publish = useCallback((next: Bookmark[]) => {
    live.current = next
    setBookmarks(next)
  }, [])

  useEffect(() => {
    publish([])
    setError(null)
    if (!trackId) return
    let stale = false
    fetchBookmarks(trackId)
      .then((loaded) => {
        if (!stale) publish(loaded.sort(byTime))
      })
      .catch((err) => {
        if (!stale) setError(String(err))
      })
    return () => {
      stale = true
    }
  }, [trackId, publish])

  // Apply a change locally, then persist the whole set; roll back on failure.
  // The rollback target is whatever was live when the change was made.
  const commit = useCallback(
    (next: Bookmark[]) => {
      if (!trackId) return
      const rollback = live.current
      const sorted = [...next].sort(byTime)
      publish(sorted)
      setError(null)
      putBookmarks(trackId, sorted).catch((err) => {
        publish(rollback)
        setError(`${err} — change reverted`)
      })
    },
    [trackId, publish],
  )

  const create = useCallback(
    (timestamp: number, snapMode: SnapMode) => {
      const entry: Bookmark = {
        id: crypto.randomUUID(),
        timestamp,
        label: '',
        snap_mode: snapMode,
      }
      commit([...live.current, entry])
      return entry.id
    },
    [commit],
  )

  const remove = useCallback(
    (id: string) => {
      commit(live.current.filter((entry) => entry.id !== id))
    },
    [commit],
  )

  // Move an existing bookmark and re-snap it with the mode selected NOW — the
  // creation-time mode is overwritten, since a shift means "put it where I'm
  // snapping today".
  const shift = useCallback(
    (id: string, timestamp: number, snapMode: SnapMode) => {
      commit(
        live.current.map((entry) =>
          entry.id === id ? { ...entry, timestamp, snap_mode: snapMode } : entry,
        ),
      )
    },
    [commit],
  )

  // Rename a bookmark. Callers commit on blur/Enter rather than per keystroke,
  // so typing a label is one write, not one write per character.
  const relabel = useCallback(
    (id: string, label: string) => {
      const current = live.current.find((entry) => entry.id === id)
      if (!current || current.label === label) return // no-op: don't rewrite the sidecar
      commit(live.current.map((entry) => (entry.id === id ? { ...entry, label } : entry)))
    },
    [commit],
  )

  return { bookmarks, error, create, remove, shift, relabel }
}
