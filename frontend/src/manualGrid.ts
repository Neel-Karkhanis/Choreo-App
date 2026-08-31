import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from './api'
import * as localDb from './localDb'
import { API_BASE } from './snap'
import type { GridData } from './types'

// Persistence for the tapped grid — the ONLY grid the app has. The backend
// stores it under data/grids/<owner_id>/<md5>.json, keyed by the audio's MD5
// hash, so a grid outlives a restart and follows the AUDIO rather than the
// filename. This is load-bearing now: with auto detection gone, a grid that
// fails to persist means the song prompts for a retap every session.
//
// The wire shape is schema v4's grid verbatim (downbeats/eight_counts are
// indices into beats) — the same arrays the analysis payload used to carry, so
// every consumer downstream of GridData is unchanged by auto's removal.
//
// SERVER-AUTHORITATIVE, INDEXEDDB AS OFFLINE CACHE ONLY: every load still
// fetches from the server first, exactly as before this file grew an
// IndexedDB mirror. The mirror (localDb.ts's `grids` store, keyed by md5 so
// it survives a rename same as the server copy does) exists for exactly one
// case — the server fetch itself fails (offline, or mid-flight on an
// installed PWA with a flaky connection) — where it lets the track open
// with its last-synced grid instead of prompting a retap. A successful
// server fetch always overwrites the mirror; the mirror never overwrites a
// successful server read. See localDb.ts's own header for why every mirror
// operation is best-effort and never throws.

interface ManualGridPayload {
  beats: number[]
  downbeats: number[]
  eight_counts: number[]
  // Legacy field from when a tap could be parked while auto detection ran.
  // Auto is gone, so a saved tap is the grid regardless of this flag — it is
  // deliberately ignored on read and never written.
  active?: boolean
}

const manualGridUrl = (trackId: string) =>
  `${API_BASE}/tracks/${encodeURIComponent(trackId)}/manual-grid`

const toGrid = (p: ManualGridPayload): GridData => ({
  beats: p.beats,
  downbeatIndices: p.downbeats,
  eightCountIndices: p.eight_counts,
})

const toPayload = (g: GridData): ManualGridPayload => ({
  beats: g.beats,
  downbeats: g.downbeatIndices,
  eight_counts: g.eightCountIndices,
})

interface ManualGridState {
  // The saved grid, or null if this track has never been tapped (or the read
  // failed — see the load effect).
  grid: GridData | null
  // True until the mount read settles. The first-run tap prompt is gated on
  // this: a track with a saved grid must NEVER flash the prompt while its
  // grid is still in flight.
  loading: boolean
  // True when `grid` came from the IndexedDB mirror because the server
  // fetch itself failed — not set on a normal successful read, and cleared
  // the moment a server read or write succeeds.
  offline: boolean
}

export interface ManualGridStore {
  grid: GridData | null
  loading: boolean
  error: string | null
  offline: boolean
  // Commit a grid (fresh tap or nudge); it becomes the live grid immediately.
  save: (grid: GridData) => void
}

/**
 * Optimistic write-through store for the tapped grid.
 *
 * Mutations update React state first and then write, rolling back to the exact
 * value that was live before on failure — the grid on screen never silently
 * disagrees with what persisted.
 *
 * StrictMode safety: every write happens in an event handler, never in an
 * effect, so a double-invoked effect cannot double-write. The only effect is
 * the mount load, which is idempotent and stale-guarded so a slow response for
 * a previous track can't land on a newer one.
 *
 * `md5` is optional only so a call site without it yet (there shouldn't be
 * one) degrades to trackId-keyed caching rather than crashing; every real
 * caller has it (LibrarySong always carries md5).
 */
export function useManualGrid(trackId: string | undefined, md5: string | undefined): ManualGridStore {
  const [state, setState] = useState<ManualGridState>({ grid: null, loading: true, offline: false })
  const [error, setError] = useState<string | null>(null)
  // The live value, readable synchronously: a rollback has to restore whatever
  // was actually on screen when the write started, not whatever this render's
  // closure happened to capture.
  const live = useRef<ManualGridState>({ grid: null, loading: true, offline: false })
  const publish = useCallback((next: ManualGridState) => {
    live.current = next
    setState(next)
  }, [])

  useEffect(() => {
    setError(null)
    if (!trackId) {
      publish({ grid: null, loading: false, offline: false })
      return
    }
    publish({ grid: null, loading: true, offline: false })
    let stale = false
    apiFetch(manualGridUrl(trackId))
      .then((res) => {
        if (!res.ok) throw new Error(`GET manual-grid -> HTTP ${res.status}`)
        return res.json() as Promise<{ manual_grid: ManualGridPayload | null }>
      })
      .then((data) => {
        if (stale) return
        const grid = data.manual_grid ? toGrid(data.manual_grid) : null
        publish({ grid, loading: false, offline: false })
        // Mirror the authoritative read, including the "never tapped" case
        // — see the deleteGrid note below for why a null grid still needs
        // its own mirror entry cleared, not just left stale.
        if (md5) {
          if (grid) void localDb.setGrid(md5, grid, Date.now())
          else void localDb.deleteGrid(md5)
        }
      })
      .catch((err) => {
        if (stale) return
        // The server is unreachable (not: "server said no grid exists" —
        // that's the branch above). Fall back to whatever was last synced,
        // rather than blanking the track and prompting a retap that would
        // just discard real, already-saved work the moment the connection
        // comes back.
        if (!md5) {
          setError(String(err))
          publish({ grid: null, loading: false, offline: false })
          return
        }
        localDb
          .getGrid(md5)
          .then((cached) => {
            if (stale) return
            if (cached) {
              setError(`${err}: showing the last synced grid (offline)`)
              publish({ grid: cached.grid, loading: false, offline: true })
            } else {
              setError(String(err))
              publish({ grid: null, loading: false, offline: false })
            }
          })
          .catch(() => {
            if (stale) return
            setError(String(err))
            publish({ grid: null, loading: false, offline: false })
          })
      })
    return () => {
      stale = true
    }
  }, [trackId, md5, publish])

  const save = useCallback(
    (next: GridData) => {
      if (!trackId) return
      const rollback = live.current
      publish({ grid: next, loading: false, offline: false })
      setError(null)
      apiFetch(manualGridUrl(trackId), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toPayload(next)),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`PUT manual-grid -> HTTP ${res.status}`)
          // Mirror only once the server has actually confirmed the write —
          // server stays authoritative, so the offline cache must never
          // hold a value the server hasn't agreed to. The on-screen grid
          // already updated above regardless; this just keeps the mirror
          // from drifting ahead of what a failed PUT would roll back.
          if (md5) void localDb.setGrid(md5, next, Date.now())
        })
        .catch((err) => {
          publish(rollback)
          setError(`${err}: tapped grid not saved`)
        })
    },
    [trackId, md5, publish],
  )

  return { grid: state.grid, loading: state.loading, error, offline: state.offline, save }
}
