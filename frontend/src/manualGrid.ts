import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE } from './snap'
import type { GridData } from './types'

// Persistence for the tapped grid — the ONLY grid the app has. The backend
// stores it under data/grids/<md5>.json, keyed by the audio's MD5 hash, so a
// grid outlives a restart and follows the AUDIO rather than the filename.
// This is load-bearing now: with auto detection gone, a grid that fails to
// persist means the song prompts for a retap every session.
//
// The wire shape is schema v4's grid verbatim (downbeats/eight_counts are
// indices into beats) — the same arrays the analysis payload used to carry, so
// every consumer downstream of GridData is unchanged by auto's removal.

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
}

export interface ManualGridStore {
  grid: GridData | null
  loading: boolean
  error: string | null
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
 */
export function useManualGrid(trackId: string | undefined): ManualGridStore {
  const [state, setState] = useState<ManualGridState>({ grid: null, loading: true })
  const [error, setError] = useState<string | null>(null)
  // The live value, readable synchronously: a rollback has to restore whatever
  // was actually on screen when the write started, not whatever this render's
  // closure happened to capture.
  const live = useRef<ManualGridState>({ grid: null, loading: true })
  const publish = useCallback((next: ManualGridState) => {
    live.current = next
    setState(next)
  }, [])

  useEffect(() => {
    setError(null)
    if (!trackId) {
      publish({ grid: null, loading: false })
      return
    }
    publish({ grid: null, loading: true })
    let stale = false
    fetch(manualGridUrl(trackId))
      .then((res) => {
        if (!res.ok) throw new Error(`GET manual-grid -> HTTP ${res.status}`)
        return res.json() as Promise<{ manual_grid: ManualGridPayload | null }>
      })
      .then((data) => {
        if (stale) return
        publish({ grid: data.manual_grid ? toGrid(data.manual_grid) : null, loading: false })
      })
      .catch((err) => {
        // A failed read lands in the tap state WITH the error visible. There is
        // no auto to fall back to, and blocking the track outright would make a
        // transient sidecar failure fatal; the surfaced error is what tells the
        // user their existing grid may still be on disk before they retap.
        if (stale) return
        setError(String(err))
        publish({ grid: null, loading: false })
      })
    return () => {
      stale = true
    }
  }, [trackId, publish])

  const save = useCallback(
    (next: GridData) => {
      if (!trackId) return
      const rollback = live.current
      publish({ grid: next, loading: false })
      setError(null)
      fetch(manualGridUrl(trackId), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toPayload(next)),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`PUT manual-grid -> HTTP ${res.status}`)
        })
        .catch((err) => {
          publish(rollback)
          setError(`${err} — tapped grid not saved`)
        })
    },
    [trackId, publish],
  )

  return { grid: state.grid, loading: state.loading, error, save }
}
