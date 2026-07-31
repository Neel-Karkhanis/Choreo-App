// Shared data shapes. These live here rather than inside a screen because the
// shell above the screens holds them now: a grid is owned by the Song, not by
// whichever view happens to be drawing it.

// Beat-grid data, already validated by the caller. downbeatIndices and
// eightCountIndices are indices into beats (schema v4), not timestamps.
export interface GridData {
  beats: number[]
  downbeatIndices: number[]
  eightCountIndices: number[]
}

// Onset data, already validated by the caller. Plain timestamps in seconds,
// independent of the beat grid — not indices, not guaranteed to land on a beat.
export interface OnsetData {
  drums: number[]
  bass: number[]
}

// Whether a song offers one screen or two. Comes from the backend manifest,
// which derives it from the file extension and lets a manifest override stand.
export type MediaKind = 'audio' | 'video'

// A library row exactly as GET /api/library returns it.
//
// `state` is the openability verdict; the three booleans are the evidence
// behind it, reported separately so the UI can explain the verdict instead of
// just asserting it. See the server's _library_entry for why missing stems do
// not make a song unopenable.
export type LibraryState = 'ready' | 'needs_tap' | 'stems_evicted'

export interface LibrarySong {
  md5: string
  id: string | null
  filename: string | null
  duration: number | null
  media_kind: MediaKind
  state: LibraryState
  grid_present: boolean
  stems_present: boolean
  original_present: boolean
}
