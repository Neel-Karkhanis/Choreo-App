import type { GridData } from './types'

export type SnapMode = 'none' | 'beat' | '4-count' | '8-count'

export const SNAP_MODES: SnapMode[] = ['none', 'beat', '4-count', '8-count']

// The mode snapping starts in until the user picks another.
export const DEFAULT_SNAP_MODE: SnapMode = 'beat'

// Modes offered from the loop's A/B handles (see LoopBoundaryHandle). 'none'
// is a valid SnapMode elsewhere (snapTime degrades to a plain clamp) but is
// deliberately not offered here — a loop handle with no snap at all isn't a
// choice this menu needs to surface.
export const BOUNDARY_SNAP_MODES: SnapMode[] = ['beat', '4-count', '8-count']

export const SNAP_MODE_LABELS: Record<SnapMode, string> = {
  none: 'No snap',
  beat: 'Beat',
  '4-count': '4 count',
  '8-count': '8 count',
}

// Same-origin in dev: Vite proxies /api to the FastAPI backend, so app
// requests never cross an origin. (The backend also sends CORS headers for
// the Vite dev origin, so pointing this at http://127.0.0.1:8000 directly
// works too.)
export const API_BASE = '/api'

// Snap anchors per mode, from data the browser already holds — snapping is
// local and instant, never a round-trip.
//
// 4-count is derived from the EIGHT-COUNT grid (every 4th beat from each
// eight-count start, i.e. counts 1 and 5 of each rendered block) rather than
// from downbeatIndices. Tapped grids mark downbeats every 4 by construction,
// so today the two agree — but anchoring on the eight-count grid keeps a
// 4-count anchor on a count-1 or count-5 of a block the user can actually see
// on the timeline, whatever downbeatIndices carries.
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
// 'nearest' — closest anchor either side.
// 'floor'   — largest anchor <= time. 'ceil' — smallest anchor >= time. These
//   exist for the LOOP, whose endpoints are a span, not a point: flooring A and
//   ceiling B makes the loop enclose whole musical units and guarantees it never
//   clips inside the range the user actually selected.
export type SnapDirection = 'nearest' | 'floor' | 'ceil'

// Anchors are compared with a small tolerance so snapping is idempotent: a time
// that already sits exactly on a boundary must floor and ceil to itself, not
// jump a whole unit because of float noise (beats come from JSON at 3dp).
const ANCHOR_EPSILON = 1e-6

// Snap a time to an anchor of the mode's grid. Defaults to NEAREST.
//
// "none" returns the raw time in every direction, so a none-mode loop
// deliberately lands BETWEEN grid lines. Every mode clamps to [0, duration]:
// with nothing to floor onto, A falls back to 0; with nothing to ceil onto, B
// rides up to the end of the track.
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
