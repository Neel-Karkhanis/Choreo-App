import type { GridData } from './types'

// Where the playhead sits in the count, derived from the grid alone.
//
// Extracted out of the Timeline so the HUD can read the count without owning
// (or re-deriving) any of this: both surfaces answer "what count is this?" from
// the same windows over the same hoisted grid, so they cannot disagree.

// One 8-count group as a time window. firstBeat indexes into beats; the k-th
// window spans [beats[ec[k]], beats[ec[k+1]]), and the last window runs to
// the track end and may hold fewer than 8 beats (ragged final group).
export interface EightCountWindow {
  firstBeat: number
  beatCount: number
  start: number
  end: number
}

export function buildEightCountWindows(
  grid: GridData | undefined,
  duration: number,
): EightCountWindow[] | null {
  if (!grid || !duration) return null
  const { beats, eightCountIndices } = grid
  return eightCountIndices.map((firstBeat, k): EightCountWindow => {
    const nextFirst = eightCountIndices[k + 1]
    return {
      firstBeat,
      beatCount: (nextFirst ?? beats.length) - firstBeat,
      start: beats[firstBeat],
      end: nextFirst !== undefined ? beats[nextFirst] : duration,
    }
  })
}

// Binary search for the window containing `time`; null before the first
// window (pickup beats precede the first downbeat, so beats[0] can sit
// outside every group) or at/past the end of the last.
export function findEightCountIndex(
  windows: EightCountWindow[],
  time: number,
): number | null {
  let lo = 0
  let hi = windows.length - 1
  let candidate = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (windows[mid].start <= time) {
      candidate = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return candidate >= 0 && time < windows[candidate].end ? candidate : null
}

/** 1-based phrase number and 1-based count within it. */
export interface CountPosition {
  phrase: number
  count: number
}

// The count under the playhead, or null outside every window (before the first
// eight-count start, or past the last beat). Counts are 1-based and label the
// beat the playhead is ON or has most recently passed — the same beat the
// Timeline's active-block labels number, by construction: both walk
// win.firstBeat + i.
export function countAt(
  windows: EightCountWindow[] | null,
  grid: GridData | undefined,
  time: number,
): CountPosition | null {
  if (!windows || !grid) return null
  const index = findEightCountIndex(windows, time)
  if (index === null) return null
  const win = windows[index]
  let count = 1
  for (let i = 0; i < win.beatCount; i++) {
    if (grid.beats[win.firstBeat + i] <= time) count = i + 1
    else break
  }
  return { phrase: index + 1, count }
}
