import { useEffect, useState } from 'react'

// How the Timeline labels each beat, on top of the plain 1..8 counts, which
// always render regardless of this setting:
//   'whole'   — just the counts (1 2 3 4 5 6 7 8), no sub-beat labels at all.
//   'and'     — adds "&" at each beat's midpoint ("1 & 2 & 3 & ..."). This IS
//               the app's original, always-on behavior (Timeline.tsx's
//               subdivision layer) before this setting existed, hence the
//               default below.
//   'e-and-a' — adds "e"/"&"/"a" at the quarter points between each beat
//               ("1 e & a 2 e & a ..."), the finer sixteenth-note grid.
// A device preference, not song data — same reasoning as theme.ts's
// dark-mode flag, accentColor.ts's accent choice, and snap.ts's default snap
// mode. Modeled directly on that last one specifically: nothing here needs a
// pre-mount "apply to the DOM" step, since this only ever feeds a useState
// INITIALIZER (Song.tsx reads it once, the moment a song's own countDisplay
// state is created) — there is no in-song control that changes it again
// afterward the way a loop handle can override snapMode, so a live-synced
// hook (theme.ts/accentColor.ts's shape) would be more machinery than this
// needs.
export type CountDisplay = 'whole' | 'and' | 'e-and-a'

export const COUNT_DISPLAY_MODES: CountDisplay[] = ['whole', 'and', 'e-and-a']

// Labels for the two actual subdivisions are 'Half-count' and 'Fourth-count'
// (naming the fraction of a beat each one adds, at the user's explicit
// request) rather than spelling out "and"/"e and a" — 'whole' isn't a
// subdivision at all, so it keeps its own plain "Whole counts" name.
export const COUNT_DISPLAY_LABELS: Record<CountDisplay, string> = {
  whole: 'Whole counts',
  and: 'Half-count',
  'e-and-a': 'Fourth-count',
}

// The FACTORY default: what a song opens with if the user has never visited
// Settings (or storage is unavailable) — 'and' so a user who never opens
// Settings sees exactly the app's pre-existing behavior, unchanged.
export const DEFAULT_COUNT_DISPLAY: CountDisplay = 'and'

const STORAGE_KEY = 'choreo-count-display'

function isCountDisplay(value: string): value is CountDisplay {
  return (COUNT_DISPLAY_MODES as string[]).includes(value)
}

export function readCountDisplay(): CountDisplay {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored && isCountDisplay(stored) ? stored : DEFAULT_COUNT_DISPLAY
  } catch {
    return DEFAULT_COUNT_DISPLAY
  }
}

export interface CountDisplayController {
  countDisplay: CountDisplay
  setCountDisplay: (mode: CountDisplay) => void
}

/**
 * Owns the live count-display preference and keeps localStorage in sync with
 * it. Settings-screen-only, mirroring useDefaultSnapMode's shape exactly —
 * nothing about an open Song reads this hook directly, only
 * readCountDisplay's plain read at mount.
 */
export function useCountDisplay(): CountDisplayController {
  const [countDisplay, setCountDisplay] = useState<CountDisplay>(readCountDisplay)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, countDisplay)
    } catch {
      // Storage unavailable — the choice still applies for this session, it
      // just won't persist across reloads.
    }
  }, [countDisplay])

  return { countDisplay, setCountDisplay }
}
