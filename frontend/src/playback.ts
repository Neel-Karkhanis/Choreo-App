import { useCallback, useEffect, useRef, useState } from 'react'
import type WaveSurfer from 'wavesurfer.js'
import type { SnapDirection } from './bookmarks'
import { PLAYBACK_RATE_MAX, PLAYBACK_RATE_MIN, type StemEngine } from './stemEngine'

// Real-time, client-side playback behaviors: A/B loop and playback speed.
// Both are thin state layers over the stem engine — the loop maps onto the
// buffer sources' NATIVE loop points and speed onto their playbackRate, so
// neither ever goes through the seek/rebuild path.

// Playback speed. 0.25x for pulling apart fast passages, up to 2x for
// skimming. Bounds come from the engine (the clamp authority). Rate
// RESAMPLES: pitch shifts with speed (chipmunk high / muddy low). That is a
// deliberate accepted tradeoff — there is no time-stretcher and no
// pitch-preservation flag; building one is out of scope by decision, not
// omission.
export const SPEED_MIN = PLAYBACK_RATE_MIN
export const SPEED_MAX = PLAYBACK_RATE_MAX
export const SPEED_STEP = 0.05
export const SPEED_DEFAULT = 1

// Loop band: a 4th visual treatment, distinct from the 8-count greys, the
// Layer 6 yellow highlight, the red/blue onsets, and the brown bookmarks.
// Translucent teal wash bracketed by solid edges at A and B.
export const LOOP_BAND_FILL = 'rgba(20, 150, 150, 0.12)'
export const LOOP_BAND_EDGE = 'rgba(0, 120, 125, 0.9)'
export const LOOP_BAND_EDGE_WIDTH = 2
export const LOOP_MINIMAP_FILL = 'rgba(20, 150, 150, 0.2)'
// A loop that is set but switched off still shows, dimmed — so you can see what
// will resume when you toggle it back on.
export const LOOP_DISABLED_OPACITY = '0.4'

export interface LoopController {
  start: number | null
  end: number | null
  enabled: boolean
  error: string | null
  // Imperative surface for a later touch/gesture pass to wire to. All take RAW
  // times and snap internally — and DIRECTIONALLY: setLoopStart floors, setLoopEnd
  // ceils, setLoop floors a and ceils b. Every caller (buttons now, drag handles
  // or tap-two-bookmarks later) gets that enclosing behavior for free.
  setLoop: (start: number, end: number) => void
  setLoopStart: (time: number) => void
  setLoopEnd: (time: number) => void
  clearLoop: () => void
  setEnabled: (enabled: boolean) => void
}

interface LoopRange {
  start: number | null
  end: number | null
}

/**
 * A/B repeat over the live playhead.
 *
 * `snap` is injected rather than imported so the loop honors whatever snap mode
 * is currently selected — the same capability bookmarks use, but asked for a
 * DIRECTION: A floors, B ceils. A loop set in "none" mode keeps raw times.
 *
 * ENFORCEMENT lives in the engine, not here: the committed region is pushed
 * into StemEngine.setLoop, which sets loop/loopStart/loopEnd NATIVELY on all
 * five playing sources (sample-accurate, click-free, no rebuild — the old
 * per-frame timeupdate + setTime mechanism died with the media element) and
 * wraps its own clock to match. Loop points are buffer-time seconds, so the
 * region is identical at every playback rate.
 *
 * Outside-region behavior (owned by the engine): pressing play, or enabling
 * the loop mid-playback, with the playhead outside [A, B) enters at A — a
 * user who turns on a loop wants the section. Seeking BEFORE A stays allowed
 * (playback runs forward into the region and wraps at B); a seek landing
 * at/past B goes to A, the same net behavior the old timeupdate yank had.
 */
export function useLoop(
  engine: StemEngine,
  snap: (time: number, direction: SnapDirection) => number,
): LoopController {
  const [range, setRange] = useState<LoopRange>({ start: null, end: null })
  const [enabled, setEnabled] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    engine.setLoop(range.start, range.end, enabled)
  }, [engine, range, enabled])

  // The endpoints AS THE USER PLACED THEM, before snapping. Kept because a swap
  // has to reorder the raw times, not the snapped ones: if A is dropped at 96.8s
  // (floored to 94.5) and B is then dropped at 86.6s, swapping the SNAPPED values
  // would ceil 94.5 — which is already a boundary — and the loop's tail would
  // stop short of the 96.8s the user actually reached for. Swapping the raws
  // instead re-derives both ends from intent: floor(86.6), ceil(96.8).
  const rawRange = useRef<LoopRange>({ start: null, end: null })

  // Normalize a candidate range.
  //
  // DIRECTIONAL snapping, unlike bookmarks: A floors and B ceils, so the loop
  // always encloses whole musical units and never clips inside the span the user
  // picked. In "none" mode both directions pass the raw time through.
  //
  // Order matters: A after B auto-SWAPS on the RAW times first (the user meant
  // the span between the two points, not an error), and only then does the new A
  // floor and the new B ceil.
  //
  // A zero-length loop is rejected: it would trap the playhead on a single
  // instant. That includes the DEGENERATE COLLAPSE where the two raw times floor
  // and ceil onto the same boundary. (Two DIFFERENT raw times inside one grid
  // unit do not collapse — they floor and ceil to the unit's two edges, which is
  // the whole point: you get the enclosing unit.)
  const commit = useCallback(
    (next: LoopRange) => {
      let rawStart = next.start
      let rawEnd = next.end
      if (rawStart !== null && rawEnd !== null && rawStart > rawEnd) {
        ;[rawStart, rawEnd] = [rawEnd, rawStart]
      }
      const start = rawStart === null ? null : snap(rawStart, 'floor')
      const end = rawEnd === null ? null : snap(rawEnd, 'ceil')
      if (start !== null && end !== null && start >= end) {
        setError('Loop needs two different points — A and B snapped to the same spot.')
        return
      }
      rawRange.current = { start: rawStart, end: rawEnd }
      setRange({ start, end })
      setError(null)
    },
    [snap],
  )

  const setLoop = useCallback((start: number, end: number) => commit({ start, end }), [commit])
  const setLoopStart = useCallback(
    (time: number) => commit({ start: time, end: rawRange.current.end }),
    [commit],
  )
  const setLoopEnd = useCallback(
    (time: number) => commit({ start: rawRange.current.start, end: time }),
    [commit],
  )
  const clearLoop = useCallback(() => {
    rawRange.current = { start: null, end: null }
    setRange({ start: null, end: null })
    setEnabled(false)
    setError(null)
  }, [])

  return {
    start: range.start,
    end: range.end,
    enabled,
    error,
    setLoop,
    setLoopStart,
    setLoopEnd,
    clearLoop,
    setEnabled,
  }
}

export interface SpeedController {
  speed: number
  setSpeed: (rate: number) => void
  resetSpeed: () => void
}

/**
 * Live playback rate. Only wall-clock speed changes: the engine's clock still
 * advances in TRACK time (rate-aware, re-anchored on every change), so beats,
 * 8-counts, bookmarks, labels, and loop A/B — all defined in track time —
 * stay exactly where they are. Nothing in the timeline may assume
 * realtime == tracktime.
 *
 * The write goes through wavesurfer's setPlaybackRate -> media.playbackRate,
 * which the engine shim maps onto all five sources live: no rebuild, no
 * position jump — a rate change bends the playback slope, never the position.
 */
export function useSpeed(wavesurfer: WaveSurfer | null): SpeedController {
  const [speed, setSpeedState] = useState(SPEED_DEFAULT)

  useEffect(() => {
    if (!wavesurfer) return
    wavesurfer.setPlaybackRate(speed)
  }, [wavesurfer, speed])

  const setSpeed = useCallback((rate: number) => {
    setSpeedState(Math.min(Math.max(rate, SPEED_MIN), SPEED_MAX))
  }, [])
  const resetSpeed = useCallback(() => setSpeedState(SPEED_DEFAULT), [])

  return { speed, setSpeed, resetSpeed }
}
