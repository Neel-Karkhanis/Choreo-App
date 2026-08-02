import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SnapDirection } from './snap'
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
// Quarter-x dials, not a continuous rate: coarser than the engine's own clamp
// resolution, but musically the increments that matter (half-time, 3/4-time).
export const SPEED_STEP = 0.25
export const SPEED_DEFAULT = 1

// Loop band: a 4th visual treatment, distinct from the 8-count greys, the
// Layer 6 yellow highlight, the red/blue onsets, and the brown bookmarks.
// Translucent teal wash bracketed by solid edges at A and B. The loop is
// always active (see useLoop), so this is drawn at full opacity always —
// there is no dimmed "set but off" state anymore.
export const LOOP_BAND_FILL = 'rgba(20, 150, 150, 0.12)'
export const LOOP_BAND_EDGE = 'rgba(0, 120, 125, 0.9)'
export const LOOP_BAND_EDGE_WIDTH = 2

/**
 * Playhead + play/pause, read straight off the engine.
 *
 * THE reason this exists: transport used to be whatever `useWavesurfer`
 * happened to report inside the Timeline, which made the Timeline the de facto
 * owner of the clock. A second screen would then have had a second wavesurfer
 * with a second copy of isPlaying/currentTime, and the two would have drifted
 * on the first seek. Subscribing to the engine instead gives every screen the
 * SAME numbers from the SAME clock, and leaves the wavesurfer instances as what
 * they should be: renderers.
 *
 * The engine emits discrete edges only (play/pause/seeked/ended/timeupdate) —
 * it deliberately does not stream a timeupdate per frame — so the position is
 * polled on rAF while playing and read from the edges otherwise. That is the
 * same shape wavesurfer's own 16ms ticker uses, for the same reason.
 */
export interface Transport {
  isPlaying: boolean
  currentTime: number
  duration: number
  play: () => void
  pause: () => void
  toggle: () => void
  // Moves the shared clock. Both wavesurfer views follow via timeupdate — no
  // view ever seeks another view.
  seek: (time: number) => void
}

export function useEngineTransport(engine: StemEngine): Transport {
  const [isPlaying, setIsPlaying] = useState(() => !engine.paused)
  const [currentTime, setCurrentTime] = useState(() => engine.currentTime)

  useEffect(() => {
    const media = engine.media
    const sync = () => setCurrentTime(engine.currentTime)
    const onPlay = () => {
      setIsPlaying(true)
      sync()
    }
    const onStop = () => {
      setIsPlaying(false)
      sync()
    }
    // Adopt the engine's state on (re)subscribe rather than trusting the
    // initializer: a new engine arrives mid-render and may already differ.
    setIsPlaying(!engine.paused)
    sync()
    media.addEventListener('play', onPlay)
    media.addEventListener('pause', onStop)
    media.addEventListener('ended', onStop)
    media.addEventListener('timeupdate', sync)
    media.addEventListener('seeked', sync)
    return () => {
      media.removeEventListener('play', onPlay)
      media.removeEventListener('pause', onStop)
      media.removeEventListener('ended', onStop)
      media.removeEventListener('timeupdate', sync)
      media.removeEventListener('seeked', sync)
    }
  }, [engine])

  // Poll the clock only while it is moving; a paused engine's position changes
  // only on seek, which fires an edge.
  useEffect(() => {
    if (!isPlaying) return
    let frame = 0
    const tick = () => {
      setCurrentTime(engine.currentTime)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [engine, isPlaying])

  const play = useCallback(() => void engine.play(), [engine])
  const pause = useCallback(() => engine.pause(), [engine])
  const toggle = useCallback(() => {
    if (engine.paused) void engine.play()
    else engine.pause()
  }, [engine])
  const seek = useCallback((time: number) => engine.seek(time), [engine])

  return useMemo(
    () => ({
      isPlaying,
      currentTime,
      duration: engine.duration,
      play,
      pause,
      toggle,
      seek,
    }),
    [isPlaying, currentTime, engine, play, pause, toggle, seek],
  )
}

export interface LoopController {
  // Always defined and always ACTIVE: the loop spans the whole track until
  // the overall timeline's A/B handles narrow it, and playback loops that
  // region unconditionally — there is no on/off switch. See useLoop.
  start: number
  end: number
  error: string | null
  // Imperative surface the overall timeline's drag handles call into. Both
  // take RAW times and snap internally — and DIRECTIONALLY: setLoopStart
  // floors, setLoopEnd ceils, setLoop floors a and ceils b — so a dragged
  // handle always lands enclosing whole musical units.
  setLoop: (start: number, end: number) => void
  setLoopStart: (time: number) => void
  setLoopEnd: (time: number) => void
}

interface LoopRange {
  start: number
  end: number
}

/**
 * A/B repeat, always on.
 *
 * A and B default to the whole track [0, duration] and playback ALWAYS loops
 * whatever region they currently describe — narrowing them (by dragging the
 * overall timeline's handles) is the only way to change what repeats, not
 * whether it repeats.
 *
 * `snap` is injected rather than imported so the loop honors the app's beat
 * grid without this hook needing to know where grids come from, asked for a
 * DIRECTION: A floors, B ceils. Snapping is always to the beat — there is no
 * user-facing mode selector — but `snap` gracefully degrades to a plain
 * clamp when there is no grid yet (see snapTime), which is what keeps A at
 * exactly 0 and B at exactly `duration` before a track has ever been tapped.
 *
 * ENFORCEMENT lives in the engine, not here: the committed region is pushed
 * into StemEngine.setLoop, which sets loop/loopStart/loopEnd NATIVELY on all
 * five playing sources (sample-accurate, click-free, no rebuild — the old
 * per-frame timeupdate + setTime mechanism died with the media element) and
 * wraps its own clock to match. Loop points are buffer-time seconds, so the
 * region is identical at every playback rate.
 *
 * Outside-region behavior (owned by the engine): pressing play, or narrowing
 * the loop mid-playback, with the playhead outside [A, B) enters at A. Seeking
 * BEFORE A stays allowed (playback runs forward into the region and wraps at
 * B); a seek landing at/past B goes to A.
 */
export function useLoop(
  engine: StemEngine,
  snap: (time: number, direction: SnapDirection) => number,
  duration: number,
): LoopController {
  const [range, setRange] = useState<LoopRange>({ start: 0, end: duration })
  const [error, setError] = useState<string | null>(null)

  // Always on — see the module doc above. Only the REGION ever changes;
  // there is nothing left to toggle.
  useEffect(() => {
    engine.setLoop(range.start, range.end, true)
  }, [engine, range])

  // The endpoints AS THE USER PLACED THEM, before snapping. Kept because a swap
  // has to reorder the raw times, not the snapped ones: if A is dropped at 96.8s
  // (floored to 94.5) and B is then dropped at 86.6s, swapping the SNAPPED values
  // would ceil 94.5 — which is already a boundary — and the loop's tail would
  // stop short of the 96.8s the user actually reached for. Swapping the raws
  // instead re-derives both ends from intent: floor(86.6), ceil(96.8).
  const rawRange = useRef<LoopRange>({ start: 0, end: duration })

  // Normalize a candidate range.
  //
  // DIRECTIONAL snapping: A floors and B ceils, so the loop always encloses
  // whole musical units and never clips inside the span the user picked.
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
      if (rawStart > rawEnd) {
        ;[rawStart, rawEnd] = [rawEnd, rawStart]
      }
      const start = snap(rawStart, 'floor')
      const end = snap(rawEnd, 'ceil')
      if (start >= end) {
        setError('Loop needs two different points — A and B snapped to the same spot.')
        return
      }
      rawRange.current = { start: rawStart, end: rawEnd }
      setRange({ start, end })
      setError(null)
    },
    [snap],
  )

  // Re-commit the RAW endpoints whenever the snap function itself changes —
  // which happens when the grid moves (a phase/count nudge, a re-tap, a tap
  // preview updating), when a grid first becomes available, or duration is
  // (re)confirmed. The raw times are the user's intent; the snapped endpoints
  // are derived state, so they follow the grid. This is what makes a phase
  // nudge audible while looping: A and B land on the corrected grid
  // immediately, mid-loop, instead of keeping the stale offset until the user
  // re-sets them.
  useEffect(() => {
    commit(rawRange.current)
  }, [commit])

  const setLoop = useCallback((start: number, end: number) => commit({ start, end }), [commit])
  const setLoopStart = useCallback(
    (time: number) => commit({ start: time, end: rawRange.current.end }),
    [commit],
  )
  const setLoopEnd = useCallback(
    (time: number) => commit({ start: rawRange.current.start, end: time }),
    [commit],
  )

  return {
    start: range.start,
    end: range.end,
    error,
    setLoop,
    setLoopStart,
    setLoopEnd,
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
 * 8-counts, and loop A/B — all defined in track time — stay exactly where
 * they are. Nothing in the timeline may assume realtime == tracktime.
 *
 * The write goes STRAIGHT to the engine, which maps it onto all five sources
 * live: no rebuild, no position jump — a rate change bends the playback slope,
 * never the position. It used to go through wavesurfer's setPlaybackRate,
 * which only ever forwarded to media.playbackRate and so landed in exactly
 * this same call; routing around it is what lets speed be owned above the
 * screens, where no wavesurfer instance exists to forward through.
 */
export function useSpeed(engine: StemEngine): SpeedController {
  const [speed, setSpeedState] = useState(SPEED_DEFAULT)

  useEffect(() => {
    engine.setPlaybackRate(speed)
  }, [engine, speed])

  const setSpeed = useCallback((rate: number) => {
    setSpeedState(Math.min(Math.max(rate, SPEED_MIN), SPEED_MAX))
  }, [])
  const resetSpeed = useCallback(() => setSpeedState(SPEED_DEFAULT), [])

  return { speed, setSpeed, resetSpeed }
}
