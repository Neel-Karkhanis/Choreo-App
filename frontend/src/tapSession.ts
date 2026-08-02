import { useCallback, useEffect, useMemo, useState } from 'react'
import type { StemEngine } from './stemEngine'
import { MIN_TAPS, fitTapGrid, gridFromFit, needsTapPrompt, type TapFit } from './tapGrid'
import type { GridData } from './types'

// The tap session: everything that happens between "this song has no count" and
// "here is its grid".
//
// It lives ABOVE the screens with the rest of the song's state, even though
// only the Timeline draws it. The reason is the preview: while a session is
// open the previewed grid IS the grid for every consumer — the loop snaps to
// it, the count readout reads it — so if the session lived inside the Timeline,
// the hoisted loop would be snapping to the saved grid while the user watched
// a different one on screen. Hoisting it keeps "the preview is not a special
// rendering path, it is the same GridData in the same slot" true.

// Refitting rebuilds every region on the timeline, so it is deferred until the
// tap stream goes quiet for this long. Doing it per-tap would block the main
// thread and delay the very pointer events whose timing the fit is measuring —
// the render would corrupt its own input. Comfortably longer than a tap gap at
// any plausible tempo (1.5s = 40 counts/min).
export const TAP_SETTLE_MS = 1500

export interface TapSession {
  tapping: boolean
  taps: number[]
  fit: TapFit | null
  error: string | null
  // The live tapped grid while the session is open, or null. Takes the saved
  // grid's place wherever a grid is consumed.
  preview: GridData | null
  enter: () => void
  exit: () => void
  record: () => void
  accept: () => void
}

export interface TapSessionOptions {
  // The clock taps are measured against. Reading the engine directly (rather
  // than a rendered currentTime) keeps taps and beats on one origin: every
  // timestamp here is ultimately compared to audio time, and a React state
  // value is a frame stale by definition.
  engine: StemEngine
  duration: number
  isPlaying: boolean
  grid: GridData | undefined
  gridLoading: boolean
  onGridTapped?: (grid: GridData) => void
}

export function useTapSession({
  engine,
  duration,
  isPlaying,
  grid,
  gridLoading,
  onGridTapped,
}: TapSessionOptions): TapSession {
  const [tapping, setTapping] = useState(false)
  const [taps, setTaps] = useState<number[]>([])
  const [fit, setFit] = useState<TapFit | null>(null)
  const [error, setError] = useState<string | null>(null)

  // First run: no saved grid means the song opens IN the tap state — a state
  // of the main view, not a modal and not a separate screen. Playback,
  // scrubbing, the waveform and the minimap are all live; the grid layers are
  // simply dormant until a fit is accepted. There is no skip. This also
  // re-opens the prompt if a save fails and rolls back to null.
  useEffect(() => {
    if (needsTapPrompt(gridLoading, grid)) setTapping(true)
  }, [gridLoading, grid])

  // One tap. The timestamp is read INSIDE the handler, from the engine's live
  // clock — Date.now() would be a different clock with its own drift against
  // the audio.
  //
  // Only while playing: paused, the engine's clock is frozen, so every tap
  // would read the same instant and the fit would have nothing to work with.
  const record = useCallback(() => {
    if (!isPlaying) return
    setTaps((prev) => [...prev, engine.currentTime])
  }, [engine, isPlaying])

  const reset = useCallback(() => {
    setTaps([])
    setFit(null)
    setError(null)
  }, [])

  const exit = useCallback(() => {
    reset()
    setTapping(false)
  }, [reset])

  const enter = useCallback(() => {
    reset()
    setTapping(true)
  }, [reset])

  // Space taps, bound at the window rather than on the pad: requiring focus
  // would make the keyboard path depend on the user having clicked the pad
  // first, which defeats the point of having a keyboard path.
  useEffect(() => {
    if (!tapping) return
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return
      const target = e.target as HTMLElement | null
      const tag = target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      e.preventDefault() // Space would otherwise scroll the page
      record()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [tapping, record])

  // Fit only once the taps STOP. Every refit rebuilds the whole timeline, and
  // doing that between taps would delay the next pointer event — the render
  // would corrupt the timing it is rendering. Deps include `taps`, so each tap
  // restarts the timer; the fit lands during the pause that follows.
  useEffect(() => {
    if (!tapping || taps.length < MIN_TAPS || !duration) return
    const timer = setTimeout(() => {
      const result = fitTapGrid(taps)
      setFit(result.ok ? result.fit : null)
      setError(result.ok ? null : result.error)
    }, TAP_SETTLE_MS)
    return () => clearTimeout(timer)
  }, [tapping, taps, duration])

  // The tapped grid, live — straight from the fit, no correction knobs left
  // to apply.
  const preview = useMemo(() => (fit && duration ? gridFromFit(fit, duration) : null), [fit, duration])

  // Accept: refit from scratch rather than trusting the settled preview, so a
  // user who accepts DURING the settle window (before a preview exists) gets
  // the same grid they would have seen. Rejection keeps the session open with
  // the reason — accepting is the only thing that can fail here, and it fails
  // loudly rather than writing a grid nobody vetted.
  const accept = useCallback(() => {
    if (!duration) return
    const result = fitTapGrid(taps)
    if (!result.ok) {
      setFit(null)
      setError(result.error)
      return
    }
    onGridTapped?.(gridFromFit(result.fit, duration))
    exit()
  }, [taps, duration, onGridTapped, exit])

  return {
    tapping,
    taps,
    fit,
    error,
    preview,
    enter,
    exit,
    record,
    accept,
  }
}
