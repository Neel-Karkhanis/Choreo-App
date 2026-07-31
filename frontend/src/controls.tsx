import { countAt, type EightCountWindow } from './eightCount'
import { formatTime } from './format'
import {
  LOOP_BAND_EDGE,
  SPEED_DEFAULT,
  SPEED_MAX,
  SPEED_MIN,
  SPEED_STEP,
  type LoopController,
  type SpeedController,
  type Transport,
} from './playback'
import { toggleStyle } from './styles'
import type { GridData } from './types'

// The transport controls, once.
//
// Every screen renders these; none of them implements one. The Timeline lays
// them out in its own rows and the HUD lays them out in a bar, but both are
// handed the SAME hoisted controllers and call the SAME imperative methods on
// them, so there is exactly one definition of what "Set A" does. Adding a
// screen must never mean re-deriving a control.
//
// Nothing here reads or writes a video element. The engine is the clock; these
// are buttons over it.

export function PlayPauseButton({
  transport,
  disabled = false,
}: {
  transport: Transport
  disabled?: boolean
}) {
  return (
    <button onClick={transport.toggle} disabled={disabled}>
      {transport.isPlaying ? 'Pause' : 'Play'}
    </button>
  )
}

/**
 * Position readout: clock time, track length, and the count under the playhead.
 *
 * The count is the point — this app is for learning choreography, where "1:23"
 * is far less useful than "phrase 6, count 3". It reads null before the first
 * eight-count start and after the last beat, and shows an em dash there rather
 * than inventing a count.
 */
export function TimeReadout({
  transport,
  grid,
  windows,
}: {
  transport: Transport
  grid: GridData | undefined
  windows: EightCountWindow[] | null
}) {
  const position = countAt(windows, grid, transport.currentTime)
  return (
    <span className="readout">
      <span className="readout-time">
        {formatTime(transport.currentTime)} / {formatTime(transport.duration)}
      </span>
      <span className="readout-count">
        {position ? `phrase ${position.phrase} · count ${position.count}` : 'count —'}
      </span>
    </span>
  )
}

/**
 * A/B loop: set each endpoint at the playhead, toggle it, clear it.
 *
 * PROVISIONAL UI — the touch/gesture model for setting loop points is not
 * decided yet (drag handles?), so these buttons are a thin shim over the loop
 * controller's imperative API. Replace the layout wholesale in the mobile pass;
 * the controller underneath does not change.
 */
export function LoopControls({
  loop,
  transport,
  disabled = false,
}: {
  loop: LoopController
  transport: Transport
  disabled?: boolean
}) {
  const nothingSet = loop.start === null && loop.end === null
  return (
    <>
      <button onClick={() => loop.setLoopStart(transport.currentTime)} disabled={disabled}>
        Set A
      </button>
      <button onClick={() => loop.setLoopEnd(transport.currentTime)} disabled={disabled}>
        Set B
      </button>
      <button
        onClick={() => loop.setEnabled(!loop.enabled)}
        disabled={loop.start === null || loop.end === null}
        aria-pressed={loop.enabled}
        style={toggleStyle(loop.enabled, LOOP_BAND_EDGE)}
      >
        Loop
      </button>
      <button onClick={loop.clearLoop} disabled={nothingSet}>
        Clear
      </button>
      <span>
        {nothingSet
          ? 'no loop'
          : `A ${loop.start !== null ? loop.start.toFixed(2) : '—'} / B ${
              loop.end !== null ? loop.end.toFixed(2) : '—'
            }`}
      </span>
    </>
  )
}

/**
 * Playback speed: live playbackRate on all five stem sources (0.25x–2x). Rate
 * RESAMPLES — pitch shifts with speed — by accepted tradeoff; there is no
 * time-stretch. Grid and loop A/B are track-time and do not move with the rate.
 */
export function SpeedControl({ speed }: { speed: SpeedController }) {
  return (
    <>
      <label>
        Speed{' '}
        <input
          type="range"
          min={SPEED_MIN}
          max={SPEED_MAX}
          step={SPEED_STEP}
          value={speed.speed}
          aria-label="Playback speed"
          onChange={(e) => speed.setSpeed(Number(e.target.value))}
        />
      </label>
      <span>{speed.speed.toFixed(2)}×</span>
      <button onClick={speed.resetSpeed} disabled={speed.speed === SPEED_DEFAULT}>
        1×
      </button>
    </>
  )
}
