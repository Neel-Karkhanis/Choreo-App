import { useRef, useState } from 'react'
import { useCloseOnOutsideClick } from './dropdown'
import { countAt, type EightCountWindow } from './eightCount'
import { formatTime } from './format'
import {
  SPEED_DEFAULT,
  SPEED_MAX,
  SPEED_MIN,
  SPEED_STEP,
  type SpeedController,
  type Transport,
} from './playback'
import { toggleStyle } from './styles'
import type { GridData } from './types'

// Every quarter-x point between the engine's bounds — both the slider's step
// and the "dial" ticks rendered on it via <datalist>.
const SPEED_DIALS: number[] = []
for (let v = SPEED_MIN; v <= SPEED_MAX + 1e-9; v += SPEED_STEP) {
  SPEED_DIALS.push(Math.round(v * 100) / 100)
}
const SPEED_ACTIVE_COLOR = 'rgba(210, 120, 20, 0.9)'

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
 * Playback speed: live playbackRate on all five stem sources (0.25x–2x). Rate
 * RESAMPLES — pitch shifts with speed — by accepted tradeoff; there is no
 * time-stretch. Grid and loop A/B are track-time and do not move with the rate.
 *
 * Collapsed behind a button showing the current rate, same as the Onset and
 * Hear dropdowns: the slider itself only needs to be reachable while actually
 * changing speed, not parked in the controls row at all times.
 */
export function SpeedControl({ speed }: { speed: SpeedController }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useCloseOnOutsideClick(open, setOpen, ref)

  return (
    <div className="dropdown" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="true"
        aria-expanded={open}
        style={toggleStyle(speed.speed !== SPEED_DEFAULT, SPEED_ACTIVE_COLOR)}
      >
        Speed: {speed.speed.toFixed(2)}× ▾
      </button>
      {open && (
        <div className="dropdown-list dropdown-list-speed" role="menu">
          <input
            type="range"
            min={SPEED_MIN}
            max={SPEED_MAX}
            step={SPEED_STEP}
            list="speed-dials"
            value={speed.speed}
            aria-label="Playback speed"
            onChange={(e) => speed.setSpeed(Number(e.target.value))}
          />
          <datalist id="speed-dials">
            {SPEED_DIALS.map((v) => (
              <option key={v} value={v} />
            ))}
          </datalist>
          <div className="speed-dial-labels">
            {SPEED_DIALS.map((v) => (
              <span key={v}>{v}×</span>
            ))}
          </div>
          <button onClick={speed.resetSpeed} disabled={speed.speed === SPEED_DEFAULT}>
            Reset to 1×
          </button>
        </div>
      )}
    </div>
  )
}
