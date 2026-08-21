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
import { STEM_MODES, type StemMode } from './stemEngine'
import { toggleStyle } from './styles'
import type { GridData } from './types'

// Every quarter-x point between the engine's bounds — both the slider's step
// and the "dial" ticks rendered on it via <datalist>.
const SPEED_DIALS: number[] = []
for (let v = SPEED_MIN; v <= SPEED_MAX + 1e-9; v += SPEED_STEP) {
  SPEED_DIALS.push(Math.round(v * 100) / 100)
}
const SPEED_ACTIVE_COLOR = 'var(--color-accent)'

// Stem mode selector's labels.
const STEM_MODE_LABELS: Record<StemMode, string> = {
  all: 'All',
  vocals: 'Vocals',
  drums: 'Drums',
  bass: 'Bass',
  instrumental: 'Instrumental',
}

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
    <button
      className="play-pause-button"
      onClick={transport.toggle}
      disabled={disabled}
      aria-label={transport.isPlaying ? 'Pause' : 'Play'}
      // The play triangle's own ink sits left-of-center in its glyph box;
      // nudge it so the button reads visually centered in both states.
      style={{ paddingLeft: transport.isPlaying ? 0 : 2 }}
    >
      {transport.isPlaying ? '❚❚' : '▶'}
    </button>
  )
}

/**
 * Position readout: clock time, track length, and the count under the playhead.
 *
 * The count is the point — this app is for learning choreography, where "1:23"
 * is far less useful than "phrase 6, count 3". It reads null before the first
 * eight-count start and after the last beat, and shows "no count" there rather
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
        {position ? `phrase ${position.phrase} · count ${position.count}` : 'no count'}
      </span>
    </span>
  )
}

/**
 * Playback speed: live playbackRate on all five stem sources (0.25x–2x). Rate
 * RESAMPLES — pitch shifts with speed — by accepted tradeoff; there is no
 * time-stretch. Grid and loop A/B are track-time and do not move with the rate.
 *
 * Collapsed behind a button (design/handoff/README.md: "speed dial"): the
 * slider itself only needs to be reachable while actually changing speed,
 * not parked in the controls row at all times. Only the trigger button is
 * restyled to the design's icon/active-state look here — the picker
 * (dropdown-list-speed below) is explicitly out of scope: the design only
 * mocks the dial's toggle, not a real picker UI, so this reuses the
 * app's pre-existing slider-based one unchanged, pending its own design pass.
 */
export function SpeedControl({ speed }: { speed: SpeedController }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useCloseOnOutsideClick(open, setOpen, ref)

  return (
    <div className="dropdown" ref={ref}>
      <button
        className="speed-dial-button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="true"
        aria-expanded={open}
        aria-label={`Playback speed: ${speed.speed.toFixed(2)}×`}
        style={toggleStyle(speed.speed !== SPEED_DEFAULT, SPEED_ACTIVE_COLOR)}
      >
        {/* design/handoff/Choreo Redesign.dc.html's speed-dial icon, ported
            verbatim (arc + hand + hub); stroke/fill use currentColor so
            toggleStyle's active-state color above drives the icon too. */}
        <svg width={15} height={15} viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4 18a8 8 0 1 1 16 0" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
          <line x1={12} y1={18} x2={16} y2={12} stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
          <circle cx={12} cy={18} r={1.3} fill="currentColor" />
        </svg>
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

/**
 * Stem mode: which of the five mixes (all/vocals/drums/bass/instrumental)
 * the engine plays and, on the Timeline, which peaks its waveform draws.
 * Every screen renders the SAME control over the SAME hoisted stemMode state,
 * so switching what you hear on the Timeline and switching it on the video
 * screen are the same action, never two copies that can disagree.
 *
 * A plain native <select> (design/handoff/README.md: "Hear" stem-mode
 * select) — the design's own five options (All/Vocals/Drums/Bass/
 * Instrumental) are exactly STEM_MODES, so this is a direct port with no
 * option dropped or invented.
 */
export function HearControl({
  stemMode,
  onStemModeChange,
  alignEnd = false,
}: {
  stemMode: StemMode
  onStemModeChange: (mode: StemMode) => void
  // Pushes this control to the row's right edge (design: Video tab's HUD
  // row has only Hear on the right; the Timeline tab's row already gets that
  // push from the Onsets select ahead of it — see Timeline.tsx).
  alignEnd?: boolean
}) {
  return (
    <label className="control-select-label" style={alignEnd ? { marginLeft: 'auto' } : undefined}>
      Hear
      <select
        className="control-select"
        value={stemMode}
        onChange={(e) => onStemModeChange(e.target.value as StemMode)}
      >
        {STEM_MODES.map((mode) => (
          <option key={mode} value={mode}>
            {STEM_MODE_LABELS[mode]}
          </option>
        ))}
      </select>
    </label>
  )
}
