import { MIN_TAPS, PHASE_NUDGE_MS, RECOMMENDED_TAPS, bpmOf, type TapFit } from './tapGrid'

// The tap overlay: a panel over the main view, deliberately NOT a separate
// screen. The waveform, the transport, and the playhead all stay live above it
// so the user can hear the song in context, jump to the drop when the intro is
// drumless, and re-hear the 1 after missing it — playback and scrubbing work
// before and during tapping.

export interface TapOverlayProps {
  taps: number[]
  isPlaying: boolean
  // The settled fit, once tapping has paused long enough to compute one. Null
  // while still tapping or when the taps don't fit a tempo.
  fit: TapFit | null
  fitError: string | null
  phaseNudgeMs: number
  countNudge: number
  onTap: () => void
  onNudgePhase: (deltaMs: number) => void
  onNudgeCount: (delta: number) => void
  onAccept: () => void
  // What abandoning the session is called and does: "Cancel" when a saved grid
  // exists to return to, "Start over" on first run — where cancelling just
  // clears the taps and the view stays in the tap state, because with auto
  // detection gone there is nowhere else for an ungridded song to go.
  cancelLabel: string
  onCancel: () => void
}

function TapOverlay({
  taps,
  isPlaying,
  fit,
  fitError,
  phaseNudgeMs,
  countNudge,
  onTap,
  onNudgePhase,
  onNudgeCount,
  onAccept,
  cancelLabel,
  onCancel,
}: TapOverlayProps) {
  const count = taps.length
  const enough = count >= MIN_TAPS
  // Where in the phrase the last tap landed. The user was told to start on a 1,
  // so this is their own count — seeing it cycle 1..8 is how they confirm they
  // are still on the phrase they think they are on.
  const beatInPhrase = count === 0 ? null : ((count - 1) % 8) + 1

  return (
    <div className="tap-panel">
      <div className="tap-panel-head">
        <strong>Tap the 1, then every count you feel.</strong>
        <span className="tap-hint">
          Tap the pad or press Space. The count you tap becomes the grid&apos;s count.
        </span>
      </div>

      <div
        className="tap-pad"
        // A div, not a button, on purpose: a focused <button> also fires on
        // Space, which would double every keyboard tap against the window-level
        // Space handler. pointerdown (not click) so the timestamp is taken at
        // the physical press — this whole feature is a timing measurement.
        role="button"
        tabIndex={-1}
        aria-label="Tap a count"
        aria-disabled={!isPlaying}
        data-armed={isPlaying}
        onPointerDown={(e) => {
          e.preventDefault()
          onTap()
        }}
      >
        {isPlaying ? (
          <>
            <span className="tap-pad-count">{beatInPhrase ?? '–'}</span>
            <span className="tap-pad-label">
              {count === 0 ? 'tap on the 1' : `${count} tap${count === 1 ? '' : 's'}`}
            </span>
          </>
        ) : (
          <span className="tap-pad-label">Press Play, then tap each count</span>
        )}
      </div>

      <div className="tap-progress">
        <progress value={Math.min(count, RECOMMENDED_TAPS)} max={RECOMMENDED_TAPS} />
        <span>
          {count} / {MIN_TAPS} minimum
          {count < RECOMMENDED_TAPS && ` · ${RECOMMENDED_TAPS} recommended`}
        </span>
      </div>
      {/*
        Not decoration: the fit is a SLOPE, so its error scales inversely with
        how long the tap window is. At 16 taps a steady tapper still lands ~1/3
        of a beat off by 3:00 in the worst decile; at 32 that drops to ~1/8.
        Hence the floor at 16 and the nagging up to 32.
      */}
      {enough && count < RECOMMENDED_TAPS && (
        <p className="tap-note">
          Enough to accept — but keep going to {RECOMMENDED_TAPS} and the grid will hold
          much better at the end of the track.
        </p>
      )}

      {fit && (
        <div className="tap-fit">
          <span>
            <strong>{bpmOf(fit).toFixed(1)}</strong> counts/min
          </span>
          <span>
            {fit.kept} taps used{fit.dropped > 0 && `, ${fit.dropped} off-beat dropped`}
          </span>
        </div>
      )}
      {fitError && <p className="error">{fitError}</p>}

      {/*
        Nudge. The fit cannot recover a CONSTANT offset (input latency + tapping
        early apply equally to every tap), and snapping to onsets is not an
        option — kicks in syncopated genres are deliberately off-grid, so
        snapping would marry beats to non-beat events exactly where this feature
        is most needed. So: the user's eyes, and two cheap knobs.
      */}
      <div className="tap-nudge">
        <span>Nudge</span>
        <button onClick={() => onNudgePhase(-PHASE_NUDGE_MS)} disabled={!fit}>
          −{PHASE_NUDGE_MS}ms
        </button>
        <button onClick={() => onNudgePhase(PHASE_NUDGE_MS)} disabled={!fit}>
          +{PHASE_NUDGE_MS}ms
        </button>
        <span className="tap-nudge-value">
          {phaseNudgeMs > 0 ? '+' : ''}
          {phaseNudgeMs}ms
        </span>
        <button onClick={() => onNudgeCount(-1)} disabled={!fit}>
          ◀ count
        </button>
        <button onClick={() => onNudgeCount(1)} disabled={!fit}>
          count ▶
        </button>
        <span className="tap-nudge-value">
          {countNudge > 0 ? '+' : ''}
          {countNudge}
        </span>
      </div>

      <div className="tap-actions">
        <button onClick={onAccept} disabled={!enough} className="tap-accept">
          Accept grid
        </button>
        <button onClick={onCancel}>{cancelLabel}</button>
        {!fit && enough && <span className="tap-hint">Stop tapping to preview the grid.</span>}
      </div>
    </div>
  )
}

export default TapOverlay
