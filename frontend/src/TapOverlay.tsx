import { MIN_TAPS, RECOMMENDED_TAPS, type TapFit } from './tapGrid'

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
  onTap: () => void
  // Starts playback. Wired to the pad itself while paused (see the pad's
  // !isPlaying branch below), since taps can't record against a frozen
  // clock — there is nothing to tap yet until the song is moving.
  onPlay: () => void
  onAccept: () => void
  // What restarting the tap count is called: "Restart" when a saved grid
  // exists to fall back to (still doesn't — clicking it just clears the taps
  // and stays in the tap state, same as "Start over" on first run; see
  // Timeline.tsx's onCancel wiring / tapSession.ts's restart). Two labels for
  // the one action because the copy differs by whether a saved grid exists.
  cancelLabel: string
  onCancel: () => void
}

function TapOverlay({
  taps,
  isPlaying,
  fit,
  fitError,
  onTap,
  onPlay,
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
          <button
            type="button"
            className="tap-pad-play"
            aria-label="Play"
            // Stops the pad's own onPointerDown from also firing onTap — a
            // no-op while paused anyway (record() checks isPlaying), but this
            // keeps the click unambiguously "start playback", nothing else.
            onPointerDown={(e) => e.stopPropagation()}
            onClick={onPlay}
          >
            ▶
          </button>
        )}
      </div>

      <div className="tap-progress">
        <progress value={Math.min(count, RECOMMENDED_TAPS)} max={RECOMMENDED_TAPS} />
        <span>
          {count} / {MIN_TAPS} minimum
          {count < RECOMMENDED_TAPS && ` · ${RECOMMENDED_TAPS} recommended`}
        </span>
      </div>
      {fitError && <p className="error">{fitError}</p>}

      <div className="tap-actions">
        <button onClick={onAccept} disabled={!enough} className="tap-accept">
          Accept
        </button>
        {!fit && enough && <span className="tap-hint">Stop tapping to preview the grid.</span>}
        {/* Pushed to the far right of the row, clear of Accept/the hint —
            it's the reset action, not the next step, so it shouldn't sit
            shoulder-to-shoulder with Accept. */}
        <button onClick={onCancel} style={{ marginLeft: 'auto' }}>
          {cancelLabel}
        </button>
      </div>
    </div>
  )
}

export default TapOverlay
