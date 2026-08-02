import { useCallback, useRef } from 'react'
import { LOOP_BAND_EDGE, LOOP_BAND_FILL, type LoopController } from './playback'

// A small dead zone (in raw seconds) enforced around each handle so a drag
// can never cross the other one. Clamped here in the component rather than
// relying on useLoop's own start>=end rejection, which would otherwise
// auto-swap A and B mid-drag — confusing for a direct-manipulation gesture,
// even though it's the right behavior for the "Set A"/"Set B" buttons.
const HANDLE_GAP_S = 0.05

/**
 * The overall timeline: a plain whole-song position bar, not a waveform view.
 * It replaced a second wavesurfer instance (a "minimap" with its own peaks,
 * 8-count shading, and a read-only viewport box) — all of that was scrapped
 * for exactly three things: a playhead, and the loop's A and B handles,
 * draggable directly on the bar. The zoomed-in Timeline above this keeps its
 * own waveform, grid, and loop-band rendering untouched; this bar only
 * answers "where in the whole song am I / where are A and B".
 *
 * A and B are ALWAYS present (see useLoop — they default to the full track)
 * and playback always loops between them; dragging is the ONLY way to change
 * them now — there are no Set A/Set B/Loop/Clear buttons anymore. Dragging
 * calls loop.setLoopStart/setLoopEnd, which is always beat-snapped, so a
 * handle can only ever come to rest on a beat.
 */
export default function TimelineOverview({
  duration,
  currentTime,
  loop,
  onSeek,
}: {
  duration: number
  currentTime: number
  loop: LoopController
  onSeek: (time: number) => void
}) {
  const trackRef = useRef<HTMLDivElement>(null)

  const timeAtClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current
      if (!el || !duration) return 0
      const rect = el.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      return fraction * duration
    },
    [duration],
  )

  const pct = (t: number) => (duration ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0)
  const playedPct = pct(currentTime)
  const aPct = pct(loop.start)
  const bPct = pct(loop.end)

  return (
    <div className="timeline-overview">
      <div
        ref={trackRef}
        className="timeline-overview-track"
        role="slider"
        aria-label="Song position"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={currentTime}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          onSeek(timeAtClientX(e.clientX))
        }}
        onPointerMove={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) onSeek(timeAtClientX(e.clientX))
        }}
      >
        <div
          className="timeline-overview-loop-band"
          style={{
            left: `${aPct}%`,
            width: `${Math.max(0, bPct - aPct)}%`,
            background: LOOP_BAND_FILL,
          }}
        />
        <div className="timeline-overview-played" style={{ width: `${playedPct}%` }} />
        <div className="timeline-overview-playhead" style={{ left: `${playedPct}%` }} />
        {(['a', 'b'] as const).map((which) => (
          <div
            key={which}
            className={`timeline-overview-handle timeline-overview-handle-${which}`}
            role="slider"
            aria-label={which === 'a' ? 'Loop start' : 'Loop end'}
            aria-valuemin={0}
            aria-valuemax={duration}
            aria-valuenow={which === 'a' ? loop.start : loop.end}
            style={{ left: `${which === 'a' ? aPct : bPct}%`, background: LOOP_BAND_EDGE }}
            onPointerDown={(e) => {
              e.stopPropagation()
              e.currentTarget.setPointerCapture(e.pointerId)
            }}
            onPointerMove={(e) => {
              if (!e.currentTarget.hasPointerCapture(e.pointerId)) return
              e.stopPropagation()
              const t = timeAtClientX(e.clientX)
              if (which === 'a') loop.setLoopStart(Math.min(t, loop.end - HANDLE_GAP_S))
              else loop.setLoopEnd(Math.max(t, loop.start + HANDLE_GAP_S))
            }}
          >
            {which.toUpperCase()}
          </div>
        ))}
      </div>
    </div>
  )
}
