import { useCallback, useRef } from 'react'
import LoopBoundaryHandle from './LoopBoundaryHandle'
import { LOOP_BAND_FILL, type LoopController } from './playback'
import type { SnapMode } from './snap'

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
 * and playback always loops between them. Dragging a handle moves it;
 * tapping one (no drag) opens a menu to pick what it snaps to — beat,
 * 4-count, or 8-count (see LoopBoundaryHandle) — there are no Set A/Set
 * B/Loop/Clear buttons anymore.
 */
export default function TimelineOverview({
  duration,
  currentTime,
  loop,
  onSeek,
  snapMode,
  onSnapModeChange,
}: {
  duration: number
  currentTime: number
  loop: LoopController
  onSeek: (time: number) => void
  snapMode: SnapMode
  onSnapModeChange: (mode: SnapMode) => void
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
          <LoopBoundaryHandle
            key={which}
            which={which}
            className={`timeline-overview-handle timeline-overview-handle-${which}`}
            leftPct={which === 'a' ? aPct : bPct}
            time={which === 'a' ? loop.start : loop.end}
            duration={duration}
            loop={loop}
            timeAtClientX={timeAtClientX}
            snapMode={snapMode}
            onSnapModeChange={onSnapModeChange}
          />
        ))}
      </div>
    </div>
  )
}
