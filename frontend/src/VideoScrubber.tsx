import { useCallback, useRef } from 'react'
import LoopBoundaryHandle from './LoopBoundaryHandle'
import { LOOP_BAND_FILL } from './playback'
import type { LoopController, Transport } from './playback'
import type { SnapMode } from './snap'

/**
 * A plain, YouTube-style scrub bar for the video screen — click or drag
 * anywhere on it to seek. Unlike the Timeline's minimap this renders no
 * waveform and no grid; it exists to answer "where am I / where are A and B",
 * not to read the music. Dragging the bar itself calls the same
 * transport.seek() the Timeline's minimap calls; dragging an A/B handle
 * calls loop.setLoopStart/setLoopEnd instead, and tapping one opens the same
 * snap-mode menu — the SAME LoopBoundaryHandle TimelineOverview renders, so
 * the loop and its snap mode can never disagree between the two screens.
 */
export default function VideoScrubber({
  transport,
  loop,
  snapMode,
  onSnapModeChange,
}: {
  transport: Transport
  loop: LoopController
  snapMode: SnapMode
  onSnapModeChange: (mode: SnapMode) => void
}) {
  const trackRef = useRef<HTMLDivElement>(null)

  const timeAtClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current
      if (!el || !transport.duration) return 0
      const rect = el.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      return fraction * transport.duration
    },
    [transport],
  )

  const duration = transport.duration || 0
  const pct = (t: number) => (duration ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0)
  const playedPct = pct(transport.currentTime)
  // Always defined now (see useLoop) — A/B default to the whole track, so the
  // band and its edge handles render from the moment a song opens.
  const aPct = pct(loop.start)
  const bPct = pct(loop.end)

  return (
    <div
      ref={trackRef}
      className="video-scrubber"
      role="slider"
      aria-label="Video position"
      aria-valuemin={0}
      aria-valuemax={duration}
      aria-valuenow={transport.currentTime}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        transport.seek(timeAtClientX(e.clientX))
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) transport.seek(timeAtClientX(e.clientX))
      }}
    >
      <div className="video-scrubber-track">
        <div
          className="video-scrubber-loop-band"
          style={{ left: `${aPct}%`, width: `${bPct - aPct}%`, background: LOOP_BAND_FILL }}
        />
        <div className="video-scrubber-played" style={{ width: `${playedPct}%` }} />
        <div className="video-scrubber-thumb" style={{ left: `${playedPct}%` }} />
        {(['a', 'b'] as const).map((which) => (
          <LoopBoundaryHandle
            key={which}
            which={which}
            className={`video-scrubber-handle video-scrubber-handle-${which}`}
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
