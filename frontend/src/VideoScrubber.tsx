import { useCallback, useRef } from 'react'
import { LOOP_BAND_EDGE, LOOP_BAND_FILL, LOOP_DISABLED_OPACITY } from './playback'
import type { LoopController, Transport } from './playback'

/**
 * A plain, YouTube-style scrub bar for the video screen — click or drag
 * anywhere on it to seek. Unlike the Timeline's minimap this renders no
 * waveform and no grid; it exists to answer "where am I / where are A and B",
 * not to read the music. Dragging calls the same transport.seek() the
 * Timeline's minimap calls, so a seek from here is indistinguishable
 * downstream from one made anywhere else.
 */
export default function VideoScrubber({
  transport,
  loop,
}: {
  transport: Transport
  loop: LoopController
}) {
  const trackRef = useRef<HTMLDivElement>(null)

  const seekFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current
      if (!el || !transport.duration) return
      const rect = el.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      transport.seek(fraction * transport.duration)
    },
    [transport],
  )

  const duration = transport.duration || 0
  const pct = (t: number) => Math.min(100, Math.max(0, (t / duration) * 100))
  const playedPct = duration ? pct(transport.currentTime) : 0
  const aPct = loop.start !== null && duration ? pct(loop.start) : null
  const bPct = loop.end !== null && duration ? pct(loop.end) : null
  const loopOpacity = loop.enabled ? 1 : LOOP_DISABLED_OPACITY

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
        seekFromClientX(e.clientX)
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) seekFromClientX(e.clientX)
      }}
    >
      <div className="video-scrubber-track">
        {aPct !== null && bPct !== null && (
          <div
            className="video-scrubber-loop-band"
            style={{ left: `${aPct}%`, width: `${bPct - aPct}%`, background: LOOP_BAND_FILL, opacity: loopOpacity }}
          />
        )}
        <div className="video-scrubber-played" style={{ width: `${playedPct}%` }} />
        {aPct !== null && (
          <div
            className="video-scrubber-marker"
            style={{ left: `${aPct}%`, background: LOOP_BAND_EDGE, opacity: loopOpacity }}
          />
        )}
        {bPct !== null && (
          <div
            className="video-scrubber-marker"
            style={{ left: `${bPct}%`, background: LOOP_BAND_EDGE, opacity: loopOpacity }}
          />
        )}
        <div className="video-scrubber-thumb" style={{ left: `${playedPct}%` }} />
      </div>
    </div>
  )
}
