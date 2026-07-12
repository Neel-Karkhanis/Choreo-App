import { useEffect, useRef } from 'react'
import { useWavesurfer } from '@wavesurfer/react'
import type WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/plugins/regions'

// Beat-grid data, already validated by the caller. downbeatIndices and
// eightCountIndices are indices into beats (schema v2), not timestamps.
export interface GridData {
  beats: number[]
  downbeatIndices: number[]
  eightCountIndices: number[]
}

interface TimelineProps {
  audioUrl: string
  grid?: GridData
  // Later layers (subdivisions, onset overlays) must position themselves via
  // this instance's own time↔pixel mapping — never independent coordinate math.
  onReady?: (wavesurfer: WaveSurfer) => void
}

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function styleRegion(el: HTMLElement | null, styles: Partial<CSSStyleDeclaration>) {
  if (el) Object.assign(el.style, styles)
}

function Timeline({ audioUrl, grid, onReady }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { wavesurfer, isReady, isPlaying, currentTime } = useWavesurfer({
    container: containerRef,
    url: audioUrl,
    height: 96,
  })

  useEffect(() => {
    if (wavesurfer && isReady) onReady?.(wavesurfer)
  }, [wavesurfer, isReady, onReady])

  // The grid is drawn with wavesurfer's RegionsPlugin rather than a hand-rolled
  // overlay: the plugin positions every element as a percentage of wavesurfer's
  // own wrapper, so the time→pixel mapping lives entirely inside wavesurfer and
  // survives container resizes with no coordinate math (and no cached pixels)
  // on our side.
  useEffect(() => {
    if (!wavesurfer || !isReady || !grid) return
    const regions = wavesurfer.registerPlugin(RegionsPlugin.create())
    const duration = wavesurfer.getDuration()
    const { beats, downbeatIndices, eightCountIndices } = grid
    const downbeats = new Set(downbeatIndices)

    // beat_this can emit a final beat at/past the audio end (e.g. exactly at
    // duration); wavesurfer can't position anything there, so such entries are
    // skipped rather than clamped onto the track edge.
    const drawable = (time: number) => time < duration

    // 8-count shading first, so tick markers paint above it. Alternate groups
    // are shaded; the last group may be partial (schema v2) and runs to the
    // end of the track.
    const boundaries = eightCountIndices.map((i) => beats[i]).filter(drawable)
    boundaries.forEach((start, n) => {
      if (n % 2 !== 0) return
      const region = regions.addRegion({
        start,
        end: boundaries[n + 1] ?? duration,
        drag: false,
        resize: false,
        color: 'rgba(110, 110, 110, 0.1)',
      })
      styleRegion(region.element, { pointerEvents: 'none' })
    })

    // Pulse ticks at every beat, bottom-anchored like a ruler; downbeats (each
    // bar's "1") are taller and darker. pointerEvents none keeps wavesurfer's
    // native click-to-seek working through the grid.
    beats.forEach((time, i) => {
      if (!drawable(time)) return
      const region = regions.addRegion({
        start: time,
        end: time,
        drag: false,
        resize: false,
      })
      const emphasized = downbeats.has(i)
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: emphasized
          ? '2px solid rgba(40, 40, 40, 0.85)'
          : '1px solid rgba(110, 110, 110, 0.55)',
        borderRadius: '0',
        height: emphasized ? '55%' : '25%',
        top: emphasized ? '45%' : '75%',
      })
    })

    return () => regions.destroy()
  }, [wavesurfer, isReady, grid])

  const duration = isReady && wavesurfer ? wavesurfer.getDuration() : 0

  return (
    <div className="timeline">
      <div ref={containerRef} />
      <div className="timeline-controls">
        <button onClick={() => wavesurfer?.playPause()} disabled={!isReady}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <span>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}

export default Timeline
