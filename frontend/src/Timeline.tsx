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

// Onset data, already validated by the caller. Plain timestamps in seconds,
// independent of the beat grid — not indices, not guaranteed to land on a beat.
export interface OnsetData {
  drums: number[]
  bass: number[]
}

interface TimelineProps {
  audioUrl: string
  grid?: GridData
  onsets?: OnsetData
  // Later layers (subdivisions, onset overlays) must position themselves via
  // this instance's own time↔pixel mapping — never independent coordinate math.
  onReady?: (wavesurfer: WaveSurfer) => void
}

// A beat's pulse tick is suppressed (not drawn) if a drum/bass onset lands
// within this many milliseconds of it — the onset marker visually absorbs
// that tick rather than drawing both on top of each other. Starting guess;
// tune against a real track if it reads too aggressive or too loose.
const ABSORPTION_TOLERANCE_MS = 60

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function styleRegion(el: HTMLElement | null, styles: Partial<CSSStyleDeclaration>) {
  if (el) Object.assign(el.style, styles)
}

function Timeline({ audioUrl, grid, onsets, onReady }: TimelineProps) {
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

    // Onset times (drums + bass) that suppress a nearby pulse tick — computed
    // once up front so the beat loop below is a cheap lookup, not an O(n*m)
    // rescan per beat.
    const drumOnsets = onsets?.drums.filter(drawable) ?? []
    const bassOnsets = onsets?.bass.filter(drawable) ?? []
    const onsetTimes = [...drumOnsets, ...bassOnsets]
    const isAbsorbed = (time: number) =>
      onsetTimes.some((onset) => Math.abs(onset - time) * 1000 < ABSORPTION_TOLERANCE_MS)

    // Pulse ticks at every beat, bottom-anchored like a ruler; downbeats (each
    // bar's "1") are taller and darker. A tick within ABSORPTION_TOLERANCE_MS
    // of a drum/bass onset is skipped — the onset marker below takes its
    // place — and this applies to downbeats too, no special-casing.
    // pointerEvents none keeps wavesurfer's native click-to-seek working
    // through the grid.
    beats.forEach((time, i) => {
      if (!drawable(time)) return
      if (isAbsorbed(time)) return
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

    // Onset markers on top of the grid: bass is the tallest marker, drums
    // shorter than bass but taller than a normal pulse tick. Independent
    // layers — both render even if they land on the same suppressed tick.
    bassOnsets.forEach((time) => {
      const region = regions.addRegion({ start: time, end: time, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: '3px solid rgba(210, 30, 30, 0.9)',
        borderRadius: '0',
        height: '90%',
        top: '5%',
      })
    })
    drumOnsets.forEach((time) => {
      const region = regions.addRegion({ start: time, end: time, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: '2px solid rgba(30, 90, 210, 0.9)',
        borderRadius: '0',
        height: '45%',
        top: '27.5%',
      })
    })

    return () => regions.destroy()
  }, [wavesurfer, isReady, grid, onsets])

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
