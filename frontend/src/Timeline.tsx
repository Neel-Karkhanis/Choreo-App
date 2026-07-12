import { useEffect, useRef } from 'react'
import { useWavesurfer } from '@wavesurfer/react'
import type WaveSurfer from 'wavesurfer.js'

interface TimelineProps {
  audioUrl: string
  // Later layers (beat grid, overlays) must position themselves via this
  // instance's own time↔pixel mapping — never independent coordinate math.
  onReady?: (wavesurfer: WaveSurfer) => void
}

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function Timeline({ audioUrl, onReady }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { wavesurfer, isReady, isPlaying, currentTime } = useWavesurfer({
    container: containerRef,
    url: audioUrl,
    height: 96,
  })

  useEffect(() => {
    if (wavesurfer && isReady) onReady?.(wavesurfer)
  }, [wavesurfer, isReady, onReady])

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
