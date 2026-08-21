import { useRef } from 'react'
import Hud from './Hud'
import type { EightCountWindow } from './eightCount'
import type { LoopController, SpeedController, Transport } from './playback'
import type { SnapMode } from './snap'
import type { StemMode } from './stemEngine'
import type { GridData } from './types'
import VideoScrubber from './VideoScrubber'
import { useVideoSync } from './videoSync'

/**
 * The video screen — a genuine second view of a Song, reachable and
 * swappable mid-playback because nothing here owns any state: engine, loop,
 * speed, grid, and playhead all live above this in the Song shell (see
 * Song.tsx), and the HUD below is the same working transport the Timeline
 * uses.
 *
 * The <video> element itself is a SLAVE, never a clock: it is always muted,
 * and its play state, position, and rate are driven entirely from
 * Transport/SpeedController by useVideoSync. StemEngine remains the sole
 * clock authority and the sole audio source in the app — see the Hud
 * comment. frontend/src/videotest.ts is the measurement rig this sync design
 * was built from.
 */
export default function VideoScreen({
  transport,
  loop,
  speed,
  grid,
  windows,
  stemMode,
  onStemModeChange,
  videoUrl,
  snapMode,
  onSnapModeChange,
}: {
  transport: Transport
  loop: LoopController
  speed: SpeedController
  grid: GridData | undefined
  windows: EightCountWindow[] | null
  stemMode: StemMode
  onStemModeChange: (mode: StemMode) => void
  videoUrl: string | null
  snapMode: SnapMode
  onSnapModeChange: (mode: SnapMode) => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const surfaceRef = useRef<HTMLDivElement>(null)
  useVideoSync(videoRef, !!videoUrl, transport, speed)

  return (
    <div className="video-screen">
      <div className="video-surface" ref={surfaceRef}>
        {videoUrl ? (
          <>
            <video ref={videoRef} className="video-pane" src={videoUrl} muted playsInline />
            <VideoScrubber
              transport={transport}
              loop={loop}
              snapMode={snapMode}
              onSnapModeChange={onSnapModeChange}
            />
            <button
              className="video-fullscreen-button"
              aria-label="Fullscreen"
              onClick={() => {
                if (document.fullscreenElement) document.exitFullscreen()
                else surfaceRef.current?.requestFullscreen()
              }}
            >
              {/* design/handoff/Choreo Redesign.dc.html's fullscreen-corners
                  icon, ported verbatim. */}
              <svg width={12} height={12} viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
                  stroke="white"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </>
        ) : (
          <div className="video-pane-empty" role="img" aria-label="no video for this track">
            <span>no video for this track</span>
          </div>
        )}
      </div>
      <Hud
        transport={transport}
        loop={loop}
        speed={speed}
        grid={grid}
        windows={windows}
        stemMode={stemMode}
        onStemModeChange={onStemModeChange}
      />
    </div>
  )
}
