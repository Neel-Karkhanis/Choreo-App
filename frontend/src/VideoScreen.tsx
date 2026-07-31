import { useRef } from 'react'
import Hud from './Hud'
import type { EightCountWindow } from './eightCount'
import type { LoopController, SpeedController, Transport } from './playback'
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
  videoUrl,
}: {
  transport: Transport
  loop: LoopController
  speed: SpeedController
  grid: GridData | undefined
  windows: EightCountWindow[] | null
  videoUrl: string | null
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  useVideoSync(videoRef, !!videoUrl, transport, speed)

  return (
    <div className="video-screen">
      {videoUrl ? (
        <>
          <video ref={videoRef} className="video-pane" src={videoUrl} muted playsInline />
          <VideoScrubber transport={transport} loop={loop} />
        </>
      ) : (
        <div className="video-pane-empty" role="img" aria-label="no video for this track">
          <span>no video for this track</span>
        </div>
      )}
      <Hud transport={transport} loop={loop} speed={speed} grid={grid} windows={windows} />
    </div>
  )
}
