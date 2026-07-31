import { useEffect, useRef } from 'react'
import type { SpeedController, Transport } from './playback'

// Drives an HTML <video> element as a SLAVE to the engine clock exposed
// through Transport/SpeedController — never the other way around. The
// element is always muted; audio stays owned entirely by the StemEngine, so
// there is exactly one clock and exactly one audio source in the app.
//
// This is the production shape of the design measured in videotest.ts: a
// loop wrap and a manual scrub both just look like a large, sudden gap
// between where the video sits and where the engine says it should be, so
// one continuous offset-correction loop handles both — nudge the video's
// playbackRate fractionally to close small drift, hard-reseek once past a
// threshold no amount of nudging would close in reasonable time.

// Multiplicative correction: the same 1% at every engine rate.
const NUDGE = 1.01
const NUDGE_ON_S = 0.03
const NUDGE_OFF_S = 0.01
const HARD_RESYNC_S = 0.25
// Below this, the video and the engine are considered in agreement — used
// for the paused-state direct seek so float rounding never causes a
// re-seek on every render.
const PAUSED_EPSILON_S = 0.05

export function useVideoSync(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  hasVideo: boolean,
  transport: Transport,
  speed: SpeedController,
): void {
  // The last (engine time, wall clock) pair seen from Transport, so the video
  // frame loop can project "what the engine reads right now" without needing
  // the raw engine object — Transport's rAF-polled currentTime is fresh
  // enough for the thresholds above.
  const anchorRef = useRef({ time: transport.currentTime, wallMs: performance.now() })
  const nudgeRef = useRef(1)

  useEffect(() => {
    anchorRef.current = { time: transport.currentTime, wallMs: performance.now() }
  }, [transport.currentTime])

  // The isPlaying EDGE: play/pause the element, and pre-seek on entering play
  // so it doesn't flash from wherever it last sat before the drift-correction
  // loop gets its first frame.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !hasVideo) return
    if (transport.isPlaying) {
      if (Math.abs(video.currentTime - transport.currentTime) > PAUSED_EPSILON_S) {
        video.currentTime = transport.currentTime
      }
      video.play().catch(() => {
        // Autoplay rejection (no user gesture yet, or a stale play() racing a
        // pause) — the next isPlaying edge or drift correction retries.
      })
    } else {
      video.pause()
    }
    // transport.currentTime deliberately excluded — this effect only reacts
    // to the isPlaying edge. Position sync while paused is the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, hasVideo, transport.isPlaying])

  // A seek while paused (scrub, loop-A jump, grid nudge) — the frame-callback
  // loop below only runs while playing, so this is the only path that moves
  // the video while paused.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !hasVideo || transport.isPlaying) return
    if (Math.abs(video.currentTime - transport.currentTime) > PAUSED_EPSILON_S) {
      video.currentTime = transport.currentTime
    }
  }, [videoRef, hasVideo, transport.isPlaying, transport.currentTime])

  // Continuous drift correction while playing, via requestVideoFrameCallback
  // where available (accurate per-displayed-frame timing), falling back to a
  // requestAnimationFrame poll otherwise.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !hasVideo || !transport.isPlaying) return

    let cancelled = false
    let handle: number | null = null

    const projected = (atWallMs: number) => {
      const { time, wallMs } = anchorRef.current
      return time + ((atWallMs - wallMs) / 1000) * speed.speed
    }

    const correct = (mediaTime: number, atWallMs: number) => {
      const offset = mediaTime - projected(atWallMs)
      if (Math.abs(offset) > HARD_RESYNC_S) {
        video.currentTime = projected(performance.now())
        nudgeRef.current = 1
      } else if (Math.abs(offset) > NUDGE_ON_S) {
        nudgeRef.current = offset > 0 ? 1 / NUDGE : NUDGE
      } else if (Math.abs(offset) < NUDGE_OFF_S) {
        nudgeRef.current = 1
      }
      const target = speed.speed * nudgeRef.current
      if (Math.abs(video.playbackRate - target) > 0.0001) video.playbackRate = target
    }

    if (typeof video.requestVideoFrameCallback === 'function') {
      const onFrame = (now: number, metadata: VideoFrameCallbackMetadata) => {
        if (cancelled) return
        correct(metadata.mediaTime, metadata.expectedDisplayTime ?? now)
        handle = video.requestVideoFrameCallback(onFrame)
      }
      handle = video.requestVideoFrameCallback(onFrame)
      return () => {
        cancelled = true
        if (handle !== null) video.cancelVideoFrameCallback(handle)
      }
    }

    const tick = () => {
      if (cancelled) return
      correct(video.currentTime, performance.now())
      handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => {
      cancelled = true
      if (handle !== null) cancelAnimationFrame(handle)
    }
  }, [videoRef, hasVideo, transport.isPlaying, speed.speed])
}
