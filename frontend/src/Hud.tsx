import { HearControl, PlayPauseButton, SpeedControl, TimeReadout } from './controls'
import type { EightCountWindow } from './eightCount'
import type { LoopController, SpeedController, Transport } from './playback'
import type { StemMode } from './stemEngine'
import type { GridData } from './types'

/**
 * The playback HUD: transport, position + count, speed, stem mode.
 *
 * IT DRIVES THE ENGINE. Not a video element — StemEngine is the sole clock
 * authority in this app; the video element on the video screen is a SLAVE
 * that follows this clock. Nothing in this component may read time from or
 * write time to a video element, or the app acquires a second clock and the
 * two start drifting the moment either one stalls.
 *
 * It also implements nothing. Every control here is the same component the
 * Timeline renders, handed the same hoisted controllers, so there is no
 * second copy of any of this state to fall out of sync. The loop itself is
 * always active; its region and snap mode are set on the overall timeline's
 * and video scrubber's own A/B handles (see LoopBoundaryHandle) — there is
 * nothing to click here.
 */
export default function Hud({
  transport,
  loop,
  speed,
  grid,
  windows,
  stemMode,
  onStemModeChange,
  disabled = false,
}: {
  transport: Transport
  loop: LoopController
  speed: SpeedController
  grid: GridData | undefined
  windows: EightCountWindow[] | null
  stemMode: StemMode
  onStemModeChange: (mode: StemMode) => void
  disabled?: boolean
}) {
  return (
    <div className="hud">
      <div className="hud-row">
        <PlayPauseButton transport={transport} disabled={disabled} />
        <TimeReadout transport={transport} grid={grid} windows={windows} />
        <SpeedControl speed={speed} />
        <HearControl stemMode={stemMode} onStemModeChange={onStemModeChange} alignEnd />
      </div>
      {loop.error && <p className="error">{loop.error}</p>}
    </div>
  )
}
