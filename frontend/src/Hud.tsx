import { LoopControls, PlayPauseButton, SpeedControl } from './controls'
import { TimeReadout } from './controls'
import type { EightCountWindow } from './eightCount'
import type { LoopController, SpeedController, Transport } from './playback'
import type { GridData } from './types'

/**
 * The playback HUD: transport, position + count, loop A/B, speed.
 *
 * IT DRIVES THE ENGINE. Not a video element — not now, and not when a real
 * video pane replaces the placeholder above it. StemEngine is the sole clock
 * authority in this app; a video element, once there is one, will be a SLAVE
 * that follows this clock. Nothing in this component may read time from or
 * write time to a video element, or the app acquires a second clock and the
 * two start drifting the moment either one stalls.
 *
 * It also implements nothing. Every control here is the same component the
 * Timeline renders, handed the same hoisted controllers — so "Set A" on this
 * screen and "Set A" on the Timeline are literally one function, and there is
 * no second copy of loop/speed state to fall out of sync.
 */
export default function Hud({
  transport,
  loop,
  speed,
  grid,
  windows,
  disabled = false,
}: {
  transport: Transport
  loop: LoopController
  speed: SpeedController
  grid: GridData | undefined
  windows: EightCountWindow[] | null
  disabled?: boolean
}) {
  return (
    <div className="hud">
      <div className="hud-row">
        <PlayPauseButton transport={transport} disabled={disabled} />
        <TimeReadout transport={transport} grid={grid} windows={windows} />
      </div>
      <div className="hud-row">
        <LoopControls loop={loop} transport={transport} disabled={disabled} />
      </div>
      <div className="hud-row">
        <SpeedControl speed={speed} />
      </div>
      {loop.error && <p className="error">{loop.error}</p>}
    </div>
  )
}
