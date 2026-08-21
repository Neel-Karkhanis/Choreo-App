import { useRef, useState } from 'react'
import { useCloseOnOutsideClick } from './dropdown'
import { type LoopController } from './playback'
import { BOUNDARY_SNAP_MODES, SNAP_MODE_LABELS, type SnapMode } from './snap'
import { toggleStyle } from './styles'

// Loop pin colors: A is yellow, B is red (both re-hued more than once at the
// user's explicit request — see index.css's --color-loop-pin-a/-b comment
// for why only the hue moves, never the lightness/chroma) — the SAME
// per-handle pair the main timeline's own A/B markers use too (see
// playback.ts's LOOP_BAND_EDGE_A/EDGE_B), so a marker on the waveform and the
// pin that set it always match. Both are var() references consumed as
// literal DOM style strings, so a theme change repaints them without any JS
// involvement.
const PIN_COLOR: Record<'a' | 'b', string> = {
  a: 'var(--color-loop-pin-a)',
  b: 'var(--color-loop-pin-b)',
}

// A small dead zone (in raw seconds) enforced around each handle so a drag
// can never cross the other one. Clamped here rather than relying on
// useLoop's own start>=end rejection, which would otherwise auto-swap A and B
// mid-drag — confusing for a direct-manipulation gesture, even though it's
// the right behavior for a programmatic setLoop call.
const HANDLE_GAP_S = 0.05

// Pointer movement below this (px) between down and up reads as a TAP, not a
// drag. Below that threshold the gesture opens the snap-mode menu instead of
// moving the handle — the same disambiguation a scrollable list uses to tell
// "tap" from "swipe".
const CLICK_MOVE_PX = 4

/**
 * One loop A/B handle, shared by TimelineOverview and VideoScrubber so the
 * drag-to-move and tap-to-choose-snap-mode gestures exist in exactly one
 * place: two independent copies of pointer-threshold math is exactly the
 * kind of drift this app's loop state was hoisted to Song.tsx to avoid (see
 * playback.ts's useLoop doc).
 *
 * Dragging calls loop.setLoopStart/setLoopEnd, unchanged from before.
 * Tapping (a pointerdown/up with negligible movement) instead opens a small
 * menu offering the snap modes the loop understands — beat, 4-count,
 * 8-count — reusing the SAME snapMode/onSnapModeChange the Song shell already
 * threads through both screens, so picking a mode from either handle, in
 * either screen, changes snapping everywhere at once.
 */
export default function LoopBoundaryHandle({
  which,
  className,
  leftPct,
  time,
  duration,
  loop,
  timeAtClientX,
  snapMode,
  onSnapModeChange,
}: {
  which: 'a' | 'b'
  className: string
  leftPct: number
  time: number
  duration: number
  loop: LoopController
  timeAtClientX: (clientX: number) => number
  snapMode: SnapMode
  onSnapModeChange: (mode: SnapMode) => void
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const handleRef = useRef<HTMLDivElement>(null)
  useCloseOnOutsideClick(menuOpen, setMenuOpen, handleRef)

  // Tracked per-gesture, not per-render: whether the CURRENT pointerdown has
  // moved past the click threshold yet.
  const dragged = useRef(false)
  const downAt = useRef({ x: 0, y: 0 })

  return (
    // Prompt 8 (touch pass): this outer box is the 44x44 invisible HIT
    // TARGET, not the visual pin — design/handoff/README.md's 14px pin is
    // well under the WCAG/Apple/Google 44px touch-target floor on its own.
    // Every pointer handler stays here unchanged; only the visible circle
    // (.loop-pin-dot, below) moved into a nested span, so the drag/tap
    // gestures and their dead-zone/threshold math are untouched — this is a
    // hit-area split, not a behavior change.
    <div
      ref={handleRef}
      className={`loop-pin-hit ${className}`}
      role="slider"
      aria-label={which === 'a' ? 'Loop start' : 'Loop end'}
      aria-valuemin={0}
      aria-valuemax={duration}
      aria-valuenow={time}
      style={{ left: `${leftPct}%` }}
      onPointerDown={(e) => {
        e.stopPropagation()
        dragged.current = false
        downAt.current = { x: e.clientX, y: e.clientY }
        e.currentTarget.setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return
        e.stopPropagation()
        if (!dragged.current) {
          const dx = e.clientX - downAt.current.x
          const dy = e.clientY - downAt.current.y
          if (Math.hypot(dx, dy) <= CLICK_MOVE_PX) return
          dragged.current = true
        }
        const t = timeAtClientX(e.clientX)
        if (which === 'a') loop.setLoopStart(Math.min(t, loop.end - HANDLE_GAP_S))
        else loop.setLoopEnd(Math.max(t, loop.start + HANDLE_GAP_S))
      }}
      onPointerUp={(e) => {
        e.stopPropagation()
        if (!dragged.current) setMenuOpen((v) => !v)
      }}
    >
      {/* The actual 14px visual pin. No pointer handlers of its own — a
          press anywhere in the 44px hit box above bubbles up to it as
          normal DOM event bubbling, so this stays a pure paint layer. */}
      <span className="loop-pin-dot" style={{ background: PIN_COLOR[which] }}>
        {which.toUpperCase()}
        {/* Nested inside the 14px dot (not the 44px hit box) so
            .loop-snap-menu's `top:100%` still opens tight against the
            VISIBLE pin, not 30px further down against the invisible hit
            target's edge. */}
        {menuOpen && (
          <div
            className="dropdown-list loop-snap-menu"
            role="menu"
            onPointerDown={(e) => e.stopPropagation()}
          >
            {BOUNDARY_SNAP_MODES.map((mode) => (
              <button
                key={mode}
                role="menuitemradio"
                aria-checked={snapMode === mode}
                onClick={() => {
                  onSnapModeChange(mode)
                  setMenuOpen(false)
                }}
                style={toggleStyle(snapMode === mode, PIN_COLOR[which])}
              >
                {SNAP_MODE_LABELS[mode]}
              </button>
            ))}
          </div>
        )}
      </span>
    </div>
  )
}
