// Display formatting shared by every control surface. One implementation, so
// the timeline and the HUD can never drift into showing the same instant two
// different ways.

/** m:ss — the readout format for playback position and track length. */
export function formatTime(totalSeconds: number): string {
  const safe = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0
  const minutes = Math.floor(safe / 60)
  const seconds = Math.floor(safe % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}
