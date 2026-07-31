import type { CSSProperties } from 'react'

// Inline style helpers shared across control surfaces. Separate from
// controls.tsx so that file exports components only — a module mixing
// components with plain exports opts out of React Fast Refresh.

/**
 * Active-state fill for toggle buttons: filled means on, in the toggle's own
 * colour so the button says which layer it controls. Placeholder styling — a
 * later design pass owns polish; only the active/inactive distinction matters.
 */
export function toggleStyle(active: boolean, color: string): CSSProperties {
  return active ? { backgroundColor: color, borderColor: color, color: 'white' } : {}
}
