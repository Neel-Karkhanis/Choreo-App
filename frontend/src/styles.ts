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

// Design tokens (design/handoff/README.md → "Design Tokens"), defined once as
// CSS custom properties in index.css (:root / :root[data-theme="dark"]) and
// read back here. This is the ONE place a token's actual color value is
// resolved for JS — everywhere else should reference the CSS variable
// directly (e.g. `var(--color-accent)`) so the browser handles light/dark for
// free. colorToken exists only for call sites that need the resolved string
// itself; it reads getComputedStyle at call time, so it always reflects
// whichever theme is active and never duplicates a token's literal value.
export type ColorToken =
  | 'pageBg'
  | 'text'
  | 'cardBg'
  | 'cardBorder'
  | 'chipBg'
  | 'inputBg'
  | 'inputBorder'
  | 'mutedText'
  | 'divider'
  | 'accent'
  | 'statusCounted'
  | 'statusNeedsCounting'
  | 'needsCountingText'
  | 'loopPinA'
  | 'loopPinB'
  | 'onsetBass'
  | 'onsetDrums'
  | 'importButtonBg'
  | 'waveform'
  | 'stemVocals'
  | 'stemInstrumental'

const CSS_VAR: Record<ColorToken, string> = {
  pageBg: '--color-page-bg',
  text: '--color-text',
  cardBg: '--color-card-bg',
  cardBorder: '--color-card-border',
  chipBg: '--color-chip-bg',
  inputBg: '--color-input-bg',
  inputBorder: '--color-input-border',
  mutedText: '--color-muted-text',
  divider: '--color-divider',
  accent: '--color-accent',
  statusCounted: '--color-status-counted',
  statusNeedsCounting: '--color-status-needs-counting',
  needsCountingText: '--color-needs-counting-text',
  loopPinA: '--color-loop-pin-a',
  loopPinB: '--color-loop-pin-b',
  onsetBass: '--color-onset-bass',
  onsetDrums: '--color-onset-drums',
  importButtonBg: '--color-import-button-bg',
  waveform: '--color-waveform',
  stemVocals: '--color-stem-vocals',
  stemInstrumental: '--color-stem-instrumental',
}

/**
 * Resolve a design token to its current computed color string, live off the
 * cascade (defaults to the document root) — so a value read here always
 * matches whichever [data-theme] is active without the caller branching on
 * darkMode itself.
 */
export function colorToken(token: ColorToken, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(CSS_VAR[token]).trim()
}
