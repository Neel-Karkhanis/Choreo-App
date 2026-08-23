import { useEffect, useState } from 'react'

// Accent color: a device preference, not song data — same reasoning as
// theme.ts's dark-mode flag, a plain localStorage string is the whole
// persistence story. This file mirrors theme.ts's shape deliberately (read/
// apply/applyStored/useX) rather than inventing a second pattern for what is,
// structurally, the same kind of setting.
const STORAGE_KEY = 'choreo-accent-color'

export interface AccentOption {
  id: string
  label: string
  hex: string
}

// Curated, not free-form (at the user's explicit request): a free color
// picker could land on a hue too close to --color-onset-bass-marker/
// --color-onset-drums-marker's ~230deg/~145deg, undoing the contrast work
// those went through (see their own comment in index.css for exactly what
// that looked like when it went wrong). Every option below sits at least
// ~70deg from both, at a similar lightness/chroma neighborhood to the
// original purple (L~0.56-0.68, C~0.14-0.25) so switching accent doesn't
// also change how BOLD the app reads, only its hue.
//
// 'purple' reproduces the app's --color-accent default (index.css :root)
// exactly, byte for byte — it's the default, and picking it back after
// trying another option must be indistinguishable from never having opened
// Settings at all. Re-hued from the original vivid #9333ea to #8d41bc —
// same ~310° hue as the user-supplied #b57edc, darkened/deepened from it,
// then nudged slightly back up in lightness from the #7e30ab that landed
// on — at their explicit request; the two must keep mirroring each other.
export const ACCENT_OPTIONS: AccentOption[] = [
  { id: 'purple', label: 'Purple', hex: '#8d41bc' },
  { id: 'rose', label: 'Rose', hex: '#c13b9f' },
  { id: 'crimson', label: 'Crimson', hex: '#d73240' },
  { id: 'amber', label: 'Amber', hex: '#cf7b26' },
]

const DEFAULT_ID = ACCENT_OPTIONS[0].id

function optionFor(id: string): AccentOption {
  return ACCENT_OPTIONS.find((option) => option.id === id) ?? ACCENT_OPTIONS[0]
}

function readStored(): string {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored && ACCENT_OPTIONS.some((option) => option.id === stored) ? stored : DEFAULT_ID
  } catch {
    return DEFAULT_ID
  }
}

// An inline style on the root element wins over index.css's stylesheet rule
// for the same custom property, so this is the whole override mechanism —
// no [data-attribute] selector or second stylesheet needed. Every Timeline
// color meant to track accent is written in index.css as color-mix(...)
// against var(--color-accent) rather than its own literal, so this one
// property set is all that's required for the change to cascade everywhere
// (buttons, the active-block wash, the loop band, the waveform's progress
// fill) — see --color-accent's own comment there.
function apply(id: string): void {
  document.documentElement.style.setProperty('--color-accent', optionFor(id).hex)
}

/**
 * Applies the stored accent to the document root. Called once, synchronously,
 * before React mounts (see main.tsx) — same reasoning as theme.ts's
 * applyStoredTheme: doing this only inside useAccentColor's effect would
 * flash the default purple first on every reload for anyone who picked
 * something else.
 */
export function applyStoredAccent(): void {
  apply(readStored())
}

export interface AccentController {
  accentId: string
  setAccentId: (id: string) => void
  options: AccentOption[]
}

/**
 * Owns the live accent choice and keeps the document root's --color-accent
 * inline style and localStorage in sync with it. Mirrors theme.ts's
 * useDarkMode shape (and playback.ts's useSpeed/useLoop before it): one
 * hook, one small object of state plus the action that changes it.
 */
export function useAccentColor(): AccentController {
  const [accentId, setAccentId] = useState<string>(readStored)

  useEffect(() => {
    apply(accentId)
    try {
      localStorage.setItem(STORAGE_KEY, accentId)
    } catch {
      // Storage unavailable (private mode, quota) — accent still applies for
      // this session, it just won't persist across reloads.
    }
  }, [accentId])

  return { accentId, setAccentId, options: ACCENT_OPTIONS }
}
