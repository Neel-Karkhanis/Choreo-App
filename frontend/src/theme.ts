import { useCallback, useEffect, useState } from 'react'

// Light/dark theme. A device preference, not song data — unlike the tapped
// grid (manualGrid.ts), it has no server-side home and never goes through
// API_BASE; a plain localStorage flag is the whole persistence story.
const STORAGE_KEY = 'choreo-dark-mode'

function readStored(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

function apply(darkMode: boolean): void {
  document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
}

/**
 * Applies the stored preference to the document root. Called once,
 * synchronously, before React mounts (see main.tsx) — doing this only inside
 * useDarkMode's effect would flash light mode first on every reload for a
 * user who picked dark, since effects run after the first paint.
 */
export function applyStoredTheme(): void {
  apply(readStored())
}

export interface ThemeController {
  darkMode: boolean
  toggleDarkMode: () => void
}

/**
 * Owns the live darkMode value and keeps the document root's data-theme
 * attribute (which index.css's token overrides key off) and localStorage in
 * sync with it. Mirrors the rest of the app's controller-hook shape (compare
 * useSpeed/useLoop in playback.ts): one hook, one small object of state plus
 * the actions that change it.
 */
export function useDarkMode(): ThemeController {
  const [darkMode, setDarkMode] = useState<boolean>(readStored)

  useEffect(() => {
    apply(darkMode)
    try {
      localStorage.setItem(STORAGE_KEY, String(darkMode))
    } catch {
      // Storage unavailable (private mode, quota) — theme still applies for
      // this session, it just won't persist across reloads.
    }
  }, [darkMode])

  const toggleDarkMode = useCallback(() => setDarkMode((v) => !v), [])

  return { darkMode, toggleDarkMode }
}
