import { useEffect, useState } from 'react'
import { apiFetch, setUnauthorizedHandler } from './api'
import Auth from './Auth'
import Library from './Library'
import { API_BASE } from './snap'
import SettingsScreen from './SettingsScreen'
import Song from './Song'
import { useDarkMode } from './theme'
import type { LibrarySong } from './types'

// The router, and nothing else. Five states: checking whether a session
// cookie is already live, Auth (signed out), the Library, Settings, and one
// open Song.
//
// A song is opened by MOUNTING Song and closed by UNMOUNTING it. That is the
// whole navigation model, and it is deliberately not a URL router or a stack:
// unmount is what guarantees the engine's AudioContext is closed, its sources
// stopped, and its event listeners dropped, because that teardown hangs off
// Song's own cleanup. Anything that kept a Song mounted "in the background"
// would keep an AudioContext alive with it. Settings carries no such
// teardown concern (no engine, no AudioContext), so it's plain mount/unmount
// like Library, just a third View variant rather than its own special case.
//
// 'auth' is the same idea applied one level up: there is no client-side
// token to check, only the httponly session cookie the browser already
// holds (or doesn't) — so "am I signed in" is answered once on mount via
// GET /auth/me, and again reactively whenever any apiFetch call anywhere in
// the app gets a 401 back (a session that lapsed mid-use), via the shared
// unauthorized handler wired up below.
type View =
  | { kind: 'checking' }
  | { kind: 'auth' }
  | { kind: 'library' }
  | { kind: 'settings' }
  | { kind: 'song'; song: LibrarySong }

export default function App() {
  const [view, setView] = useState<View>({ kind: 'checking' })
  const { darkMode } = useDarkMode()

  useEffect(() => {
    setUnauthorizedHandler(() => setView({ kind: 'auth' }))
    return () => setUnauthorizedHandler(null)
  }, [])

  useEffect(() => {
    apiFetch(`${API_BASE}/auth/me`)
      .then((res) => setView({ kind: res.ok ? 'library' : 'auth' }))
      .catch(() => setView({ kind: 'auth' }))
  }, [])

  if (view.kind === 'checking') {
    return (
      <main className="settings">
        <p>Loading…</p>
      </main>
    )
  }

  if (view.kind === 'auth') {
    return <Auth onSignedIn={() => setView({ kind: 'library' })} />
  }

  if (view.kind === 'song') {
    return (
      <Song
        // Keyed by md5 so opening a different song is a fresh mount rather
        // than a re-render that would hand the new song the old song's state.
        key={view.song.md5}
        song={view.song}
        onExit={() => setView({ kind: 'library' })}
        darkMode={darkMode}
      />
    )
  }

  if (view.kind === 'settings') {
    return (
      <SettingsScreen
        onExit={() => setView({ kind: 'library' })}
        onSignedOut={() => setView({ kind: 'auth' })}
      />
    )
  }

  return (
    <Library
      onOpen={(song) => setView({ kind: 'song', song })}
      onOpenSettings={() => setView({ kind: 'settings' })}
    />
  )
}
