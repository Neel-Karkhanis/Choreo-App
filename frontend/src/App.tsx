import { useEffect, useState } from 'react'
import { ensureDevice } from './device'
import Library from './Library'
import SettingsScreen from './SettingsScreen'
import Song from './Song'
import { useDarkMode } from './theme'
import type { LibrarySong } from './types'

// The router, and nothing else. Four states: bootstrapping the device
// identity, an error if that failed, the Library, Settings, and one open
// Song.
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
// There is no sign-in screen: the app has no accounts. 'checking' is only
// ever the one moment on startup where the device-id cookie is being
// established (see device.ts) — every subsequent /api/* call rides on it
// automatically, with nothing left client-side to check.
type View =
  | { kind: 'checking' }
  | { kind: 'error'; message: string }
  | { kind: 'library' }
  | { kind: 'settings' }
  | { kind: 'song'; song: LibrarySong }

export default function App() {
  const [view, setView] = useState<View>({ kind: 'checking' })
  const { darkMode } = useDarkMode()

  useEffect(() => {
    ensureDevice()
      .then(() => setView({ kind: 'library' }))
      .catch((err) => setView({ kind: 'error', message: String(err) }))
  }, [])

  if (view.kind === 'checking') {
    return (
      <main className="settings">
        <p>Loading…</p>
      </main>
    )
  }

  if (view.kind === 'error') {
    return (
      <main className="settings">
        <p className="error">{view.message}</p>
        <button type="button" onClick={() => window.location.reload()}>
          Retry
        </button>
      </main>
    )
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
    return <SettingsScreen onExit={() => setView({ kind: 'library' })} />
  }

  return (
    <Library
      onOpen={(song) => setView({ kind: 'song', song })}
      onOpenSettings={() => setView({ kind: 'settings' })}
    />
  )
}
