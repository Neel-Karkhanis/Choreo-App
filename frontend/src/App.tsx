import { useState } from 'react'
import Library from './Library'
import SettingsScreen from './SettingsScreen'
import Song from './Song'
import { useDarkMode } from './theme'
import type { LibrarySong } from './types'

// The router, and nothing else. Three levels: the Library, Settings, and one
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
type View = { kind: 'library' } | { kind: 'settings' } | { kind: 'song'; song: LibrarySong }

export default function App() {
  const [view, setView] = useState<View>({ kind: 'library' })
  const { darkMode } = useDarkMode()

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
