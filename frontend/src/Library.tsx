import { useCallback, useEffect, useRef, useState } from 'react'
import { formatTime } from './format'
import { API_BASE } from './snap'
import type { LibrarySong, LibraryState } from './types'

// LEVEL 1 of the app: every song the backend has ever known, and the door into
// each one.
//
// The three states below are modelled explicitly and are NEVER collapsed into
// "openable / not openable". Each one means something different to the person
// reading it, and one of them — STEMS EVICTED — exists specifically to keep a
// promise: your tapped grid is still on disk, it is keyed to the audio's md5,
// and re-importing the same file gets it back untouched. Hiding that row, or
// deleting the grid behind it, would silently destroy the one artifact in this
// app that cannot be recomputed.

const STATE_LABEL: Record<LibraryState, string> = {
  ready: 'Ready',
  needs_tap: 'Needs a count',
  stems_evicted: 'File missing',
}

function StateNote({ song }: { song: LibrarySong }) {
  if (song.state === 'stems_evicted') {
    return (
      <p className="song-note">
        The audio file is no longer in the library folder, so this song can&apos;t be opened.
        {song.grid_present ? (
          <>
            {' '}
            <strong>Your tapped count is still saved.</strong> Import{' '}
            {song.filename ? <code>{song.filename}</code> : 'the same file'} again and it comes
            back exactly as you left it — the grid is keyed to the audio itself, not to the
            filename.
          </>
        ) : (
          ' Import it again to restore it.'
        )}
      </p>
    )
  }
  if (song.state === 'needs_tap') {
    return <p className="song-note">Opens straight into tap mode — count it once and it sticks.</p>
  }
  if (!song.stems_present) {
    return (
      <p className="song-note">
        Stems were cleared from the cache. Opening this song re-separates them first, which takes
        a few minutes; your tapped count is unaffected.
      </p>
    )
  }
  return null
}

export default function Library({ onOpen }: { onOpen: (song: LibrarySong) => void }) {
  const [songs, setSongs] = useState<LibrarySong[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(() => {
    setLoading(true)
    return fetch(`${API_BASE}/library`)
      .then((res) => {
        if (!res.ok) throw new Error(`GET library -> HTTP ${res.status}`)
        return res.json() as Promise<{ songs: LibrarySong[] }>
      })
      .then((data) => {
        setSongs(data.songs)
        setError(null)
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Import is a real browser upload: the file is POSTed, saved into the
  // library folder, and registered. Importing a file the app already has is
  // the documented way back from "File missing" — identical bytes hash to the
  // same md5, so the existing grid is picked straight back up.
  const onPick = async (file: File | undefined) => {
    if (!file) return
    setImporting(true)
    setError(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const res = await fetch(`${API_BASE}/import`, { method: 'POST', body })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail ?? `POST import -> HTTP ${res.status}`)
      }
      await refresh()
    } catch (err) {
      setError(String(err))
    } finally {
      setImporting(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <main className="library">
      <header className="library-head">
        <h1>Choreo</h1>
        <div>
          <input
            ref={fileRef}
            type="file"
            accept="audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.mp4,.mov,.webm,.mkv,.avi"
            hidden
            onChange={(e) => void onPick(e.target.files?.[0])}
          />
          <button onClick={() => fileRef.current?.click()} disabled={importing}>
            {importing ? 'Importing…' : 'Import new song'}
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {loading && songs.length === 0 && <p>Loading library…</p>}
      {!loading && songs.length === 0 && !error && (
        <p>No songs yet. Import one to get started.</p>
      )}

      <ul className="song-list">
        {songs.map((song) => (
          <li key={song.md5} className={`song-row is-${song.state}`}>
            <div className="song-row-main">
              <button
                className="song-open"
                // The evicted state is the one state that genuinely cannot be
                // opened: the engine plays the original file, and no amount of
                // cached derived data substitutes for it.
                disabled={song.state === 'stems_evicted'}
                onClick={() => onOpen(song)}
              >
                {song.filename ?? song.id ?? `(unnamed · ${song.md5.slice(0, 8)})`}
              </button>
              <span className={`song-state is-${song.state}`}>{STATE_LABEL[song.state]}</span>
              {song.media_kind === 'video' && <span className="song-kind">video</span>}
              {song.duration !== null && (
                <span className="song-duration">{formatTime(song.duration)}</span>
              )}
            </div>
            <StateNote song={song} />
          </li>
        ))}
      </ul>
    </main>
  )
}
