import { useCallback, useEffect, useRef, useState } from 'react'
import Logo from './Logo'
import { formatTime } from './format'
import { API_BASE } from './snap'
import type { LibrarySong } from './types'

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
//
// The design (design/handoff/README.md "1. Library") only models a binary dot
// (counted/needs-counting) with a single nullable note string. Rather than
// collapsing stems_evicted into one of those two, it gets a third dot color
// (var(--color-muted-text), since the design has no third hue for it) and
// keeps its own richer, branching note text — see StateNote.
function dotState(song: LibrarySong): 'ready' | 'needs_tap' | 'stems_evicted' {
  return song.state
}

function StateNote({ song }: { song: LibrarySong }) {
  if (song.state === 'stems_evicted') {
    return (
      <span className="song-note">
        The audio file is no longer in the library folder, so this song can&apos;t be opened.
        {song.grid_present ? (
          <>
            {' '}
            <strong>Your tapped count is still saved.</strong> Import{' '}
            {song.filename ? <code>{song.filename}</code> : 'the same file'} again and it comes
            back exactly as you left it. The grid is keyed to the audio itself, not to the
            filename.
          </>
        ) : (
          ' Import it again to restore it.'
        )}
      </span>
    )
  }
  if (song.state === 'needs_tap') {
    return (
      <span className="song-note" data-tone="amber">
        Needs counts
      </span>
    )
  }
  if (!song.stems_present) {
    return (
      <span className="song-note">
        Stems were cleared from the cache. Opening this song re-separates them first, which takes
        a few minutes; your tapped count is unaffected.
      </span>
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
        <div className="library-brand">
          <Logo />
          <h1 className="library-wordmark">horeo</h1>
        </div>
        <div>
          <input
            ref={fileRef}
            type="file"
            accept="audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.mp4,.mov,.webm,.mkv,.avi"
            hidden
            onChange={(e) => void onPick(e.target.files?.[0])}
          />
          <button className="import-button" onClick={() => fileRef.current?.click()} disabled={importing}>
            {importing ? 'Importing…' : 'Import song'}
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
          <li key={song.md5}>
            <button
              type="button"
              className="song-row"
              // The evicted state is the one state that genuinely cannot be
              // opened: the engine plays the original file, and no amount of
              // cached derived data substitutes for it.
              disabled={song.state === 'stems_evicted'}
              onClick={() => onOpen(song)}
            >
              <span className="song-dot" data-state={dotState(song)} aria-hidden="true" />
              <span className="song-row-text">
                <span className="song-title">
                  {song.filename ?? song.id ?? `(unnamed · ${song.md5.slice(0, 8)})`}
                </span>
                <StateNote song={song} />
              </span>
              <span className="song-meta">
                {song.media_kind === 'video' && <span className="song-kind-flag">VIDEO</span>}
                {song.duration !== null && (
                  <span className="song-duration">{formatTime(song.duration)}</span>
                )}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </main>
  )
}
