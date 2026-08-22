import { useCallback, useEffect, useRef, useState } from 'react'
import { formatTime } from './format'
import Logo from './Logo'
import { API_BASE } from './snap'
import type { LibrarySong } from './types'

// Settings is its own SCREEN (App.tsx's View, alongside Library/Song), not a
// dropdown — at the user's explicit request, reversed from an earlier
// dropdown-popover version once it grew a second option (beat snapping,
// alongside accent color): a popover reads fine for one small control, but
// two independent settings belong on a real screen, not stacked in a menu.
// This button is just the trigger; SettingsScreen.tsx owns everything past
// the click. No design mock covers a Settings control at all (the redesign
// doesn't have one yet), so the icon is invented, same as the Timeline's own
// canvas palette elsewhere in this app — but at the user's explicit request,
// no longer an invented shape of its own (a hub + eight spokes, which read
// as more of a sun/loading glyph than a gear): this is the generic settings
// symbol every icon set converges on — an 8-tooth gear with a hole punched
// out of the center — generated algorithmically (not hand-plotted, and not
// recalled from any specific icon library) via a small Python script:
// alternating outer/inner radius points around the circle for the teeth,
// plus a same-path circle for the hole, cut out via fillRule="evenodd".
// Filled rather than stroked (unlike SpeedControl's own icon) because a
// gear's teeth are what read as "gear" at 15px, and a stroke outline of
// this many vertices gets muddy that small — solid fill stays legible.
function SettingsButton({ onOpen }: { onOpen: () => void }) {
  return (
    <button className="speed-dial-button" onClick={onOpen} aria-label="Settings">
      <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path
          fillRule="evenodd"
          d="M 20 12 L 19.85 13.56 L 22 13.99 L 20.48 17.67 L 18.65 16.44 L 17.66 17.66 L 16.44 18.65 L 17.67 20.48 L 13.99 22 L 13.56 19.85 L 12 20 L 10.44 19.85 L 10.01 22 L 6.33 20.48 L 7.56 18.65 L 6.34 17.66 L 5.35 16.44 L 3.52 17.67 L 2 13.99 L 4.15 13.56 L 4 12 L 4.15 10.44 L 2 10.01 L 3.52 6.33 L 5.35 7.56 L 6.34 6.34 L 7.56 5.35 L 6.33 3.52 L 10.01 2 L 10.44 4.15 L 12 4 L 13.56 4.15 L 13.99 2 L 17.67 3.52 L 16.44 5.35 L 17.66 6.34 L 18.65 7.56 L 20.48 6.33 L 22 10.01 L 19.85 10.44 Z
             M 8.6 12 A 3.4 3.4 0 1 0 15.4 12 A 3.4 3.4 0 1 0 8.6 12 Z"
        />
      </svg>
    </button>
  )
}

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

export default function Library({
  onOpen,
  onOpenSettings,
}: {
  onOpen: (song: LibrarySong) => void
  onOpenSettings: () => void
}) {
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
        <div className="library-head-actions">
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
          <SettingsButton onOpen={onOpenSettings} />
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
