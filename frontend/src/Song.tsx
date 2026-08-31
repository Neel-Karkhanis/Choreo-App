import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiFetch } from './api'
import Logo from './Logo'
import Timeline from './Timeline'
import VideoScreen from './VideoScreen'
import { readCountDisplay, type CountDisplay } from './countDisplay'
import { buildEightCountWindows } from './eightCount'
import { useManualGrid, type ManualGridStore } from './manualGrid'
import { useEngineTransport, useLoop, useSpeed } from './playback'
import { API_BASE, readDefaultSnapMode, snapTime, type SnapDirection, type SnapMode } from './snap'
import { DEFAULT_STEM_MODE, StemEngine, loadStems, type StemMode } from './stemEngine'
import { useTapSession } from './tapSession'
import type { GridData, LibrarySong, OnsetData } from './types'

// LEVEL 2 of the app: one song, open. Only one exists at a time, and leaving it
// unmounts this subtree — which is what tears the engine (and its AudioContext)
// down. There is no engine cache and no "keep it warm" path: a second live
// engine would be a second AudioContext playing a second copy of the audio.

// Parallel arrays: t[i] is an onset timestamp in seconds, strength[i] the
// onset-envelope value at that onset. strength is plumbed by schema v4 but
// deliberately unused by the UI so far.
interface StemOnsets {
  t: number[]
  strength: number[]
}

// Mirrors the analysis JSON contract: stems and onsets only. The grid
// (beats/downbeats/eight_counts) comes EXCLUSIVELY from the user's tap fit —
// auto beat detection is gone. If an older payload still carries grid fields,
// they are ignored by construction: nothing here reads them, and nothing may
// fall back to them.
interface Analysis {
  id: string
  filename: string
  schema_version: number
  duration: number
  onsets: { drums: StemOnsets; bass: StemOnsets }
  audio_url: string
  video_url: string | null
  stems: { name: string; url: string }[]
}

const isNumbers = (xs: unknown): xs is number[] =>
  Array.isArray(xs) && xs.every((x) => typeof x === 'number' && Number.isFinite(x))

// Onsets are independent of the beat grid, so validation failures here
// shouldn't block the grid from rendering — reported separately. Only the
// timestamps flow to the Timeline; strength is contract-checked but unused.
function onsetsFromAnalysis(a: Analysis): { onsets?: OnsetData; onsetsError?: string } {
  const isStemOnsets = (s: StemOnsets | undefined) =>
    !!s && isNumbers(s.t) && isNumbers(s.strength) && s.t.length === s.strength.length
  if (!a.onsets || !isStemOnsets(a.onsets.drums) || !isStemOnsets(a.onsets.bass)) {
    return {
      onsetsError: 'analysis.onsets is not { drums|bass: { t: number[], strength: number[] } }',
    }
  }
  return { onsets: { drums: a.onsets.drums.t, bass: a.onsets.bass.t } }
}

// Which view of the song is on screen. An audio song only ever has 'timeline';
// a video song can swap. This is the ONLY thing that changes on a swap — no
// engine, loop, speed, grid, or playhead state is keyed to it.
type Screen = 'timeline' | 'video'

// Shared by both of a song's loading paragraphs below — the slow server-side
// analysis fetch (source separation + onset detection; minutes on a first
// run) and the client-side stem decode that follows it — so the same cycling
// status replaces both old static lines instead of just the first. Purely
// cosmetic: neither step reports real progress, so this doesn't track actual
// pipeline stage, just gives the wait something to look at. Each mount starts
// back at index 0, so the sequence restarts when loading hands off to the
// stem decode.
const LOADING_MESSAGES = ['Loading timeline…', 'Separating stems…', 'Mapping onsets…']
const LOADING_MESSAGE_INTERVAL_MS = 4000

// Ring geometry for ProgressRing below. A module constant (not derived
// inline per render) since both the SVG radius and the dasharray/dashoffset
// math need the exact same circumference.
const PROGRESS_RING_RADIUS = 24
const PROGRESS_RING_CIRCUMFERENCE = 2 * Math.PI * PROGRESS_RING_RADIUS

// Demucs' own separation progress, drawn as a ring that fills clockwise from
// 12 o'clock. `fraction` is done/total straight off the backend's
// /analysis/progress poll (see the progress effect in Song() below) — real
// per-chunk completion, the same signal Demucs' own CLI bar reads from, not
// a simulated animation. Purple to match the app's one accent color
// (--color-accent), same hue as the active 8-count and the tap-mode buttons.
function ProgressRing({ fraction }: { fraction: number }) {
  const offset = PROGRESS_RING_CIRCUMFERENCE * (1 - fraction)
  return (
    <svg
      className="progress-ring"
      width="56"
      height="56"
      viewBox="0 0 56 56"
      role="img"
      aria-label={`${Math.round(fraction * 100)}% separated`}
    >
      <circle className="progress-ring-track" cx="28" cy="28" r={PROGRESS_RING_RADIUS} />
      <circle
        className="progress-ring-fill"
        cx="28"
        cy="28"
        r={PROGRESS_RING_RADIUS}
        strokeDasharray={PROGRESS_RING_CIRCUMFERENCE}
        strokeDashoffset={offset}
      />
    </svg>
  )
}

function LoadingStatus({
  // Off for the stem decode: that step is client-side and normally fast, so
  // the "couple minutes" warning belongs only to the slow server-side
  // analysis fetch above it.
  showSubtext = true,
  // Demucs separation progress for a FRESH song. Left undefined for the stem
  // decode's LoadingStatus (that step isn't Demucs — there's no real
  // percentage to show) and stays null for a cached song's near-instant
  // reload, since a cache hit never starts a Demucs run for the progress
  // effect to observe. Either way, no `progress` means no ring: this is the
  // one thing that decides whether the ring renders at all.
  progress,
}: {
  showSubtext?: boolean
  progress?: { done: number; total: number } | null
}) {
  const [index, setIndex] = useState(0)
  useEffect(() => {
    const timer = setInterval(
      () => setIndex((i) => (i + 1) % LOADING_MESSAGES.length),
      LOADING_MESSAGE_INTERVAL_MS,
    )
    return () => clearInterval(timer)
  }, [])
  return (
    <div className="song-loading">
      {progress && progress.total > 0 && (
        <ProgressRing fraction={Math.min(1, progress.done / progress.total)} />
      )}
      {/* Keyed by index so each message change is a fresh DOM node — that's
          what makes .song-loading-message's CSS animation replay on every
          swap instead of only once on the very first mount. */}
      <p key={index} className="song-loading-message">
        {LOADING_MESSAGES[index]}
      </p>
      {showSubtext && <p className="song-loading-subtext">This may take a couple minutes.</p>}
    </div>
  )
}

export default function Song({
  song,
  onExit,
  darkMode,
}: {
  song: LibrarySong
  onExit: () => void
  darkMode: boolean
}) {
  const trackId = song.id ?? undefined
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [engine, setEngine] = useState<StemEngine | null>(null)
  const [stemError, setStemError] = useState<string | null>(null)
  // Non-fatal: a local IndexedDB cache write hit its storage quota (see
  // stemEngine.ts's onCacheWarning). Playback is unaffected — the buffer
  // already loaded and decoded fine — this just says the stem won't be
  // there next time without a re-download.
  const [cacheWarning, setCacheWarning] = useState<string | null>(null)
  // Demucs' live separation progress for the in-flight analysis fetch below;
  // see the polling effect further down and ProgressRing/LoadingStatus above.
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)

  // The tapped grid for this audio — the only grid there is. Loaded in parallel
  // with the analysis (a tiny sidecar read against a slow pipeline), so it is
  // always in hand before stems finish decoding and the timeline first paints;
  // a track with no saved grid opens in the tap state.
  const manual = useManualGrid(trackId, song.md5)

  useEffect(() => {
    if (!trackId) return
    let stale = false
    setLoading(true)
    setAnalysis(null)
    setError(null)
    setProgress(null)
    apiFetch(`${API_BASE}/tracks/${encodeURIComponent(trackId)}/analysis`)
      .then((res) => {
        if (!res.ok) throw new Error(`GET analysis -> HTTP ${res.status}`)
        return res.json() as Promise<Analysis>
      })
      .then((data) => {
        if (!stale) setAnalysis(data)
      })
      .catch((err) => {
        if (!stale) setError(String(err))
      })
      .finally(() => {
        if (!stale) setLoading(false)
      })
    return () => {
      stale = true
    }
  }, [trackId])

  // Polls Demucs' live separation progress while the analysis fetch above is
  // in flight. A cached track's analysis returns before this ever observes
  // {active: true} — that alone is what keeps the ring out of a cached
  // reload, with no separate "is this cached" flag needed. Once a run does
  // go active, `progress` is left at its last value on later {active: false}
  // polls (rather than reset to null) so the ring doesn't blink out the
  // moment Demucs finishes but analysis is still finishing up beat/onset
  // detection behind it.
  useEffect(() => {
    if (!trackId || !loading) return
    let stale = false
    const poll = () => {
      apiFetch(`${API_BASE}/tracks/${encodeURIComponent(trackId)}/analysis/progress`)
        .then((res) => (res.ok ? (res.json() as Promise<{ active: boolean; done?: number; total?: number }>) : null))
        .then((data) => {
          if (stale || !data || !data.active) return
          setProgress({ done: data.done ?? 0, total: data.total ?? 0 })
        })
        .catch(() => {})
    }
    poll()
    const timer = setInterval(poll, 400)
    return () => {
      stale = true
      clearInterval(timer)
    }
  }, [trackId, loading])

  // Eager stem load: all five buffers fetched and decoded through the single
  // loadStems seam before any screen (and thus playback) exists. The abort +
  // stale guard means leaving the song — or StrictMode's dev double-mount —
  // cancels the in-flight fetches and never constructs a stray engine.
  useEffect(() => {
    if (!analysis) return
    let stale = false
    const controller = new AbortController()
    setEngine(null)
    setStemError(null)
    setCacheWarning(null)
    loadStems(analysis.id, song.md5, controller.signal, (message) => {
      if (!stale) setCacheWarning(message)
    })
      .then((buffers) => {
        if (stale) return
        const next = new StemEngine(buffers)
        // Dev-only debug handle: lets driver scripts assert engine internals.
        if (import.meta.env.DEV) {
          ;(window as unknown as { __stemEngine?: StemEngine }).__stemEngine = next
        }
        setEngine(next)
      })
      .catch((err) => {
        if (!stale && !controller.signal.aborted) setStemError(String(err))
      })
    return () => {
      stale = true
      controller.abort()
    }
    // song.md5 deliberately excluded — App.tsx keys Song by song.md5, so a
    // change to it always remounts this component from scratch rather than
    // re-running this effect; within one mount it's invariant.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis])

  // The engine owns an AudioContext; close it when the engine is replaced or
  // this song is left (separate effect so the loader above never races its own
  // cleanup). This is the whole teardown story: unmounting Song runs this.
  useEffect(() => {
    if (!engine) return
    return () => engine.destroy()
  }, [engine])

  const { onsets, onsetsError } = useMemo(
    () => (analysis ? onsetsFromAnalysis(analysis) : {}),
    [analysis],
  )

  // Extension stripped for aesthetics, same as the Library row's own title
  // (Library.tsx's SongRow) — song.id already has none (the backend derives
  // it from the file's Path.stem), so only the song.filename branch ever
  // needs the regex.
  const title = song.filename ? song.filename.replace(/\.[^./]+$/, '') : (song.id ?? song.md5)

  return (
    <main className="song">
      <div className="song-scroll">
        <header className="song-head">
          <div className="song-head-left">
            <button className="song-back" onClick={onExit} aria-label="Back to library">
              ←
            </button>
            <h1>{title}</h1>
          </div>
          <Logo size={29} />
        </header>
        {error && <p className="error">{error}</p>}
        {onsetsError && <p className="error">Onset overlays unavailable: {onsetsError}</p>}
        {stemError && <p className="error">Stems unavailable: {stemError}</p>}
        {cacheWarning && <p className="settings-section-note">{cacheWarning}</p>}
        {manual.error && <p className="error">Tapped grid: {manual.error}</p>}
        {loading && <LoadingStatus progress={progress} />}
        {analysis && !engine && !stemError && <LoadingStatus showSubtext={false} />}
        {engine && (
          // Keyed by the engine so the shell's hoisted state is rebuilt exactly
          // when the audio it describes is — never on a screen swap. The nav
          // bar below is INSIDE this same gate, not a sibling of it: `engine`
          // truthy is exactly "the timeline view is actually present and
          // loaded" (stems decoded, SongShell/Timeline mounted) — before that
          // (the analysis fetch, then the stem decode) the nav must not show
          // at all, only the two LoadingStatus screens above.
          <>
            <SongShell
              key={analysis?.id ?? song.md5}
              engine={engine}
              manual={manual}
              onsets={onsets}
              mediaKind={song.media_kind}
              videoUrl={analysis?.video_url ?? null}
              darkMode={darkMode}
            />
            {/* Design's "Library / [song name]" tab bar. Only "Library"
                actually navigates (onExit) — the song tab is the current
                screen, shown active/inert, not a jump-back-in-from-elsewhere
                control (that would need "last open song" state App.tsx
                deliberately doesn't keep). A normal in-flow element, not
                fixed to the viewport — at the user's explicit request, it
                scrolls away with the rest of the page instead of hovering
                over it, sitting directly below the Timeline card
                (waveform/controls) rather than pinned to the bottom edge. */}
            <nav className="song-bottom-tabs" aria-label="Navigate">
              <button type="button" className="song-bottom-tab" onClick={onExit}>
                Library
              </button>
              <button type="button" className="song-bottom-tab is-active" aria-current="page">
                {title}
              </button>
            </nav>
          </>
        )}
      </div>
    </main>
  )
}

/**
 * THE STATE OWNER.
 *
 * Everything a screen could possibly want to remember lives here: the engine
 * handle, the transport, the loop, the speed, the snap mode, the stem mode, the
 * grid, and the tap session. Screens below are VIEWS — they render this state
 * and call back into it, and they own nothing but their own local rendering
 * toggles.
 *
 * That is not a style preference. Two screens each holding their own loop and
 * speed state is the specific failure this shell exists to prevent: set a loop
 * on the Timeline, swap to Video, and the loop would be gone — or worse, still
 * running in the engine while the HUD showed "no loop". Hoisting also means the
 * swap does not need to preserve anything, because the swap does not touch
 * anything: it changes which component is mounted, and nothing else.
 */
function SongShell({
  engine,
  manual,
  onsets,
  mediaKind,
  videoUrl,
  darkMode,
}: {
  engine: StemEngine
  manual: ManualGridStore
  onsets: OnsetData | undefined
  mediaKind: LibrarySong['media_kind']
  videoUrl: string | null
  darkMode: boolean
}) {
  const [screen, setScreen] = useState<Screen>('timeline')
  const [stemMode, setStemMode] = useState<StemMode>(DEFAULT_STEM_MODE)

  // Duration comes from the engine, which holds it as a readonly constant taken
  // from the decoded original buffer. It is deliberately NOT read back off a
  // wavesurfer instance per render: a media element refines an MP3's duration
  // after it finishes parsing, and that refinement used to ripple through every
  // memo the grid effects depend on and tear down a live RegionsPlugin
  // mid-commit. A frozen number cannot do that.
  const duration = engine.duration

  const transport = useEngineTransport(engine)
  const speed = useSpeed(engine)

  const grid = manual.grid ?? undefined
  const tap = useTapSession({
    engine,
    duration,
    isPlaying: transport.isPlaying,
    grid,
    gridLoading: manual.loading,
    onGridTapped: manual.save,
  })

  // THE grid, for every consumer. A tap preview simply takes the saved grid's
  // place — it is not a special rendering path, it is the same GridData in the
  // same slot, which is exactly why previewing a tapped grid shows precisely
  // what accepting it will produce. Because this is resolved HERE, the loop
  // snaps to the preview too; if the preview lived inside the Timeline, the
  // loop up here would still be snapping to the old grid.
  const effectiveGrid: GridData | undefined = tap.preview ?? grid

  const windows = useMemo(
    () => buildEightCountWindows(effectiveGrid, duration),
    [effectiveGrid, duration],
  )

  // The snap mode is picked by tapping either loop handle in either screen
  // (see LoopBoundaryHandle) and lives here, once, so both screens' handles
  // agree on what a tap changed. Whatever mode is active, the loop still
  // snaps DIRECTIONALLY (floors A, ceils B) so it encloses whole musical
  // units, and snapTime still degrades to a plain clamp when there is no
  // grid yet, which is what lets A/B default to the track's true start/end
  // before a song has ever been tapped.
  // Initial value only — reads the user's Settings-screen preference (or the
  // factory default if they've never set one) once, at mount. A song already
  // open does NOT react to a later Settings change; it takes effect the next
  // time a song is opened, same as accent/dark-mode's own "applies from here
  // forward" behavior.
  const [snapMode, setSnapMode] = useState<SnapMode>(readDefaultSnapMode)
  const snapToMode = useCallback(
    (time: number, direction: SnapDirection) => snapTime(time, snapMode, effectiveGrid, duration, direction),
    [snapMode, effectiveGrid, duration],
  )

  // Same "initial value only" reasoning as snapMode just above — unlike
  // snapMode, nothing in-song ever calls a setter for this, since Settings is
  // its only control surface (no loop-handle-style override exists), so
  // there is no setCountDisplay to hold onto here at all.
  const [countDisplay] = useState<CountDisplay>(readCountDisplay)
  // The loop drives the ENGINE (native buffer-source looping). Same clock as
  // everything else, and one instance for every screen.
  const loop = useLoop(engine, snapToMode, duration)

  // An audio song has one screen; a video song has two, user-swappable. The
  // swap is a render choice, nothing more.
  const screens: Screen[] = mediaKind === 'video' ? ['timeline', 'video'] : ['timeline']
  const active = screens.includes(screen) ? screen : 'timeline'

  return (
    <>
      {screens.length > 1 && (
        <nav className="screen-tabs" role="group" aria-label="Screen">
          {screens.map((name) => (
            <button
              key={name}
              onClick={() => setScreen(name)}
              aria-pressed={active === name}
              className={active === name ? 'screen-tab is-active' : 'screen-tab'}
            >
              {name === 'timeline' ? 'Timeline' : 'Video'}
            </button>
          ))}
        </nav>
      )}
      {active === 'timeline' ? (
        <Timeline
          engine={engine}
          transport={transport}
          loop={loop}
          speed={speed}
          duration={duration}
          grid={effectiveGrid}
          hasSavedGrid={!!grid}
          windows={windows}
          onsets={onsets}
          stemMode={stemMode}
          onStemModeChange={setStemMode}
          snapMode={snapMode}
          onSnapModeChange={setSnapMode}
          tap={tap}
          darkMode={darkMode}
          countDisplay={countDisplay}
        />
      ) : (
        <VideoScreen
          transport={transport}
          loop={loop}
          speed={speed}
          grid={effectiveGrid}
          windows={windows}
          stemMode={stemMode}
          onStemModeChange={setStemMode}
          videoUrl={videoUrl}
          snapMode={snapMode}
          onSnapModeChange={setSnapMode}
        />
      )}
    </>
  )
}
