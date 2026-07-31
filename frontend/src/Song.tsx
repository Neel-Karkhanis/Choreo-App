import { useCallback, useEffect, useMemo, useState } from 'react'
import Timeline from './Timeline'
import VideoScreen from './VideoScreen'
import { buildEightCountWindows } from './eightCount'
import { useManualGrid, type ManualGridStore } from './manualGrid'
import { useEngineTransport, useLoop, useSpeed } from './playback'
import { API_BASE, DEFAULT_SNAP_MODE, snapTime, type SnapDirection, type SnapMode } from './snap'
import { DEFAULT_STEM_MODE, StemEngine, loadStems, type StemMode } from './stemEngine'
import { nudgeGrid } from './tapGrid'
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

export default function Song({ song, onExit }: { song: LibrarySong; onExit: () => void }) {
  const trackId = song.id ?? undefined
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [engine, setEngine] = useState<StemEngine | null>(null)
  const [stemError, setStemError] = useState<string | null>(null)

  // The tapped grid for this audio — the only grid there is. Loaded in parallel
  // with the analysis (a tiny sidecar read against a slow pipeline), so it is
  // always in hand before stems finish decoding and the timeline first paints;
  // a track with no saved grid opens in the tap state.
  const manual = useManualGrid(trackId)

  useEffect(() => {
    if (!trackId) return
    let stale = false
    setLoading(true)
    setAnalysis(null)
    setError(null)
    fetch(`${API_BASE}/tracks/${encodeURIComponent(trackId)}/analysis`)
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
    loadStems(analysis.id, controller.signal)
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

  const title = song.filename ?? song.id ?? song.md5

  return (
    <main className="song">
      <header className="song-head">
        <button onClick={onExit}>← Library</button>
        <h1>{title}</h1>
      </header>
      {error && <p className="error">{error}</p>}
      {onsetsError && <p className="error">Onset overlays unavailable: {onsetsError}</p>}
      {stemError && <p className="error">Stems unavailable: {stemError}</p>}
      {manual.error && <p className="error">Tapped grid: {manual.error}</p>}
      {loading && <p>Analyzing {title}… (first run on a new track takes minutes)</p>}
      {analysis && !engine && !stemError && (
        <p>Loading stems… (decoding five buffers; sizes logged to console)</p>
      )}
      {engine && (
        // Keyed by the engine so the shell's hoisted state is rebuilt exactly
        // when the audio it describes is — never on a screen swap.
        <SongShell
          key={analysis?.id ?? song.md5}
          engine={engine}
          manual={manual}
          onsets={onsets}
          mediaKind={song.media_kind}
          videoUrl={analysis?.video_url ?? null}
        />
      )}
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
}: {
  engine: StemEngine
  manual: ManualGridStore
  onsets: OnsetData | undefined
  mediaKind: LibrarySong['media_kind']
  videoUrl: string | null
}) {
  const [screen, setScreen] = useState<Screen>('timeline')
  const [snapMode, setSnapMode] = useState<SnapMode>(DEFAULT_SNAP_MODE)
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

  // Loop points snap according to the snap mode selected right now —
  // DIRECTIONALLY (the loop floors A and ceils B so it encloses whole musical
  // units). In "none" mode both keep the raw playhead time.
  const snapToMode = useCallback(
    (time: number, direction: SnapDirection) =>
      snapTime(time, snapMode, effectiveGrid, duration, direction),
    [snapMode, effectiveGrid, duration],
  )
  // The loop drives the ENGINE (native buffer-source looping). Same clock as
  // everything else, and one instance for every screen.
  const loop = useLoop(engine, snapToMode)

  // Nudge the SAVED grid — a core control, not a tap-mode afterthought. Every
  // grid in the app carries the tap session's constant touch→audio latency
  // (50–150ms) plus early-tap bias, and that offset lands directly on loop
  // points; the fix has to be reachable while the song plays and while the loop
  // runs, so the correction is HEARD landing rather than adjusted blind. Each
  // press re-derives the grid from its own fit (phase slides, or the "1"
  // re-labels — the fitted tempo is never touched) and writes through the same
  // save path as an accepted tap, so the correction persists like any other.
  const onNudgeGrid = useCallback(
    (deltaMs: number, deltaCount: number) => {
      if (!grid) return
      const next = nudgeGrid(grid, duration, deltaMs / 1000, deltaCount)
      if (next) manual.save(next)
    },
    [grid, duration, manual],
  )

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
          snapMode={snapMode}
          onSnapModeChange={setSnapMode}
          stemMode={stemMode}
          onStemModeChange={setStemMode}
          tap={tap}
          onNudgeGrid={onNudgeGrid}
        />
      ) : (
        <VideoScreen
          transport={transport}
          loop={loop}
          speed={speed}
          grid={effectiveGrid}
          windows={windows}
          videoUrl={videoUrl}
        />
      )}
    </>
  )
}
