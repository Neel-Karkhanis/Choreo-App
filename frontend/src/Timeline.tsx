import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useWavesurfer } from '@wavesurfer/react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/plugins/regions'
import {
  DEFAULT_SNAP_MODE,
  SNAP_MODES,
  snapTime,
  useBookmarks,
  type Bookmark,
  type SnapDirection,
  type SnapMode,
} from './bookmarks'
import {
  LOOP_BAND_EDGE,
  LOOP_BAND_EDGE_WIDTH,
  LOOP_BAND_FILL,
  LOOP_DISABLED_OPACITY,
  LOOP_MINIMAP_FILL,
  SPEED_DEFAULT,
  SPEED_MAX,
  SPEED_MIN,
  SPEED_STEP,
  useLoop,
  useSpeed,
} from './playback'
import {
  DEFAULT_STEM_MODE,
  STEM_MODES,
  type StemEngine,
  type StemMode,
} from './stemEngine'

// Beat-grid data, already validated by the caller. downbeatIndices and
// eightCountIndices are indices into beats (schema v2), not timestamps.
export interface GridData {
  beats: number[]
  downbeatIndices: number[]
  eightCountIndices: number[]
}

// Onset data, already validated by the caller. Plain timestamps in seconds,
// independent of the beat grid — not indices, not guaranteed to land on a beat.
export interface OnsetData {
  drums: number[]
  bass: number[]
}

interface TimelineProps {
  // The fully-loaded stem engine: audio source, clock authority, and peaks
  // provider. The Timeline never fetches or decodes audio itself — both
  // wavesurfer instances are views over this engine's shim + peak data.
  engine: StemEngine
  // The track's id — the filename stem. This is exactly the key the backend's
  // bookmark sidecar is named after (tracks/<id>.bookmarks.json), so it must
  // be passed through verbatim or bookmarks save and load under different keys.
  trackId: string
  grid?: GridData
  onsets?: OnsetData
  // Later layers (subdivisions, onset overlays) must position themselves via
  // this instance's own time↔pixel mapping — never independent coordinate math.
  onReady?: (wavesurfer: WaveSurfer) => void
}

// A beat's pulse tick is suppressed (not drawn) if a drum/bass onset lands
// within this many milliseconds of it — the onset marker visually absorbs
// that tick rather than drawing both on top of each other. Starting guess;
// tune against a real track if it reads too aggressive or too loose.
const ABSORPTION_TOLERANCE_MS = 60

// Zoomed slice: the visible window spans this many eight-counts. The window
// is musical, not temporal — on-screen seconds vary with tempo by design.
const SLICE_EIGHT_COUNTS = 2

// Scale factor on onset marker widths (1 = full width: 3px bass, 2px drums).
// Kept as a knob for legibility tuning in dense passages; the thinned 0.67
// variant read too faint, so markers render at full width.
const ONSET_WIDTH_SCALE = 1

// Layer 6: the 8-count block under the playhead gets a faint yellow wash and
// a border. Both are translucent so the waveform, onset markers, and playhead
// stay readable through them; tune or revert here.
const ACTIVE_BLOCK_FILL = 'rgba(250, 200, 40, 0.15)'
const ACTIVE_BLOCK_BORDER = '1px solid rgba(200, 150, 20, 0.6)'

// Overview minimap: a fixed strip under the main view with the whole track
// visible at once. It is a second VIEW of the main instance's clock (shared
// media element) and a seek surface; it never owns transport.
const MINIMAP_HEIGHT = 40
// Fit the entire duration to the container width — the minimap never scrolls.
const MINIMAP_FILL_PARENT = true
// Global playhead line on the minimap (the instance's own cursor).
const MINIMAP_PLAYHEAD_COLOR = 'rgba(40, 40, 40, 0.9)'
const MINIMAP_PLAYHEAD_WIDTH = 2
// Read-only viewport box marking the main view's visible time-slice.
const VIEWPORT_BOX_FILL = 'rgba(40, 40, 40, 0.07)'
const VIEWPORT_BOX_BORDER = '1px solid rgba(40, 40, 40, 0.7)'

// Bookmarks are user-authored, so they get their own hue: brown. Deliberately
// distinct from drums-blue, bass-red, and the Layer 6 yellow highlight.
const BOOKMARK_HUE = 'rgb(124, 74, 32)'
const BOOKMARK_SELECTED_HUE = 'rgb(196, 118, 51)'
// Main (detail) view glyph: a full-height stem with a tappable diamond at the
// bottom edge — clear of the Layer 6 count labels along the top edge.
const BOOKMARK_STEM_WIDTH = 2
const BOOKMARK_GLYPH_SIZE = 12
const BOOKMARK_GLYPH_SHAPE = 'rotate(45deg)' // square rotated into a diamond
// Minimap tick: a plain full-height line, tappable to seek.
const BOOKMARK_TICK_WIDTH = 2
// A long note would run across neighbouring beats, so the on-timeline text is
// cut to this many characters. Display only — the full label is what's stored,
// what the tooltip shows, and what the info panel edits.
const BOOKMARK_LABEL_MAX_CHARS = 5
// The user's typed label, drawn beside the glyph on the main view. Sits on the
// bottom edge, clear of the Layer 6 count labels along the top.
const BOOKMARK_LABEL_STYLE: Partial<CSSStyleDeclaration> = {
  fontSize: '11px',
  lineHeight: '1',
  fontWeight: '600',
  // White halo keeps the label legible over waveform peaks, same trick the
  // Layer 6 count numbers use.
  textShadow: '0 0 3px rgba(255, 255, 255, 0.9), 0 0 1px rgba(255, 255, 255, 0.9)',
}

// Count-number styling: top edge of the block, nudged right of the beat line
// so the digit clears the (up to 3px wide) onset markers on the beat itself.
// Pulse ticks are bottom-anchored, so there's no vertical collision either.
const COUNT_LABEL_STYLE: Partial<CSSStyleDeclaration> = {
  padding: '2px 0 0 4px',
  fontSize: '11px',
  lineHeight: '1',
  color: 'rgba(130, 95, 0, 0.95)',
  // White halo keeps digits legible where waveform peaks reach the top edge.
  textShadow: '0 0 3px rgba(255, 255, 255, 0.9), 0 0 1px rgba(255, 255, 255, 0.9)',
  fontVariantNumeric: 'tabular-nums',
}

// Subdivisions: the half-beat midpoint of every beat gap. These sit at the
// BOTTOM of the visual hierarchy — downbeat > beat > subdivision — so they are
// shorter and lower-contrast than a pulse tick (25% tall, alpha 0.55), which is
// itself shorter than a downbeat (55%, alpha 0.85). If a subdivision ever reads
// as loud as a beat, the beat grid stops being legible: that's the whole risk
// of this layer, so tune these three together.
//
// Bottom-anchored like the pulse ticks, so the top of the strip stays clear for
// count labels; the top offset is derived from the height, never hardcoded.
const SUBDIVISION_TICK_HEIGHT_PCT = 12
const SUBDIVISION_TICK_WIDTH = 1
const SUBDIVISION_TICK_COLOR = 'rgba(120, 120, 120, 0.3)'

// "&" labels, subordinate to the integer counts: same family and anchoring,
// but smaller and lighter. Same 4px right-nudge off the tick, so a bass onset
// landing on a midpoint doesn't sit under the glyph.
const SUBDIVISION_LABEL_STYLE: Partial<CSSStyleDeclaration> = {
  padding: '3px 0 0 4px',
  fontSize: '9px',
  lineHeight: '1',
  color: 'rgba(150, 125, 60, 0.7)',
  textShadow: '0 0 3px rgba(255, 255, 255, 0.9), 0 0 1px rgba(255, 255, 255, 0.9)',
}

// Stem mode selector (PROVISIONAL UI — see the JSX comment at the render
// site). Active mode fills dark, distinct from the red/blue onset toggles.
const STEM_MODE_LABELS: Record<StemMode, string> = {
  all: 'All',
  vocals: 'Vocals',
  drums: 'Drums',
  bass: 'Bass',
  instrumental: 'Instrumental',
}
const STEM_MODE_ACTIVE_COLOR = 'rgba(40, 40, 40, 0.9)'

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function styleRegion(el: HTMLElement | null, styles: Partial<CSSStyleDeclaration>) {
  if (el) Object.assign(el.style, styles)
}

// Layer 2's 8-count shading builder, shared verbatim by the main view and the
// minimap: regions are TIME-defined, so the same definitions render at any
// px-per-sec with no pixel recomputation. Alternate groups are shaded; the
// last group may be partial (schema v2) and runs to the end of the track.
// Times at/past duration are skipped (beat_this can emit a final beat there).
// pointerEvents none so shades never swallow clicks — wavesurfer's native
// click-to-seek on the main view, the seek handler on the minimap.
function addEightCountShading(
  regions: RegionsPlugin,
  grid: GridData,
  duration: number,
): Region[] {
  const boundaries = grid.eightCountIndices
    .map((i) => grid.beats[i])
    .filter((time) => time < duration)
  const added: Region[] = []
  boundaries.forEach((start, n) => {
    if (n % 2 !== 0) return
    const region = regions.addRegion({
      start,
      end: boundaries[n + 1] ?? duration,
      drag: false,
      resize: false,
      color: 'rgba(110, 110, 110, 0.1)',
    })
    styleRegion(region.element, { pointerEvents: 'none' })
    added.push(region)
  })
  return added
}

// Cut a bookmark's note down to BOOKMARK_LABEL_MAX_CHARS for the timeline
// glyph, with an ellipsis when there's more. Purely visual: the untruncated
// label stays on the record, in the glyph's tooltip, and in the edit field.
// A trailing space in the cut would render as a gap before the ellipsis.
function truncateLabel(label: string): string {
  if (label.length <= BOOKMARK_LABEL_MAX_CHARS) return label
  return `${label.slice(0, BOOKMARK_LABEL_MAX_CHARS).trimEnd()}…`
}

// One 8-count group as a time window. firstBeat indexes into beats; the k-th
// window spans [beats[ec[k]], beats[ec[k+1]]), and the last window runs to
// the track end and may hold fewer than 8 beats (ragged final group).
interface EightCountWindow {
  firstBeat: number
  beatCount: number
  start: number
  end: number
}

// Binary search for the window containing `time`; null before the first
// window (pickup beats precede the first downbeat, so beats[0] can sit
// outside every group) or at/past the end of the last.
function findEightCountIndex(windows: EightCountWindow[], time: number): number | null {
  let lo = 0
  let hi = windows.length - 1
  let candidate = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (windows[mid].start <= time) {
      candidate = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return candidate >= 0 && time < windows[candidate].end ? candidate : null
}

// Onset toggle buttons: active fills with the stem's marker color, inactive is
// the plain default button. Placeholder styling — a later design pass owns
// polish; only the active/inactive distinction matters here.
function toggleStyle(active: boolean, color: string): CSSProperties {
  return active ? { backgroundColor: color, borderColor: color, color: 'white' } : {}
}

function Timeline({ engine, trackId, grid, onsets, onReady }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const minimapRef = useRef<HTMLDivElement>(null)
  // Snap mode applies to the NEXT bookmark created (and to any shift) — it
  // never re-snaps bookmarks already placed.
  const [snapMode, setSnapMode] = useState<SnapMode>(DEFAULT_SNAP_MODE)
  const [selectedBookmarkId, setSelectedBookmarkId] = useState<string | null>(null)
  const bookmarkStore = useBookmarks(trackId)
  // Which stems are audible (and which waveform renders). Pure engine state:
  // switching mode ramps five gains and swaps waveform pixels — it must never
  // touch the grid, overlays, bookmarks, or the loop (all track properties).
  const [stemMode, setStemMode] = useState<StemMode>(DEFAULT_STEM_MODE)
  // Mirror for effects that need the CURRENT mode without re-running on mode
  // changes (the minimap is created once; its later re-skins come from the
  // mode effect, not from re-creation).
  const stemModeRef = useRef(stemMode)
  stemModeRef.current = stemMode
  // The minimap instance is kept in state (alongside its RegionsPlugin) so
  // the mode effect can re-render its waveform on stem switches.
  const [minimap, setMinimap] = useState<WaveSurfer | null>(null)
  const [minimapRegions, setMinimapRegions] = useState<RegionsPlugin | null>(null)
  // Session-only visibility toggles for the onset overlays; both start off,
  // so the baseline view is pulse ticks + 8-count shading with no markers.
  const [bassVisible, setBassVisible] = useState(false)
  const [drumsVisible, setDrumsVisible] = useState(false)
  // Which 8-count window the playhead is in (null = outside every window),
  // and the grid's RegionsPlugin instance so the Layer 6 effect can add its
  // regions to the SAME plugin — one coordinate space, one paint layer.
  const [currentEightCountIndex, setCurrentEightCountIndex] = useState<number | null>(null)
  const [gridRegions, setGridRegions] = useState<RegionsPlugin | null>(null)
  const { wavesurfer, isPlaying, currentTime } = useWavesurfer({
    container: containerRef,
    // NO url and NO HTMLAudioElement — the engine's media shim is the clock,
    // and precomputed peaks + duration put wavesurfer on its render-without-
    // audio path: nothing is fetched, nothing is decoded, no second buffer
    // copy exists. All three values are reference-stable per engine, so the
    // hook never re-creates the instance. (The component mounts with
    // DEFAULT_STEM_MODE, so those peaks are the correct initial render; mode
    // switches re-skin via the mode effect below, never via options.)
    media: engine.media,
    peaks: engine.peaksFor(DEFAULT_STEM_MODE),
    duration: engine.duration,
    height: 96,
    // Playback follow in the zoomed slice uses wavesurfer's own scrolling —
    // never a hand-rolled scroll transform (single time↔pixel authority).
    autoScroll: true,
    autoCenter: true,
  })

  // Readiness is derived HERE, not taken from the hook. With no URL to fetch
  // and no audio to decode, wavesurfer's load completes within microtasks of
  // creation — its 'ready' event fires BEFORE @wavesurfer/react's subscription
  // effect runs (a later commit), so the hook's isReady never turns true.
  // decodedData is assigned synchronously right before 'ready' is emitted,
  // so "already ready" is exactly getDecodedData() !== null; the event
  // subscription covers the (never-observed) slow path.
  const [isReady, setIsReady] = useState(false)
  useEffect(() => {
    if (!wavesurfer) return
    setIsReady(wavesurfer.getDecodedData() !== null)
    const unsubscribe = wavesurfer.on('ready', () => setIsReady(true))
    return () => {
      unsubscribe()
      setIsReady(false)
    }
  }, [wavesurfer])

  useEffect(() => {
    if (wavesurfer && isReady) onReady?.(wavesurfer)
  }, [wavesurfer, isReady, onReady])

  // Duration is CAPTURED ONCE, when wavesurfer reports ready — never re-read on
  // each render. The browser refines an MP3's media.duration after it has fully
  // parsed the file (seeking to the end triggers exactly that), and reading it
  // per render turned that refinement into a duration CHANGE. That rippled into
  // the memos the grid effect depends on, so the grid effect tore its
  // RegionsPlugin down and built a new one — while, in the same commit, every
  // effect that also depends on duration (the active-block labels, the bookmark
  // glyphs, the loop band) re-ran still holding the OLD, destroyed plugin and
  // called addRegion on it: "WaveSurfer is not initialized", and the whole app
  // unmounted. Freezing duration keeps the grid stable; a sub-millisecond
  // correction to the track length is worth nothing next to that.
  const [duration, setDuration] = useState(0)
  useEffect(() => {
    if (!wavesurfer || !isReady) return
    setDuration(wavesurfer.getDuration())
  }, [wavesurfer, isReady])

  // Loop points snap through the SAME capability bookmarks use, honoring the
  // snap mode selected right now — but DIRECTIONALLY (the loop floors A and
  // ceils B so it encloses whole musical units, where a bookmark takes the
  // nearest grid line). In "none" mode both keep the raw playhead time.
  const snapToMode = useCallback(
    (time: number, direction: SnapDirection) =>
      snapTime(time, snapMode, grid, duration, direction),
    [snapMode, grid, duration],
  )
  // The loop drives the ENGINE (native buffer-source looping); speed drives
  // the shared shim through wavesurfer. Same clock either way.
  const loop = useLoop(engine, snapToMode)
  const { speed, setSpeed, resetSpeed } = useSpeed(wavesurfer)

  // Per-tick absorption tags, computed once per analysis: nearBass/nearDrums
  // mark beats with a bass/drum onset within ABSORPTION_TOLERANCE_MS — the
  // same distance rule Layer 3 applied inline, cached here so render-time
  // suppression is a lookup gated on which stems are currently visible.
  const tickTags = useMemo(() => {
    if (!grid || !onsets || !duration) return null
    const near = (onsetTimes: number[]) => {
      const drawable = onsetTimes.filter((t) => t < duration)
      return grid.beats.map((beat) =>
        drawable.some((onset) => Math.abs(onset - beat) * 1000 < ABSORPTION_TOLERANCE_MS),
      )
    }
    return { nearBass: near(onsets.bass), nearDrums: near(onsets.drums) }
  }, [grid, onsets, duration])

  // Subdivisions: ONE tick at the temporal midpoint of each consecutive beat
  // pair — interpolated in TIME, so wavesurfer still owns time→pixel and
  // zoom/scroll come for free. Half-beat only (quarter-beat "e & a" is a
  // separate layer, deliberately not built here).
  //
  // The last beat has no successor, so it gets no trailing subdivision — the
  // loop stops at beats.length - 1 rather than extrapolating past the track.
  //
  // Unlike pulse ticks, subdivisions are NOT absorbed by nearby onsets. The
  // absorption rule exists because an onset marker and a beat tick are
  // collinear and read as one smeared line; a subdivision can't smear into
  // them because it lives in a different vertical band (the bottom
  // SUBDIVISION_TICK_HEIGHT_PCT%, while onset markers span 27.5–72.5%). And
  // suppressing them would be actively wrong: on a drum-dense block it deletes
  // most of the grid, and a metric ruler with holes in it is not a ruler.
  const subdivisions = useMemo(() => {
    if (!grid || !duration || grid.beats.length < 2) return null
    const { beats } = grid
    const midpoints: number[] = []
    for (let i = 0; i < beats.length - 1; i++) {
      const time = (beats[i] + beats[i + 1]) / 2
      if (time < duration) midpoints.push(time)
    }
    return midpoints
  }, [grid, duration])

  // 8-count windows for active-block tracking, derived once per analysis.
  const eightCountWindows = useMemo(() => {
    if (!grid || !duration) return null
    const { beats, eightCountIndices } = grid
    return eightCountIndices.map((firstBeat, k): EightCountWindow => {
      const nextFirst = eightCountIndices[k + 1]
      return {
        firstBeat,
        beatCount: (nextFirst ?? beats.length) - firstBeat,
        start: beats[firstBeat],
        end: nextFirst !== undefined ? beats[nextFirst] : duration,
      }
    })
  }, [grid, duration])

  // The RegionsPlugin is created ONCE per wavesurfer instance and outlives every
  // layer drawn into it. It deliberately does NOT get torn down when the grid
  // re-renders: every layer (grid, active-block labels, bookmarks, loop band)
  // adds regions to this one plugin, and several of them re-run on the same
  // dependency (duration). If a grid re-render destroyed and replaced the
  // plugin, those other effects would re-run in the SAME commit still holding
  // the old, destroyed instance — React has not flushed the new one into state
  // yet — and calling addRegion on it throws "WaveSurfer is not initialized",
  // unmounting the app. Stable plugin, per-layer region cleanup: no such window.
  useEffect(() => {
    if (!wavesurfer || !isReady) return
    const regions = wavesurfer.registerPlugin(RegionsPlugin.create())
    setGridRegions(regions)
    return () => {
      setGridRegions(null)
      regions.destroy()
    }
  }, [wavesurfer, isReady])

  // The grid is drawn with wavesurfer's RegionsPlugin rather than a hand-rolled
  // overlay: the plugin positions every element as a percentage of wavesurfer's
  // own wrapper, so the time→pixel mapping lives entirely inside wavesurfer and
  // survives container resizes with no coordinate math (and no cached pixels)
  // on our side. This effect owns only the regions it adds — it removes exactly
  // those on cleanup and never touches the plugin's lifetime.
  useEffect(() => {
    if (!gridRegions || !grid || !duration) return
    const regions = gridRegions
    const added: Region[] = []
    const addRegion = (params: Parameters<RegionsPlugin['addRegion']>[0]) => {
      const region = regions.addRegion(params)
      added.push(region)
      return region
    }
    const { beats, downbeatIndices } = grid
    const downbeats = new Set(downbeatIndices)

    // beat_this can emit a final beat at/past the audio end (e.g. exactly at
    // duration); wavesurfer can't position anything there, so such entries are
    // skipped rather than clamped onto the track edge.
    const drawable = (time: number) => time < duration

    // 8-count shading first, so tick markers paint above it (shared builder,
    // also drawn on the minimap).
    added.push(...addEightCountShading(regions, grid, duration))

    // A tick is suppressed only when it's near an onset from a CURRENTLY
    // VISIBLE stem (tags precomputed in tickTags with the Layer 3 rule).
    // With both toggles off nothing is suppressed — every drawable tick
    // renders; that full-grid baseline is intended.
    const suppressed = (i: number) =>
      tickTags !== null &&
      ((bassVisible && tickTags.nearBass[i]) || (drumsVisible && tickTags.nearDrums[i]))

    // Subdivision hairlines at the half-beat midpoints, drawn BEFORE the pulse
    // ticks so a beat always paints over a subdivision, never the reverse.
    // They share the beat ticks' bottom anchoring and z-index (they never
    // coincide in time, so they cannot overlap) but are shorter and fainter —
    // the subordination that keeps the beat grid readable.
    subdivisions?.forEach((time) => {
      const region = addRegion({ start: time, end: time, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${SUBDIVISION_TICK_WIDTH}px solid ${SUBDIVISION_TICK_COLOR}`,
        borderRadius: '0',
        height: `${SUBDIVISION_TICK_HEIGHT_PCT}%`,
        top: `${100 - SUBDIVISION_TICK_HEIGHT_PCT}%`,
        zIndex: '3',
      })
    })

    // Pulse ticks at every beat, bottom-anchored like a ruler; downbeats (each
    // bar's "1") are taller and darker. A suppressed tick (downbeats included,
    // no special-casing) is skipped — the onset marker takes its place.
    // pointerEvents none keeps wavesurfer's native click-to-seek working
    // through the grid.
    beats.forEach((time, i) => {
      if (!drawable(time)) return
      if (suppressed(i)) return
      const region = addRegion({
        start: time,
        end: time,
        drag: false,
        resize: false,
      })
      const emphasized = downbeats.has(i)
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: emphasized
          ? '2px solid rgba(40, 40, 40, 0.85)'
          : '1px solid rgba(110, 110, 110, 0.55)',
        borderRadius: '0',
        height: emphasized ? '55%' : '25%',
        top: emphasized ? '45%' : '75%',
        zIndex: '3',
      })
    })

    // Onset markers on top of the grid, drawn only for visible stems: bass is
    // the tallest marker, drums shorter than bass but taller than a normal
    // pulse tick. Independent layers — both render even if they land on the
    // same suppressed tick, and coincident bass+drum onsets (kick+bass on a
    // downbeat) both draw. Explicit z-index stacks bass over drums over ticks.
    if (drumsVisible && onsets) {
      onsets.drums.filter(drawable).forEach((time) => {
        const region = addRegion({ start: time, end: time, drag: false, resize: false })
        styleRegion(region.element, {
          pointerEvents: 'none',
          borderLeft: `${2 * ONSET_WIDTH_SCALE}px solid rgba(30, 90, 210, 0.9)`,
          borderRadius: '0',
          height: '45%',
          top: '27.5%',
          zIndex: '4',
        })
      })
    }
    if (bassVisible && onsets) {
      onsets.bass.filter(drawable).forEach((time) => {
        const region = addRegion({ start: time, end: time, drag: false, resize: false })
        styleRegion(region.element, {
          pointerEvents: 'none',
          borderLeft: `${3 * ONSET_WIDTH_SCALE}px solid rgba(210, 30, 30, 0.9)`,
          borderRadius: '0',
          height: '90%',
          top: '5%',
          zIndex: '5',
        })
      })
    }

    // Remove only what this effect drew; the plugin itself (and every other
    // layer's regions in it) is left alone.
    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, grid, duration, onsets, tickTags, subdivisions, bassVisible, drumsVisible])

  // Track which 8-count the playhead is in. The handler runs per timeupdate
  // but is just a binary search; state only changes when the index changes,
  // so nothing re-renders per frame. Seeks/scrubs need no special-casing —
  // timeupdate fires and the search lands wherever the playhead is.
  useEffect(() => {
    if (!wavesurfer || !isReady || !eightCountWindows) return
    const track = (time: number) => {
      const index = findEightCountIndex(eightCountWindows, time)
      setCurrentEightCountIndex((prev) => (prev === index ? prev : index))
    }
    track(wavesurfer.getCurrentTime())
    return wavesurfer.on('timeupdate', track)
  }, [wavesurfer, isReady, eightCountWindows])

  // Layer 6: highlight the active 8-count and number its beats 1..n. Extends
  // the Layer 2 regions (same plugin, wavesurfer-owned coordinates), so Layer
  // 4 scroll applies for free. Re-renders only when the active index or the
  // grid regions change — never per frame.
  useEffect(() => {
    if (!gridRegions || !grid || !eightCountWindows) return
    if (currentEightCountIndex === null) return
    const win = eightCountWindows[currentEightCountIndex]
    // beat_this can emit a final beat at/past the audio end; a window
    // starting there can't be positioned (same rule as the grid's drawable).
    if (!(win.start < duration)) return
    const added: Region[] = []

    // Fill via the region's own color, border via style. zIndex 2 keeps it
    // above the alternating 8-count shading (painted in DOM order) but below
    // pulse ticks (3) and onset markers (4/5).
    const highlight = gridRegions.addRegion({
      start: win.start,
      end: Math.min(win.end, duration),
      drag: false,
      resize: false,
      color: ACTIVE_BLOCK_FILL,
    })
    styleRegion(highlight.element, {
      pointerEvents: 'none',
      border: ACTIVE_BLOCK_BORDER,
      borderRadius: '0',
      boxSizing: 'border-box',
      zIndex: '2',
    })
    added.push(highlight)

    // Labels render unconditionally: the main view's zoom is fixed, always
    // dense enough that 1..8 digits can't collide.
    // Every group is labeled from 1: the backend (eight_count_grouping)
    // trims beats to the first downbeat and chunks by 8 from there, so a
    // group always starts on count 1 and only the LAST can be partial. If
    // the backend ever emits a partial FIRST group (a pickup group not
    // starting on count 1), this sequential labeling would be wrong.
    for (let i = 0; i < win.beatCount; i++) {
      const time = grid.beats[win.firstBeat + i]
      if (!(time < duration)) continue
      const label = gridRegions.addRegion({
        start: time,
        end: time,
        drag: false,
        resize: false,
        content: String(i + 1),
      })
      styleRegion(label.element, { pointerEvents: 'none', border: 'none', zIndex: '6' })
      if (label.content) Object.assign(label.content.style, COUNT_LABEL_STYLE)
      added.push(label)
    }

    // "&" on the midpoints BETWEEN this block's beats, giving
    // "1 & 2 & ... 7 & 8" — beatCount - 1 ampersands, so a ragged final group
    // gets only the "&"s that actually exist between its beats, unpadded.
    //
    // Note this stops at the block's last beat: the midpoint between count 8
    // and the NEXT block's count 1 draws a tick but no "&", because the label
    // run is specified to end on 8. Off the active block nothing is labeled at
    // all — the loop only ever walks the active window.
    for (let i = 0; i < win.beatCount - 1; i++) {
      const time = (grid.beats[win.firstBeat + i] + grid.beats[win.firstBeat + i + 1]) / 2
      if (!(time < duration)) continue
      const label = gridRegions.addRegion({
        start: time,
        end: time,
        drag: false,
        resize: false,
        content: '&',
      })
      styleRegion(label.element, { pointerEvents: 'none', border: 'none', zIndex: '6' })
      if (label.content) Object.assign(label.content.style, SUBDIVISION_LABEL_STYLE)
      added.push(label)
    }

    return () => {
      // A grid re-render destroys the whole plugin (removing every region)
      // before this cleanup sees the new plugin instance; only regions still
      // live need explicit removal.
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, grid, eightCountWindows, currentEightCountIndex, duration])

  // FIXED zoom (the app's only zoom — runtime zoom in/out is out of scope):
  // set wavesurfer's OWN zoom level once so SLICE_EIGHT_COUNTS eight-counts
  // fill the container; overview navigation is the minimap's job. Deriving
  // the level from musical duration is fine — it only configures wavesurfer's
  // pxPerSec, after which wavesurfer stays the sole authority for every
  // position. The median span between consecutive eight-count starts absorbs
  // tempo drift (spans between starts are always full groups; only the
  // segment after the last start can be partial, and it isn't a span).
  useEffect(() => {
    if (!wavesurfer || !isReady || !grid) return
    const { beats, eightCountIndices } = grid
    const starts = eightCountIndices.map((i) => beats[i])
    if (starts.length < 2) return // too little structure to size a slice; keep full-song view
    const spans = starts.slice(1).map((t, n) => t - starts[n])
    const median = spans.sort((a, b) => a - b)[Math.floor(spans.length / 2)]
    if (!(median > 0)) return
    const applyZoom = () => {
      const width = containerRef.current?.clientWidth
      if (width) wavesurfer.zoom(width / (SLICE_EIGHT_COUNTS * median))
    }
    applyZoom()
    // Re-derive the zoom level when the container resizes so the slice stays
    // SLICE_EIGHT_COUNTS wide; wavesurfer re-renders and repositions markers.
    const observer = new ResizeObserver(applyZoom)
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [wavesurfer, isReady, grid])

  // Minimap: a SECOND wavesurfer instance over the SAME engine shim — one
  // clock, two views. Playback, seeks, and timeupdate all live on the shared
  // engine, so both views follow automatically. Peaks and duration come
  // precomputed from the engine, which keeps construction side-effect free:
  // with no URL and peaks supplied, wavesurfer fetches and decodes nothing.
  // destroy() returns early for external media, so teardown never pauses
  // playback. Created with the CURRENT mode's peaks (ref, not dep) so a mode
  // switch never rebuilds it — later re-skins come from the mode effect.
  useEffect(() => {
    if (!wavesurfer || !isReady || !minimapRef.current) return
    const instance = WaveSurfer.create({
      container: minimapRef.current,
      media: engine.media,
      peaks: engine.peaksFor(stemModeRef.current),
      duration: engine.duration,
      height: MINIMAP_HEIGHT,
      fillParent: MINIMAP_FILL_PARENT,
      minPxPerSec: 0,
      // Transport contention guard (pitfall 1): the minimap never seeks,
      // plays, or pauses — interaction is disabled here and pointer events
      // are handled by seekMainTo against the MAIN instance only. Its cursor
      // doubles as the global playhead line at the shared currentTime.
      interact: false,
      dragToSeek: false,
      autoplay: false,
      autoScroll: false,
      autoCenter: false,
      cursorColor: MINIMAP_PLAYHEAD_COLOR,
      cursorWidth: MINIMAP_PLAYHEAD_WIDTH,
    })
    const regions = instance.registerPlugin(RegionsPlugin.create())
    setMinimap(instance)
    setMinimapRegions(regions)
    return () => {
      setMinimap(null)
      setMinimapRegions(null)
      instance.destroy()
    }
  }, [wavesurfer, isReady, engine])

  // Stem mode → (a) audible gains, ramped inside the engine, and (b) the
  // rendered waveform on both views, swapped by handing the renderer the
  // mode's precomputed peak buffer directly — the same internal path zoom()
  // takes, so regions survive untouched. NOTHING else may react to the mode:
  // beats, 8-counts, onsets, count labels, subdivisions, and bookmarks are
  // properties of the TRACK, not the stem, and none of their effects depend
  // on stemMode. Neither instance is destroyed or reloaded here (a reload
  // would flip isReady and churn every overlay effect).
  useEffect(() => {
    engine.setMode(stemMode)
    const display = engine.displayBufferFor(stemMode)
    if (wavesurfer && isReady) void wavesurfer.getRenderer().render(display)
    if (minimap) void minimap.getRenderer().render(display)
  }, [engine, stemMode, wavesurfer, isReady, minimap])

  // Minimap structure: the same 8-count shading the main view draws, from the
  // same builder — shades only. Ticks, onset markers, and count labels are
  // deliberately absent: at whole-track density they're sub-pixel mush.
  useEffect(() => {
    if (!minimapRegions || !grid || !duration) return
    const added = addEightCountShading(minimapRegions, grid, duration)
    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [minimapRegions, grid, duration])

  // Read-only viewport box: the time-slice the main view currently shows,
  // derived from the main renderer's scroll geometry and drawn as a
  // non-interactive region on the minimap (time-defined, so a minimap resize
  // repositions it for free). Updated on the main view's scroll AND on
  // timeupdate; it is an indicator only — not draggable.
  useEffect(() => {
    if (!wavesurfer || !minimapRegions || !duration) return
    let box: Region | null = null
    const update = () => {
      const scrollEl = wavesurfer.getWrapper().parentElement
      if (!scrollEl || !scrollEl.scrollWidth) return
      const start = Math.max(0, (scrollEl.scrollLeft / scrollEl.scrollWidth) * duration)
      const end = Math.min(
        duration,
        ((scrollEl.scrollLeft + scrollEl.clientWidth) / scrollEl.scrollWidth) * duration,
      )
      if (box && !box.isRemoved) {
        box.setOptions({ start, end })
      } else {
        box = minimapRegions.addRegion({
          start,
          end,
          drag: false,
          resize: false,
          color: VIEWPORT_BOX_FILL,
        })
        styleRegion(box.element, {
          pointerEvents: 'none',
          border: VIEWPORT_BOX_BORDER,
          borderRadius: '0',
          boxSizing: 'border-box',
          zIndex: '2',
        })
      }
    }
    update()
    const unsubscribe = [wavesurfer.on('scroll', update), wavesurfer.on('timeupdate', update)]
    return () => {
      unsubscribe.forEach((u) => u())
      if (box && !box.isRemoved) box.remove()
    }
  }, [wavesurfer, minimapRegions, duration])

  // Bookmark glyphs on the MAIN (detail) view: a brown stem with a tappable
  // diamond at the bottom edge. Bookmarks are positioned by TIME like every
  // other region, so a "none"-mode bookmark renders exactly where it was
  // stored — between beat markers — and is never pulled onto a grid line.
  // Two bookmarks on the same grid point simply draw on top of each other.
  useEffect(() => {
    if (!gridRegions || !duration) return
    const added: Region[] = []
    bookmarkStore.bookmarks.forEach((entry) => {
      if (!(entry.timestamp < duration)) return // can't position at/past the end
      const selected = entry.id === selectedBookmarkId
      const color = selected ? BOOKMARK_SELECTED_HUE : BOOKMARK_HUE

      // Diamond + the user's label, side by side at the bottom edge. The whole
      // group is the hit target: the region element is pointerEvents:none so it
      // never blocks click-to-seek on the waveform, and this content opts back
      // in — a 2px stem alone would be nearly unclickable.
      const glyph = document.createElement('div')
      Object.assign(glyph.style, {
        position: 'absolute',
        bottom: '0',
        left: `${-BOOKMARK_GLYPH_SIZE / 2}px`,
        display: 'flex',
        alignItems: 'center',
        gap: `${BOOKMARK_GLYPH_SIZE / 2}px`,
        whiteSpace: 'nowrap',
        pointerEvents: 'auto',
        cursor: 'pointer',
      })
      glyph.title = entry.label || 'Bookmark (no label)'
      const diamond = document.createElement('div')
      Object.assign(diamond.style, {
        flex: '0 0 auto',
        width: `${BOOKMARK_GLYPH_SIZE}px`,
        height: `${BOOKMARK_GLYPH_SIZE}px`,
        backgroundColor: color,
        transform: BOOKMARK_GLYPH_SHAPE,
      })
      glyph.appendChild(diamond)
      if (entry.label) {
        const text = document.createElement('span')
        // Truncated for the timeline; the tooltip above carries the full note.
        text.textContent = truncateLabel(entry.label)
        Object.assign(text.style, BOOKMARK_LABEL_STYLE, { color })
        glyph.appendChild(text)
      }
      glyph.addEventListener('click', (e) => {
        e.stopPropagation()
        setSelectedBookmarkId((prev) => (prev === entry.id ? null : entry.id))
      })
      const region = gridRegions.addRegion({
        start: entry.timestamp,
        end: entry.timestamp,
        drag: false,
        resize: false,
        content: glyph,
      })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${BOOKMARK_STEM_WIDTH}px solid ${color}`,
        borderRadius: '0',
        height: '100%',
        top: '0',
        zIndex: '7', // above ticks, onsets, and Layer 6 labels
      })
      added.push(region)
    })
    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, duration, bookmarkStore.bookmarks, selectedBookmarkId])

  // Bookmark ticks on the MINIMAP: navigation targets in the overview. Tapping
  // one seeks to the bookmark's exact stored time (not the clicked pixel), via
  // the same seek path the strip uses. stopPropagation keeps the strip's own
  // pointer handler from also seeking to the click x.
  useEffect(() => {
    if (!minimapRegions || !duration) return
    const added: Region[] = []
    bookmarkStore.bookmarks.forEach((entry) => {
      if (!(entry.timestamp < duration)) return
      const selected = entry.id === selectedBookmarkId
      const region = minimapRegions.addRegion({
        start: entry.timestamp,
        end: entry.timestamp,
        drag: false,
        resize: false,
      })
      styleRegion(region.element, {
        pointerEvents: 'auto',
        cursor: 'pointer',
        borderLeft: `${BOOKMARK_TICK_WIDTH}px solid ${
          selected ? BOOKMARK_SELECTED_HUE : BOOKMARK_HUE
        }`,
        borderRadius: '0',
        height: '100%',
        top: '0',
        zIndex: '3',
      })
      region.element?.addEventListener('pointerdown', (e) => {
        e.stopPropagation()
        seekMainToTime(entry.timestamp)
      })
      added.push(region)
    })
    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
    // seekMainToTime closes over wavesurfer/duration, both in the deps already.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minimapRegions, duration, bookmarkStore.bookmarks, selectedBookmarkId, wavesurfer])

  // Loop markers on BOTH views. The A and B EDGES render INDEPENDENTLY: A shows
  // the moment A is set, B the moment B is set — neither waits on the other, so
  // you see each endpoint land as you place it. The teal wash between them is a
  // separate fill that only exists once both endpoints do (a fill needs a span).
  // Everything is time-defined, so the main view's zoom/scroll and the minimap's
  // fit-to-width place it with no pixel math; adding or clearing mid-playback
  // touches nothing the playhead or viewport box depend on; non-interactive, so
  // it can't eat the minimap's seek gesture.
  useEffect(() => {
    if (!duration) return
    const added: Region[] = []
    // A loop that's set but switched off stays visible, dimmed.
    const opacity = loop.enabled ? '1' : LOOP_DISABLED_OPACITY

    // A single vertical edge line at one endpoint, on one plugin.
    const edge = (regions: RegionsPlugin | null, time: number, zIndex: string) => {
      if (!regions) return
      const at = Math.min(time, duration)
      const region = regions.addRegion({ start: at, end: at, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${LOOP_BAND_EDGE_WIDTH}px solid ${LOOP_BAND_EDGE}`,
        borderRadius: '0',
        height: '100%',
        top: '0',
        opacity,
        zIndex,
      })
      added.push(region)
    }
    // The wash spanning A..B, drawn only when BOTH endpoints exist.
    const fill = (regions: RegionsPlugin | null, color: string, zIndex: string) => {
      if (!regions || loop.start === null || loop.end === null) return
      const region = regions.addRegion({
        start: loop.start,
        end: Math.min(loop.end, duration),
        drag: false,
        resize: false,
        color,
      })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderRadius: '0',
        opacity,
        zIndex,
      })
      added.push(region)
    }

    // Fill first, then edges over it. Minimap stays under the viewport box (z2).
    fill(gridRegions, LOOP_BAND_FILL, '2')
    fill(minimapRegions, LOOP_MINIMAP_FILL, '1')
    if (loop.start !== null) {
      edge(gridRegions, loop.start, '3')
      edge(minimapRegions, loop.start, '1')
    }
    if (loop.end !== null) {
      edge(gridRegions, loop.end, '3')
      edge(minimapRegions, loop.end, '1')
    }

    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, minimapRegions, duration, loop.start, loop.end, loop.enabled])

  // The one seek path: move the MAIN instance — the single transport owner;
  // the shared clock moves both views. Manual seeks don't auto-scroll a paused
  // main view (autoScroll follows playback), so it's centered explicitly;
  // during playback the next autoCenter lands on the same target.
  const seekMainToTime = (time: number) => {
    if (!wavesurfer || !duration) return
    const clamped = Math.min(Math.max(time, 0), duration)
    wavesurfer.setTime(clamped)
    const scrollEl = wavesurfer.getWrapper().parentElement
    if (scrollEl && scrollEl.scrollWidth > scrollEl.clientWidth) {
      wavesurfer.setScroll(
        (clamped / duration) * scrollEl.scrollWidth - scrollEl.clientWidth / 2,
      )
    }
  }

  // Minimap seek surface: map pointer x → time, then seek. Clamped, so a drag
  // that overshoots the strip pins to the track edges.
  const seekMainToX = (clientX: number, surface: HTMLDivElement) => {
    if (!duration) return
    const rect = surface.getBoundingClientRect()
    if (!rect.width) return
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
    seekMainToTime(fraction * duration)
  }

  // Creating a bookmark selects it, so its info panel opens straight away —
  // shift/delete/close/note are all one step from the Bookmark button.
  const createBookmark = () => {
    if (!wavesurfer || !duration) return
    const id = bookmarkStore.create(
      snapTime(wavesurfer.getCurrentTime(), snapMode, grid, duration),
      snapMode,
    )
    setSelectedBookmarkId(id)
  }

  // SHIFT: move the selected bookmark to the playhead, re-snapped with the
  // mode selected NOW ("move it the way I'm snapping today"). In "none" mode
  // that lands on the raw playhead time, off-grid.
  const shiftSelected = () => {
    if (!wavesurfer || !duration || !selectedBookmarkId) return
    bookmarkStore.shift(
      selectedBookmarkId,
      snapTime(wavesurfer.getCurrentTime(), snapMode, grid, duration),
      snapMode,
    )
  }

  const deleteSelected = () => {
    if (!selectedBookmarkId) return
    bookmarkStore.remove(selectedBookmarkId)
    setSelectedBookmarkId(null)
  }

  const selectedBookmark =
    bookmarkStore.bookmarks.find((entry) => entry.id === selectedBookmarkId) ?? null

  return (
    <div className="timeline">
      <div ref={containerRef} />
      <div
        ref={minimapRef}
        className="timeline-minimap"
        onPointerDown={(e) => {
          // Capture so a drag keeps seeking even when the pointer leaves the
          // strip; x is clamped, so overshoot pins to the track edges.
          e.currentTarget.setPointerCapture(e.pointerId)
          seekMainToX(e.clientX, e.currentTarget)
        }}
        onPointerMove={(e) => {
          if (e.buttons & 1) seekMainToX(e.clientX, e.currentTarget)
        }}
      />
      <div className="timeline-controls">
        <button onClick={() => wavesurfer?.playPause()} disabled={!isReady}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <span>
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <button
          onClick={() => setBassVisible((v) => !v)}
          disabled={!onsets}
          aria-pressed={bassVisible}
          style={toggleStyle(bassVisible, 'rgba(210, 30, 30, 0.9)')}
        >
          Bass
        </button>
        <button
          onClick={() => setDrumsVisible((v) => !v)}
          disabled={!onsets}
          aria-pressed={drumsVisible}
          style={toggleStyle(drumsVisible, 'rgba(30, 90, 210, 0.9)')}
        >
          Drums
        </button>
      </div>
      {/*
        PROVISIONAL UI — stem mode selector. Five mutually exclusive taps over
        the engine's mode table; the active mode is shown filled. Tap only, by
        design: NO hover and NO drag interactions — this app is mobile-first
        and the touch/gesture pass comes later (drag interactions would need a
        rewrite, not a refactor). Replace this row wholesale in that pass; the
        engine's setMode underneath does not change.
      */}
      <div className="timeline-controls" role="group" aria-label="Stem mode">
        <span>Hear:</span>
        {STEM_MODES.map((mode) => (
          <button
            key={mode}
            onClick={() => setStemMode(mode)}
            aria-pressed={stemMode === mode}
            style={toggleStyle(stemMode === mode, STEM_MODE_ACTIVE_COLOR)}
          >
            {STEM_MODE_LABELS[mode]}
          </button>
        ))}
      </div>
      <div className="timeline-controls">
        <label>
          Snap{' '}
          <select
            value={snapMode}
            onChange={(e) => setSnapMode(e.target.value as SnapMode)}
            aria-label="Snap mode"
          >
            {SNAP_MODES.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={createBookmark}
          disabled={!isReady}
          style={{ backgroundColor: BOOKMARK_HUE, borderColor: BOOKMARK_HUE, color: 'white' }}
        >
          Bookmark
        </button>
        <span>
          {bookmarkStore.bookmarks.length} bookmark
          {bookmarkStore.bookmarks.length === 1 ? '' : 's'}
        </span>
      </div>
      {/*
        PROVISIONAL UI — placeholder trigger surface only. The touch/gesture model
        for setting loop points is not decided yet (drag handles? tap two
        bookmarks?), so these buttons are a thin shim over the loop engine's
        imperative API (setLoop / setLoopStart / setLoopEnd / clearLoop /
        setEnabled). Replace this row wholesale in the mobile pass; the engine
        underneath does not change.
      */}
      <div className="timeline-controls">
        <button onClick={() => loop.setLoopStart(currentTime)} disabled={!isReady}>
          Set A
        </button>
        <button onClick={() => loop.setLoopEnd(currentTime)} disabled={!isReady}>
          Set B
        </button>
        <button
          onClick={() => loop.setEnabled(!loop.enabled)}
          disabled={loop.start === null || loop.end === null}
          aria-pressed={loop.enabled}
          style={toggleStyle(loop.enabled, LOOP_BAND_EDGE)}
        >
          Loop
        </button>
        <button onClick={loop.clearLoop} disabled={loop.start === null && loop.end === null}>
          Clear
        </button>
        <span>
          {loop.start !== null || loop.end !== null
            ? `A ${loop.start !== null ? loop.start.toFixed(2) : '—'} / B ${
                loop.end !== null ? loop.end.toFixed(2) : '—'
              }`
            : 'no loop'}
        </span>
        {/*
          Speed: live playbackRate on all five stem sources (0.25x-2x). Rate
          RESAMPLES — pitch shifts with speed — by accepted tradeoff; no
          time-stretch. Grid, bookmarks, and loop A/B are track-time and do
          not move with the rate.
        */}
        <label>
          Speed{' '}
          <input
            type="range"
            min={SPEED_MIN}
            max={SPEED_MAX}
            step={SPEED_STEP}
            value={speed}
            aria-label="Playback speed"
            onChange={(e) => setSpeed(Number(e.target.value))}
          />
        </label>
        <span>{speed.toFixed(2)}×</span>
        <button onClick={resetSpeed} disabled={speed === SPEED_DEFAULT}>
          1×
        </button>
      </div>
      {loop.error && <p className="error">{loop.error}</p>}
      {selectedBookmark && (
        <BookmarkInfo
          // Keyed by id so switching bookmarks resets the label draft rather
          // than carrying the previous bookmark's typing into this one.
          key={selectedBookmark.id}
          bookmark={selectedBookmark}
          snapMode={snapMode}
          onSeek={() => seekMainToTime(selectedBookmark.timestamp)}
          onShift={shiftSelected}
          onDelete={deleteSelected}
          onRelabel={(label) => bookmarkStore.relabel(selectedBookmark.id, label)}
          onClose={() => setSelectedBookmarkId(null)}
        />
      )}
      {bookmarkStore.error && <p className="error">Bookmarks: {bookmarkStore.error}</p>}
    </div>
  )
}

// Info panel for the tapped bookmark: its id, time, and the mode it was
// created with — plus an editable label, SHIFT (move to the playhead,
// re-snapped by the mode selected NOW, which may differ from the creation
// mode), and DELETE.
function BookmarkInfo({
  bookmark,
  snapMode,
  onSeek,
  onShift,
  onDelete,
  onRelabel,
  onClose,
}: {
  bookmark: Bookmark
  snapMode: SnapMode
  onSeek: () => void
  onShift: () => void
  onDelete: () => void
  onRelabel: (label: string) => void
  onClose: () => void
}) {
  // The input is a local draft: it commits on blur or Enter, so typing a label
  // is ONE write, not one PUT per keystroke. Escape abandons the edit. The
  // draft resyncs if the stored label changes underneath us — which is how a
  // rolled-back failed write puts the old text back in the box.
  const [draft, setDraft] = useState(bookmark.label)
  useEffect(() => setDraft(bookmark.label), [bookmark.label])

  // Escape blurs the input, and blur is the commit trigger — without this flag
  // the blur handler would save the very text Escape just abandoned (it still
  // sees the pre-Escape draft in its closure).
  const abandoning = useRef(false)
  const commitDraft = () => {
    if (abandoning.current) {
      abandoning.current = false
      setDraft(bookmark.label)
      return
    }
    onRelabel(draft)
  }

  return (
    <div className="bookmark-info" style={{ borderColor: BOOKMARK_HUE }}>
      <span>
        <code>{bookmark.id.slice(0, 8)}</code> @ {bookmark.timestamp.toFixed(3)}s (snap:{' '}
        {bookmark.snap_mode})
      </span>
      <input
        value={draft}
        placeholder="Add a note…"
        aria-label="Bookmark label"
        // The panel is keyed by bookmark id, so this mounts fresh each time one
        // is opened: the note field is ready to type into immediately after
        // hitting Bookmark.
        autoFocus
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commitDraft}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.currentTarget.blur() // blur commits; relabel de-dupes the repeat
          } else if (e.key === 'Escape') {
            abandoning.current = true
            e.currentTarget.blur()
          }
        }}
      />
      <button onClick={onSeek}>Go to</button>
      <button onClick={onShift}>Shift to playhead ({snapMode})</button>
      <button onClick={onDelete}>Delete</button>
      <button onClick={onClose}>Close</button>
    </div>
  )
}

export default Timeline
