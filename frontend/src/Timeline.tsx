import { useEffect, useMemo, useRef, useState } from 'react'
import { useWavesurfer } from '@wavesurfer/react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/plugins/regions'
import { PlayPauseButton, SpeedControl, TimeReadout } from './controls'
import { useCloseOnOutsideClick } from './dropdown'
import { toggleStyle } from './styles'
import { findEightCountIndex, type EightCountWindow } from './eightCount'
import {
  LOOP_BAND_EDGE,
  LOOP_BAND_EDGE_WIDTH,
  LOOP_BAND_FILL,
  type LoopController,
  type SpeedController,
  type Transport,
} from './playback'
import { DEFAULT_STEM_MODE, STEM_MODES, type StemEngine, type StemMode } from './stemEngine'
import type { TapSession } from './tapSession'
import TimelineOverview from './TimelineOverview'
import type { GridData, OnsetData } from './types'
import TapOverlay from './TapOverlay'

// Re-exported for the modules that still name these through the Timeline.
// They are defined in types.ts now: the shell owns them, not this screen.
export type { GridData, OnsetData } from './types'

interface TimelineProps {
  // The fully-loaded stem engine: audio source, clock authority, and peaks
  // provider. The Timeline never fetches or decodes audio itself — both
  // wavesurfer instances are views over this engine's shim + peak data.
  engine: StemEngine
  // ALL of the state below is owned by the Song shell above this screen. This
  // component is a VIEW: it renders that state and calls back into it. It
  // deliberately holds no transport, loop, speed, snap, stem-mode, or grid
  // state of its own, so swapping to another screen and back cannot lose or
  // fork any of it.
  transport: Transport
  loop: LoopController
  speed: SpeedController
  duration: number
  // THE grid to draw: the tap preview when a session is open, the saved grid
  // otherwise, undefined when neither exists. Nothing below this line knows a
  // grid can be tapped.
  grid?: GridData
  // Whether a PERSISTED grid exists — distinct from `grid`, which may be a
  // preview. Gates the re-tap control, which acts on the saved grid.
  hasSavedGrid: boolean
  windows: EightCountWindow[] | null
  onsets?: OnsetData
  stemMode: StemMode
  onStemModeChange: (mode: StemMode) => void
  tap: TapSession
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

// Tap markers: where the user's taps landed, in a hue used nowhere else
// (green — clear of drums-blue, bass-red, the Layer 6 yellow, and the loop's
// teal). Full height and above every grid layer, because while tapping these
// ARE the subject.
const TAP_MARKER_COLOR = 'rgba(20, 150, 60, 0.9)'
const TAP_MARKER_WIDTH = 2

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

function styleRegion(el: HTMLElement | null, styles: Partial<CSSStyleDeclaration>) {
  if (el) Object.assign(el.style, styles)
}

// Layer 2's 8-count shading builder, shared verbatim by the main view and the
// minimap: regions are TIME-defined, so the same definitions render at any
// px-per-sec with no pixel recomputation. Alternate groups are shaded; the
// last group may be partial and runs to the end of the track. Times at/past
// duration are skipped (gridFromFit's 3dp rounding can land the final beat
// there). pointerEvents none so shades never swallow clicks — wavesurfer's
// native click-to-seek on the main view, the seek handler on the minimap.
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

function Timeline({
  engine,
  transport,
  loop,
  speed,
  duration,
  grid,
  hasSavedGrid,
  windows,
  onsets,
  stemMode,
  onStemModeChange,
  tap,
  onReady,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  // Session-only visibility toggles for the onset overlays; both start off,
  // so the baseline view is pulse ticks + 8-count shading with no markers.
  // These are the one kind of state this screen legitimately owns: they change
  // nothing about the song, only what this view draws.
  const [bassVisible, setBassVisible] = useState(false)
  const [drumsVisible, setDrumsVisible] = useState(false)
  // The Bass/Drums toggles above live behind a single "Onset" dropdown so the
  // controls row doesn't grow with every stem this app learns to detect.
  const [onsetMenuOpen, setOnsetMenuOpen] = useState(false)
  const onsetMenuRef = useRef<HTMLDivElement>(null)
  useCloseOnOutsideClick(onsetMenuOpen, setOnsetMenuOpen, onsetMenuRef)
  // Same pattern for the stem mode selector, collapsed behind a "Hear" dropdown.
  const [hearMenuOpen, setHearMenuOpen] = useState(false)
  const hearMenuRef = useRef<HTMLDivElement>(null)
  useCloseOnOutsideClick(hearMenuOpen, setHearMenuOpen, hearMenuRef)
  // Which 8-count window the playhead is in (null = outside every window),
  // and the grid's RegionsPlugin instance so the Layer 6 effect can add its
  // regions to the SAME plugin — one coordinate space, one paint layer.
  const [currentEightCountIndex, setCurrentEightCountIndex] = useState<number | null>(null)
  const [gridRegions, setGridRegions] = useState<RegionsPlugin | null>(null)
  const { wavesurfer } = useWavesurfer({
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

  // The RegionsPlugin is created ONCE per wavesurfer instance and outlives every
  // layer drawn into it. It deliberately does NOT get torn down when the grid
  // re-renders: every layer (grid, active-block labels, loop band) adds
  // regions to this one plugin, and several of them re-run on the same
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

  // Tap markers: the one thing that IS live per tap. One region each, so the
  // per-tap cost stays flat no matter how long the grid is.
  useEffect(() => {
    if (!gridRegions || !duration || !tap.tapping) return
    const added: Region[] = []
    tap.taps.forEach((time) => {
      if (!(time < duration)) return
      const region = gridRegions.addRegion({
        start: time,
        end: time,
        drag: false,
        resize: false,
      })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${TAP_MARKER_WIDTH}px solid ${TAP_MARKER_COLOR}`,
        borderRadius: '0',
        height: '100%',
        top: '0',
        zIndex: '7', // above every grid layer: while tapping, these are the subject
      })
      added.push(region)
    })
    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, duration, tap.taps, tap.tapping])

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

    // gridFromFit's 3dp rounding can land the final beat at/past the audio
    // end; wavesurfer can't position anything there, so such entries are
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
    //
    // Onsets are measured from the audio and are true regardless of what the
    // grid makes of them. When the grid is the thing in doubt they are the
    // evidence you check it against, so they keep full contrast.
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
  //
  // Ask the clock, NEVER the event's payload. Wavesurfer emits timeupdate from
  // a reactive effect subscribed to BOTH currentTime and isPlaying, and it
  // passes the cached currentTime — a store refreshed only by media
  // 'timeupdate' events, which the engine's shim deliberately does not stream
  // (it emits discrete edges only). So the store idles at its initial 0, and
  // PAUSING — an isPlaying change — re-fires the effect with that stale 0. Read
  // literally, that says "the playhead is at 0:00, outside every 8-count", and
  // the highlight tore itself down on every pause while the audio sat happily
  // mid-track. getCurrentTime() goes straight to the engine's live clock and is
  // always right, paused or not.
  useEffect(() => {
    if (!wavesurfer || !isReady || !windows) return
    const track = () => {
      const index = findEightCountIndex(windows, wavesurfer.getCurrentTime())
      setCurrentEightCountIndex((prev) => (prev === index ? prev : index))
    }
    track()
    return wavesurfer.on('timeupdate', track)
  }, [wavesurfer, isReady, windows])

  // Layer 6: highlight the active 8-count and number its beats 1..n. Extends
  // the Layer 2 regions (same plugin, wavesurfer-owned coordinates), so Layer
  // 4 scroll applies for free. Re-renders only when the active index or the
  // grid regions change — never per frame.
  useEffect(() => {
    if (!gridRegions || !grid || !windows) return
    // Off entirely while tapping. This layer asserts a count — a yellow block
    // sliding under the playhead with 1..8 written on it — and during tap mode
    // that count is the one the user is in the middle of replacing, so it is
    // both wrong and loud, right where they need to watch their own markers.
    // The pad's own 1..8 readout is the count that matters here; the phrase
    // boundaries stay legible through Layer 2's shading and the downbeat ticks,
    // so the ±1 count nudge is still judgeable without this.
    if (tap.tapping) return
    if (currentEightCountIndex === null) return
    const win = windows[currentEightCountIndex]
    // A grid's final beat can land at/past the audio end (3dp rounding); a
    // window starting there can't be positioned (same rule as the grid's
    // drawable).
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
    // Every group is labeled from 1: gridFromFit marks an eight-count start
    // every 8 beats from the tapped (or count-nudged) "1", so a group always
    // starts on count 1 and only the LAST can be partial. If a grid ever
    // carried a partial FIRST group (a pickup group not starting on count 1),
    // this sequential labeling would be wrong.
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
      styleRegion(label.element, {
        pointerEvents: 'none',
        border: 'none',
        zIndex: '6',
      })
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
      styleRegion(label.element, {
        pointerEvents: 'none',
        border: 'none',
        zIndex: '6',
      })
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
  }, [gridRegions, grid, windows, currentEightCountIndex, duration, tap.tapping])

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

  // Stem mode → (a) audible gains, ramped inside the engine, and (b) the
  // rendered waveform, swapped by handing the renderer the mode's precomputed
  // peak buffer directly — the same internal path zoom() takes, so regions
  // survive untouched. NOTHING else may react to the mode: beats, 8-counts,
  // onsets, count labels, and subdivisions are properties of the TRACK, not
  // the stem, and none of their effects depend on stemMode. The instance is
  // never destroyed or reloaded here (a reload would flip isReady and churn
  // every overlay effect).
  useEffect(() => {
    engine.setMode(stemMode)
    const display = engine.displayBufferFor(stemMode)
    if (wavesurfer && isReady) void wavesurfer.getRenderer().render(display)
  }, [engine, stemMode, wavesurfer, isReady])

  // Loop markers on the main view. The always-active loop (see useLoop) is
  // drawn at full opacity — there is no "set but off" state to dim anymore.
  // The A and B EDGES render INDEPENDENTLY: A shows the moment A is set, B
  // the moment B is set — neither waits on the other, so you see each
  // endpoint land as you place it. The teal wash between them is a separate
  // fill. Everything is time-defined, so the main view's zoom/scroll place it
  // with no pixel math; non-interactive, so it can't eat the view's own
  // click-to-seek. The overall timeline below draws its own copy of A/B (see
  // TimelineOverview) since it isn't a wavesurfer instance and has no
  // regions plugin to share.
  useEffect(() => {
    if (!gridRegions || !duration) return
    const added: Region[] = []

    // A single vertical edge line at one endpoint.
    const edge = (time: number, zIndex: string) => {
      const at = Math.min(time, duration)
      const region = gridRegions.addRegion({ start: at, end: at, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${LOOP_BAND_EDGE_WIDTH}px solid ${LOOP_BAND_EDGE}`,
        borderRadius: '0',
        height: '100%',
        top: '0',
        zIndex,
      })
      added.push(region)
    }
    // The wash spanning A..B.
    const region = gridRegions.addRegion({
      start: loop.start,
      end: Math.min(loop.end, duration),
      drag: false,
      resize: false,
      color: LOOP_BAND_FILL,
    })
    styleRegion(region.element, {
      pointerEvents: 'none',
      borderRadius: '0',
      zIndex: '2',
    })
    added.push(region)
    edge(loop.start, '3')
    edge(loop.end, '3')

    return () => {
      added.forEach((region) => {
        if (!region.isRemoved) region.remove()
      })
    }
  }, [gridRegions, duration, loop.start, loop.end])

  // The one seek path: move the shared clock through the hoisted transport —
  // never a view seeking another view. Manual seeks don't auto-scroll a paused
  // main view (autoScroll follows playback), so it's centered explicitly;
  // during playback the next autoCenter lands on the same target. The centering
  // is the only part of this that is a VIEW concern, which is why it lives here
  // and the seek itself does not.
  const seekAndCenter = (time: number) => {
    if (!wavesurfer || !duration) return
    const clamped = Math.min(Math.max(time, 0), duration)
    transport.seek(clamped)
    const scrollEl = wavesurfer.getWrapper().parentElement
    if (scrollEl && scrollEl.scrollWidth > scrollEl.clientWidth) {
      wavesurfer.setScroll(
        (clamped / duration) * scrollEl.scrollWidth - scrollEl.clientWidth / 2,
      )
    }
  }

  return (
    <div className="timeline">
      <div ref={containerRef} />
      <TimelineOverview duration={duration} currentTime={transport.currentTime} loop={loop} onSeek={seekAndCenter} />
      <div className="timeline-controls">
        <PlayPauseButton transport={transport} disabled={!isReady} />
        <TimeReadout transport={transport} grid={grid} windows={windows} />
        <div className="dropdown" ref={onsetMenuRef}>
          <button
            onClick={() => setOnsetMenuOpen((v) => !v)}
            disabled={!onsets}
            aria-haspopup="true"
            aria-expanded={onsetMenuOpen}
            style={toggleStyle(bassVisible || drumsVisible, 'rgba(90, 40, 160, 0.9)')}
          >
            Onset ▾
          </button>
          {onsetMenuOpen && (
            <div className="dropdown-list" role="menu">
              <button
                role="menuitemcheckbox"
                onClick={() => setBassVisible((v) => !v)}
                aria-pressed={bassVisible}
                style={toggleStyle(bassVisible, 'rgba(210, 30, 30, 0.9)')}
              >
                Bass
              </button>
              <button
                role="menuitemcheckbox"
                onClick={() => setDrumsVisible((v) => !v)}
                aria-pressed={drumsVisible}
                style={toggleStyle(drumsVisible, 'rgba(30, 90, 210, 0.9)')}
              >
                Drums
              </button>
            </div>
          )}
        </div>
        {/*
          Re-tap is PERMANENT: always here, next to the transport, never locked
          away once a grid exists — not behind a settings menu and not
          conditional on anything tripping. Users get the count wrong sometimes
          and must be able to redo it without re-importing the song. Hidden only
          during first run, where tap mode is already forced open and there is
          no grid to exit back to.
        */}
        {hasSavedGrid && (
          <button
            className="tap-enter"
            onClick={tap.tapping ? tap.exit : tap.enter}
            disabled={!isReady}
            aria-pressed={tap.tapping}
          >
            {tap.tapping ? 'Exit tap mode' : 'Re-tap the count'}
          </button>
        )}
      </div>
      {tap.tapping && (
        <TapOverlay
          taps={tap.taps}
          isPlaying={transport.isPlaying}
          fit={tap.fit}
          fitError={tap.error}
          onTap={tap.record}
          onAccept={tap.accept}
          // First run has no grid to fall back to, so "cancel" just clears the
          // session and stays in the tap state (nothing was persisted — there
          // is no draft). With a grid, cancel returns to it untouched.
          cancelLabel={hasSavedGrid ? 'Cancel' : 'Start over'}
          onCancel={hasSavedGrid ? tap.exit : tap.enter}
        />
      )}
      {/*
        PROVISIONAL UI — stem mode selector. Five mutually exclusive taps over
        the engine's mode table; the active mode is shown filled. Tap only, by
        design: NO hover and NO drag interactions — this app is mobile-first
        and the touch/gesture pass comes later (drag interactions would need a
        rewrite, not a refactor). Replace this row wholesale in that pass; the
        engine's setMode underneath does not change.
      */}
      <div className="timeline-controls" role="group" aria-label="Stem mode">
        <div className="dropdown" ref={hearMenuRef}>
          <button
            onClick={() => setHearMenuOpen((v) => !v)}
            aria-haspopup="true"
            aria-expanded={hearMenuOpen}
            style={toggleStyle(stemMode !== 'all', STEM_MODE_ACTIVE_COLOR)}
          >
            Hear: {STEM_MODE_LABELS[stemMode]} ▾
          </button>
          {hearMenuOpen && (
            <div className="dropdown-list" role="menu">
              {STEM_MODES.map((mode) => (
                <button
                  key={mode}
                  role="menuitemradio"
                  aria-checked={stemMode === mode}
                  onClick={() => {
                    onStemModeChange(mode)
                    setHearMenuOpen(false)
                  }}
                  style={toggleStyle(stemMode === mode, STEM_MODE_ACTIVE_COLOR)}
                >
                  {STEM_MODE_LABELS[mode]}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="timeline-controls">
        <SpeedControl speed={speed} />
      </div>
      {loop.error && <p className="error">{loop.error}</p>}
    </div>
  )
}

export default Timeline
