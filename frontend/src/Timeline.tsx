import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { useWavesurfer } from '@wavesurfer/react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/plugins/regions'
import { HearControl, PlayPauseButton, SpeedControl, TimeReadout } from './controls'
import { colorToken, type ColorToken } from './styles'
import { findEightCountIndex, type EightCountWindow } from './eightCount'
import {
  LOOP_BAND_EDGE_A,
  LOOP_BAND_EDGE_B,
  LOOP_BAND_EDGE_WIDTH,
  type LoopController,
  type SpeedController,
  type Transport,
} from './playback'
import type { SnapMode } from './snap'
import { DEFAULT_STEM_MODE, type StemEngine, type StemMode } from './stemEngine'
import type { TapSession } from './tapSession'
import TimelineOverview from './TimelineOverview'
import type { GridData, OnsetData } from './types'
import TapOverlay from './TapOverlay'

// Re-exported for the modules that still name these through the Timeline.
// They are defined in types.ts now: the shell owns them, not this screen.
export type { GridData, OnsetData } from './types'

// Per-stem waveform coloring — invented; the design doesn't cover this. Only
// the unplayed bars (waveColor) vary; progressColor/cursorColor stay constant
// (a faded --color-waveform-progress purple / text) regardless of stemMode,
// so the played/unplayed distinction reads the same no matter what's
// isolated. 'drums'/'bass' deliberately reuse
// the onset marker tokens (same instrument, same hue, in both the waveform
// and its onset ticks); 'vocals'/'instrumental' get their own invented tokens
// since nothing else in the app already means those. 'all' reuses the plain
// waveform token — the released mix has no isolated instrument to tint it.
const STEM_WAVE_TOKEN: Record<StemMode, ColorToken> = {
  all: 'waveform',
  vocals: 'stemVocals',
  drums: 'onsetDrums',
  bass: 'onsetBass',
  instrumental: 'stemInstrumental',
}

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
  // Only consumed to retrigger the waveform's color redraw (see the
  // stemMode/darkMode effect below) — wavesurfer's canvas colors are set
  // once at option-set time and never re-read on their own, unlike the
  // region-drawn layers below it, which are plain DOM elements that repaint
  // automatically off the CSS cascade the moment [data-theme] changes.
  darkMode: boolean
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
  snapMode: SnapMode
  onSnapModeChange: (mode: SnapMode) => void
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

// Same idea, one layer up: when both onset overlays are on and a drum onset
// lands within this many milliseconds of a bass onset, the drum marker is
// dropped and only the bass marker draws — bass takes priority, at the
// user's explicit request, since two markers this close read as clutter
// rather than two distinct hits. Only ever drops drums; bass always renders
// when visible. At the app's fixed zoom (16 beats fill the container width)
// this works out to roughly 3-5px on screen across a normal tempo range —
// close enough to visually collide, not so wide it erases a real
// kick-then-bass gap. Starting guess; tune against a real track.
const DRUM_BASS_MERGE_TOLERANCE_MS = 50

// Onset markers outside the currently-active 8-count get a thin ring in the
// page/card background color — third attempt at the "ugly outside the
// highlight" complaint. First was INACTIVE_ONSET_OPACITY (fading toward
// transparent), second was a brightness() darken; both reverted at the
// user's explicit request, and the marker color itself reverted alongside
// them back to the plain --color-onset-bass-marker/--color-onset-drums-
// marker fill with no active/inactive distinction — this ring is a purely
// ADDITIVE treatment on top of that unchanged base color, not another
// recolor. Measured earlier that the complaint isn't a contrast problem
// (WCAG contrast against the plain grey wash actually measured slightly
// HIGHER than against the active block's purple wash, ~3.1 vs ~3.0) but a
// color-harmony one — this is the same halo TECHNIQUE bubbledLine's `halo`
// param already offers elsewhere (loop boundary markers), but deliberately
// NOT that hardcoded white: a white ring was tried on onset markers
// specifically and dropped at the user's explicit request (see bubbledLine's
// own comment) because it read as a literal glow rather than separation.
// var(--color-card-bg) instead blends into whichever background is actually
// behind the marker — grey waveform or purple wash alike — same reasoning as
// why --color-waveform-progress mixes toward --color-card-bg rather than a
// fixed white/black.
const INACTIVE_ONSET_RING = '0 0 0 1px var(--color-card-bg)'

// Zoomed slice: the visible window spans this many eight-counts. The window
// is musical, not temporal — on-screen seconds vary with tempo by design.
const SLICE_EIGHT_COUNTS = 2

// The audio waveform's OWN drawn height, in px. Briefly shrunk 33% (to 64) at
// the user's explicit request, then reverted back to 96, then grown 33% from
// THAT (96 * 1.33 = 127.68, rounded to 128) — this is "the overall timeline"
// getting bigger per that request, consistently with the same box this
// constant has meant every other time "the timeline" came up (e.g. the
// WAVEFORM_BAR_FRACTION 60/40 split just below is 60/40 of THIS figure, not
// of TIMELINE_TRACK_HEIGHT). COUNT_LABEL_BAND_HEIGHT deliberately does NOT
// grow alongside it — the request was to grow the timeline, not the label
// band sitting above it, and 22px was already comfortable for its 11px/9px
// text. Every grid/onset/loop-marker region below still positions itself
// relative to this figure (via waveformPct below where it matters), so they
// all rescale automatically with no other change whenever this is retuned.
const WAVEFORM_HEIGHT = 128

// A band ABOVE the waveform, reserved for the 1..8/"&" count labels — at the
// user's explicit request, reversed from an earlier below-the-waveform
// placement to sit on top instead, and pulled fully clear of the 8-count
// shading/active-block washes (see addEightCountShading and Layer 6 below,
// both now explicitly confined to the waveform's own zone via waveformTopPct/
// waveformPct(100) rather than their old default full-box fill) — the labels
// stand on genuinely plain card background, not on any purple/grey tint, and
// still land exactly above the beat they represent via the same
// wavesurfer-positioned regions everything else here uses. This is NOT extra
// blank canvas tacked onto WAVEFORM_HEIGHT's own box: it's genuine added
// height on wavesurfer's instance (see the height/barAlign/barHeight options
// below), which is why the minimap/controls beneath the timeline shift down
// too — ordinary block layout, no separate "move the slider" code needed.
const COUNT_LABEL_BAND_HEIGHT = 22

// wavesurfer's actual rendered height: the label band now on top plus the
// waveform's own box below it. barAlign:'bottom' + barHeight (below) pin the
// drawn bars to exactly the BOTTOM WAVEFORM_HEIGHT px of this taller box, so
// the waveform itself looks and sizes identically to before this band
// existed — it's just pushed down by COUNT_LABEL_BAND_HEIGHT px, with the
// band living in the space that opens up above it.
const TIMELINE_TRACK_HEIGHT = WAVEFORM_HEIGHT + COUNT_LABEL_BAND_HEIGHT

// The waveform's DRAWN BARS are capped to this fraction of WAVEFORM_HEIGHT —
// at the user's explicit request (sketched by hand: a waveform confined to
// the bottom 60% of its own box, with the top 40% reserved and handed to the
// onset markers instead of the bars ever reaching it — since retuned to
// 65/35, still the same split, just less reserved space). barAlign stays
// 'bottom' (see the wavesurfer options below) so the bars still end flush
// with the very bottom of the box; only how far UP they're allowed to reach
// changes.
const WAVEFORM_BAR_FRACTION = 0.65

// The reserved strip that opens up at the TOP of the waveform's own box once
// the bars stop short of it — in the same "percent of WAVEFORM_HEIGHT" unit
// waveformPct/waveformTopPct below already use, so it composes with them
// directly. Onset markers hang from this strip's own top edge (== the
// waveform box's top edge, waveformTopPct(0) — the same anchor the
// beat/downbeat/subdivision ticks already use) down to this boundary, never
// reaching into the bars' own WAVEFORM_BAR_FRACTION share.
const ONSET_ZONE_HEIGHT_PCT = (1 - WAVEFORM_BAR_FRACTION) * 100

// Ticks (downbeat 55/beat 25/subdivision SUBDIVISION_TICK_HEIGHT_PCT, all
// "% of WAVEFORM_HEIGHT") were sized back when they had the whole waveform
// box to themselves, well past ONSET_ZONE_HEIGHT_PCT — downbeat's 55% alone
// already spilled past the reserved strip into the bars' own share. At the
// user's explicit request, all three now scale down together so the
// TALLEST (downbeat) exactly fits the reserved strip, the same ceiling
// onset markers already respect — keeping their original ratios to each
// other exactly as before (downbeat is still 55/25 = 2.2x a beat tick, a
// beat tick still SUBDIVISION_TICK_HEIGHT_PCT/25 = ~2.08x a subdivision).
// A derived scale rather than three re-picked numbers, so retuning
// ONSET_ZONE_HEIGHT_PCT later keeps ticks correctly fitted with no other
// change here, same reasoning as every other derived constant in this file.
const TICK_SCALE = ONSET_ZONE_HEIGHT_PCT / 55

// Counts 1, 4, and 5 of every 8-count read as fainter than every other beat
// tick — at the user's explicit request, first just 1 and 4, "slightly"
// (0.7), then fainter again from there (0.5), then 5 folded in to match 1
// (see faintCountTicks in the grid effect below for exactly why). Layered
// on top of whatever border/height that tick already has — 1 and 5 stay
// taller/darker as downbeats, just at reduced opacity, they don't drop to
// plain-beat styling.
const FAINT_COUNT_TICK_OPACITY = 0.5

// Every tick/onset-marker top|height below was authored as a "% of the
// waveform's own box" back when that box WAS the whole wavesurfer instance.
// Now that instance is taller (TIMELINE_TRACK_HEIGHT, to fit the label band
// above it) AND the waveform's own top edge has shifted down by
// COUNT_LABEL_BAND_HEIGHT px, a bare percentage needs two corrections: HEIGHT
// only needs rescaling (waveformPct — a span's size doesn't care where it
// starts), but TOP needs that same rescale PLUS the band's own offset added
// back in (waveformTopPct), or every mark would creep up into the label band
// by exactly COUNT_LABEL_BAND_HEIGHT px. Both preserve the exact prior pixel
// geometry against the unchanged waveform with no other numbers below
// needing to change.
const waveformPct = (percentOfWaveform: number) =>
  `${(percentOfWaveform * WAVEFORM_HEIGHT) / TIMELINE_TRACK_HEIGHT}%`
const waveformTopPct = (percentOfWaveform: number) =>
  `${(COUNT_LABEL_BAND_HEIGHT * 100 + percentOfWaveform * WAVEFORM_HEIGHT) / TIMELINE_TRACK_HEIGHT}%`

// Scale factor on onset marker widths (1 = full width: 3px bass, 2px drums).
// Kept as a knob for legibility tuning in dense passages; the thinned 0.67
// variant read too faint, so markers render at full width.
const ONSET_WIDTH_SCALE = 1

// Layer 6: the 8-count block under the playhead gets a purple (accent) wash
// and a border, on top of Layer 2's plain grey wash on every other group —
// active reads as purple, everything else reads as grey. Both are
// translucent so the waveform, onset markers, and playhead stay readable
// through them; tune or revert here.
//
// These, and every other color below except the waveform/progress/cursor
// colors further down, are plain DOM regions (RegionsPlugin renders each as
// a styled <div>, never a canvas draw call) — a literal var(--x) string
// resolves against the live cascade exactly like any other CSS value, so a
// [data-theme] flip repaints them for free with no JS involvement and no
// redraw call. Only the waveform itself is canvas — see the
// waveColor/progressColor/cursorColor options and the reskin effect below,
// which DO need one (wavesurfer never re-reads options on its own).
const ACTIVE_BLOCK_FILL = 'var(--color-active-block-fill)'
const ACTIVE_BLOCK_BORDER = '1px solid var(--color-active-block-border)'

// Count-number styling: at the user's explicit request, moved OUT of the
// waveform into the COUNT_LABEL_BAND_HEIGHT strip ABOVE it (used to sit at
// the block's top edge, on top of the drawn waveform itself; briefly tried
// below the waveform instead, then reversed back to above — also at the
// user's explicit request). The region element is still the block's
// full-height default (top 0/height 100%, unset here), so this padding-top
// is what actually places the digit — a small gap from the very top of the
// (now-reserved, unwashed) band, well clear of the waveform which only
// starts at COUNT_LABEL_BAND_HEIGHT. Horizontally centered on the beat
// itself via translateX(-50%) — same technique bubbledLine's markers use to
// center on their own zero-width anchor (see its comment) — rather than the
// right-nudge this used to have: that nudge existed to clear onset markers
// that used to share this same horizontal position, and at the user's
// explicit request it was reported as reading visibly off-alignment once
// the onset markers themselves moved into their own reserved zone below,
// leaving nothing left to clear. Dark slate, its own token
// (--color-count-label-text) rather than the Library's amber
// --color-needs-counting-text, so this can be restyled without touching
// that unrelated "Needs counting" note. No halo: that existed only to fight
// waveform peaks the labels used to sit on top of — this now sits on
// genuinely plain card background (the 8-count/active-block washes are
// explicitly kept out of this band — see their own comments), which needs no
// contrast rescue.
const COUNT_LABEL_STYLE: Partial<CSSStyleDeclaration> = {
  padding: '4px 0 0 0',
  // -50% keeps it centered on the beat (see the comment above); the extra
  // +2px on top of that nudges it right of center — at the user's explicit
  // request, the running offset has gone -10, -5, -3, -1, and now +2px
  // (this latest request moved both this and SUBDIVISION_LABEL_STYLE by
  // +3px together again, back from the prior round's numbered-counts-only
  // divergence), each time "move back N right" from wherever it last was.
  transform: 'translateX(calc(-50% + 2px))',
  fontSize: '11px',
  lineHeight: '1',
  color: 'var(--color-count-label-text)',
  fontVariantNumeric: 'tabular-nums',
}

// Subdivisions: the half-beat midpoint of every beat gap. These sit at the
// BOTTOM of the visual hierarchy — downbeat > beat > subdivision — so they are
// shorter and lower-contrast than a pulse tick (25 * TICK_SCALE tall, alpha
// 0.55), which is itself shorter than a downbeat (55 * TICK_SCALE, alpha
// 0.85) — the literal 25/55 (and SUBDIVISION_TICK_HEIGHT_PCT) are the design
// ratios; TICK_SCALE is what fits them to the current reserved strip, see
// its own comment. If a subdivision ever reads as loud as a beat, the beat
// grid stops being legible: that's the whole risk of this layer, so tune
// these three together.
//
// Top-anchored (against the waveform's own box — see waveformPct/
// waveformTopPct), at the user's explicit request — right up against the
// count-label band above, so the label and its own tick read as one unit
// instead of the tick sitting off at the far end of the waveform. All three
// tick tiers (downbeat/beat/subdivision) share that same top: 0 and differ
// only in height, so the taller ones simply hang further down into the
// waveform rather than floating at some derived offset.
const SUBDIVISION_TICK_HEIGHT_PCT = 12
const SUBDIVISION_TICK_WIDTH = 1
const SUBDIVISION_TICK_COLOR = 'var(--color-subdivision-tick)'

// "&" labels, subordinate to the integer counts: same family and anchoring
// (same above-waveform band, same no-more-halo reasoning, same centering —
// see COUNT_LABEL_STYLE), smaller, and — at the user's explicit request —
// the SAME dark-slate color as the integer counts (--color-count-label-text)
// rather than their own lighter amber. Same 1px stagger below the counts'
// own padding-top they always had.
const SUBDIVISION_LABEL_STYLE: Partial<CSSStyleDeclaration> = {
  padding: '5px 0 0 0',
  // Moved +3px right alongside COUNT_LABEL_STYLE this round (that request
  // named both again), landing back at dead-center: -3px + 3px = 0, so no
  // extra offset needed beyond the -50% centering itself.
  transform: 'translateX(-50%)',
  fontSize: '9px',
  lineHeight: '1',
  color: 'var(--color-count-label-text)',
}

// Tap markers: where the user's taps landed, in a hue used nowhere else
// (green — clear of drums-red, bass-blue, and the accent purple now shared by
// Layer 6's active-block highlight and the loop band). Full height and above
// every grid layer, because while tapping these
// ARE the subject.
const TAP_MARKER_COLOR = 'var(--color-tap-marker)'
const TAP_MARKER_WIDTH = 2

// Loop boundary markers (see the effect near the bottom of this file, and
// playback.ts's LOOP_BAND_EDGE_WIDTH comment for why these replaced a bare
// hairline): the badge is the circle sitting astride each boundary's bar,
// sized well past LOOP_BAND_EDGE_WIDTH so it always reads as a distinct "pin
// head", not just a fat end-cap on the line.
const LOOP_BAND_BADGE_SIZE = 16
// Badge center, as a % down from the top of the waveform itself (converted
// via waveformTopPct at its use site — the waveform's own top edge is no
// longer the box's top edge now that the label band sits above it) — 80%
// "up" the timeline (from the bottom), at the user's explicit request,
// rather than sitting flush with the very top edge.
const LOOP_BAND_BADGE_TOP_PCT = 20

function styleRegion(el: HTMLElement | null, styles: Partial<CSSStyleDeclaration>) {
  if (el) Object.assign(el.style, styles)
}

// A "bubbled" line: a solid, centered, rounded bar rather than a bare
// hairline border — shared by every marker layer that wants to read as a
// distinct line rather than a thin rule (onset markers below, and the loop
// boundary markers further down). A zero-width addRegion box (start===end)
// needs no centering of its own for the old borderLeft technique, since a
// 0-width box's left edge already sits exactly on the anchor point; giving
// the box real width via `background` instead moves its left edge there, so
// it now needs translateX(-50%) to stay centered on that same point.
function bubbledLine(width: number, color: string, halo = false): Partial<CSSStyleDeclaration> {
  return {
    background: color,
    width: `${width}px`,
    transform: 'translateX(-50%)',
    borderRadius: `${Math.max(1, width / 2)}px`,
    // An optional thin white halo, briefly added to EVERY bubbled line (at
    // the user's explicit request) for markers that were reading as blended
    // into the waveform's purple "played" fill (progressColor, set below)
    // once playback/seeking passed over them. NOT a stacking fix —
    // regions-container already paints above wavesurfer's canvas/progress
    // layers (verified: it carries its own higher z-index, wavesurfer's own
    // doing, not ours), so these were never actually hidden UNDER the
    // waveform; the blend was purely a hue clash (bass's blue sits close to
    // the accent purple), which no z-index can fix — only contrast can.
    // Same technique COUNT_LABEL_STYLE/SUBDIVISION_LABEL_STYLE below already
    // use via textShadow for the same problem, ported to boxShadow here
    // since this is a filled bar, not text. Removed again from the onset
    // markers specifically, also at the user's explicit request — callers
    // opt in per marker layer rather than this being unconditional.
    ...(halo ? { boxShadow: '0 0 0 1px white' } : null),
  }
}

// Layer 2's 8-count shading builder: regions are TIME-defined, so the same
// definitions render at any px-per-sec with no pixel recomputation. EVERY
// group gets the grey wash (not alternating groups — the active group under
// the playhead is what's meant to stand out, via Layer 6's purple highlight
// drawn on top of it; every other group reads as plain grey). The last group
// may be partial and runs to the end of the track. Times at/past duration
// are skipped (gridFromFit's 3dp rounding can land the final beat there).
// pointerEvents none so shades never swallow clicks — wavesurfer's native
// click-to-seek on the main view. Explicitly confined to the waveform's own
// zone (top/height via waveformTopPct/waveformPct rather than the region
// default's full-box fill) so it stops at the label band above rather than
// washing over it — at the user's explicit request, the count labels stand
// on genuinely plain card background, not this tint.
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
    const region = regions.addRegion({
      start,
      end: boundaries[n + 1] ?? duration,
      drag: false,
      resize: false,
      color: 'var(--color-eight-count-shade)',
    })
    styleRegion(region.element, {
      pointerEvents: 'none',
      top: waveformTopPct(0),
      height: waveformPct(100),
    })
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
  snapMode,
  onSnapModeChange,
  tap,
  onReady,
  darkMode,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  // Session-only visibility toggles for the onset overlays; both start off,
  // so the baseline view is pulse ticks + 8-count shading with no markers.
  // These are the one kind of state this screen legitimately owns: they change
  // nothing about the song, only what this view draws.
  const [bassVisible, setBassVisible] = useState(false)
  const [drumsVisible, setDrumsVisible] = useState(false)
  // Which 8-count window the playhead is in (null = outside every window),
  // and the grid's RegionsPlugin instance so the Layer 6 effect can add its
  // regions to the SAME plugin — one coordinate space, one paint layer.
  const [currentEightCountIndex, setCurrentEightCountIndex] = useState<number | null>(null)
  const [gridRegions, setGridRegions] = useState<RegionsPlugin | null>(null)

  // waveColor/progressColor/cursorColor MUST be computed exactly once, at
  // mount, and never again from a live stemMode/darkMode read: @wavesurfer/
  // react's useWavesurfer effect depends on every option VALUE (see its
  // source), so an option that changes value between renders tears down and
  // recreates the whole WaveSurfer instance — destroying every RegionsPlugin
  // region with it. That is exactly the failure this file's peaks comment
  // above already warns about for stem switching ("never via options"); the
  // same rule applies to color. Stem- and theme-driven recoloring instead
  // goes through wavesurfer.setOptions() in the effect below, the one
  // wavesurfer API that updates options AND repaints without recreating
  // anything.
  const initialWaveColors = useMemo(
    () => ({
      waveColor: colorToken(STEM_WAVE_TOKEN[DEFAULT_STEM_MODE]),
      progressColor: colorToken('waveformProgress'),
      cursorColor: colorToken('text'),
    }),
    [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
  )

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
    // The instance is TIMELINE_TRACK_HEIGHT tall (the count-label band above
    // + the waveform's own box below it — see that constant's comment).
    // barAlign/barHeight pin the actually-drawn bars to the BOTTOM
    // WAVEFORM_BAR_FRACTION share of WAVEFORM_HEIGHT — i.e. the bottom 60%
    // of the waveform's own box, touching the very bottom of the whole
    // instance — leaving two genuinely blank strips: the label band above
    // (reserved for the label regions added in the grid effect below) and
    // now also the waveform box's own top 40% (reserved for the onset
    // markers, added in that same effect).
    height: TIMELINE_TRACK_HEIGHT,
    barAlign: 'bottom',
    barHeight: (WAVEFORM_BAR_FRACTION * WAVEFORM_HEIGHT) / TIMELINE_TRACK_HEIGHT,
    // Playback follow in the zoomed slice uses wavesurfer's own scrolling —
    // never a hand-rolled scroll transform (single time↔pixel authority).
    autoScroll: true,
    autoCenter: true,
    ...initialWaveColors,
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

  // Index-aligned with onsets.drums: true where that drum onset is within
  // DRUM_BASS_MERGE_TOLERANCE_MS of a (drawable) bass onset, and so should be
  // dropped in favor of the bass marker when both overlays are visible. Only
  // compared against drawable bass onsets — a bass onset past the audio end
  // is never actually rendered, so it shouldn't be able to absorb anything.
  const drumsAbsorbedByBass = useMemo(() => {
    if (!onsets || !duration) return null
    const drawableBass = onsets.bass.filter((t) => t < duration)
    return onsets.drums.map((t) =>
      drawableBass.some((b) => Math.abs(b - t) * 1000 < DRUM_BASS_MERGE_TOLERANCE_MS),
    )
  }, [onsets, duration])

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
  // collinear and read as one smeared line; a subdivision avoids that mostly
  // on TIME, not height, now — since the onset-marker zone redesign (bass/
  // drums hang from the same top edge as subdivisions, through 0–
  // ONSET_ZONE_HEIGHT_PCT%/half that) the old height-based separation this
  // comment used to describe is gone; a subdivision at 0–SUBDIVISION_TICK_
  // HEIGHT_PCT% now sits fully inside both onset zones whenever their TIMES
  // happen to coincide. That stays an edge case rather than the common
  // case only because a subdivision's time (an exact beat midpoint) rarely
  // lands on a detected onset's time, not because of any remaining vertical
  // gap. And suppressing them would be actively wrong regardless: on a
  // drum-dense block it deletes most of the grid, and a metric ruler with
  // holes in it is not a ruler.
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
    const { beats, downbeatIndices, eightCountIndices } = grid
    const downbeats = new Set(downbeatIndices)
    // Counts 1 (== every eight-count start), 4, and 5 of every group get a
    // dedicated fainter treatment below (FAINT_COUNT_TICK_OPACITY), layered
    // on top of whichever tier (downbeat/beat) they already belong to.
    // Built up in two requests: count 4 first (at the user's explicit
    // request), then count 5 added to match — "the 5 count is still dark,
    // so match [it] to the 1 count marker," since count 5 is ALSO a
    // downbeat here (this grid's downbeats land on both the "1" AND the "5"
    // of each 8-count — a downbeat every 4 beats/bar, an 8-count spanning 2
    // bars) and was reading unfaded next to a now-faded count 1. Count 4 is
    // neither a downbeat nor an eight-count start, so none of this reuses
    // `downbeats` above — every position here is computed straight from
    // eightCountIndices.
    const faintCountTicks = new Set([
      ...eightCountIndices, // count 1
      ...eightCountIndices.map((start) => start + 3), // count 4
      ...eightCountIndices.map((start) => start + 4), // count 5
    ])

    // gridFromFit's 3dp rounding can land the final beat at/past the audio
    // end; wavesurfer can't position anything there, so such entries are
    // skipped rather than clamped onto the track edge.
    const drawable = (time: number) => time < duration

    // Onset-marker ring (see INACTIVE_ONSET_RING): the active window's own
    // bounds, same source Layer 6 uses for its highlight, so "in the active
    // block" means the same thing here as it does there. null (outside every
    // window, or mid-tap where Layer 6 itself goes dark too) rings
    // everything — there's no "the one block that's emphasized" to exempt
    // when nothing is.
    const activeWindow =
      currentEightCountIndex !== null && windows ? windows[currentEightCountIndex] : null
    const inActiveWindow = (time: number) =>
      activeWindow !== null &&
      time >= activeWindow.start &&
      time < Math.min(activeWindow.end, duration)

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
    // They share the beat ticks' top anchoring and z-index (they never
    // coincide in time, so they cannot overlap) but are shorter and fainter —
    // the subordination that keeps the beat grid readable.
    subdivisions?.forEach((time) => {
      const region = addRegion({ start: time, end: time, drag: false, resize: false })
      styleRegion(region.element, {
        pointerEvents: 'none',
        borderLeft: `${SUBDIVISION_TICK_WIDTH}px solid ${SUBDIVISION_TICK_COLOR}`,
        borderRadius: '0',
        height: waveformPct(SUBDIVISION_TICK_HEIGHT_PCT * TICK_SCALE),
        top: waveformTopPct(0),
        zIndex: '3',
      })
    })

    // Pulse ticks at every beat, top-anchored like a ruler hanging from the
    // count-label band above — at the user's explicit request, moved up to
    // sit with the labels rather than at the waveform's far (bottom) edge;
    // downbeats (each bar's "1") are taller and darker, so they hang further
    // down into the waveform than a plain beat rather than starting higher
    // up (both start at the same top: 0). A suppressed tick (downbeats
    // included, no special-casing) is skipped — the onset marker takes its
    // place. pointerEvents none keeps wavesurfer's native click-to-seek
    // working through the grid.
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
          ? '2px solid var(--color-downbeat-tick)'
          : '1px solid var(--color-beat-tick)',
        borderRadius: '0',
        height: emphasized ? waveformPct(55 * TICK_SCALE) : waveformPct(25 * TICK_SCALE),
        top: waveformTopPct(0),
        zIndex: '3',
        opacity: faintCountTicks.has(i) ? String(FAINT_COUNT_TICK_OPACITY) : '1',
      })
    })

    // Onset markers, drawn only for visible stems, hang from the SAME top
    // edge the ticks use (waveformTopPct(0)) down through ONSET_ZONE_HEIGHT_
    // PCT — the strip reserved above the waveform's own bars (see
    // WAVEFORM_BAR_FRACTION) — at the user's explicit request (sketched by
    // hand), rather than spanning down into the bars themselves the way they
    // used to. Bass fills the whole reserved strip (tallest, most
    // prominent); drums gets half that (shorter, still hanging from the same
    // top). Both render even if they land on the same suppressed tick — it's
    // only each other a close drum+bass pair competes with, via
    // drumsAbsorbedByBass just below (bass always wins). Explicit z-index
    // still stacks bass over drums over ticks for the cases that don't merge.
    //
    // Onsets are measured from the audio and are true regardless of what the
    // grid makes of them. When the grid is the thing in doubt they are the
    // evidence you check it against, so their fill color never moves — only
    // a separating ring appears, via inActiveWindow/INACTIVE_ONSET_RING
    // above, and only outside the block currently being counted.
    if (drumsVisible && onsets) {
      onsets.drums.forEach((time, i) => {
        if (!drawable(time)) return
        // Bass takes priority when both overlays are on: a drum onset this
        // close to a bass onset is dropped rather than drawn on top of/next
        // to it (see DRUM_BASS_MERGE_TOLERANCE_MS). Gated on bassVisible —
        // with bass off there's no marker to defer to, so nothing should be
        // suppressed for a mark the user can't even see.
        if (bassVisible && drumsAbsorbedByBass?.[i]) return
        const region = addRegion({ start: time, end: time, drag: false, resize: false })
        styleRegion(region.element, {
          pointerEvents: 'none',
          // design/handoff/README.md: drums onset -> red. (Previously blue —
          // this and the bass marker below were swapped relative to the
          // design; fixed here per the explicit mapping.) Re-hued more than
          // once since, at the user's explicit request — a deliberate
          // deviation from the design mock; see index.css's
          // --color-onset-drums comment. Bubbled, also at the user's
          // request, rather than the bare hairline this used to be — see
          // bubbledLine's comment. Reads the -marker variant, not
          // --color-onset-drums itself — see that token's own comment for
          // why the marker needed its own contrast-tuned (theme-dependent)
          // shade once the purple around it faded, rather than reusing the
          // base hue (the stem waveform coloring and the Onsets dropdown
          // text still read the full-saturation base token, unaffected).
          ...bubbledLine(2 * ONSET_WIDTH_SCALE, 'var(--color-onset-drums-marker)'),
          // Was half the reserved zone (ONSET_ZONE_HEIGHT_PCT / 2); grown
          // 25% from that at the user's explicit request — still
          // comfortably under the zone's own ceiling (bass, which already
          // fills it), so this is purely "drums reads more prominent now,"
          // not a fit concern the way the ticks below were.
          height: waveformPct((ONSET_ZONE_HEIGHT_PCT / 2) * 1.25),
          top: waveformTopPct(0),
          zIndex: '4',
          boxShadow: inActiveWindow(time) ? 'none' : INACTIVE_ONSET_RING,
        })
      })
    }
    if (bassVisible && onsets) {
      onsets.bass.filter(drawable).forEach((time) => {
        const region = addRegion({ start: time, end: time, drag: false, resize: false })
        styleRegion(region.element, {
          pointerEvents: 'none',
          // design/handoff/README.md: bass onset -> blue. (Previously red —
          // see the note on the drums marker above.) Same faded -marker
          // variant as drums, same reasoning.
          ...bubbledLine(3 * ONSET_WIDTH_SCALE, 'var(--color-onset-bass-marker)'),
          height: waveformPct(ONSET_ZONE_HEIGHT_PCT),
          top: waveformTopPct(0),
          zIndex: '5',
          boxShadow: inActiveWindow(time) ? 'none' : INACTIVE_ONSET_RING,
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
  }, [
    gridRegions,
    grid,
    duration,
    onsets,
    tickTags,
    drumsAbsorbedByBass,
    subdivisions,
    bassVisible,
    drumsVisible,
    windows,
    currentEightCountIndex,
  ])

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
    // pulse ticks (3) and onset markers (4/5). Explicitly confined to the
    // waveform's own zone (top/height via waveformTopPct/waveformPct), same
    // reasoning as addEightCountShading — this wash stops at the label band
    // above it rather than tinting it too.
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
      top: waveformTopPct(0),
      height: waveformPct(100),
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

    // "&" on the midpoints BETWEEN beats, giving "1 & 2 & ... 7 & 8 &" —
    // beatCount ampersands, including the trailing one after count 8 (the
    // midpoint between this block's last beat and the very next beat in the
    // grid, whichever block that belongs to — normally the next block's own
    // count 1). At the user's explicit request: that trailing "&" was
    // previously skipped (the loop stopped at beatCount - 1, reasoning the
    // label run "ends on 8"), which silently dropped the one count dancers
    // actually rely on most to catch the next phrase's downbeat. A ragged
    // final group still only gets the "&"s that actually exist between (and
    // just after) its beats — nextTime is undefined past the grid's last
    // beat, so the very end of the song simply has none to add, unpadded.
    for (let i = 0; i < win.beatCount; i++) {
      const nextTime = grid.beats[win.firstBeat + i + 1]
      if (nextTime === undefined) continue
      const time = (grid.beats[win.firstBeat + i] + nextTime) / 2
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

  // Stem mode → (a) audible gains, ramped inside the engine, (b) the
  // rendered waveform's peak buffer, swapped by handing the renderer the
  // mode's precomputed buffer directly — the same internal path zoom() takes,
  // so regions survive untouched — and (c), now, the waveform's colors.
  // Beats, 8-counts, onsets, count labels, and subdivisions are still
  // properties of the TRACK, not the stem, and none of THEIR effects depend
  // on stemMode; only the waveform's own paint does.
  //
  // darkMode is also a dependency for exactly one reason: it is the explicit
  // redraw path required because wavesurfer never re-reads waveColor/
  // progressColor/cursorColor on its own — they're canvas fillStyle values,
  // baked in at the moment they're set, not a live CSS binding the way a DOM
  // region's `var(--x)` style is (see the region colors above, which need no
  // such path). setOptions() is the one wavesurfer API that both updates
  // options and repaints (WaveSurfer.prototype.setOptions: "Set new
  // wavesurfer options and re-render it") — reusing it here, rather than
  // feeding a changing color into useWavesurfer's own options object above,
  // matters: that object's every value is a dependency of @wavesurfer/
  // react's create-effect, so a changed value there tears down and recreates
  // the whole instance (see initialWaveColors' comment) instead of just
  // repainting it.
  //
  // Ordering: setOptions's own internal repaint runs against the OLD peak
  // buffer (whatever was already loaded), immediately followed by
  // getRenderer().render(display) against the NEW one — one wasted
  // intermediate paint, but the frame that actually lands on screen carries
  // both the new peaks and the new colors together. The instance itself is
  // never destroyed or reloaded here (a reload would flip isReady and churn
  // every overlay effect).
  useEffect(() => {
    engine.setMode(stemMode)
    const display = engine.displayBufferFor(stemMode)
    if (wavesurfer && isReady) {
      wavesurfer.setOptions({
        waveColor: colorToken(STEM_WAVE_TOKEN[stemMode]),
        progressColor: colorToken('waveformProgress'),
        cursorColor: colorToken('text'),
      })
      void wavesurfer.getRenderer().render(display)
    }
  }, [engine, stemMode, darkMode, wavesurfer, isReady])

  // Loop markers on the main view: no wash (see playback.ts's
  // LOOP_BAND_FILL/EDGE_A/EDGE_B comment for why that was removed), and each
  // boundary drawn as a labeled marker — a wider bar with a small circular
  // A/B badge sitting astride its top edge, like a flag on a pole — rather
  // than the bare hairline this used to be. Colored to match its own drag
  // handle (LoopBoundaryHandle.tsx), so a marker here always reads as the
  // same A/B as the pin that set it.
  //
  // Each boundary renders only while it's off its default track edge (A off
  // 0, B off duration) — at the user's explicit request: a marker fixed at
  // the very start/end of every waveform states the obvious and reads as
  // clutter, same complaint that got the old unlabeled A line dropped
  // entirely; a LABELED marker fixes that complaint for wherever the user
  // actually drags a boundary to, but not for the untouched default position.
  //
  // Everything is time-defined, so the main view's zoom/scroll place it with
  // no pixel math; non-interactive, so it can't eat the view's own
  // click-to-seek. The overview below still draws its own copy of both A and
  // B (see TimelineOverview) since it isn't a wavesurfer instance and has no
  // regions plugin to share.
  useEffect(() => {
    if (!gridRegions || !duration) return
    const added: Region[] = []

    const marker = (time: number, letter: 'A' | 'B', color: string) => {
      const at = Math.min(Math.max(time, 0), duration)

      // The badge: an HTMLElement (not a string) passed as the region's
      // `content` — RegionsPlugin uses an HTMLElement content verbatim
      // (appendChild, no wrapping), unlike a string, which it pads into an
      // inline-block div (see the count-label layer above). aria-hidden
      // since this is a decorative echo of the A/B already announced by the
      // real slider handles (LoopBoundaryHandle's role="slider" aria-label).
      const badge = document.createElement('span')
      badge.textContent = letter
      badge.setAttribute('aria-hidden', 'true')
      Object.assign(badge.style, {
        position: 'absolute',
        top: waveformTopPct(LOOP_BAND_BADGE_TOP_PCT),
        left: '0',
        // translate(-50%, -50%) centers the badge ON that top% point (both
        // axes), rather than hanging it below/right of it.
        transform: 'translate(-50%, -50%)',
        width: `${LOOP_BAND_BADGE_SIZE}px`,
        height: `${LOOP_BAND_BADGE_SIZE}px`,
        borderRadius: '50%',
        background: color,
        color: 'white',
        fontSize: '9px',
        fontWeight: '700',
        lineHeight: `${LOOP_BAND_BADGE_SIZE}px`,
        textAlign: 'center',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.35)',
      })

      const region = gridRegions.addRegion({
        start: at,
        end: at,
        drag: false,
        resize: false,
        content: badge,
      })
      styleRegion(region.element, {
        pointerEvents: 'none',
        ...bubbledLine(LOOP_BAND_EDGE_WIDTH, color, true),
        height: '100%',
        top: '0',
        // Above every other grid layer (ticks 3, active-block 2, onsets 4/5,
        // count labels 6, tap markers 7) — a loop boundary is persistent,
        // load-bearing state, not a transient annotation, and its badge
        // reads as broken if a beat tick or onset marker draws over it.
        zIndex: '8',
      })
      added.push(region)
    }

    if (loop.start > 0) marker(loop.start, 'A', LOOP_BAND_EDGE_A)
    if (loop.end < duration) marker(loop.end, 'B', LOOP_BAND_EDGE_B)

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

  // The design's "Onsets" select (Off/Bass/Drums/Both) is a VIEW over
  // bassVisible/drumsVisible, not a third piece of state: it reads as a
  // derived value and writes by dispatching to the same two setters the
  // suppression logic above already depends on independently. Collapsing
  // them into one enum's own state would break tick suppression for
  // whichever combination the enum didn't preserve (see GAP_AUDIT.md §4.5).
  const onsetSelectValue: 'off' | 'bass' | 'drums' | 'both' =
    bassVisible && drumsVisible ? 'both' : bassVisible ? 'bass' : drumsVisible ? 'drums' : 'off'
  const onOnsetSelectChange = (value: string) => {
    setBassVisible(value === 'bass' || value === 'both')
    setDrumsVisible(value === 'drums' || value === 'both')
  }

  return (
    <>
      <div className="timeline">
        <div
          ref={containerRef}
          className="timeline-waveform"
          // Wavesurfer's own playhead cursor is a shadow-DOM element (part=
          // "cursor") sized to its host's FULL height by wavesurfer itself —
          // the whole TIMELINE_TRACK_HEIGHT box, count-label band included,
          // not just the waveform's own zone. At the user's explicit
          // request ("the playhead shoots through and above the actual
          // timeline box"), constrained to the waveform's own zone via
          // these two custom properties: CSS custom properties (unlike
          // selectors) cross a shadow boundary, so index.css's
          // `.timeline-waveform > div::part(cursor)` rule can read them
          // from outside the shadow root wavesurfer creates in here.
          style={
            {
              '--cursor-top': waveformTopPct(0),
              '--cursor-height': waveformPct(100),
            } as CSSProperties
          }
        />
        <div className="timeline-card-divider" />
        <TimelineOverview
          duration={duration}
          currentTime={transport.currentTime}
          loop={loop}
          onSeek={seekAndCenter}
          snapMode={snapMode}
          onSnapModeChange={onSnapModeChange}
        />
        <div className="timeline-controls">
          <PlayPauseButton transport={transport} disabled={!isReady} />
          <TimeReadout transport={transport} grid={grid} windows={windows} />
          {/*
            The design tucks this button below-left of the row via a negative
            margin, sized for its fixed 407px-wide frame. At the 380px width
            this app is verified against, the row wraps to a second line
            (Onsets/Hear don't fit next to play+time+count) and that negative
            offset then lands ON TOP of the wrapped line instead of empty
            space beneath it — so it renders inline here instead. Only the
            button's own look (icon, active/inactive color) is a design
            requirement; its exact offset position was not (README only
            mocks the toggle, not a picker or its placement).
          */}
          <SpeedControl speed={speed} />
          {/*
            Re-tap is PERMANENT: always here, next to the transport, never
            locked away once a grid exists — not behind a settings menu and
            not conditional on anything tripping. Users get the count wrong
            sometimes and must be able to redo it without re-importing the
            song. Hidden only during first run, where tap mode is already
            forced open and there is no grid to exit back to.
          */}
          {hasSavedGrid && (
            <button
              className="tap-enter"
              onClick={tap.tapping ? tap.exit : tap.enter}
              disabled={!isReady}
              aria-pressed={tap.tapping}
            >
              {tap.tapping ? 'Exit tap mode' : 'Re-tap the counts'}
            </button>
          )}
          <label className="control-select-label" style={{ marginLeft: 'auto' }}>
            Onsets
            <select
              className="control-select"
              value={onsetSelectValue}
              disabled={!onsets}
              onChange={(e) => onOnsetSelectChange(e.target.value)}
            >
              <option value="off">Off</option>
              <option value="bass" style={{ color: 'var(--color-onset-bass)' }}>
                Bass
              </option>
              {/* Reads the -marker variant here, not the base token (unlike
                  Bass just above, whose base and marker still share the
                  same blue) — drums' marker was re-hued to red while its
                  base token stayed green (still the drums STEM waveform
                  color; see --color-onset-drums-marker's own comment for
                  why those two are deliberately separate tokens), so
                  reading the base here left this dropdown showing green
                  for what's now a red marker. At the user's explicit
                  request. */}
              <option value="drums" style={{ color: 'var(--color-onset-drums-marker)' }}>
                Drums
              </option>
              <option value="both">Both</option>
            </select>
          </label>
          <HearControl stemMode={stemMode} onStemModeChange={onStemModeChange} />
        </div>
        {loop.error && <p className="error">{loop.error}</p>}
      </div>
      {tap.tapping && (
        <TapOverlay
          taps={tap.taps}
          isPlaying={transport.isPlaying}
          fit={tap.fit}
          fitError={tap.error}
          onTap={tap.record}
          onPlay={transport.play}
          onAccept={tap.accept}
          // Both labels now do the same kind of thing: clear the taps and stay
          // IN the tap state, never dropping back out to a saved grid (at the
          // user's explicit request — Restart used to call tap.exit, which
          // left tap mode entirely). "Start over" on first run and "Restart"
          // with a saved grid are just two names for the same restart action;
          // they stay separate props only because the label text differs.
          cancelLabel={hasSavedGrid ? 'Restart' : 'Start over'}
          onCancel={hasSavedGrid ? tap.restart : tap.enter}
        />
      )}
    </>
  )
}

export default Timeline
