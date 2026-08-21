import type { GridData } from './types'

// Manual tap grid: fit a constant-tempo beat grid to timestamps the user tapped
// against the audio clock. With auto detection removed, this is the ONLY way a
// grid comes to exist.
//
// This module is the only place that knows a grid comes from taps. Its output
// is a plain GridData, so every consumer downstream (8-count shading, onset
// overlays, count labels, subdivision ticks, minimap, loop snapping) stays
// unaware of where beats[] came from. That is the whole point: provenance dies
// here.

// A fit is a SLOPE, and slope error scales inversely with the length of the tap
// window. Eight taps is not enough — jitter that reads as harmless at 0:10
// propagates into audible drift by the end of a 3-minute track. 16 is the floor;
// 32 roughly halves the residual tempo error and is what the UI asks for.
export const MIN_TAPS = 16
export const RECOMMENDED_TAPS = 32

// One nudge step. Touch→audio latency (50-150ms on mobile) plus human negative
// mean asynchrony (tapping ~20-80ms early) put a CONSTANT offset on every tap
// that the fit cannot see: it is in the phase, not the residuals. 10ms is fine
// enough to land inside that band without a long press-and-hold. Shared by the
// in-overlay nudge and the main-view nudge — one step size, one meaning.
export const PHASE_NUDGE_MS = 10

// A tap further than this fraction of a period from its predicted time is not a
// late tap, it's a wrong one (a stumble, a double, a missed entry). Quarter of a
// beat is the widest a tap can be off and still plausibly mean the beat it was
// assigned to.
const OUTLIER_RESIDUAL_RATIO = 0.25

// If more than this fraction of taps are outliers, the tapping itself was
// inconsistent — the surviving taps would fit a clean line through noise and
// produce a confidently wrong grid. Reject and ask for a retry instead.
const MAX_DROPPED_RATIO = 0.3

export interface TapFit {
  // Fitted time of the user's FIRST tap — their "1". The grid's phase origin.
  phase: number
  // Seconds per tapped count. This is the level the USER tapped at: whatever
  // they felt as the count becomes the grid's count.
  period: number
  kept: number
  dropped: number
}

export type TapFitResult = { ok: true; fit: TapFit } | { ok: false; error: string }

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Least-squares fit of t ≈ phase + period * idx. Returns null when the taps
// carry no slope information (fewer than two, or all on one index).
function regress(idx: number[], t: number[]): { phase: number; period: number } | null {
  const n = idx.length
  if (n < 2) return null
  const meanIdx = idx.reduce((a, b) => a + b, 0) / n
  const meanT = t.reduce((a, b) => a + b, 0) / n
  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    const d = idx[i] - meanIdx
    num += d * (t[i] - meanT)
    den += d * d
  }
  if (den === 0) return null
  const period = num / den
  return { period, phase: meanT - period * meanIdx }
}

/**
 * Fit tempo and phase to a run of tap timestamps (seconds, on the audio clock).
 *
 * Deliberately does NOT take duration: a fit is a line, not a grid. Turning it
 * into beats is gridFromFit's job, which lets the nudge controls re-derive the
 * grid without refitting.
 */
export function fitTapGrid(taps: number[]): TapFitResult {
  if (taps.length < MIN_TAPS) {
    return { ok: false, error: `Need at least ${MIN_TAPS} taps (you tapped ${taps.length}).` }
  }
  const t = [...taps].sort((a, b) => a - b)

  // Rough period first, so taps can be assigned to counts before anything is
  // fitted. Median (not mean) so a single stumble doesn't move it.
  const gaps: number[] = []
  for (let i = 1; i < t.length; i++) gaps.push(t[i] - t[i - 1])
  const roughPeriod = median(gaps)
  if (!(roughPeriod > 0)) {
    return { ok: false, error: 'Those taps all landed on the same instant. Is the track playing?' }
  }

  // Index assignment is what makes the fit robust to human error: a SKIPPED
  // count leaves a gap in idx rather than pulling every later tap one count
  // early, and a doubled tap collides on an index instead of shearing the line.
  const idx = t.map((x) => Math.round((x - t[0]) / roughPeriod))

  const first = regress(idx, t)
  if (!first || !(first.period > 0)) {
    return { ok: false, error: 'Could not read a tempo from those taps. Try again.' }
  }

  // Outlier rejection doubles as the repair for a misfiled tap. `roughPeriod`
  // carries ~0.3% error, which over a long tap run can accumulate past ±0.5 and
  // file a late tap under the wrong count; that tap then sits a full period off
  // the fitted line and is dropped here, so the refit below still runs on
  // correctly-filed taps. (Measured: re-assigning indices off the first fit
  // instead does NOT help — a sheared fit just re-derives the same shear.)
  const keep: number[] = []
  for (let i = 0; i < t.length; i++) {
    const residual = t[i] - (first.phase + first.period * idx[i])
    if (Math.abs(residual) <= OUTLIER_RESIDUAL_RATIO * first.period) keep.push(i)
  }
  const dropped = t.length - keep.length
  if (dropped / t.length > MAX_DROPPED_RATIO) {
    return {
      ok: false,
      error: `Taps were too uneven: ${dropped} of ${t.length} missed the beat. Try again, and keep it steady.`,
    }
  }

  // Refit ONCE on the survivors. Not iterated to convergence: repeated rejection
  // walks toward whatever subset is most self-consistent, which on genuinely
  // sloppy tapping is a confident fit to a handful of taps.
  const refit = regress(
    keep.map((i) => idx[i]),
    keep.map((i) => t[i]),
  )
  if (!refit || !(refit.period > 0)) {
    return { ok: false, error: 'Could not read a tempo from those taps. Try again.' }
  }

  return {
    ok: true,
    fit: { phase: refit.phase, period: refit.period, kept: keep.length, dropped },
  }
}

/**
 * Expand a fit into a full-track GridData.
 *
 * `phaseNudgeSeconds` slides every beat (the nudge control's ±10ms, correcting
 * the constant touch-latency + negative-mean-asynchrony offset the fit cannot
 * see). `countNudge` moves which beat is the "1" by whole counts, without
 * touching the beat times themselves.
 */
export function gridFromFit(
  fit: TapFit,
  duration: number,
  phaseNudgeSeconds = 0,
  countNudge = 0,
): GridData {
  const { period } = fit
  const phase = fit.phase + phaseNudgeSeconds

  // k runs NEGATIVE where the audio starts before the user's first tap — they
  // are told to start on a 1, not to start at 0:00, so the grid has to be
  // extrapolated backwards to cover the intro.
  const kMin = Math.ceil((0 - phase) / period)
  const kMax = Math.floor((duration - phase) / period)
  const beats: number[] = []
  for (let k = kMin; k <= kMax; k++) {
    // 3dp to match the backend's own rounding, so a tapped grid and an auto
    // grid carry the same precision downstream.
    beats.push(Math.round((phase + k * period) * 1000) / 1000)
  }

  // Array index of the user's "1". kMin is where the array starts in k-space, so
  // fit-index 0 sits at -kMin; the count nudge just re-labels which beat is 1.
  const one = -kMin + countNudge
  const downbeatIndices: number[] = []
  const eightCountIndices: number[] = []
  for (let j = 0; j < beats.length; j++) {
    // Modulo that stays non-negative: `one` can fall outside [0, beats.length).
    const rel = (((j - one) % 8) + 8) % 8
    if (rel === 0) eightCountIndices.push(j)
    if (rel % 4 === 0) downbeatIndices.push(j)
  }

  return { beats, downbeatIndices, eightCountIndices }
}

export const bpmOf = (fit: TapFit): number => 60 / fit.period

/**
 * Recover fit parameters from a saved grid. The fit itself is never persisted —
 * only beats[] — so the main-view nudge re-derives it here instead of refitting
 * from taps that no longer exist.
 *
 * Every saved grid is gridFromFit output: constant period, phrase-aligned
 * eight-counts, beats rounded to 3dp. The endpoint slope bounds the
 * accumulated rounding error at ±1ms/(n-1) — far below the 10ms nudge step —
 * so the reconstruction preserves the fitted tempo to well under audibility.
 */
export function fitFromGrid(grid: GridData): TapFit | null {
  const { beats, eightCountIndices, downbeatIndices } = grid
  if (beats.length < 2) return null
  const period = (beats[beats.length - 1] - beats[0]) / (beats.length - 1)
  if (!(period > 0)) return null
  // Phase anchors on a "1" (the first eight-count start), so regenerating the
  // grid keeps every phrase boundary exactly where the user accepted it.
  const anchor = eightCountIndices[0] ?? downbeatIndices[0] ?? 0
  // kept/dropped describe a tap session; a reconstructed fit has none.
  return { phase: beats[anchor], period, kept: 0, dropped: 0 }
}

/**
 * Nudge a SAVED grid: phase by seconds, count labeling by whole counts.
 * Reconstructs the fit and regenerates through gridFromFit, so the semantics
 * are identical to the in-overlay nudge — phase slides, the "1" re-labels,
 * and the fitted tempo is never touched.
 */
export function nudgeGrid(
  grid: GridData,
  duration: number,
  phaseNudgeSeconds: number,
  countNudge: number,
): GridData | null {
  if (!duration) return null
  const fit = fitFromGrid(grid)
  return fit && gridFromFit(fit, duration, phaseNudgeSeconds, countNudge)
}

// The first-run gate: prompt for taps only when the persisted-grid read has
// SETTLED and found nothing. While the read is in flight the timeline renders
// dormant — flashing the prompt at a user whose grid is about to load would
// contradict "a tapped song never prompts again".
export const needsTapPrompt = (
  gridLoading: boolean,
  grid: GridData | null | undefined,
): boolean => !gridLoading && !grid
