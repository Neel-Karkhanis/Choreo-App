import { describe, expect, it } from 'vitest'
import {
  fitFromGrid,
  fitTapGrid,
  gridFromFit,
  needsTapPrompt,
  nudgeGrid,
} from './tapGrid'

// A saved grid exactly as the app persists one: gridFromFit output, beats
// rounded to 3dp. The period is deliberately NOT a round number so the 3dp
// rounding actually exercises the reconstruction (a 0.5s period would round
// losslessly and hide any drift).
const PHASE = 10.007
const PERIOD = 0.483
const DURATION = 187.36
const FIT = { phase: PHASE, period: PERIOD, kept: 32, dropped: 0 }
const saved = gridFromFit(FIT, DURATION)

// Reconstruction wobble bound: 3dp storage rounding (±0.5ms) on each end plus
// the endpoint-slope error accumulated across the track (~1ms) plus the fresh
// rounding of regenerated beats (±0.5ms).
const TOLERANCE = 0.004

describe('needsTapPrompt', () => {
  it('never prompts while the persisted-grid read is in flight', () => {
    expect(needsTapPrompt(true, null)).toBe(false)
  })

  it('never prompts once a saved grid is in hand — a tapped song never prompts again', () => {
    expect(needsTapPrompt(false, saved)).toBe(false)
  })

  it('prompts when the read settles and finds nothing', () => {
    expect(needsTapPrompt(false, null)).toBe(true)
    expect(needsTapPrompt(false, undefined)).toBe(true)
  })
})

describe('fitFromGrid', () => {
  it('recovers the fitted tempo from a stored grid', () => {
    const fit = fitFromGrid(saved)
    expect(fit).not.toBeNull()
    expect(Math.abs(fit!.period - PERIOD)).toBeLessThan(1e-4)
  })

  it('anchors the phase on an eight-count start, so phrases stay put', () => {
    const fit = fitFromGrid(saved)
    expect(fit!.phase).toBe(saved.beats[saved.eightCountIndices[0]])
  })

  it('rejects a grid too short to carry a slope', () => {
    expect(
      fitFromGrid({ beats: [1.0], downbeatIndices: [0], eightCountIndices: [0] }),
    ).toBeNull()
  })
})

describe('nudgeGrid', () => {
  it('reproduces the stored grid when nudged by nothing', () => {
    const next = nudgeGrid(saved, DURATION, 0, 0)!
    expect(next.beats.length).toBe(saved.beats.length)
    next.beats.forEach((b, i) => {
      expect(Math.abs(b - saved.beats[i])).toBeLessThan(TOLERANCE)
    })
    expect(next.eightCountIndices).toEqual(saved.eightCountIndices)
    expect(next.downbeatIndices).toEqual(saved.downbeatIndices)
  })

  it('slides every beat by the phase nudge', () => {
    const next = nudgeGrid(saved, DURATION, 0.01, 0)!
    expect(next.beats.length).toBe(saved.beats.length)
    next.beats.forEach((b, i) => {
      expect(Math.abs(b - saved.beats[i] - 0.01)).toBeLessThan(TOLERANCE)
    })
  })

  it('never touches the fitted tempo', () => {
    const next = nudgeGrid(saved, DURATION, 0.01, 0)!
    const before = fitFromGrid(saved)!
    const after = fitFromGrid(next)!
    expect(Math.abs(after.period - before.period)).toBeLessThan(1e-4)
  })

  it('re-labels the "1" on a count nudge without moving the beats', () => {
    const next = nudgeGrid(saved, DURATION, 0, 1)!
    expect(next.beats.length).toBe(saved.beats.length)
    next.beats.forEach((b, i) => {
      expect(Math.abs(b - saved.beats[i])).toBeLessThan(TOLERANCE)
    })
    // Every phrase boundary moves exactly one COUNT later (one index), and the
    // downbeat structure stays every-4 against the new "1".
    saved.eightCountIndices.forEach((i) => {
      if (i + 1 < next.beats.length) expect(next.eightCountIndices).toContain(i + 1)
    })
    expect(next.eightCountIndices.every((i) => next.downbeatIndices.includes(i))).toBe(true)
  })

  it('phase and count nudges accumulate across saves without drifting the tempo', () => {
    // Ten successive ±10ms corrections, each re-derived from the previous
    // save — the worst case for reconstruction error compounding.
    let grid = saved
    for (let n = 0; n < 10; n++) {
      grid = nudgeGrid(grid, DURATION, n % 2 ? -0.01 : 0.01, 0)!
    }
    const fit = fitFromGrid(grid)!
    expect(Math.abs(fit.period - PERIOD)).toBeLessThan(1e-4)
  })
})

describe('fitTapGrid → gridFromFit → fitFromGrid', () => {
  it('round-trips a clean tap session into a recoverable grid', () => {
    const taps = Array.from({ length: 32 }, (_, k) => PHASE + k * PERIOD)
    const result = fitTapGrid(taps)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const grid = gridFromFit(result.fit, DURATION)
    const recovered = fitFromGrid(grid)!
    expect(Math.abs(recovered.period - PERIOD)).toBeLessThan(1e-4)
  })
})
