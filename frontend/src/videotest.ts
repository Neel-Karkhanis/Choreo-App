// MEASUREMENT RIG — not the feature. Answers the blocking-risk question for
// video support: how long does a <video> seek-to-first-presented-frame take at
// every loop wrap, on the ACTUAL audio path (the real StemEngine with native
// looping sources), in a real browser, on a real dance video?
//
// Two modes, selected by query param:
//   ?mode=wrap  — engine loops [A, A+len); on every wrap the video is seeked
//                 once and requestVideoFrameCallback timestamps the first
//                 post-seek frame. Collects N wraps, reports latency stats.
//   ?mode=run   — no loop; engine plays from 0 for runSec seconds at `rate`;
//                 the video free-runs with multiplicative playbackRate nudging
//                 and per-displayed-frame A/V offset is recorded.
//
// The slave logic here (seek once per wrap, free-run between, nudge
// playbackRate multiplicatively, video never consulted for time) is the
// prototype of the production design — measured first, built second.
//
// Params: track (track id), video (URL), mode, loopA, loopLen, rate, wraps,
// runSec. Results land in window.__vt and in the #results <pre> as JSON;
// __vt.done flips true when the run completes.

import { getAudioContext } from './audioUnlock'
import { loadStems, StemEngine } from './stemEngine'

const params = new URLSearchParams(location.search)
const trackId = params.get('track') ?? ''
const videoUrl = params.get('video') ?? ''
const mode = params.get('mode') ?? 'wrap'
const loopA = Number(params.get('loopA') ?? '10')
const loopLen = Number(params.get('loopLen') ?? '4')
const rate = Number(params.get('rate') ?? '1')
const wrapTarget = Number(params.get('wraps') ?? '50')
const runSec = Number(params.get('runSec') ?? '120')

const statusEl = document.getElementById('status') as HTMLParagraphElement
const startEl = document.getElementById('start') as HTMLButtonElement
const resultsEl = document.getElementById('results') as HTMLPreElement
const video = document.getElementById('video') as HTMLVideoElement

// Drift correction constants (prototype of the production values).
// Correction is MULTIPLICATIVE on the engine rate: rate * 1.01 is the same 1%
// correction at every speed. Engaged past 30 ms of offset, released under
// 10 ms, hard-reseek past 250 ms.
const NUDGE = 1.01
const NUDGE_ON_S = 0.03
const NUDGE_OFF_S = 0.01
const HARD_RESYNC_S = 0.25

interface WrapSample {
  wrap: number
  detectLagMs: number // engine time past loopStart when rAF saw the wrap
  seekedEventMs: number // seek issue -> 'seeked' event
  firstFrameMs: number // seek issue -> first post-seek frame PRESENTED
  preWrapOffsetMs: number // A/V offset just before this wrap (intra-cycle drift)
}

interface OffsetSample {
  t: number // seconds since run start (wall)
  offsetMs: number // video mediaTime - engine time, at frame presentation
  nudge: number
}

const state = {
  wraps: [] as WrapSample[],
  offsets: [] as OffsetSample[],
  stalls: 0,
  hardResyncs: 0,
  done: false,
  error: null as string | null,
}
;(window as unknown as { __vt: typeof state }).__vt = state

function quantiles(xs: number[]) {
  if (xs.length === 0) return null
  const s = [...xs].sort((a, b) => a - b)
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))]
  return {
    n: s.length,
    min: +s[0].toFixed(1),
    median: +q(0.5).toFixed(1),
    p90: +q(0.9).toFixed(1),
    p99: +q(0.99).toFixed(1),
    max: +s[s.length - 1].toFixed(1),
  }
}

function report(extra: Record<string, unknown> = {}) {
  const summary = {
    mode,
    trackId,
    videoUrl,
    rate,
    ...(mode === 'wrap'
      ? {
          loopA,
          loopLen,
          wrapsCollected: state.wraps.length,
          firstFrameMs: quantiles(state.wraps.map((w) => w.firstFrameMs)),
          seekedEventMs: quantiles(state.wraps.map((w) => w.seekedEventMs)),
          detectLagMs: quantiles(state.wraps.map((w) => w.detectLagMs)),
          preWrapOffsetMs: quantiles(state.wraps.map((w) => w.preWrapOffsetMs)),
        }
      : {
          runSec,
          samples: state.offsets.length,
          offsetAbsMs: quantiles(state.offsets.map((o) => Math.abs(o.offsetMs))),
          // per-30s-bucket max |offset|: accumulation would show as a rising
          // sequence, a hold as a flat one.
          bucketsMaxAbsMs: Object.entries(
            state.offsets.reduce<Record<string, number>>((acc, o) => {
              const k = String(Math.floor(o.t / 30) * 30)
              acc[k] = Math.max(acc[k] ?? 0, Math.abs(o.offsetMs))
              return acc
            }, {}),
          ).map(([k, v]) => [Number(k), +v.toFixed(1)]),
        }),
    stalls: state.stalls,
    hardResyncs: state.hardResyncs,
    done: state.done,
    error: state.error,
    ...extra,
  }
  resultsEl.textContent = JSON.stringify(summary, null, 2)
  return summary
}

async function main() {
  if (!trackId || !videoUrl) {
    statusEl.textContent = 'missing ?track= or ?video='
    return
  }
  statusEl.textContent = `loading stems for ${trackId}…`
  // No md5 available from this debug harness's ?track=/?video= query params
  // — undefined just means "skip the IndexedDB cache", same as before this
  // file's stemEngine.ts grew one.
  const buffers = await loadStems(trackId, undefined)
  // The driver runs Chrome with --autoplay-policy=no-user-gesture-required, so
  // this lazily-created context needs no gesture to come up running.
  const engine = new StemEngine(buffers, getAudioContext())
  statusEl.textContent = 'stems loaded; loading video…'

  video.src = videoUrl
  video.preload = 'auto'
  await new Promise<void>((resolve, reject) => {
    video.onloadeddata = () => resolve()
    video.onerror = () => reject(new Error('video load failed'))
  })
  statusEl.textContent = `ready (video ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(1)}s)`
  startEl.disabled = false

  const start = () => {
    startEl.disabled = true
    if (mode === 'wrap') runWrapTest(engine).catch(fail)
    else runOffsetTest(engine).catch(fail)
  }
  startEl.onclick = start
  // Driver runs Chrome with --autoplay-policy=no-user-gesture-required, so
  // auto-start; the button is the fallback for a human at the page.
  start()
}

function fail(err: unknown) {
  state.error = String(err)
  state.done = true
  report()
}

// Video mediaTime -> engine-time offset at (approximately) the moment the
// frame is displayed. rVFC gives the frame's mediaTime plus when it is
// expected on glass; the engine clock is projected forward to that instant.
function offsetAt(
  engine: StemEngine,
  now: number,
  metadata: VideoFrameCallbackMetadata,
): number {
  const engineAtDisplay =
    engine.currentTime + ((metadata.expectedDisplayTime - now) / 1000) * engine.getPlaybackRate()
  return metadata.mediaTime - engineAtDisplay
}

async function runWrapTest(engine: StemEngine) {
  engine.setLoop(loopA, loopA + loopLen, true)
  engine.setPlaybackRate(rate)
  engine.seek(loopA)
  await engine.play()
  await video.play()
  video.playbackRate = rate
  video.currentTime = engine.currentTime

  let prevT = engine.currentTime
  let lastOffsetMs = 0
  let seekPending = false
  let seekIssuedAt = 0
  let wrapDetectLagMs = 0
  let seekedEventMs = 0

  video.addEventListener('seeked', () => {
    if (seekPending) seekedEventMs = performance.now() - seekIssuedAt
  })

  const onFrame = (now: number, metadata: VideoFrameCallbackMetadata) => {
    if (seekPending && Math.abs(metadata.mediaTime - loopA) < loopLen / 2) {
      // First PRESENTED frame after the wrap seek — the visible freeze ends
      // here. presentationTime is in the performance.now() timebase.
      state.wraps.push({
        wrap: state.wraps.length + 1,
        detectLagMs: +wrapDetectLagMs.toFixed(1),
        seekedEventMs: +seekedEventMs.toFixed(1),
        firstFrameMs: +(metadata.presentationTime - seekIssuedAt).toFixed(1),
        preWrapOffsetMs: +(lastOffsetMs).toFixed(1),
      })
      seekPending = false
      statusEl.textContent = `wrap ${state.wraps.length}/${wrapTarget}`
      report()
    }
    if (!seekPending) {
      const off = offsetAt(engine, now, metadata) * 1000
      // Skip the frame that straddles the wrap itself (video mediaTime still
      // pre-wrap, engine clock already wrapped): it reads as ~loopLen of
      // offset and is a discontinuity artifact, not drift.
      if (Math.abs(off) < 1000) lastOffsetMs = off
    }
    video.requestVideoFrameCallback(onFrame)
  }
  video.requestVideoFrameCallback(onFrame)

  const tick = () => {
    const t = engine.currentTime
    if (t < prevT - loopLen / 2 && !seekPending) {
      // Engine wrapped (native source loop; the clock wraps with it). Seek
      // the video ONCE to the engine's current position; free-run until the
      // next wrap.
      wrapDetectLagMs = ((t - loopA) / rate) * 1000
      seekIssuedAt = performance.now()
      seekedEventMs = -1
      seekPending = true
      video.currentTime = t
    }
    prevT = t
    if (state.wraps.length < wrapTarget) {
      requestAnimationFrame(tick)
    } else {
      engine.pause()
      video.pause()
      state.done = true
      statusEl.textContent = 'done'
      report()
    }
  }
  requestAnimationFrame(tick)
}

async function runOffsetTest(engine: StemEngine) {
  engine.setPlaybackRate(rate)
  engine.seek(0)
  await engine.play()
  await video.play()
  video.playbackRate = rate
  video.currentTime = engine.currentTime

  const t0 = performance.now()
  let nudgeMul = 1

  video.addEventListener('waiting', () => {
    state.stalls += 1
  })

  const onFrame = (now: number, metadata: VideoFrameCallbackMetadata) => {
    const offset = offsetAt(engine, now, metadata)
    state.offsets.push({
      t: +((now - t0) / 1000).toFixed(2),
      offsetMs: +(offset * 1000).toFixed(2),
      nudge: nudgeMul,
    })

    if (Math.abs(offset) > HARD_RESYNC_S) {
      state.hardResyncs += 1
      video.currentTime = engine.currentTime
      nudgeMul = 1
    } else if (Math.abs(offset) > NUDGE_ON_S) {
      // video ahead (offset > 0) -> run it fractionally slower; behind ->
      // faster. Multiplicative: the same 1% at every engine rate.
      nudgeMul = offset > 0 ? 1 / NUDGE : NUDGE
    } else if (Math.abs(offset) < NUDGE_OFF_S) {
      nudgeMul = 1
    }
    const targetRate = engine.getPlaybackRate() * nudgeMul
    if (video.playbackRate !== targetRate) video.playbackRate = targetRate

    if ((now - t0) / 1000 < runSec) {
      video.requestVideoFrameCallback(onFrame)
    } else {
      engine.pause()
      video.pause()
      state.done = true
      statusEl.textContent = 'done'
      report()
    }
    if (state.offsets.length % 60 === 0) {
      statusEl.textContent = `run ${((now - t0) / 1000).toFixed(0)}/${runSec}s`
      report()
    }
  }
  video.requestVideoFrameCallback(onFrame)
}

main().catch(fail)
