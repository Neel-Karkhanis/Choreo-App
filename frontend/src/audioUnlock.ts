// iOS audio unlock, done once per session from the first real user gesture.
// Two platform quirks:
//
//   1. An AudioContext created OUTSIDE a user gesture starts life `suspended`
//      on iOS, and a later resume() is not a dependable way back. So the app's
//      single AudioContext is created HERE — synchronously inside the first
//      touchstart / click / keydown — and never inside loadStems' async chain
//      (see Song.tsx). getAudioContext() hands that one instance to every
//      StemEngine. This is the load-bearing fix and it works.
//
//   2. A pure Web Audio graph (which the stem engine is — there is no <audio>
//      element anywhere in its sound path) is silenced by the hardware ringer
//      switch / Silent Mode on iOS. There is no reliable programmatic way out
//      of that from a Web Audio-only page — a muted looping <audio> hint was
//      tried and did NOT help on a real device, so it was removed rather than
//      left dead in the page. The app handles this in the UI instead: while
//      playback is demonstrably running it shows a one-time, dismissible
//      "check Silent Mode" notice on iOS (see SilentModeNotice in Song.tsx).
//
// If the whole approach is abandoned, delete this file and the
// installAudioUnlock call in main.tsx; StemEngine would then need
// `new AudioContext()` back.

let sharedCtx: AudioContext | null = null
let unlocked = false

/**
 * The app's one and only AudioContext. Created lazily, but in practice the
 * first caller is always inside a user gesture — either installAudioUnlock's
 * listeners below, or StemEngine.play() (which only ever runs from a tap) —
 * so on iOS it comes up `running`, not `suspended`.
 */
export function getAudioContext(): AudioContext {
  if (!sharedCtx) sharedCtx = new AudioContext({ sampleRate: 44100 })
  return sharedCtx
}

function unlock(): void {
  if (unlocked) return
  unlocked = true
  document.removeEventListener('touchstart', unlock, true)
  document.removeEventListener('click', unlock, true)
  document.removeEventListener('keydown', unlock, true)

  const ctx = getAudioContext()
  void ctx.resume()

  // A bare resume() is not a reliable unlock on iOS; a buffer actually
  // rendered to the destination is. One inaudible sample is enough, and the
  // `unlocked` guard keeps this to exactly once per session.
  try {
    const src = ctx.createBufferSource()
    src.buffer = ctx.createBuffer(1, 1, 22050)
    src.connect(ctx.destination)
    src.start(0)
  } catch {
    // Non-fatal: resume() above may still take, and StemEngine.play() calls
    // resume() again on every play.
  }
}

/** Register the one-shot gesture listeners. Call once, at startup (main.tsx). */
export function installAudioUnlock(): void {
  if (typeof document === 'undefined') return
  const opts = { once: true, capture: true } as const
  document.addEventListener('touchstart', unlock, opts)
  document.addEventListener('click', unlock, opts)
  document.addEventListener('keydown', unlock, opts)
}
