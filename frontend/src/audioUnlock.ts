// iOS audio unlock, done once per session from the first real user gesture.
// Two platform quirks, one fix each:
//
//   1. An AudioContext created OUTSIDE a user gesture starts life `suspended`
//      on iOS, and a later resume() is not a dependable way back. So the app's
//      single AudioContext is created HERE — synchronously inside the first
//      touchstart / click / keydown — and never inside loadStems' async chain
//      (see Song.tsx). getAudioContext() hands that one instance to every
//      StemEngine.
//
//   2. A pure Web Audio graph (which the stem engine is — there is no <audio>
//      element anywhere in its sound path) is silenced by the hardware ringer
//      switch on iOS. A muted, looping <audio> element that is merely *playing*
//      — never routed through Web Audio — takes this page's audio out of that
//      behavior for the rest of the session. It is a session hint and nothing
//      else: it is not connected to any node and nothing reads from it.
//
// If the <audio> hint turns out not to help on a real device, delete it (the
// `try` block in unlock()) rather than leaving a dead element in the page. If
// the whole approach is abandoned, delete this file and the installAudioUnlock
// call in main.tsx; StemEngine would then need `new AudioContext()` back.

let sharedCtx: AudioContext | null = null
let unlocked = false

// ~0.15s of 8-bit silence, mono, 8 kHz. Inlined so the ringer-switch hint has
// no network dependency and works on first paint / offline.
const SILENT_WAV =
  'data:audio/wav;base64,UklGRtQEAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YbAEAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIA='

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

  // Ringer-switch hint (quirk 2). Best-effort — the AudioContext unlock above
  // is the load-bearing part. NOT routed through Web Audio by design.
  try {
    const el = document.createElement('audio')
    el.src = SILENT_WAV
    el.loop = true
    el.muted = true
    el.setAttribute('playsinline', '')
    el.setAttribute('aria-hidden', 'true')
    el.style.display = 'none'
    document.body.appendChild(el)
    void el.play().catch(() => {})
  } catch {
    // No <audio> / DOM unavailable — skip the hint.
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
