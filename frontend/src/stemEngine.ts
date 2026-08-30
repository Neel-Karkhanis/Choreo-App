import { apiFetch } from './api'
import { API_BASE } from './snap'

// The Web Audio stem engine: the app's audio source AND clock authority.
//
// Replaces the shared HTMLAudioElement wholesale. Five AudioBuffers (original
// mix + four Demucs stems) play through five parallel GainNodes into one
// AudioContext; a "stem mode" is nothing but a gain vector, so switching what
// you hear can never desync anything — all five sources run continuously and
// only their gains move. audioContext.currentTime is the one clock; both
// wavesurfer instances consume it through a minimal media-element shim
// (engine.media), so every existing time→pixel consumer (playhead, grid
// highlight, minimap viewport, loop) reads this clock unchanged.
//
// Wavesurfer never decodes or fetches anything: both instances are created
// with `media` + `peaks` + `duration` and no URL, which its v7 load path
// treats as "render w/o audio" — peaks are wrapped as a faux buffer, the
// fetch and decodeAudioData branches are skipped entirely (verified against
// wavesurfer.js@7.12.10 dist/wavesurfer.js loadAudio).

export const STEM_NAMES = ['original', 'vocals', 'drums', 'bass', 'other'] as const
export type StemName = (typeof STEM_NAMES)[number]

export type StemBuffers = Record<StemName, AudioBuffer>

// User-facing playback modes. `other` is a buffer but not a mode: it exists
// because instrumental = drums + bass + other, derived in the graph — there
// is no separate instrumental file loaded.
export const STEM_MODES = ['all', 'vocals', 'drums', 'bass', 'instrumental'] as const
export type StemMode = (typeof STEM_MODES)[number]
export const DEFAULT_STEM_MODE: StemMode = 'all'

// The mode table IS the feature: one gain value per stem source, mutually
// exclusive modes, "all" plays the released mix with every stem silent.
// Adding or removing a mode is one line here (plus MODE_DISPLAY below).
const MODE_GAINS: Record<StemMode, Record<StemName, number>> = {
  all:          { original: 1, vocals: 0, drums: 0, bass: 0, other: 0 },
  vocals:       { original: 0, vocals: 1, drums: 0, bass: 0, other: 0 },
  drums:        { original: 0, vocals: 0, drums: 1, bass: 0, other: 0 },
  bass:         { original: 0, vocals: 0, drums: 0, bass: 1, other: 0 },
  instrumental: { original: 0, vocals: 0, drums: 1, bass: 1, other: 1 },
}

// Which buffers compose the RENDERED waveform per mode. Multi-buffer modes
// (instrumental) draw the peak envelope of the summed signals, matching what
// the graph actually plays.
const MODE_DISPLAY: Record<StemMode, StemName[]> = {
  all: ['original'],
  vocals: ['vocals'],
  drums: ['drums'],
  bass: ['bass'],
  instrumental: ['drums', 'bass', 'other'],
}

// Per-buffer decode targets. Stems are analysis/isolation audio, not the
// release: mono at reduced rates cuts the in-memory footprint (this is
// memory-profiled on mobile later — see the byte-size logs in loadStems).
// Mixed rates in one graph are fine: AudioBuffer carries its own rate and
// the source node resamples on playback.
const TARGET_FORMATS: Record<StemName, { channels: number; sampleRate: number }> = {
  original: { channels: 2, sampleRate: 44100 },
  vocals: { channels: 1, sampleRate: 22050 },
  drums: { channels: 1, sampleRate: 22050 },
  other: { channels: 1, sampleRate: 22050 },
  bass: { channels: 1, sampleRate: 8000 },
}

// Gain moves are ramped, never assigned: a direct .value jump is a step
// discontinuity in the output signal and clicks audibly on every switch.
const GAIN_RAMP_SECONDS = 0.015

// Playback rate bounds (§speed): resampling only — pitch shifts with rate by
// deliberate tradeoff; there is no time-stretcher and no pitch flag.
export const PLAYBACK_RATE_MIN = 0.25
export const PLAYBACK_RATE_MAX = 2

// Seek duck: a rebuild-causing seek hard-stops five sources mid-signal, which
// clicks. The duck gain ramps the summed output to 0 over this window, the
// rebuild happens in silence, and the ramp back up makes the whole seek an
// imperceptible dip instead of a pop. Rapid drag-seeks just re-arm the timer,
// so a drag burst collapses into ONE rebuild at the final position.
const SEEK_DUCK_SECONDS = 0.005

// Sources are scheduled slightly in the future so all five start() calls in
// one synchronous block share the same start sample — synced by construction.
// A `when` already in the past would let the browser start each source "asap",
// which is only quantum-aligned, not guaranteed identical across five nodes.
const SCHEDULE_LEAD_SECONDS = 0.005

// Waveform peak resolution. The main view's fixed zoom lands around
// 100–150 px/sec, so 200 peaks/sec keeps ≥1 peak per pixel there while the
// whole set (5 modes × one Float32Array) stays around a megabyte per track.
const PEAKS_PER_SECOND = 200

// End-of-track detection tolerance: onended callbacks dispatch a little after
// the buffer truly exhausts, and the clock is quantum-granular.
const ENDED_EPSILON_SECONDS = 0.05

const trackUrl = (trackId: string, name: StemName) => {
  const id = encodeURIComponent(trackId)
  return name === 'original'
    ? `${API_BASE}/tracks/${id}/audio`
    : `${API_BASE}/tracks/${id}/stems/${name}`
}

// Decode one fetched file straight to its target format. decodeAudioData
// resamples to the decoding context's rate (44.1k — an exact no-op for the
// 44.1k source files); the OfflineAudioContext render then downmixes and
// resamples in one pass: its destination has an explicit channel count, so a
// stereo source into a mono context downmixes by the standard speaker rules.
async function decodeToTarget(
  data: ArrayBuffer,
  target: { channels: number; sampleRate: number },
): Promise<AudioBuffer> {
  const decoded = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(data)
  if (decoded.numberOfChannels === target.channels && decoded.sampleRate === target.sampleRate) {
    return decoded
  }
  const offline = new OfflineAudioContext(
    target.channels,
    Math.ceil(decoded.duration * target.sampleRate),
    target.sampleRate,
  )
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()
  return offline.startRendering()
}

const bufferBytes = (buffer: AudioBuffer) =>
  buffer.length * buffer.numberOfChannels * Float32Array.BYTES_PER_ELEMENT

/**
 * THE load seam: everything that needs stem audio goes through here, keyed by
 * trackId only. A future library feature reimplements this function (or its
 * fetches) and nothing else changes. Loads and decodes all five buffers
 * eagerly, in parallel, to their target formats, and logs real decoded byte
 * sizes (mobile memory profiling wants numbers, not estimates).
 */
export async function loadStems(trackId: string, signal?: AbortSignal): Promise<StemBuffers> {
  const entries = await Promise.all(
    STEM_NAMES.map(async (name) => {
      const url = trackUrl(trackId, name)
      const res = await apiFetch(url, { signal })
      if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`)
      const buffer = await decodeToTarget(await res.arrayBuffer(), TARGET_FORMATS[name])
      return [name, buffer] as const
    }),
  )
  const buffers = Object.fromEntries(entries) as StemBuffers
  let total = 0
  for (const name of STEM_NAMES) {
    const b = buffers[name]
    const bytes = bufferBytes(b)
    total += bytes
    console.log(
      `[stems] ${name}: ${b.numberOfChannels}ch @ ${b.sampleRate}Hz, ` +
        `${b.length} samples, ${(bytes / 1024 / 1024).toFixed(2)} MB decoded`,
    )
  }
  console.log(`[stems] total decoded: ${(total / 1024 / 1024).toFixed(2)} MB`)
  return buffers
}

// Scale a peak set back into [-1, 1] if the mix summed past full scale, so
// wavesurfer's own in-place normalize never rewrites our cached arrays by a
// data-dependent factor mid-flight.
function normalizePeaks(peaks: Float32Array): Float32Array {
  let max = 0
  for (const v of peaks) if (v > max) max = v
  if (max > 1) for (let i = 0; i < peaks.length; i++) peaks[i] /= max
  return peaks
}

// Max-abs peak envelope of one buffer (any channel count, max across
// channels — a stereo mix must not cancel L against R).
function peaksOf(buffer: AudioBuffer, duration: number): Float32Array {
  const bucketCount = Math.max(1, Math.ceil(duration * PEAKS_PER_SECOND))
  const peaks = new Float32Array(bucketCount)
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c)
    const samplesPerBucket = buffer.sampleRate / PEAKS_PER_SECOND
    for (let i = 0; i < data.length; i++) {
      const k = Math.min(bucketCount - 1, Math.floor(i / samplesPerBucket))
      const a = Math.abs(data[i])
      if (a > peaks[k]) peaks[k] = a
    }
  }
  return normalizePeaks(peaks)
}

// Peak envelope of the SUM of several mono buffers — what instrumental mode
// actually sounds like. Buffers carry different rates (drums/other 22.05k,
// bass 8k), so the sum is sampled on the fastest buffer's grid with
// nearest-sample lookup into the slower ones; phase-accurate enough for a
// display envelope.
function peaksOfMix(buffers: AudioBuffer[], duration: number): Float32Array {
  const masterRate = Math.max(...buffers.map((b) => b.sampleRate))
  const channels = buffers.map((b) => b.getChannelData(0))
  const steps = buffers.map((b) => b.sampleRate / masterRate)
  const bucketCount = Math.max(1, Math.ceil(duration * PEAKS_PER_SECOND))
  const peaks = new Float32Array(bucketCount)
  const samplesPerBucket = masterRate / PEAKS_PER_SECOND
  const totalSamples = Math.ceil(duration * masterRate)
  for (let i = 0; i < totalSamples; i++) {
    let v = 0
    for (let s = 0; s < channels.length; s++) {
      const idx = Math.round(i * steps[s])
      if (idx < channels[s].length) v += channels[s][idx]
    }
    const a = Math.abs(v)
    const k = Math.min(bucketCount - 1, Math.floor(i / samplesPerBucket))
    if (a > peaks[k]) peaks[k] = a
  }
  return normalizePeaks(peaks)
}

// Faux AudioBuffer over precomputed peaks — the same shape wavesurfer's own
// Decoder.createBuffer produces from the `peaks` option, handed straight to
// Renderer.render() on mode switches. No audio samples, no decoding.
function displayBufferOf(peaks: Float32Array, duration: number): AudioBuffer {
  return {
    duration,
    length: peaks.length,
    sampleRate: peaks.length / duration,
    numberOfChannels: 1,
    getChannelData: () => peaks,
    copyFromChannel: AudioBuffer.prototype.copyFromChannel,
    copyToChannel: AudioBuffer.prototype.copyToChannel,
  } as unknown as AudioBuffer
}

type ShimListener = (event: Event) => void

/**
 * Graph + clock, one object — they are inseparable because seeking IS a graph
 * rebuild (AudioBufferSourceNode is one-shot: it cannot be seeked, restarted,
 * or reused, so seek/pause/resume all mean "discard all five sources, build
 * five new ones, start them together").
 *
 * Clock surface: currentTime / duration / paused / play() / pause() / seek(t),
 * with addEventListener/removeEventListener as the subscribe mechanism (media
 * event names, since wavesurfer is the main subscriber). Track time is
 * startOffset + (ctx.currentTime - startAt) * rate — RATE-AWARE, and every
 * rate or loop change RE-ANCHORS (recompute t under the old parameters, reset
 * startAt/startOffset, then apply the new ones) so history is never
 * retroactively rescaled. With a loop active the sources wrap THEMSELVES
 * (native loop/loopStart/loopEnd — no seek path, no rebuild), so the clock
 * wraps track time modulo the loop region with the same piecewise mapping the
 * sources follow: t = raw while raw < loopEnd, then loopStart +
 * ((raw - loopEnd) mod loopLen). Loop-point exactness across mixed sample
 * rates was measured before this was built: ≤0.004 ms inter-stem deviation
 * over 220+ wraps with non-sample-aligned bounds, no accumulation (the
 * browser tracks loop position in double precision, not per-source sample
 * grids).
 *
 * Wavesurfer consumes all of this through `engine.media`, a minimal shim
 * implementing exactly the media-element surface wavesurfer v7 reads
 * (verified against dist/player.js + dist/wavesurfer.js): property gets on
 * currentTime/duration/paused/ended/seeking/volume/muted/playbackRate/src/
 * currentSrc, a currentTime SET (that is wavesurfer's whole seek path),
 * play/pause/canPlayType/addEventListener/removeEventListener, and the events
 * play/pause/ended/timeupdate/seeking/seeked/volumechange. Wavesurfer runs its
 * own 16 ms ticker between play and pause events, polling currentTime — the
 * shim only needs discrete event edges, not a timeupdate stream.
 */
export class StemEngine {
  readonly duration: number
  /** The media-element surface both wavesurfer instances receive via `media`. */
  readonly media: HTMLMediaElement

  private readonly ctx: AudioContext
  private readonly buffers: StemBuffers
  private readonly gains: Record<StemName, GainNode>
  private readonly masterGain: GainNode
  // Separate from masterGain (volume/mute) so seek ducking and user volume
  // never fight over one AudioParam.
  private readonly duckGain: GainNode
  // Shortest buffer across the five: a loopEnd beyond this would make that
  // source wrap early at its own buffer end and desync the set, so the
  // effective loop end is clamped here for sources AND clock alike.
  private readonly minBufferDuration: number

  private sources: AudioBufferSourceNode[] = []
  // Guards overlapping rebuilds: onended callbacks from a discarded source
  // generation must never mutate clock state. Rebuilds themselves are fully
  // synchronous (buffers are pre-decoded), so two source sets can never be
  // mid-construction at once — the generation check covers the async
  // callbacks that outlive their sources.
  private generation = 0
  private startAt = 0 // ctx.currentTime the current sources started at
  private startOffset = 0 // track time at startAt; the position itself when paused
  private _paused = true
  private _ended = false
  private _volume = 1
  private _muted = false
  private mode: StemMode = DEFAULT_STEM_MODE
  // Playback rate: applied identically to all five sources (a multiplier, so
  // mixed sample rates are unaffected) and folded into the clock.
  private rate = 1
  // Loop region in TRACK/BUFFER seconds — rate-independent by construction,
  // which is exactly why a loop set at 1x covers the same musical region at
  // 0.5x. null/false = no loop.
  private loopStartTime: number | null = null
  private loopEndTime: number | null = null
  private loopOn = false
  // A rebuild-causing seek defers its rebuild by SEEK_DUCK_SECONDS so the
  // sources stop in silence; this timer is the pending rebuild. Rapid seeks
  // replace it (one rebuild per burst); pause/destroy cancel it and restore
  // the duck gain, so it can never wedge the output at 0.
  private pendingSeekTimer: ReturnType<typeof setTimeout> | null = null

  private listeners = new Map<string, { fn: ShimListener; once: boolean }[]>()
  private peaksCache = new Map<StemMode, Float32Array[]>()
  private displayCache = new Map<StemMode, AudioBuffer>()

  constructor(buffers: StemBuffers) {
    this.buffers = buffers
    // The original mix is the authoritative track length; stems can differ by
    // a few ms of codec padding and simply end when their buffers run out.
    this.duration = buffers.original.duration
    this.minBufferDuration = Math.min(...STEM_NAMES.map((n) => buffers[n].duration))
    // 44.1k context so the original buffer plays without a resample.
    this.ctx = new AudioContext({ sampleRate: 44100 })
    // 5 stem gains -> masterGain (volume/mute) -> duckGain (seek ramp) -> out
    this.duckGain = this.ctx.createGain()
    this.duckGain.connect(this.ctx.destination)
    this.masterGain = this.ctx.createGain()
    this.masterGain.connect(this.duckGain)
    this.gains = {} as Record<StemName, GainNode>
    for (const name of STEM_NAMES) {
      const gain = this.ctx.createGain()
      gain.gain.value = MODE_GAINS[this.mode][name]
      gain.connect(this.masterGain)
      this.gains[name] = gain
    }
    this.media = this.createMediaShim()
  }

  // ---- clock ----

  // Track time advances at rate x wall time from the last anchor, and wraps
  // modulo the loop region exactly as the natively-looping sources do: linear
  // until raw time first reaches loopEnd, then periodic inside [start, end).
  // Every parameter change (rate, loop region, seek) re-anchors, so this
  // formula only ever runs with constants that were true for the whole epoch.
  get currentTime(): number {
    if (this._paused) return this.startOffset
    const raw =
      this.startOffset + Math.max(0, this.ctx.currentTime - this.startAt) * this.rate
    const loop = this.activeLoop()
    if (loop && raw >= loop.end) {
      return loop.start + ((raw - loop.end) % (loop.end - loop.start))
    }
    return Math.min(raw, this.duration)
  }

  // The loop region the SOURCES are actually wrapping on, or null. End is
  // clamped to the shortest buffer so all five wrap at the same instant even
  // when stems carry a few ms less codec padding than the original.
  private activeLoop(): { start: number; end: number } | null {
    if (!this.loopOn || this.loopStartTime === null || this.loopEndTime === null) return null
    const start = this.loopStartTime
    const end = Math.min(this.loopEndTime, this.minBufferDuration)
    return start < end ? { start, end } : null
  }

  // Fold the elapsed epoch into startOffset/startAt so rate or loop changes
  // apply from NOW instead of rescaling history. No-op while paused (the
  // offset already IS the position).
  private reanchor(): void {
    if (this._paused) return
    const t = this.currentTime
    this.startAt = this.ctx.currentTime
    this.startOffset = t
  }

  get paused(): boolean {
    return this._paused
  }

  get ended(): boolean {
    return this._ended
  }

  play(): Promise<void> {
    if (!this._paused) return Promise.resolve()
    const loop = this.activeLoop()
    if (loop && (this.startOffset < loop.start || this.startOffset >= loop.end)) {
      // A user who presses play with a loop on wants the SECTION — from
      // outside the region, playback enters at A rather than running past
      // loopEnd (where a native loop would never wrap).
      this.startOffset = loop.start
    } else if (this.currentTime >= this.duration) {
      // Media-element semantics: play() at the end restarts from the top.
      this.startOffset = 0
    }
    this._ended = false
    this._paused = false
    // Autoplay policy leaves a fresh context suspended until a user gesture.
    // resume() is async, but scheduling against a suspended context is safe:
    // ctx.currentTime is frozen until resume, and the sources start on that
    // same frozen-then-running timeline — no drift window.
    void this.ctx.resume()
    this.startSources(this.startOffset)
    this.emit('play')
    return Promise.resolve()
  }

  pause(): void {
    if (this._paused) return
    const t = this.currentTime
    // A pending ducked rebuild must not fire into the paused state, and the
    // duck gain must never be left at 0 (the next play would be silent).
    this.cancelPendingSeek()
    this.stopSources()
    this.startOffset = t
    this._paused = true
    this.emit('pause')
  }

  seek(t: number): void {
    let clamped = Math.min(Math.max(t, 0), this.duration)
    const loop = this.activeLoop()
    if (loop && clamped >= loop.end) {
      // Loop re-catch, same behavior the media-era loop had: a seek landing
      // at/past B goes to A. Seeking BEFORE A stays allowed — playback runs
      // forward into the region and wraps natively at B.
      clamped = loop.start
    }
    if (!loop && !this._paused && clamped >= this.duration) {
      // Seeking to the very end mid-playback ends playback, like a media
      // element would.
      this.cancelPendingSeek()
      this.stopSources()
      this.startOffset = this.duration
      this._paused = true
      this._ended = true
      this.emit('seeking')
      this.emit('timeupdate')
      this.emit('seeked')
      this.emit('pause')
      this.emit('ended')
      return
    }
    this._ended = false
    if (this._paused) {
      this.startOffset = clamped
    } else {
      this.duckedRebuild(clamped)
    }
    // Real elements fire this triple on a seek; wavesurfer's timeupdate
    // handler is what repaints the cursor while paused.
    this.emit('seeking')
    this.emit('timeupdate')
    this.emit('seeked')
  }

  // ---- graph ----

  /** Ramped gain move to the mode's vector — data lookup, no branching. */
  setMode(mode: StemMode): void {
    this.mode = mode
    const now = this.ctx.currentTime
    for (const name of STEM_NAMES) {
      const param = this.gains[name].gain
      param.cancelScheduledValues(now)
      param.setValueAtTime(param.value, now)
      param.linearRampToValueAtTime(MODE_GAINS[mode][name], now + GAIN_RAMP_SECONDS)
    }
  }

  getMode(): StemMode {
    return this.mode
  }

  /**
   * Live playback rate on all five sources — a multiplier, so the mixed
   * buffer sample rates are unaffected and the sources stay locked together.
   * No rebuild and no waveform discontinuity: an AudioParam rate change bends
   * the playback slope, it never jumps the playback position. Pitch shifts
   * with rate (resampling) — the accepted tradeoff; there is no time-stretch.
   */
  setPlaybackRate(rate: number): void {
    const clamped = Math.min(Math.max(rate, PLAYBACK_RATE_MIN), PLAYBACK_RATE_MAX)
    if (clamped === this.rate) return
    // Re-anchor under the OLD rate first: the naive formula would rescale
    // the entire elapsed epoch and jump the playhead.
    this.reanchor()
    this.rate = clamped
    for (const source of this.sources) source.playbackRate.value = clamped
    this.emit('ratechange')
  }

  getPlaybackRate(): number {
    return this.rate
  }

  /**
   * Native A/B loop on all five sources. loopStart/loopEnd are BUFFER-time
   * seconds, so one pair of values serves every sample rate and the region is
   * rate-independent. Enabling, disabling, and moving A/B are all applied
   * LIVE on playing sources — never a rebuild; the only seek this can issue
   * is the outside-region entry jump below.
   */
  setLoop(start: number | null, end: number | null, enabled: boolean): void {
    // Fold the epoch under the OLD loop mapping first — the wrapped position
    // is real; the raw pre-wrap time is not.
    this.reanchor()
    this.loopStartTime = start
    this.loopEndTime = end
    this.loopOn = enabled
    const loop = this.activeLoop()
    for (const source of this.sources) {
      source.loop = loop !== null
      if (loop) {
        source.loopStart = loop.start
        source.loopEnd = loop.end
      }
    }
    // A user enabling a loop wants the section: with the playhead outside
    // [A, B) during playback, enter at A (a source past loopEnd would
    // otherwise never wrap). Paused outside is handled by play().
    if (loop && !this._paused) {
      const t = this.startOffset
      if (t < loop.start || t >= loop.end) this.seek(loop.start)
    }
  }

  /** Cached `peaks` option value for a mode — reference-stable, so react
   *  hooks and wavesurfer option-identity checks never see churn. */
  peaksFor(mode: StemMode): Float32Array[] {
    let cached = this.peaksCache.get(mode)
    if (!cached) {
      const parts = MODE_DISPLAY[mode].map((name) => this.buffers[name])
      cached = [
        parts.length === 1 ? peaksOf(parts[0], this.duration) : peaksOfMix(parts, this.duration),
      ]
      this.peaksCache.set(mode, cached)
    }
    return cached
  }

  /** Cached faux AudioBuffer for Renderer.render() on mode switches. */
  displayBufferFor(mode: StemMode): AudioBuffer {
    let cached = this.displayCache.get(mode)
    if (!cached) {
      cached = displayBufferOf(this.peaksFor(mode)[0], this.duration)
      this.displayCache.set(mode, cached)
    }
    return cached
  }

  destroy(): void {
    this.cancelPendingSeek()
    this.stopSources()
    this.listeners.clear()
    void this.ctx.close()
  }

  // ---- internals ----

  // Rebuild-causing seek while playing. The click in a hard seek is stopping
  // five sources mid-signal, so the stop must happen at silence: ramp the
  // duck gain to 0 over SEEK_DUCK_SECONDS, rebuild inside that silence, ramp
  // back. The CLOCK re-anchors to the target immediately (the playhead must
  // track a drag in real time); the audio follows within ~10 ms. A newer seek
  // during the window just re-anchors and re-arms the timer — a whole minimap
  // drag collapses into one rebuild at the final position, and every timer
  // path ends in a ramp back to 1, so the duck can never stick at 0.
  private duckedRebuild(offset: number): void {
    const now = this.ctx.currentTime
    const duck = this.duckGain.gain
    duck.cancelScheduledValues(now)
    duck.setValueAtTime(duck.value, now)
    duck.linearRampToValueAtTime(0, now + SEEK_DUCK_SECONDS)
    this.startAt = now
    this.startOffset = offset
    if (this.pendingSeekTimer !== null) clearTimeout(this.pendingSeekTimer)
    this.pendingSeekTimer = setTimeout(() => {
      this.pendingSeekTimer = null
      // The clock advanced during the duck window; start where it is NOW so
      // audio rejoins the playhead seamlessly.
      this.startSources(this.currentTime)
      const t = this.ctx.currentTime
      duck.cancelScheduledValues(t)
      duck.setValueAtTime(0, t)
      duck.linearRampToValueAtTime(1, t + SCHEDULE_LEAD_SECONDS + SEEK_DUCK_SECONDS)
    }, SEEK_DUCK_SECONDS * 1000)
  }

  // Abandon a pending ducked rebuild (pause/destroy). The duck gain snaps
  // back to 1 — nothing is audible in the paused state, and leaving it at 0
  // would silence the next play().
  private cancelPendingSeek(): void {
    if (this.pendingSeekTimer === null) return
    clearTimeout(this.pendingSeekTimer)
    this.pendingSeekTimer = null
    const now = this.ctx.currentTime
    const duck = this.duckGain.gain
    duck.cancelScheduledValues(now)
    duck.setValueAtTime(1, now)
  }

  // The hot path: stop and discard all five sources, build and start five new
  // ones at a common future context time with a common offset. Mode gains
  // live on the persistent GainNodes, so they carry across every rebuild
  // untouched; rate and loop parameters are re-applied to the new sources.
  // Synchronous end to end — generations keep stale onended callbacks from
  // touching anything.
  private startSources(offset: number): void {
    this.stopSources()
    const generation = ++this.generation
    const when = this.ctx.currentTime + SCHEDULE_LEAD_SECONDS
    const loop = this.activeLoop()
    for (const name of STEM_NAMES) {
      const buffer = this.buffers[name]
      if (offset >= buffer.duration) continue // this stem has nothing left
      const source = this.ctx.createBufferSource()
      source.buffer = buffer
      source.playbackRate.value = this.rate
      if (loop) {
        source.loop = true
        source.loopStart = loop.start
        source.loopEnd = loop.end
      }
      source.connect(this.gains[name])
      source.start(when, offset)
      if (name === 'original') {
        // The original carries the authoritative duration, so ITS exhaustion
        // is end-of-track. (A looping source never ends; if the loop is later
        // disabled live, this fires at the buffer end as usual.)
        source.onended = () => this.onNaturalEnd(generation)
      }
      this.sources.push(source)
    }
    this.startAt = when
    this.startOffset = offset
  }

  private stopSources(): void {
    for (const source of this.sources) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // never started or already stopped — either is fine, it's discarded
      }
      source.disconnect()
    }
    this.sources = []
  }

  private onNaturalEnd(generation: number): void {
    // Stale generations (seek/pause rebuilt the graph after this source was
    // discarded but before its callback dispatched) must not end playback,
    // and neither may the old sources of a seek whose rebuild is still inside
    // its duck window (they stop at the rebuild, not at track end).
    if (generation !== this.generation || this._paused) return
    if (this.pendingSeekTimer !== null) return
    if (this.currentTime < this.duration - ENDED_EPSILON_SECONDS) return
    this.stopSources()
    this.startOffset = this.duration
    this._paused = true
    this._ended = true
    // Media elements report pause -> ended; wavesurfer stops its ticker on
    // pause and emits finish on ended.
    this.emit('pause')
    this.emit('timeupdate')
    this.emit('ended')
  }

  private setVolume(v: number): void {
    this._volume = Math.min(Math.max(v, 0), 1)
    this.applyMasterGain()
    this.emit('volumechange')
  }

  private setMuted(m: boolean): void {
    this._muted = m
    this.applyMasterGain()
    this.emit('volumechange')
  }

  private applyMasterGain(): void {
    const target = this._muted ? 0 : this._volume
    const now = this.ctx.currentTime
    const param = this.masterGain.gain
    param.cancelScheduledValues(now)
    param.setValueAtTime(param.value, now)
    param.linearRampToValueAtTime(target, now + GAIN_RAMP_SECONDS)
  }

  private on(type: string, fn: ShimListener, options?: boolean | AddEventListenerOptions): void {
    const once = typeof options === 'object' && options !== null && options.once === true
    const list = this.listeners.get(type) ?? []
    list.push({ fn, once })
    this.listeners.set(type, list)
  }

  private off(type: string, fn: ShimListener): void {
    const list = this.listeners.get(type)
    if (!list) return
    const index = list.findIndex((entry) => entry.fn === fn)
    if (index !== -1) list.splice(index, 1)
  }

  private emit(type: string): void {
    const list = this.listeners.get(type)
    if (!list || list.length === 0) return
    const snapshot = [...list]
    for (const entry of snapshot) {
      if (entry.once) this.off(type, entry.fn)
      entry.fn(new Event(type))
    }
  }

  private createMediaShim(): HTMLMediaElement {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const engine = this
    const shim = {
      // Setting currentTime IS wavesurfer's seek path (Player.setTime).
      get currentTime() {
        return engine.currentTime
      },
      set currentTime(t: number) {
        engine.seek(t)
      },
      get duration() {
        return engine.duration
      },
      get paused() {
        return engine._paused
      },
      get ended() {
        return engine._ended
      },
      // Seeks are synchronous rebuilds; there is never an async seeking window.
      seeking: false,
      get volume() {
        return engine._volume
      },
      set volume(v: number) {
        engine.setVolume(v)
      },
      get muted() {
        return engine._muted
      },
      set muted(m: boolean) {
        engine.setMuted(m)
      },
      get playbackRate() {
        return engine.rate
      },
      set playbackRate(rate: number) {
        engine.setPlaybackRate(rate)
      },
      // Accepted and ignored: rate resamples (pitch shifts with speed) by
      // deliberate tradeoff — there is no time-stretch path to toggle.
      preservesPitch: false,
      // Empty src is load-bearing: wavesurfer's constructor reads it to decide
      // there is no URL to load, and its setSrc('', undefined) no-ops.
      currentSrc: '',
      src: '',
      canPlayType: () => '',
      play: () => engine.play(),
      pause: () => engine.pause(),
      addEventListener: (type: string, fn: ShimListener, options?: boolean | AddEventListenerOptions) =>
        engine.on(type, fn, options),
      removeEventListener: (type: string, fn: ShimListener) => engine.off(type, fn),
    }
    return shim as unknown as HTMLMediaElement
  }
}
