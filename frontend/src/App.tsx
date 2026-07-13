import { useEffect, useMemo, useState } from 'react'
import type WaveSurfer from 'wavesurfer.js'
import Timeline, { type GridData, type OnsetData } from './Timeline'

interface Track {
  id: string
  filename: string
}

interface Stem {
  name: string
  url: string
}

// Parallel arrays: t[i] is an onset timestamp in seconds, strength[i] the
// onset-envelope value at that onset. strength is plumbed by schema v4 but
// deliberately unused by the UI so far.
interface StemOnsets {
  t: number[]
  strength: number[]
}

// Mirrors the analysis JSON contract (schema_version 4): downbeats and
// eight_counts are integer indices into beats, not timestamps. onsets.drums
// and onsets.bass are timestamp/strength pairs, independent of the beat grid.
interface Analysis {
  id: string
  filename: string
  schema_version: number
  duration: number
  tempo: number | null
  beats: number[]
  downbeats: number[]
  eight_counts: number[]
  onsets: { drums: StemOnsets; bass: StemOnsets }
  audio_url: string
  stems: Stem[]
}

const isNumbers = (xs: unknown): xs is number[] =>
  Array.isArray(xs) && xs.every((x) => typeof x === 'number' && Number.isFinite(x))

// Grid data must match the schema v2 contract exactly; on any mismatch we
// render an error instead of the grid rather than inferring a shape.
function gridFromAnalysis(a: Analysis): { grid?: GridData; gridError?: string } {
  if (!isNumbers(a.beats)) return { gridError: 'analysis.beats is not number[]' }
  const isIndices = (xs: unknown): xs is number[] =>
    Array.isArray(xs) &&
    xs.every((x) => Number.isInteger(x) && x >= 0 && x < a.beats.length)
  if (!isIndices(a.downbeats))
    return { gridError: 'analysis.downbeats is not an array of indices into beats' }
  if (!isIndices(a.eight_counts))
    return { gridError: 'analysis.eight_counts is not an array of indices into beats' }
  return {
    grid: {
      beats: a.beats,
      downbeatIndices: a.downbeats,
      eightCountIndices: a.eight_counts,
    },
  }
}

// Onsets are independent of the beat grid, so validation failures here
// shouldn't block the grid from rendering — reported separately. Only the
// timestamps flow to the Timeline; strength is contract-checked but unused.
function onsetsFromAnalysis(a: Analysis): { onsets?: OnsetData; onsetsError?: string } {
  const isStemOnsets = (s: StemOnsets | undefined) =>
    !!s && isNumbers(s.t) && isNumbers(s.strength) && s.t.length === s.strength.length
  if (!a.onsets || !isStemOnsets(a.onsets.drums) || !isStemOnsets(a.onsets.bass)) {
    return {
      onsetsError:
        'analysis.onsets is not { drums|bass: { t: number[], strength: number[] } }',
    }
  }
  return { onsets: { drums: a.onsets.drums.t, bass: a.onsets.bass.t } }
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`)
  return res.json()
}

function App() {
  const [tracks, setTracks] = useState<Track[]>([])
  const [selected, setSelected] = useState<Track | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Single source of truth for time↔pixel mapping; later layers (beat grid,
  // overlays) read duration/zoom/scroll from this instance.
  const [, setWavesurfer] = useState<WaveSurfer | null>(null)
  const { grid, gridError } = useMemo(
    () => (analysis ? gridFromAnalysis(analysis) : {}),
    [analysis],
  )
  const { onsets, onsetsError } = useMemo(
    () => (analysis ? onsetsFromAnalysis(analysis) : {}),
    [analysis],
  )

  useEffect(() => {
    fetchJson<{ tracks: Track[] }>('/api/tracks')
      .then((data) => setTracks(data.tracks))
      .catch((err) => setError(String(err)))
  }, [])

  useEffect(() => {
    if (!selected) return
    let stale = false
    setLoading(true)
    setAnalysis(null)
    setError(null)
    fetchJson<Analysis>(`/api/tracks/${encodeURIComponent(selected.id)}/analysis`)
      .then((data) => {
        console.log('analysis:', data)
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
  }, [selected])

  return (
    <main>
      <h1>Choreo</h1>
      {error && <p className="error">{error}</p>}
      <ul>
        {tracks.map((track) => (
          <li key={track.id}>
            <button onClick={() => setSelected(track)}>
              {track.id === selected?.id ? '▶ ' : ''}
              {track.filename}
            </button>
          </li>
        ))}
      </ul>
      {loading && (
        <p>Analyzing {selected?.filename}… (first run on a new track takes minutes)</p>
      )}
      {analysis && (
        <section>
          <p>
            {analysis.filename} — {analysis.tempo ?? '?'} BPM, {analysis.beats.length}{' '}
            beats, {analysis.eight_counts.length} eight-counts (full JSON in console)
          </p>
          {gridError && <p className="error">Beat grid unavailable: {gridError}</p>}
          {onsetsError && <p className="error">Onset overlays unavailable: {onsetsError}</p>}
          <Timeline
            key={analysis.id}
            audioUrl={analysis.audio_url}
            grid={grid}
            onsets={onsets}
            onReady={setWavesurfer}
          />
        </section>
      )}
    </main>
  )
}

export default App
