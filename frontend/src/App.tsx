import { useEffect, useMemo, useState } from 'react'
import type WaveSurfer from 'wavesurfer.js'
import Timeline, { type GridData } from './Timeline'

interface Track {
  id: string
  filename: string
}

interface Stem {
  name: string
  url: string
}

// Mirrors the analysis JSON contract (schema_version 2): downbeats and
// eight_counts are integer indices into beats, not timestamps.
interface Analysis {
  id: string
  filename: string
  schema_version: number
  duration: number
  tempo: number | null
  beats: number[]
  downbeats: number[]
  eight_counts: number[]
  audio_url: string
  stems: Stem[]
}

// Grid data must match the schema v2 contract exactly; on any mismatch we
// render an error instead of the grid rather than inferring a shape.
function gridFromAnalysis(a: Analysis): { grid?: GridData; gridError?: string } {
  const isNumbers = (xs: unknown): xs is number[] =>
    Array.isArray(xs) && xs.every((x) => typeof x === 'number' && Number.isFinite(x))
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
          <Timeline
            key={analysis.id}
            audioUrl={analysis.audio_url}
            grid={grid}
            onReady={setWavesurfer}
          />
        </section>
      )}
    </main>
  )
}

export default App
