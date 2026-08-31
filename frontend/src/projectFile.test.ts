import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  buildProjectExport,
  checkDurationMismatch,
  importProjectFile,
  parseProjectFile,
  PROJECT_SCHEMA_VERSION,
} from './projectFile'
import type { LibrarySong } from './types'

const SONG: LibrarySong = {
  md5: 'a'.repeat(32),
  id: 'My Song',
  filename: 'My Song.mp3',
  duration: 187.36,
  media_kind: 'audio',
  state: 'ready',
  grid_present: true,
  stems_present: true,
  original_present: true,
}

const GRID_WIRE = {
  beats: [0.5, 1.1, 1.7, 2.3],
  downbeats: [0, 2],
  eight_counts: [0],
}

const jsonResponse = (body: unknown, ok = true) =>
  ({ ok, status: ok ? 200 : 400, json: () => Promise.resolve(body) }) as Response

describe('checkDurationMismatch', () => {
  it('warns when durations disagree beyond tolerance', () => {
    const warning = checkDurationMismatch(180, 240)
    expect(warning).not.toBeNull()
    expect(warning).toMatch(/240\.0s/)
    expect(warning).toMatch(/180\.0s/)
  })

  it('does not warn for durations within tolerance (encoder rounding)', () => {
    expect(checkDurationMismatch(180.0, 180.2)).toBeNull()
  })

  it('does not warn when there is nothing to compare against (no existing song)', () => {
    expect(checkDurationMismatch(undefined, 240)).toBeNull()
    expect(checkDurationMismatch(null, 240)).toBeNull()
  })

  it('does not warn when the import itself carries no duration', () => {
    expect(checkDurationMismatch(180, null)).toBeNull()
  })

  it('respects a custom tolerance', () => {
    expect(checkDurationMismatch(180, 181, 2)).toBeNull()
    expect(checkDurationMismatch(180, 183, 2)).not.toBeNull()
  })
})

describe('parseProjectFile', () => {
  it('parses a well-formed file', () => {
    const text = JSON.stringify({
      schema_version: PROJECT_SCHEMA_VERSION,
      exported_at: '2026-01-01T00:00:00.000Z',
      track: { md5: SONG.md5, filename: SONG.filename, media_kind: 'audio', duration: 187.36 },
      manual_grid: { beats: [1], downbeatIndices: [0], eightCountIndices: [0] },
    })
    const parsed = parseProjectFile(text)
    expect(parsed.track.md5).toBe(SONG.md5)
    expect(parsed.manual_grid?.beats).toEqual([1])
  })

  it('rejects invalid JSON', () => {
    expect(() => parseProjectFile('not json{{{')).toThrow(/valid JSON/)
  })

  it('rejects a wrong schema_version', () => {
    const text = JSON.stringify({ schema_version: 999, track: { md5: SONG.md5 } })
    expect(() => parseProjectFile(text)).toThrow(/version/)
  })

  it('rejects a file with no track info', () => {
    const text = JSON.stringify({ schema_version: PROJECT_SCHEMA_VERSION })
    expect(() => parseProjectFile(text)).toThrow(/track/)
  })

  it('rejects a non-object top level', () => {
    expect(() => parseProjectFile('42')).toThrow()
    expect(() => parseProjectFile('null')).toThrow()
  })
})

describe('buildProjectExport / importProjectFile round trip', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('excludes owner_id and any audio, and round-trips the grid losslessly', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ manual_grid: GRID_WIRE }))
    vi.stubGlobal('fetch', fetchMock)

    const exported = await buildProjectExport(SONG)

    // The whole point of the export shape: no owner_id, no audio bytes,
    // ever — only metadata + grid.
    expect(JSON.stringify(exported)).not.toMatch(/owner_id/i)
    expect(exported.track.md5).toBe(SONG.md5)
    expect(exported.track.filename).toBe(SONG.filename)
    expect(exported.manual_grid).toEqual({
      beats: GRID_WIRE.beats,
      downbeatIndices: GRID_WIRE.downbeats,
      eightCountIndices: GRID_WIRE.eight_counts,
    })

    // Round-trip through parseProjectFile (as if it had been written to
    // disk and read back from a file input) and then importProjectFile.
    const reparsed = parseProjectFile(JSON.stringify(exported))
    const importFetch = vi.fn().mockResolvedValue(jsonResponse({ md5: SONG.md5 }))
    vi.stubGlobal('fetch', importFetch)

    await importProjectFile(reparsed)

    const [, init] = importFetch.mock.calls[0] as [string, RequestInit]
    const sentBody = JSON.parse(init.body as string)
    expect(sentBody.manual_grid).toEqual(GRID_WIRE) // exactly the wire shape the server expects
    expect(sentBody.track.md5).toBe(SONG.md5)
    expect(JSON.stringify(sentBody)).not.toMatch(/owner_id/i)
  })

  it('exports manual_grid: null for a song with no tapped grid', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ manual_grid: null })))
    const exported = await buildProjectExport(SONG)
    expect(exported.manual_grid).toBeNull()
  })

  it('throws with the server detail on a failed import', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'Unsupported project schema_version' }, false)))
    const file = parseProjectFile(
      JSON.stringify({ schema_version: PROJECT_SCHEMA_VERSION, track: { md5: SONG.md5 }, manual_grid: null }),
    )
    await expect(importProjectFile(file)).rejects.toThrow(/Unsupported project schema_version/)
  })
})
