import { apiFetch } from './api'
import { API_BASE } from './snap'
import type { GridData, LibrarySong, MediaKind } from './types'

// Phase 4: no account means no password-reset-style recovery — export/import
// is the mitigation. A project file is one song's md5-identified metadata
// plus its tapped grid, nothing else:
//
//   - No stem audio, and no source track audio either — only the source
//     FILENAME is recorded. Stems already have their own local, evictable
//     cache (localDb.ts); an export duplicating them would just be a second
//     copy that goes stale. Re-importing the actual file through the
//     ordinary "Import song" flow (POST /api/import) is what reconnects
//     the audio — identical bytes hash to the same md5 they always did,
//     which is what lets that re-import pick this project's grid back up.
//   - No owner_id, ever. It never appears in a ProjectFile, is never read
//     from one on import, and the server-side counterpart (server.py's
//     import_project) never reads it from the body either — see
//     identity.py's BEARER CREDENTIAL note on why that field must never
//     leave the cookie/header/device-mirror it already lives in.
//
// schema_version here IS the backend's ANALYSIS_SCHEMA_VERSION — the same
// version that already governs the grid's own downbeats/eight_counts-as-
// indices shape (schema v4) — not a second export-format version to keep
// in sync by hand. A project file is a schema v4 grid plus just enough
// track metadata to re-anchor it; there is nothing else versioned here.
export const PROJECT_SCHEMA_VERSION = 4

export interface ProjectFile {
  schema_version: number
  exported_at: string
  track: {
    md5: string
    filename: string | null
    media_kind: MediaKind
    duration: number | null
  }
  manual_grid: GridData | null
}

interface ManualGridPayload {
  beats: number[]
  downbeats: number[]
  eight_counts: number[]
}

const toGrid = (p: ManualGridPayload): GridData => ({
  beats: p.beats,
  downbeatIndices: p.downbeats,
  eightCountIndices: p.eight_counts,
})

const toPayload = (g: GridData) => ({
  beats: g.beats,
  downbeats: g.downbeatIndices,
  eight_counts: g.eightCountIndices,
})

/**
 * Builds one song's exportable project state. Reads the grid directly by
 * md5 (server.py's GET /api/library/{md5}/manual-grid) rather than through
 * the track_id-based route manualGrid.ts uses — that one needs a live
 * source file to resolve a track_id from; this works the same for a
 * stems_evicted row with no source file at all, which is exactly the row
 * a user is most likely to want to export before it's lost for good.
 */
export async function buildProjectExport(song: LibrarySong): Promise<ProjectFile> {
  const res = await apiFetch(`${API_BASE}/library/${encodeURIComponent(song.md5)}/manual-grid`)
  if (!res.ok) throw new Error(`GET manual-grid -> HTTP ${res.status}`)
  const data = (await res.json()) as { manual_grid: ManualGridPayload | null }
  return {
    schema_version: PROJECT_SCHEMA_VERSION,
    exported_at: new Date().toISOString(),
    track: {
      md5: song.md5,
      filename: song.filename,
      media_kind: song.media_kind,
      duration: song.duration,
    },
    manual_grid: data.manual_grid ? toGrid(data.manual_grid) : null,
  }
}

/**
 * Triggers a browser download of an already-built export — a plain Blob +
 * object URL + a detached `<a download>`, the ordinary way a page saves a
 * file it built client-side (this runs in the real deployed app, never
 * inside a sandboxed preview that would block it).
 */
export function downloadProjectFile(file: ProjectFile): void {
  const stem = (file.track.filename ?? file.track.md5).replace(/\.[^./]+$/, '')
  const blob = new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${stem}.choreo-project.json`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/**
 * Parses and validates a project file's raw text — untrusted, it came from
 * a file picker — into a ProjectFile, or throws a descriptive Error.
 * Deliberately strict about schema_version: an older or newer export must
 * fail loudly here, in one clear place, rather than reaching the server
 * and getting rejected by a validator (ManualGrid's shape check) that was
 * never meant to double as this file's format check.
 */
export function parseProjectFile(text: string): ProjectFile {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error('Not a valid JSON file')
  }
  if (typeof data !== 'object' || data === null) {
    throw new Error('Not a Choreo project file')
  }
  const obj = data as Record<string, unknown>
  if (obj.schema_version !== PROJECT_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported project version ${JSON.stringify(obj.schema_version)} ` +
        `(this app understands version ${PROJECT_SCHEMA_VERSION})`,
    )
  }
  const track = obj.track as Record<string, unknown> | undefined
  if (!track || typeof track.md5 !== 'string' || track.md5.length === 0) {
    throw new Error('Project file is missing its track information')
  }
  return obj as unknown as ProjectFile
}

/**
 * DURATION MISMATCH WARNING — why this exists at all:
 *
 * Schema v4's grid stores `beats` (timestamps) plus `downbeatIndices`/
 * `eightCountIndices` — positions INTO `beats`, not timestamps themselves.
 * That grid is only meaningful against audio whose beat structure it was
 * actually tapped against. Import audio of a different length (a
 * re-encode, a trimmed re-upload, or simply the wrong song sharing a
 * filename) under the same md5-adjacent flow, and the indices are all
 * still perfectly valid array positions — nothing throws, nothing 404s,
 * nothing looks wrong at the API layer. The 8-counts just silently point
 * at the wrong moments in the music. There is no way to detect this from
 * the grid's own shape; duration is the only cheap signal available, and
 * even that isn't proof (a legitimate re-encode can shift duration by a
 * few ms without actually invalidating the grid) — so this warns, it
 * never blocks. See Library.tsx for where the warning actually surfaces.
 */
export function checkDurationMismatch(
  existingDuration: number | null | undefined,
  importedDuration: number | null,
  toleranceSeconds = 0.5,
): string | null {
  if (existingDuration == null || importedDuration == null) return null
  if (Math.abs(existingDuration - importedDuration) <= toleranceSeconds) return null
  return (
    `This project was exported from a ${importedDuration.toFixed(1)}s track, but the version already ` +
    `on this device is ${existingDuration.toFixed(1)}s. The imported grid's beat positions may not ` +
    `line up with this audio.`
  )
}

/**
 * Sends a parsed, already-confirmed project file to the server. Creates or
 * refreshes the library entry for this md5 and, if the file carries one,
 * REPLACES the tapped grid wholesale — same semantics as a normal save,
 * no merge. Callers are expected to confirm with the user before calling
 * this (see Library.tsx) since it can silently overwrite an existing grid;
 * the server itself does not ask.
 */
export async function importProjectFile(file: ProjectFile): Promise<{ md5: string }> {
  const res = await apiFetch(`${API_BASE}/library/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema_version: file.schema_version,
      track: file.track,
      manual_grid: file.manual_grid ? toPayload(file.manual_grid) : null,
    }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ?? `POST import -> HTTP ${res.status}`)
  }
  return res.json() as Promise<{ md5: string }>
}
