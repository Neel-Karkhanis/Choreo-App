import { openDB, type DBSchema, type IDBPDatabase } from 'idb'
import type { GridData } from './types'

// The one IndexedDB database in the app, and the one place that talks to it
// directly — everything else (device.ts, manualGrid.ts, stemEngine.ts) goes
// through the functions below, never `idb`/`indexedDB` itself.
//
// Three stores, three different reliability contracts:
//
//   meta  — the device id mirror. A fallback for when Safari's ITP evicts
//           the signed cookie (see device.ts); the server is still what
//           issues and validates the real id, this is just where a
//           returning browser finds its own copy to hand back.
//   grids — a cache of the tapped grid, server-authoritative (manualGrid.ts
//           already fetches fresh from the server on every load; this is
//           purely an offline/instant-paint fallback, never the source of
//           truth).
//   stems — separated-audio Opus blobs. The ONLY store that is not a mirror
//           of anything durable: stems are evictable derived data even on
//           the SERVER (see server.py's cache/ tree), so losing this store
//           just means re-requesting separation, not losing anything.
//
// iOS IndexedDB has known transaction-failure and data-loss bugs (WebKit
// bugs #197050, #226547, among others) well beyond ordinary quota errors —
// so every exported function here catches everything, not just the errors
// you'd expect, and resolves to a "nothing here" value instead of
// rejecting. Every caller can therefore treat "IndexedDB is broken on this
// device" identically to "there's nothing cached yet": fall back to the
// server, or to a stem re-request. The one exception is setStem's
// QuotaExceededError, which the caller needs to know about specifically to
// show a real message (see stemEngine.ts) rather than silently dropping a
// stem that will just 404 from cache next time.

const DB_NAME = 'choreo'
const DB_VERSION = 1
const DEVICE_ID_KEY = 'deviceId'

interface ChoreoDB extends DBSchema {
  meta: {
    key: string
    value: { key: string; value: string }
  }
  grids: {
    key: string // md5
    value: {
      md5: string
      beats: number[]
      downbeatIndices: number[]
      eightCountIndices: number[]
      updatedAt: number
    }
  }
  stems: {
    key: string // `${md5}:${stemName}`
    value: {
      key: string
      blob: Blob
      updatedAt: number
    }
  }
}

let dbPromise: Promise<IDBPDatabase<ChoreoDB>> | null = null

function getDb(): Promise<IDBPDatabase<ChoreoDB>> {
  if (!dbPromise) {
    dbPromise = openDB<ChoreoDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta', { keyPath: 'key' })
        if (!db.objectStoreNames.contains('grids')) db.createObjectStore('grids', { keyPath: 'md5' })
        if (!db.objectStoreNames.contains('stems')) db.createObjectStore('stems', { keyPath: 'key' })
      },
    })
  }
  return dbPromise
}

const stemKey = (md5: string, stemName: string) => `${md5}:${stemName}`

// ---- device id mirror ------------------------------------------------

export async function getDeviceId(): Promise<string | null> {
  try {
    const db = await getDb()
    const row = await db.get('meta', DEVICE_ID_KEY)
    return row?.value ?? null
  } catch (err) {
    console.warn('[localDb] getDeviceId failed, treating as no mirror', err)
    return null
  }
}

export async function setDeviceId(id: string): Promise<void> {
  try {
    const db = await getDb()
    await db.put('meta', { key: DEVICE_ID_KEY, value: id })
  } catch (err) {
    // Best-effort: the cookie (server-set, HttpOnly) is still the durable
    // copy for this session. Losing the mirror only matters the NEXT time
    // the cookie itself is gone too.
    console.warn('[localDb] setDeviceId failed, mirror not updated', err)
  }
}

// ---- tapped grid cache -------------------------------------------------

export interface CachedGrid {
  grid: GridData
  updatedAt: number
}

export async function getGrid(md5: string): Promise<CachedGrid | null> {
  try {
    const db = await getDb()
    const row = await db.get('grids', md5)
    if (!row) return null
    return {
      grid: {
        beats: row.beats,
        downbeatIndices: row.downbeatIndices,
        eightCountIndices: row.eightCountIndices,
      },
      updatedAt: row.updatedAt,
    }
  } catch (err) {
    console.warn('[localDb] getGrid failed, falling back to server', err)
    return null
  }
}

export async function setGrid(md5: string, grid: GridData, updatedAt: number): Promise<void> {
  try {
    const db = await getDb()
    await db.put('grids', {
      md5,
      beats: grid.beats,
      downbeatIndices: grid.downbeatIndices,
      eightCountIndices: grid.eightCountIndices,
      updatedAt,
    })
  } catch (err) {
    // The server write (manualGrid.ts's PUT) is what actually persists a
    // tap. A failed mirror write just means the next offline load re-fetches
    // from the server instead of painting instantly from cache.
    console.warn('[localDb] setGrid failed, offline cache not updated', err)
  }
}

// Clears a mirrored grid — used when the server authoritatively reports
// "no grid" (e.g. a delete/retap on another device), so a stale mirror entry
// can't outlive the thing it was caching and get served as a false offline
// fallback later.
export async function deleteGrid(md5: string): Promise<void> {
  try {
    const db = await getDb()
    await db.delete('grids', md5)
  } catch (err) {
    console.warn('[localDb] deleteGrid failed', err)
  }
}

// ---- stem blob cache -----------------------------------------------------

export async function getStem(md5: string, stemName: string): Promise<Blob | null> {
  try {
    const db = await getDb()
    const row = await db.get('stems', stemKey(md5, stemName))
    return row?.blob ?? null
  } catch (err) {
    console.warn(`[localDb] getStem(${stemName}) failed, will re-request from server`, err)
    return null
  }
}

export type SetStemResult =
  | { ok: true }
  // Distinguished from a generic failure: the caller needs to show a real
  // message for this one, not just log and move on — see stemEngine.ts.
  | { ok: false; reason: 'quota' | 'error'; error: unknown }

export async function setStem(md5: string, stemName: string, blob: Blob): Promise<SetStemResult> {
  try {
    const db = await getDb()
    await db.put('stems', { key: stemKey(md5, stemName), blob, updatedAt: Date.now() })
    return { ok: true }
  } catch (err) {
    const isQuota =
      (err instanceof DOMException && err.name === 'QuotaExceededError') ||
      (err instanceof Error && err.name === 'QuotaExceededError')
    if (isQuota) {
      console.warn(`[localDb] setStem(${stemName}) hit storage quota`, err)
      return { ok: false, reason: 'quota', error: err }
    }
    console.warn(`[localDb] setStem(${stemName}) failed`, err)
    return { ok: false, reason: 'error', error: err }
  }
}
