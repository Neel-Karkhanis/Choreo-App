import { beforeEach, describe, expect, it, vi } from 'vitest'

// Covers exactly the thing a "reads from cache, falls through on miss"
// description leaves out: whether a miss actually WRITES the fetched bytes
// back to the cache. Without that write, the cache is permanently empty and
// the QuotaExceededError path below is unreachable — this file exists
// because that gap was real once (a correct write path, but no test that
// would have caught it going missing).
//
// Mocks localDb (no real IndexedDB in this test environment, same reasoning
// as device.test.ts) and fetch directly, and calls fetchStemBytes rather
// than loadStems so no Web Audio decode is involved.
vi.mock('./localDb', () => ({
  getStem: vi.fn(),
  setStem: vi.fn(),
}))

import { fetchStemBytes } from './stemEngine'
import * as localDb from './localDb'

const MD5 = 'deadbeefdeadbeefdeadbeefdeadbeef'
const TRACK_ID = 'my-track'

function blobResponse(bytes: Uint8Array, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 404,
    blob: () => Promise.resolve(new Blob([bytes as BlobPart])),
  } as unknown as Response
}

describe('fetchStemBytes', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('writes the fetched blob to the cache on a miss, for a cacheable stem', async () => {
    vi.mocked(localDb.getStem).mockResolvedValue(null)
    vi.mocked(localDb.setStem).mockResolvedValue({ ok: true })
    const fetchMock = vi.fn().mockResolvedValue(blobResponse(new Uint8Array([1, 2, 3])))
    vi.stubGlobal('fetch', fetchMock)

    await fetchStemBytes(TRACK_ID, MD5, 'drums', undefined, undefined)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(localDb.setStem).toHaveBeenCalledTimes(1)
    const [calledMd5, calledName, calledBlob] = vi.mocked(localDb.setStem).mock.calls[0]
    expect(calledMd5).toBe(MD5)
    expect(calledName).toBe('drums')
    expect(calledBlob).toBeInstanceOf(Blob)

    vi.unstubAllGlobals()
  })

  it('never writes (or fetches through the cache path at all) for `original`', async () => {
    vi.mocked(localDb.getStem).mockResolvedValue(null)
    const fetchMock = vi.fn().mockResolvedValue(blobResponse(new Uint8Array([1])))
    vi.stubGlobal('fetch', fetchMock)

    await fetchStemBytes(TRACK_ID, MD5, 'original', undefined, undefined)

    expect(localDb.getStem).not.toHaveBeenCalled()
    expect(localDb.setStem).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1) // still fetched — just never cached

    vi.unstubAllGlobals()
  })

  it('skips the network entirely on a cache hit', async () => {
    vi.mocked(localDb.getStem).mockResolvedValue(new Blob([new Uint8Array([9, 9]) as BlobPart]))
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await fetchStemBytes(TRACK_ID, MD5, 'vocals', undefined, undefined)

    expect(fetchMock).not.toHaveBeenCalled()
    expect(localDb.setStem).not.toHaveBeenCalled()

    vi.unstubAllGlobals()
  })

  it('surfaces a real message on QuotaExceededError, without throwing', async () => {
    vi.mocked(localDb.getStem).mockResolvedValue(null)
    vi.mocked(localDb.setStem).mockResolvedValue({
      ok: false,
      reason: 'quota',
      error: new DOMException('quota', 'QuotaExceededError'),
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(blobResponse(new Uint8Array([1]))))

    const onCacheWarning = vi.fn()
    const bytes = await fetchStemBytes(TRACK_ID, MD5, 'bass', undefined, onCacheWarning)

    expect(bytes).toBeInstanceOf(ArrayBuffer) // the fetch itself still succeeded
    expect(onCacheWarning).toHaveBeenCalledTimes(1)
    expect(onCacheWarning.mock.calls[0][0]).toMatch(/bass/i)

    vi.unstubAllGlobals()
  })

  it('does not call onCacheWarning for a non-quota cache error', async () => {
    vi.mocked(localDb.getStem).mockResolvedValue(null)
    vi.mocked(localDb.setStem).mockResolvedValue({ ok: false, reason: 'error', error: new Error('x') })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(blobResponse(new Uint8Array([1]))))

    const onCacheWarning = vi.fn()
    await fetchStemBytes(TRACK_ID, MD5, 'other', undefined, onCacheWarning)

    expect(onCacheWarning).not.toHaveBeenCalled()

    vi.unstubAllGlobals()
  })

  it('skips the cache entirely when md5 is undefined', async () => {
    const fetchMock = vi.fn().mockResolvedValue(blobResponse(new Uint8Array([1])))
    vi.stubGlobal('fetch', fetchMock)

    await fetchStemBytes(TRACK_ID, undefined, 'drums', undefined, undefined)

    expect(localDb.getStem).not.toHaveBeenCalled()
    expect(localDb.setStem).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)

    vi.unstubAllGlobals()
  })
})
