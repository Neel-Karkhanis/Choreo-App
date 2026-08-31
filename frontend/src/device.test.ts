import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The actual cookie-wins DECISION happens entirely server-side (see
// src/api/identity.py's issue_device and tests/test_identity.py) — this
// module can't read the HttpOnly cookie to make that call itself. What
// belongs here is the orchestration: does ensureDevice send whatever the
// IndexedDB mirror holds, and does it write back whatever the server
// answers with, even when that differs from what was sent (the case where
// the mirror was stale and the cookie won). Mocking localDb rather than
// pulling in a real IndexedDB environment (this project's vitest config has
// none) keeps this test aimed at that orchestration, not at IndexedDB itself
// — localDb.ts's own contract (never throws, degrades to null) is simple
// enough to take as given here.
vi.mock('./localDb', () => ({
  getDeviceId: vi.fn(),
  setDeviceId: vi.fn(),
}))

import { ensureDevice } from './device'
import * as localDb from './localDb'

const jsonResponse = (body: unknown, ok = true) =>
  ({
    ok,
    status: ok ? 200 : 400,
    json: () => Promise.resolve(body),
  }) as Response

describe('ensureDevice', () => {
  beforeEach(() => {
    // clearAllMocks (not restoreAllMocks): these are vi.fn()s created by the
    // vi.mock('./localDb', ...) factory above, not spies on a real
    // implementation, so there is nothing to "restore" — only call history
    // to reset between tests.
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the mirrored id when one exists, and stores back whatever the server answers', async () => {
    vi.mocked(localDb.getDeviceId).mockResolvedValue('11111111-1111-4111-8111-111111111111')
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ device_id: '11111111-1111-4111-8111-111111111111' }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await ensureDevice()

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(JSON.parse(init.body as string)).toEqual({
      device_id: '11111111-1111-4111-8111-111111111111',
    })
    expect(localDb.setDeviceId).toHaveBeenCalledWith('11111111-1111-4111-8111-111111111111')
  })

  it('omits device_id on a real first visit (no mirror yet)', async () => {
    vi.mocked(localDb.getDeviceId).mockResolvedValue(null)
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ device_id: 'freshly-minted' }))
    vi.stubGlobal('fetch', fetchMock)

    await ensureDevice()

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(JSON.parse(init.body as string)).toEqual({ device_id: undefined })
    expect(localDb.setDeviceId).toHaveBeenCalledWith('freshly-minted')
  })

  it("overwrites a stale mirror with the server's answer when the cookie won", async () => {
    // The mirror thinks it's device B; the server's live cookie says A —
    // this is exactly the "cookie wins, differs from the mirror" case.
    vi.mocked(localDb.getDeviceId).mockResolvedValue('22222222-2222-4222-8222-222222222222')
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ device_id: '11111111-1111-4111-8111-111111111111' }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await ensureDevice()

    // The mirror is corrected to the cookie's id, not left at what was sent.
    expect(localDb.setDeviceId).toHaveBeenCalledWith('11111111-1111-4111-8111-111111111111')
    expect(localDb.setDeviceId).not.toHaveBeenCalledWith('22222222-2222-4222-8222-222222222222')
  })

  it('throws on a non-ok response and never touches the mirror', async () => {
    vi.mocked(localDb.getDeviceId).mockResolvedValue(null)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({}, false)))

    await expect(ensureDevice()).rejects.toThrow(/HTTP 400/)
    expect(localDb.setDeviceId).not.toHaveBeenCalled()
  })
})
