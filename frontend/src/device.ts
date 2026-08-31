import { apiFetch } from './api'
import * as localDb from './localDb'
import { API_BASE } from './snap'

// Anonymous, device-scoped identity — the replacement for accounts. The
// server issues a signed, HttpOnly UUIDv4 cookie the browser can neither
// read nor forge (see src/api/identity.py); this is the one call that asks
// for one. Every other /api/* call then rides on that cookie automatically
// (same-origin, see api.ts's own comment on credentials), so nothing else
// in the app needs to know the id exists.
//
// RECONCILIATION, COOKIE WINS: called once by App on every startup, not
// just a real first visit. This module's only job is to hand the server
// whatever id IndexedDB has mirrored (if any) and store back whatever the
// server says is actually live — the comparison between "what the cookie
// says" and "what the mirror says" happens entirely server-side (see
// identity.issue_device), because the cookie is HttpOnly and this code can
// never read its value to compare locally. Concretely:
//
//   - Cookie present and valid: the server ignores the mirrored id
//     entirely and reaffirms the cookie's own id. If that differs from
//     what we sent (a stale mirror), we overwrite the mirror with the
//     server's answer below — the cookie always wins.
//   - Cookie missing/invalid (Safari's ITP evicted it) but the mirror has
//     a valid id: the server re-cookies that id. This is the actual
//     recovery path the mirror exists for.
//   - Neither: the server mints a fresh id. This is a real first visit,
//     or a device whose mirror was ALSO lost (e.g. Safari's eviction
//     takes the origin's entire storage at once, cookie and IndexedDB
//     together, for anything short of an installed PWA) — an
//     unavoidable, acknowledged data-loss case (see InstallPrompt.tsx for
//     the mitigation: installed home-screen apps are exempt from that
//     eviction).
//
// Every call returns a real answer — the local mirror read is wrapped in
// localDb.ts's own error handling and degrades to "no mirror" on failure,
// same as any other IndexedDB read in this app.
export async function ensureDevice(): Promise<void> {
  const mirrored = await localDb.getDeviceId()

  const res = await apiFetch(`${API_BASE}/device`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_id: mirrored ?? undefined }),
  })
  if (!res.ok) {
    throw new Error(`Failed to establish a device identity -> HTTP ${res.status}`)
  }
  const data = (await res.json()) as { device_id: string }
  // Always write back, even when it matches what we sent: this is what
  // corrects a stale mirror to the cookie's real id when the two differ.
  await localDb.setDeviceId(data.device_id)
}
