// Thin wrapper around fetch for every /api call in the app. There is no
// session to lapse and no sign-in screen to fall back to — every call site
// only needs `fetch(` swapped for `apiFetch(`, kept as its own function
// (rather than inlined) so a cross-cutting concern (retry, telemetry, a
// device-id recovery hook) has one place to land later without touching
// every call site again.
//
// No credentials option is set here on purpose: API_BASE is always a
// same-origin relative path (the Vite dev proxy in dev, Caddy in
// production — see snap.ts), so the browser's default fetch credentials
// mode ("same-origin") already sends the device-id cookie without this
// needing to opt in explicitly.
export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  return fetch(input, init)
}
