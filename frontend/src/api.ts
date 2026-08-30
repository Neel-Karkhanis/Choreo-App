// Thin wrapper around fetch for every /api call in the app. The one thing
// it adds over a bare fetch: on a 401 (no session cookie, or an expired
// one) it tells App to fall back to the sign-in screen, so a request made
// after a session lapses doesn't just fail silently in whichever screen
// happened to make it. Everything else — method, body, how a call site
// reads a non-2xx response — is unchanged from a bare fetch call, so every
// existing call site only needs `fetch(` swapped for `apiFetch(`.
//
// No credentials option is set here on purpose: API_BASE is always a
// same-origin relative path (the Vite dev proxy in dev, Caddy in
// production — see snap.ts), so the browser's default fetch credentials
// mode ("same-origin") already sends the session cookie without this
// needing to opt in explicitly.
type UnauthorizedListener = () => void

let onUnauthorized: UnauthorizedListener | null = null

export function setUnauthorizedHandler(handler: UnauthorizedListener | null) {
  onUnauthorized = handler
}

export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(input, init)
  if (res.status === 401) {
    onUnauthorized?.()
  }
  return res
}
