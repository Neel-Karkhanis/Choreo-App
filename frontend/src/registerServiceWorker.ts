// Registers public/sw.js — see that file's own header for what it does and
// (deliberately) doesn't cache. A missing serviceWorker API (older Safari,
// some in-app browsers) or a rejected registration just means no SW and no
// installability boost; the app itself doesn't depend on one being present.
//
// Skipped outside a production build: `vite dev` serves unbundled modules
// that change on every save, and a SW sitting in front of that — even one as
// deliberately network-first as sw.js — is a well-known source of "why is
// my editor change not showing up" confusion during local development, for
// zero installability benefit (nobody installs a dev server as a PWA).
export function registerServiceWorker(): void {
  if (!import.meta.env.PROD) return
  if (!('serviceWorker' in navigator)) return
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      console.warn('[sw] registration failed', err)
    })
  })
}
