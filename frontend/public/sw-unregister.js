// Kill switch for sw.js — NOT deployed by default. See README's Service
// Worker Kill Switch section for how to actually use this: it does nothing
// on its own until you deliberately serve it in place of sw.js.
//
// If the real service worker (sw.js) ever ships a bug that gets a client
// stuck (serving something wrong/stale in a way a normal reload can't fix,
// since the whole point of a SW is that it sits in front of normal
// reloads), the fix is not "ship a corrected sw.js" — the browser only
// checks a SW for updates on navigation, through the OLD, possibly-broken
// SW's own fetch handler, which may itself be the thing stopping the new
// script from ever being fetched. The fix is to replace sw.js's CONTENT
// with THIS file's content (same URL, same scope, so the browser's normal
// "bytes changed -> update" check still fires) — this file's only job is
// to unregister itself and delete every cache this app ever created, which
// runs regardless of what the previous (broken) SW was doing, because
// installing a new SW at the same URL always supersedes the old one.
//
// After deploying this, browsers that visit the app go through ONE
// installation of this worker (which immediately removes itself), and
// every subsequent load has no SW at all — back to plain, uncached
// requests, same as the app behaved before sw.js ever existed. Once
// confirmed clean, revert to shipping the real sw.js again; this file
// does not need to be removed from the repo between incidents.

self.addEventListener('install', () => {
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys()
      await Promise.all(keys.map((key) => caches.delete(key)))
      await self.registration.unregister()
      // Force every open tab under this scope to reload once the
      // unregister above completes, so a client that never gets a manual
      // reload still recovers on its own.
      const clientsList = await self.clients.matchAll({ type: 'window' })
      for (const client of clientsList) {
        client.navigate(client.url)
      }
    })(),
  )
})
