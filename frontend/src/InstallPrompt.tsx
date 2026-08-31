import { useEffect, useState } from 'react'

// PWA install nudge — a correctness feature, not polish: an installed
// home-screen app is exempt from Safari's 7-day script-writable-storage
// eviction (see localDb.ts's header and device.ts), so this is what keeps a
// device's tapped grids and cached stems alive across a couple weeks of not
// opening the app, rather than relying on the signed cookie alone.
//
// No design mock covers this at all (same as Settings/the row menu's icons —
// see SettingsScreen's own comment), so the banner is deliberately plain:
// one line of text, one action, one dismiss.
//
// Two real install paths, handled differently because the platforms don't
// offer the same API:
//   - Chrome/Edge/Android fire `beforeinstallprompt`, which this listens for
//     and can trigger programmatically via its own .prompt().
//   - iOS Safari has no such event at all — installation is a manual Share
//     sheet action, so the banner there is instructions, not a button.
// Everywhere else (a desktop browser with no install affordance, or a
// browser that already dispatched neither event by the time this renders)
// shows nothing rather than a dead-end banner.

const DISMISSED_KEY = 'choreo-install-dismissed'

function isStandalone(): boolean {
  try {
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      // iOS Safari's own flag — it never fires `display-mode: standalone`
      // the same way Chrome does.
      (navigator as unknown as { standalone?: boolean }).standalone === true
    )
  } catch {
    return false
  }
}

function isIOS(): boolean {
  return /iphone|ipad|ipod/i.test(navigator.userAgent)
}

function readDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISSED_KEY) === 'true'
  } catch {
    return false
  }
}

function persistDismissed(): void {
  try {
    localStorage.setItem(DISMISSED_KEY, 'true')
  } catch {
    // Best effort, same as theme.ts/accentColor.ts's own storage writes —
    // the banner just re-shows next session instead of staying dismissed.
  }
}

// Not in lib.dom.d.ts (a Chrome-only, non-standard event) — minimal shape
// this component actually reads.
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

export default function InstallPrompt() {
  const [dismissed, setDismissed] = useState(readDismissed)
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null)
  const [standalone, setStandalone] = useState(isStandalone)

  useEffect(() => {
    const onBeforeInstallPrompt = (e: Event) => {
      // Chrome's default mini-infobar is suppressed; this banner is the
      // only install affordance so there's exactly one, not two competing
      // prompts.
      e.preventDefault()
      setDeferredPrompt(e as BeforeInstallPromptEvent)
    }
    const onInstalled = () => {
      setStandalone(true)
      setDeferredPrompt(null)
    }
    window.addEventListener('beforeinstallprompt', onBeforeInstallPrompt)
    window.addEventListener('appinstalled', onInstalled)
    return () => {
      window.removeEventListener('beforeinstallprompt', onBeforeInstallPrompt)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [])

  if (standalone || dismissed) return null

  const dismiss = () => {
    persistDismissed()
    setDismissed(true)
  }

  if (deferredPrompt) {
    return (
      <div className="install-prompt" role="note">
        <span>Install Choreo so your tapped grids survive even when you don't open it for a while.</span>
        <button
          type="button"
          onClick={() => {
            void deferredPrompt.prompt()
            void deferredPrompt.userChoice.then(() => setDeferredPrompt(null))
          }}
        >
          Install
        </button>
        <button type="button" className="install-prompt-dismiss" onClick={dismiss} aria-label="Dismiss">
          ×
        </button>
      </div>
    )
  }

  if (isIOS()) {
    return (
      <div className="install-prompt" role="note">
        <span>
          Add Choreo to your Home Screen so your tapped grids survive even when you don't open it for
          a while: tap <strong>Share</strong>, then <strong>Add to Home Screen</strong>.
        </span>
        <button type="button" className="install-prompt-dismiss" onClick={dismiss} aria-label="Dismiss">
          ×
        </button>
      </div>
    )
  }

  return null
}
