import { useState } from 'react'
import { apiFetch } from './api'
import Logo from './Logo'
import { API_BASE } from './snap'

// LEVEL 0, ahead of Library in App's view stack: the one screen shown to
// anyone without a live session, and the only way into the app besides an
// already-valid cookie. No design mock covers accounts at all — the
// redesign predates multi-user hosting — so, like Settings and the row
// menu's icons, this screen's layout is invented, kept deliberately plain
// to match the rest of the app's placeholder-styling-pending-a-design-pass
// look (see SettingsScreen's own comment).
//
// No forgot-password flow: there is no email-sending infra in this app, so
// signup and login are deliberately the only two doors in.
type Mode = 'login' | 'signup'

export default function Auth({ onSignedIn }: { onSignedIn: () => void }) {
  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [tosAccepted, setTosAccepted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const switchMode = (next: Mode) => {
    setMode(next)
    setError(null)
  }

  const submit = async () => {
    if (mode === 'signup' && !tosAccepted) {
      setError('You must accept the Terms of Use to create an account.')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const path = mode === 'login' ? '/auth/login' : '/auth/signup'
      const body =
        mode === 'login' ? { email, password } : { email, password, tos_accepted: tosAccepted }
      // Deliberately the bare apiFetch, not a route past App's own
      // unauthorized handler: a failed login is a normal 401 this form
      // already displays inline, not a session that just expired.
      const res = await apiFetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail ?? `${mode} -> HTTP ${res.status}`)
      }
      onSignedIn()
    } catch (err) {
      setError(String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="settings">
      <header className="library-head">
        <div className="library-brand">
          <Logo />
          <h1 className="library-wordmark">horeo</h1>
        </div>
      </header>

      <section className="settings-section">
        <h2 className="settings-section-title">
          {mode === 'login' ? 'Sign in' : 'Create an account'}
        </h2>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            void submit()
          }}
          style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxWidth: '22rem' }}
        >
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            Email
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            Password
            <input
              type="password"
              required
              minLength={8}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {mode === 'signup' && (
            <label style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <input
                type="checkbox"
                checked={tosAccepted}
                onChange={(e) => setTosAccepted(e.target.checked)}
              />
              <span>
                I own the rights to what I upload, or have permission to use it, and I accept the{' '}
                <a href="/tos.html" target="_blank" rel="noreferrer">
                  Terms of Use
                </a>
                .
              </span>
            </label>
          )}

          {error && <p className="error">{error}</p>}

          <button type="submit" disabled={submitting}>
            {submitting ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        <p className="settings-section-note">
          {mode === 'login' ? (
            <>
              Don&apos;t have an account?{' '}
              <button type="button" onClick={() => switchMode('signup')}>
                Create one
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button type="button" onClick={() => switchMode('login')}>
                Sign in
              </button>
            </>
          )}
        </p>
      </section>
    </main>
  )
}
