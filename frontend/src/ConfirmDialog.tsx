import { useEffect } from 'react'
import type { ReactNode } from 'react'

// A generic yes/no confirmation gate for actions that destroy state with no
// way back (currently just "Redo counts" in Song.tsx). Deliberately tiny and
// content-agnostic — every word is a prop — so a second destructive action
// elsewhere in the app can reuse this instead of hand-rolling its own overlay.

export interface ConfirmDialogProps {
  open: boolean
  title: string
  body: ReactNode
  cancelLabel?: string
  confirmLabel: string
  // The default/secondary action — also fired by Escape and a backdrop click,
  // so every dismissal path funnels through the one handler a caller must
  // treat as "nothing happened" (see Song.tsx's onCancel: it touches no state).
  onCancel: () => void
  // The destructive action. Only ever reachable via the confirm button itself
  // — never wired to Escape/backdrop — so an accidental dismissal can never
  // double as an accept.
  onConfirm: () => void
}

function ConfirmDialog({
  open,
  title,
  body,
  cancelLabel = 'Cancel',
  confirmLabel,
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  // Escape cancels — never confirms — matching the backdrop click below.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onCancel])

  if (!open) return null

  return (
    <div className="confirm-dialog-backdrop" onClick={onCancel}>
      <div
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        // Stops a click inside the card from bubbling to the backdrop's
        // cancel-everything handler above.
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-dialog-title">{title}</h2>
        <p>{body}</p>
        <div className="confirm-dialog-actions">
          {/* Cancel is the default action — first in DOM/tab order, plain
              styling — since discarding a count grid is the one this dialog
              exists to guard against, not the one to make easy. */}
          <button type="button" className="confirm-dialog-cancel" onClick={onCancel} autoFocus>
            {cancelLabel}
          </button>
          <button type="button" className="confirm-dialog-confirm" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ConfirmDialog
