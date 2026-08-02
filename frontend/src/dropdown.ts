import { useEffect, type RefObject } from 'react'

// Shared by every button-triggered dropdown in the app (Onset, Hear, Speed):
// closes an open menu on an outside pointer-down, so each behaves like a
// normal menu rather than staying pinned open until its own button is
// pressed again.
export function useCloseOnOutsideClick(
  open: boolean,
  setOpen: (open: boolean) => void,
  ref: RefObject<HTMLElement | null>,
) {
  useEffect(() => {
    if (!open) return
    const onPointerDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open, setOpen, ref])
}
