import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { applyStoredTheme } from './theme'

// Runs before the first React commit so a stored dark-mode preference is on
// the root element before anything paints — doing this only inside
// useDarkMode's effect would flash light mode first on every reload.
applyStoredTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
