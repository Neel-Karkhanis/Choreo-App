import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind one address (not both IPv4 and IPv6) and refuse to start if 5173
    // is already taken, rather than silently landing on the other IP stack
    // or bumping to 5174. A stale `npm run dev` left running then fails loud
    // instead of leaving two servers answering "localhost:5173" at random.
    host: '127.0.0.1',
    strictPort: true,
    // Forward API calls to the FastAPI backend so the app can use the
    // backend's relative URLs (audio_url, stem urls) as-is, with no CORS.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
