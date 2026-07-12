import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the FastAPI backend so the app can use the
    // backend's relative URLs (audio_url, stem urls) as-is, with no CORS.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
