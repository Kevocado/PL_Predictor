import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Bind all interfaces, not just localhost — needed to reach this from
    // another device (phone over Tailscale, or another machine on the LAN).
    host: true,
  },
})
