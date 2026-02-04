import react from '@vitejs/plugin-react-swc'
import { defineConfig } from 'vite'
import { viteSingleFile } from "vite-plugin-singlefile"

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteSingleFile()
  ],
  build: {
    // 1) avoid code splitting
    // 2) keep single chunk
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      }
    }
  }
})
