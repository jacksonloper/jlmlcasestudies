import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'resolve-external-deps',
      enforce: 'pre',
      resolveId(source, importer) {
        // If importing from case directories, resolve dependencies from frontend/node_modules
        if (importer && (importer.includes('/case1/frontend/') || importer.includes('/case2/frontend/'))) {
          if (!source.startsWith('.') && !source.startsWith('/')) {
            // This is a package import, resolve it relative to frontend's node_modules
            return this.resolve(source, path.join(__dirname, 'src', 'App.jsx'), { skipSelf: true })
          }
        }
        return null
      }
    }
  ],
  resolve: {
    alias: {
      '@case1': path.resolve(__dirname, '../case1/frontend'),
      '@case2': path.resolve(__dirname, '../case2/frontend'),
    },
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
