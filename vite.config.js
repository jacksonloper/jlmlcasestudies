import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// https://vite.dev/config/
export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? '/jlmlcasestudies/' : '/',
  plugins: [
    react(),
    {
      name: 'resolve-external-deps',
      enforce: 'pre',
      resolveId(source, importer) {
        // If importing from case directories, resolve dependencies from node_modules
        if (importer && (importer.includes('/case1/frontend/') || importer.includes('/case2/frontend/') || importer.includes('/case3/frontend/'))) {
          if (!source.startsWith('.') && !source.startsWith('/')) {
            // This is a package import, resolve it relative to root's node_modules
            return this.resolve(source, path.join(__dirname, 'src', 'App.jsx'), { skipSelf: true })
          }
        }
        return null
      }
    }
  ],
  resolve: {
    alias: {
      '@case1': path.resolve(__dirname, 'case1/frontend'),
      '@case2': path.resolve(__dirname, 'case2/frontend'),
      '@case3': path.resolve(__dirname, 'case3/frontend'),
    },
  },
  server: {
    fs: {
      allow: ['.'],
    },
  },
})
