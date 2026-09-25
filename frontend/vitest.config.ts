import { defineConfig } from "vitest/config";

// Kept apart from vite.config.ts: vitest 2 bundles its own Vite types, which
// don't accept the app's Vite 8 plugins. Tests need no plugins — esbuild's
// automatic JSX runtime is enough for the React components.
export default defineConfig({
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test-setup.ts",
  },
});
