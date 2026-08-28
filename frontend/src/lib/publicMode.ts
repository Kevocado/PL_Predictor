// True only in the public, password-gated build (Dockerfile bakes
// VITE_PUBLIC_MODE=true at build time) — hides admin controls the backend
// also hard-blocks in that mode (see api/routes.py::_admin_only). Local dev
// never sets this, so nothing here changes the private/full app.
export const PUBLIC_MODE = import.meta.env.VITE_PUBLIC_MODE === "true";
