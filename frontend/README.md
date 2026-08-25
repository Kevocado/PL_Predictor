# PL Predictor frontend

React 19, TypeScript, Vite, and Tailwind client for the FastAPI service in
`../src/pl_predictor/api/main.py`.

## Development

```bash
npm install
npm run dev
```

Vite prints the local URL (normally `http://127.0.0.1:5173`; it selects the
next free port when that port is busy). The API must be running on port `8000`.
Set `VITE_API_BASE_URL` only when the API is on another host.

```bash
npm run build
```

Builds the production bundle and type-checks the client.

## UI responsibilities

- `pages/FixturesPage.tsx` renders gameweeks and polls only while official
  player outcomes are pending.
- `components/FixtureModal.tsx` keeps pre-match projections, completed-match
  reviews, and player-call review provenance visibly distinct.
- `components/TeamHub.tsx` and `components/PlayerHub.tsx` are descriptive
  analytics only; they must not change prediction or value-bet decisions.
- `components/TeamBadge.tsx` resolves local crest assets and has an initials
  fallback. See `public/badges/README.md` before adding assets.

Keep API types in `src/types.ts` aligned with Pydantic schemas. For any UI/API
change, run `npm run build` and the focused backend tests.
