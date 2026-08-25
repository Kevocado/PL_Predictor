# Club crest assets

`TeamBadge` serves club crests from `/badges/`. Keep the source SVGs in the
repository-level `badges/` directory and copy the same files here for Vite to
serve them. The app makes no remote image request and falls back to a
club-colour initials badge if a file is missing or cannot render.

Use the exact filenames in `frontend/src/lib/teamColors.ts`'s `CREST_FILES`
map, such as `arsenal-logo-footylogos.svg`,
`manchester-city-logo-footylogos.svg`, and
`nottingham-forest-logo-footylogos.svg`. Update that map and add the matching
asset before supporting another club. Keep SVGs square with transparent
padding where necessary so crests align consistently.
