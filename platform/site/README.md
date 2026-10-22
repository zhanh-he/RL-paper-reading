# Published sites

`index.html` is the RL catalog at the existing GitHub Pages root; `amt/index.html` is the independent music-transcription catalog. Both pages are generated from `platform/templates/catalog.html`; their sources are `platform/data/rl/literature.csv` and `platform/data/amt/literature.csv`. Run `platform/scripts/sync-catalog.mjs` through the normal build. Preserve existing paper deep links and the root URL; the combined JSON endpoint is read-only compatibility data.

Verified experiment demos go under [demos](demos/README.md). A cross-experiment `overview/` website is deferred until comparable, publishable results exist. Until then, the repository README is the overview; no empty result pages or invented numbers are published.
