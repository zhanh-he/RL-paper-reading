# Repository Agent Protocol

This file is the first-read instruction for any agent working in this repository.

## Required Read Order

1. Read this `AGENTS.md`.
2. Pull/rebase `main`, then read the three `upd_request/Update_request_*.txt` files. Do not process pending requests during unrelated repository-structure work.
3. Read `README.md`, `platform/docs/EDITING.md`, and `platform/data/literature.csv` before changing catalog behavior.
4. Read `platform/docs/maintainers/OBSIDIAN_SYNC_AND_RESEARCH_MAP.md` only for maintainer work, weekly knowledge-base sync, or explicit Obsidian requests.

## Sources Of Truth

- `platform/data/literature.csv` is the only editable source for the shared literature catalog.
- `platform/site/data/literature.json` is generated and must not be edited manually.
- `README.md` is a short, manually maintained team entrypoint. Do not add a generated catalog table to it; link to the interactive site instead.
- Run `npm --prefix platform run build` after catalog changes and `npm --prefix platform run check` before committing.
- Keep `id` stable after publication. Use only the controlled domain/workstream labels documented in `platform/docs/EDITING.md`; keep `keywords` to at most three useful terms.
- A trailing `*` marks a provisional rating pending reread, for example `4/5*`.
- Ratings use `zhanh_rating` and `other_rating`. The latter includes explicit ratings from Felix, Jiahao and other contributors; identify the rater in `other_note`. Preserve existing ratings when a request provides no new score.
- Keep `zhanh_note`, legacy `felix_note`, `other_note`, and `hanyu_note` separately attributed. A request explicitly relaying another person's rating is sufficient evidence to record that rating; do not infer ratings from enthusiastic wording or research-pack ranks.
- `author_org` is the first-listed author's institution(s) as stated in the paper, or the official owning team for a collective paper/resource. Use `author_org_url` for its primary source. Never replace an academic first author with a sponsoring company merely for convenience.
- `MusicFoundation` is reserved for music-understanding/representation models such as MERT/MuQ, not song generators. `MusicGen(LLM)` covers token-LM or LM+renderer generation (including compact LMs; not a size claim). Non-LM generators with a verified reference size use `MusicGen(Small)` (<1B), `MusicGen(Mid)` (1–3B), or `MusicGen(Large)` (>3B). `MusicGen` is the explicit unknown/mixed-scope fallback. These are catalog conventions, not universal scientific categories; keep at most two visible labels (domain + workstream).
- Record model size, numerical reference `model_params_b`, speed, measurement scope and primary sources separately. Do not confuse trainable adapter size with full inference size, token counts with parameter counts, NFE with latency, rendering-only with end-to-end runtime, or inverse RTF conventions. Author-reported timings are not our own measurements; missing data stays blank.
- Inaccessible reading leads may be retained as explicitly unverified Web entries with blank year and ratings. Never invent their titles, author affiliations, publication dates, code or demos.

## Update Request Protocol

1. Synchronize with `origin/main` before editing. Rebase or fast-forward; preserve concurrent work.
2. For an update-processing task, check the Zhanh, Other and reserved Hanyu `upd_request/Update_request_*.txt` files. Other is the shared intake for Felix, Jiahao and other contributors. Treat a placeholder-only file as `no-action`; an unrelated structural task must leave pending text untouched.
3. For literature requests, verify title, year, venue, publication status, and links against primary sources. Do not invent code/demo links.
4. Apply new literature and shared-metadata changes to the CSV and synchronize the corresponding Obsidian catalog/cards in the same maintenance task when the vault is accessible. Preserve internal insights and attribution; never upload private notes. A UI-only change needs no vault rewrite. Report an inaccessible vault as pending synchronization, not complete agreement.
5. Archive the original request in `upd_request/done_requests/` with an AWST timestamp. Preserve its text verbatim and append `processed_at`, `status`, and a concise result summary.
6. Recreate the contributor's blank TXT template at the same path so its GitHub edit link stays stable.
7. Build, validate, review the diff, rebase again if the remote moved, commit, push, and verify the GitHub Action.

## Experiment Publishing

- Public audio policy: this repository, GitHub Pages and demo release assets may contain only MP3 audio. Encode listening copies with libmp3lame at 192 kbps / 48 kHz, without loudness normalization. Never commit or upload WAV/FLAC/OGG/M4A/AAC/AIFF audio. Keep lossless training/scoring originals outside the repository on experiment machines or private recovery storage. Record separate original and MP3 SHA256 hashes; lossy listening copies must not be described as byte-identical scored audio. MIDI and image assets are allowed. Run the MP3-only build check before publication.

- `platform/site/index.html` and `platform/data/literature.csv` remain the literature site's entry and source of truth. The deployed Pages URL stays stable although the repository source moved.
- Task code and run records live under `music-trans/` or `music-gen/`; public demos live under `platform/site/demos/`. Keep a run's model, data split, reward version, optimizer, independent evaluation and provenance together.
- Do not publish private vault notes, restricted data, checkpoints, credentials or unlicensed audio. Label external paper results, local measurements and plans distinctly. Do not create dummy reward implementations or demo pages that look like completed experiments.

## README Audience

`README.md` is for Zhanh, Felix, and Hanyu. Keep it limited to what the team needs to browse and update this repository. Do not put local filesystem paths, Obsidian topology, agent-only instructions, or long research maps there.

## Obsidian And Mac Mini

- GitHub CSV is authoritative for shared literature identities, ratings, labels, links, affiliations and quantitative metadata. Obsidian is authoritative for core research insights, detailed reading notes and private summaries. New shared literature must agree in both places, not wait silently for a later weekly pass.
- Use the scoped `platform/scripts/sync-obsidian.mjs` with an explicit vault path. It updates marked shared blocks and preserves detailed prose. Read vault rules and confirm Obsidian Sync when available; check for concurrent changes immediately before writes.
- Follow `platform/docs/maintainers/OBSIDIAN_SYNC_AND_RESEARCH_MAP.md`; never delete unrelated vault content and never expose private notes in the public catalog.
- The public CSV and the Obsidian catalog should agree on paper identity and shared metadata. Obsidian may contain richer internal links, detailed reading cards, experiment context, and private judgments.

## Safety And Git Hygiene

- Do not revert changes made by other contributors.
- Do not rewrite published history or force-push.
- Keep request archives immutable after completion except to correct an obvious archival error.
- A catalog task is complete only after generated views are current, validation passes, requests are archived, and the remote workflow succeeds.
