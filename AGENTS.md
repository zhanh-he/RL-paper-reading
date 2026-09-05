# Repository Agent Protocol

This file is the first-read instruction for any agent working in this repository.

## Required Read Order

1. Read this `AGENTS.md`.
2. Pull/rebase `main`, then read every root-level `Update_request_*.txt` file.
3. Read `README.md`, `EDITING.md`, and `data/literature.csv` before changing catalog behavior.
4. Read `docs/maintainers/OBSIDIAN_SYNC_AND_RESEARCH_MAP.md` only for maintainer work, weekly knowledge-base sync, or explicit Obsidian requests.

## Sources Of Truth

- `data/literature.csv` is the only editable source for the shared literature catalog.
- `site/data/literature.json` is generated and must not be edited manually.
- `README.md` is a short, manually maintained team entrypoint. Do not add a generated catalog table to it; link to the interactive site instead.
- Run `npm run build` after catalog changes and `npm run check` before committing.
- Keep `id` stable after publication. Use only the controlled domain/workstream labels documented in `EDITING.md`; keep `keywords` to at most three useful terms.
- Never fill or change `felix_rating` unless Felix requested it. Preserve each person's independent rating.

## Update Request Protocol

1. Synchronize with `origin/main` before editing. Rebase or fast-forward; preserve concurrent work.
2. Process every root `Update_request_*.txt`. Treat a placeholder-only file as checked with status `no-action`.
3. For literature requests, verify title, year, venue, publication status, and links against primary sources. Do not invent code/demo links.
4. Apply changes to the shared CSV and, when explicitly requested or during the weekly sync, the Obsidian vault.
5. Archive the original request in `Done_UPD_request/` as `Update_request_<name>_YYYYMMDD_HHMMSS_AWST.txt`.
6. Preserve the original request verbatim and append `processed_at`, `status`, and a concise result summary.
7. Recreate the root request file from the blank template so its stable GitHub edit link continues to work.
8. Build, validate, review the diff, rebase again if the remote moved, commit, push, and verify the GitHub Action.

## README Audience

`README.md` is for Zhanh, Felix, and Hanyu. Keep it limited to what the team needs to browse and update this repository. Do not put local filesystem paths, Obsidian topology, agent-only instructions, or long research maps there.

## Obsidian And Mac Mini

- Routine GitHub edits do not require immediate Obsidian synchronization.
- Run the private-vault synchronization when Zhanh explicitly asks, when a request asks for a detailed note, or during the usual weekly maintenance pass.
- Follow `docs/maintainers/OBSIDIAN_SYNC_AND_RESEARCH_MAP.md`; never delete unrelated vault content and never expose private notes in the public catalog.
- The public CSV and the Obsidian catalog should agree on paper identity and shared metadata. Obsidian may contain richer internal links, detailed reading cards, experiment context, and private judgments.

## Safety And Git Hygiene

- Do not revert changes made by other contributors.
- Do not rewrite published history or force-push.
- Keep request archives immutable after completion except to correct an obvious archival error.
- A catalog task is complete only after generated views are current, validation passes, requests are archived, and the remote workflow succeeds.
