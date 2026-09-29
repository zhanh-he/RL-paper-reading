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
- Keep `zhanh_note`, `felix_note`, and `hanyu_note` separately attributed. Never fill or change another contributor's note, and never change Felix's rating unless Felix requested it; preserve each person's independent judgment.

## Update Request Protocol

1. Synchronize with `origin/main` before editing. Rebase or fast-forward; preserve concurrent work.
2. For an update-processing task, check all three `upd_request/Update_request_*.txt` files. Treat a placeholder-only file as `no-action`; an unrelated structural task must leave pending text untouched.
3. For literature requests, verify title, year, venue, publication status, and links against primary sources. Do not invent code/demo links.
4. Apply changes to the shared CSV and, when explicitly requested or during the weekly sync, the Obsidian vault.
5. Archive the original request in `upd_request/done_requests/` with an AWST timestamp. Preserve its text verbatim and append `processed_at`, `status`, and a concise result summary.
6. Recreate the contributor's blank TXT template at the same path so its GitHub edit link stays stable.
7. Build, validate, review the diff, rebase again if the remote moved, commit, push, and verify the GitHub Action.

## Experiment Publishing

- `platform/site/index.html` and `platform/data/literature.csv` remain the literature site's entry and source of truth. The deployed Pages URL stays stable although the repository source moved.
- Task code and run records live under `music-trans/` or `music-gen/`; public demos live under `platform/site/demos/`. Keep a run's model, data split, reward version, optimizer, independent evaluation and provenance together.
- Do not publish private vault notes, restricted data, checkpoints, credentials or unlicensed audio. Label external paper results, local measurements and plans distinctly. Do not create dummy reward implementations or demo pages that look like completed experiments.

## README Audience

`README.md` is for Zhanh, Felix, and Hanyu. Keep it limited to what the team needs to browse and update this repository. Do not put local filesystem paths, Obsidian topology, agent-only instructions, or long research maps there.

## Obsidian And Mac Mini

- Routine GitHub edits do not require immediate Obsidian synchronization.
- Run the private-vault synchronization when Zhanh explicitly asks, when a request asks for a detailed note, or during the usual weekly maintenance pass.
- Follow `platform/docs/maintainers/OBSIDIAN_SYNC_AND_RESEARCH_MAP.md`; never delete unrelated vault content and never expose private notes in the public catalog.
- The public CSV and the Obsidian catalog should agree on paper identity and shared metadata. Obsidian may contain richer internal links, detailed reading cards, experiment context, and private judgments.

## Safety And Git Hygiene

- Do not revert changes made by other contributors.
- Do not rewrite published history or force-push.
- Keep request archives immutable after completion except to correct an obvious archival error.
- A catalog task is complete only after generated views are current, validation passes, requests are archived, and the remote workflow succeeds.
