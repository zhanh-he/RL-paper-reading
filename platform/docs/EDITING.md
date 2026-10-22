# Updating the literature catalog

The two catalogs have independent sources: `platform/data/rl/literature.csv` and
`platform/data/amt/literature.csv`. RL is published at the Pages root; AMT at `/amt/`.
Both use `platform/templates/catalog.html` and the same table controls. Generated pages/JSON
must not be edited manually. `README.md` stays a manually maintained team entrypoint.

## Catalog ownership

- **RL**: MusicGen, music evaluation, reward, RL and supporting cross-domain methods. Beat/genre tools used to evaluate generation stay here; they are not automatically AMT.
- **AMT**: audio-to-note transcription, multi-instrument transcription, choral transcription, multipitch estimation and voice assignment. `MusicTranscription` is the domain label; `RL` can still be its workstream if the paper actually studies RL.
- Keep one owning catalog per paper, stable IDs, all ratings, attributed comments and links. Global validation rejects duplicate IDs/titles across the two files. Cross-field papers use links, not copied editable records.
- `platform/site/data/{rl,amt}-literature.json` are the separate public tables. The old combined `literature.json` is kept only for integrations and resolving pre-split `?paper=` links.

## Recommended no-install workflow

1. Choose RL or AMT in the [request guide](../../upd_request/README.md), then open the corresponding Zhanh, Other (Felix / Jiahao / colleagues) or Hanyu TXT in `upd_request/rl/` or `upd_request/amt/`.
2. Add papers, method ideas, rating changes or corrections below the marker. Keep each person's rating and judgment attributed.
3. Commit the TXT through GitHub's normal web editor. No GitHub App is required.
4. The maintainer/agent verifies the request, archives the original in the same domain's `done_requests/` with an AWST timestamp and processing receipt, then recreates the blank TXT at the stable path.

Historical root archives stay unchanged. The three old root inboxes remain compatibility entry points; if someone adds text there, route it by subject without deleting it. A new method becomes task code under `music-trans/` or `music-gen/` only when implemented.

Contributors who prefer Git may edit either source CSV directly and open a pull
request. Pages CMS remains optional for accounts allowed to authorize its GitHub App.

## Field rules

- `domain`: one of `MusicFoundation`, `MusicGen(LLM)`, `MusicGen(Small)`, `MusicGen(Mid)`, `MusicGen(Large)`, `MusicGen`, `MusicEval`, `MusicTranscription`, `MIR`, `AudioGen`, `Speech`, `SpeechEnhance`,
  `AudioLLM`, `LLM`, `CV`, `MachineLearning`, `SourceSep`, `Multimodal`, or `Other`.
- `workstream`: one of `Reward`, `RL`, `Reward-n-RL`, or `Other`.
- `other_rating` and `zhanh_rating`: blank or a value such as `4/5` or `4.5/5`; append `*` when the rating is provisional and awaiting reread. `other_rating` replaces the former Felix column; identify its contributor in `other_note`. A missing score stays blank.
- `zhanh_note`, legacy `felix_note`, `other_note`, and `hanyu_note`: independently attributed project judgments. Start shared Other notes with the contributor's name, for example `Felix:` or `Jiahao:`.
- `MusicFoundation` means music-understanding representations (MERT/MuQ), not music generators. `MusicGen(LLM)` means token language models or LM+renderer systems, regardless of size and whether text-pretrained. Non-LM generators use `Small` (<1B), `Mid` (1–3B) or `Large` (>3B) based on a sourced reference configuration. Unknown/ambiguous scopes stay `MusicGen`; company size and product marketing are not evidence.
- `model_size`: reference version and component-wise parameter count; distinguish main generator, encoder, codec, renderer and trainable adapters. `model_params_b` is an optional positive numerical reference in billions used for sorting; its scope must match the textual description. A family may have several sizes, so its domain never replaces the component list.
- `generation_speed`: author-reported latency, RTF or explicitly labeled relative speed/NFE; not a locally measured benchmark. `measurement_context` must disclose hardware, duration, sampling/precision/batch where known, component scope, RTF direction and any missing conditions. NFE is not wall-clock speed. Never infer end-to-end timing from renderer-only claims.
- `model_size_url` and `generation_speed_url` are mandatory primary-source links whenever the corresponding quantitative text is filled. Missing/unverified quantities stay blank, not zero. Different-paper timings are not a fair speed leaderboard.
- `author_org`: short institution/team names for the first-listed author only; multiple affiliations of that author may be retained. Collective resources use their official team. `author_org_url` links the supporting primary source. Unverified affiliations are marked `待核验`.
- `year`: publication year for a paper, or documented release year for a resource. Undated/inaccessible Web leads may leave it blank; do not substitute the year they were added to this catalog.
- `keywords`: at most three useful terms separated with `；`.
- URL columns: blank or complete `https://` links.
- `github_note`: use `unofficial` only when the linked repository is unofficial.
- `id`: a stable, unique ASCII identifier. Do not change it after publication.
- `obsidian_target`: optional private-vault routing metadata; it is excluded from
  the public JSON used by the site.

## Direct data workflow

GitHub's CSV editor remains available at
[Edit RL data](https://github.com/zhanh-he/RL-paper-reading/edit/main/platform/data/rl/literature.csv) or
[Edit AMT data](https://github.com/zhanh-he/RL-paper-reading/edit/main/platform/data/amt/literature.csv).
After a local edit, run from the repository root:

```bash
npm --prefix platform run build
npm --prefix platform run check
```

The workflow rejects unsupported labels, duplicate IDs or titles, malformed URLs,
invalid ratings, and keyword lists longer than three terms. The public site normally
updates within a minute after a successful merge to `main`.

## UI Validation

Run `npm --prefix platform run test:catalog` for catalog and note-preservation checks.
With Playwright and its dedicated testing browser installed, run
`npm --prefix platform run test:catalog:ui` for desktop/mobile rendering, filters,
exports, catalog switching, request paths and legacy paper redirects. If Playwright
is provided by a shared runtime, set `PLAYWRIGHT_MODULE` to its module path.
`CATALOG_BASE_URL` optionally targets the deployed site instead of an ephemeral local server.
Never point browser automation at the user's installed Google Chrome or personal profile.
