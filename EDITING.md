# Updating the literature catalog

The shared catalog lives in `data/literature.csv`. The interactive site's JSON is
generated from it. `README.md` stays short and is maintained manually as the team entrypoint.

## Recommended no-install workflow

1. Open [`update-request/`](update-request/README.md) and use the literature or new-method template.
2. Add one request per file with a descriptive name. Keep the author's rating and judgment attributed.
3. Commit the new file through GitHub's normal web editor. No GitHub App is required.
4. The maintainer/agent verifies and applies the change, then archives the request under `update-request/archive/` with processing status.

The legacy root `Update_request_*.txt` files remain in place, including any pending text. They use the previous `Done_UPD_request/` archive convention until explicitly processed; this layout change does not process or discard them.

Contributors who prefer Git may edit `data/literature.csv` directly and open a pull
request. Pages CMS remains optional for accounts allowed to authorize its GitHub App.

## Field rules

- `domain`: one of `MusicGen`, `MusicEval`, `MIR`, `AudioGen`, `Speech`, `SpeechEnhance`,
  `AudioLLM`, `LLM`, `CV`, `MachineLearning`, `SourceSep`, `Multimodal`, or `Other`.
- `workstream`: one of `Reward`, `RL`, `Reward-n-RL`, or `Other`.
- `felix_rating` and `zhanh_rating`: blank or a value such as `4/5` or `4.5/5`; append `*` when the rating is provisional and awaiting reread.
- `zhanh_note`, `felix_note`, and `hanyu_note`: independently attributed project judgments. Never put one person's interpretation in the other person's field.
- `keywords`: at most three useful terms separated with `；`.
- URL columns: blank or complete `https://` links.
- `github_note`: use `unofficial` only when the linked repository is unofficial.
- `id`: a stable, unique ASCII identifier. Do not change it after publication.
- `obsidian_target`: optional private-vault routing metadata; it is excluded from
  the public JSON used by the site.

## Direct data workflow

GitHub's CSV editor remains available at
[Edit raw data](https://github.com/zhanh-he/RL-paper-reading/edit/main/data/literature.csv).
After a local edit, run:

```bash
npm run build
npm run check
```

The workflow rejects unsupported labels, duplicate IDs or titles, malformed URLs,
invalid ratings, and keyword lists longer than three terms. The public site normally
updates within a minute after a successful merge to `main`.
