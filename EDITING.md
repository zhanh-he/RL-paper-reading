# Editing the literature catalog

The shared catalog lives in `data/literature.csv`. `README.md` and the interactive
site are generated views; do not edit the generated Markdown table directly.

## Recommended workflow

1. Open [Pages CMS](https://app.pagescms.org/) and sign in with GitHub.
2. Install the Pages CMS GitHub App for `zhanh-he/RL-paper-reading` only.
3. Open **Literature catalog**. Pages CMS presents the CSV as an editable grid.
4. Add or edit rows, then save. The save creates a Git commit in this repository.
5. The `Validate and publish catalog` workflow validates the row, regenerates the
   README table, and updates the interactive site.

Felix needs collaborator access to this repository before signing in. The GitHub
App should be restricted to this repository rather than all repositories.

## Field rules

- `domain`: one of `MusicGen`, `MusicEval`, `MIR`, `AudioGen`, `SpeechEnhance`,
  `AudioLLM`, `LLM`, `CV`, `ML`, `SourceSep`, `Multimodal`, or `Other`.
- `workstream`: one of `Reward`, `RL`, `Reward-n-RL`, or `Other`.
- `felix_rating` and `zhanh_rating`: blank or a value such as `4/5` or `4.5/5`.
- `keywords`: at most three useful terms separated with `；`.
- URL columns: blank or complete `https://` links.
- `github_note`: use `unofficial` only when the linked repository is unofficial.
- `id`: a stable, unique ASCII identifier. Do not change it after publication.
- `obsidian_target`: optional private-vault routing metadata; it is excluded from
  the public JSON used by the site.

## Raw-data fallback

GitHub's CSV editor remains available at
[Edit raw data](https://github.com/zhanh-he/RL-paper-reading/edit/main/data/literature.csv).
After a local edit, run:

```bash
npm run build
npm run check
```

The workflow rejects unsupported labels, duplicate IDs or titles, malformed URLs,
invalid ratings, and keyword lists longer than three terms.
