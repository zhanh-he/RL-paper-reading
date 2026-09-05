# Updating the literature catalog

The shared catalog lives in `data/literature.csv`. `README.md` and the interactive
site are generated views; do not edit the generated Markdown table directly.

## Recommended no-install workflow

1. Open your root request file: `Update_request_zhanh.txt`,
   `Update_request_felix.txt`, or `Update_request_hanyu.txt`.
2. Add the requested paper, rating change, correction, or question below the marker.
3. Commit the request through GitHub's normal web editor. No GitHub App is required.
4. The maintainer/agent verifies and applies the change, then moves the completed
   request to `Done_UPD_request/` with an AWST timestamp.
5. A fresh blank request file is recreated at the same path for the next update.

Contributors who prefer Git may edit `data/literature.csv` directly and open a pull
request. Pages CMS remains optional for accounts allowed to authorize its GitHub App.

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
