# Updating the literature catalog

The shared catalog lives in `platform/data/literature.csv`. The interactive site's JSON is
generated from it. `README.md` stays short and is maintained manually as the team entrypoint.

## Recommended no-install workflow

1. Open your stable request file in `upd_request/`: [Zhanh](../../upd_request/Update_request_zhanh.txt), [Felix](../../upd_request/Update_request_felix.txt), or [Hanyu](../../upd_request/Update_request_hanyu.txt).
2. Add papers, method ideas, rating changes or corrections below the marker. Keep each person's rating and judgment attributed.
3. Commit the TXT through GitHub's normal web editor. No GitHub App is required.
4. The maintainer/agent verifies the request, archives the original in `upd_request/done_requests/` with an AWST timestamp and processing receipt, then recreates the blank TXT at the stable path.

The existing pending request text was moved without processing or rewriting it. A new method becomes task code under `music-trans/` or `music-gen/` only when implemented; there is no second request workflow.

Contributors who prefer Git may edit `platform/data/literature.csv` directly and open a pull
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
[Edit raw data](https://github.com/zhanh-he/RL-paper-reading/edit/main/platform/data/literature.csv).
After a local edit, run from the repository root:

```bash
npm --prefix platform run build
npm --prefix platform run check
```

The workflow rejects unsupported labels, duplicate IDs or titles, malformed URLs,
invalid ratings, and keyword lists longer than three terms. The public site normally
updates within a minute after a successful merge to `main`.
