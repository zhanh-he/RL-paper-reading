# Updating the literature catalog

The shared catalog lives in `platform/data/literature.csv`. The interactive site's JSON is
generated from it. `README.md` stays short and is maintained manually as the team entrypoint.

## Recommended no-install workflow

1. Open your stable request file in `upd_request/`: [Zhanh](../../upd_request/Update_request_zhanh.txt), [Other](../../upd_request/Update_request_other.txt) (Felix, Jiahao and other contributors), or [Hanyu](../../upd_request/Update_request_hanyu.txt) (reserved).
2. Add papers, method ideas, rating changes or corrections below the marker. Keep each person's rating and judgment attributed.
3. Commit the TXT through GitHub's normal web editor. No GitHub App is required.
4. The maintainer/agent verifies the request, archives the original in `upd_request/done_requests/` with an AWST timestamp and processing receipt, then recreates the blank TXT at the stable path.

The existing pending request text was moved without processing or rewriting it. A new method becomes task code under `music-trans/` or `music-gen/` only when implemented; there is no second request workflow.

Contributors who prefer Git may edit `platform/data/literature.csv` directly and open a pull
request. Pages CMS remains optional for accounts allowed to authorize its GitHub App.

## Field rules

- `domain`: one of `MusicFoundation`, `MusicGen(LLM)`, `MusicGen(Small)`, `MusicGen(Mid)`, `MusicGen(Large)`, `MusicGen`, `MusicEval`, `MIR`, `AudioGen`, `Speech`, `SpeechEnhance`,
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
[Edit raw data](https://github.com/zhanh-he/RL-paper-reading/edit/main/platform/data/literature.csv).
After a local edit, run from the repository root:

```bash
npm --prefix platform run build
npm --prefix platform run check
```

The workflow rejects unsupported labels, duplicate IDs or titles, malformed URLs,
invalid ratings, and keyword lists longer than three terms. The public site normally
updates within a minute after a successful merge to `main`.
