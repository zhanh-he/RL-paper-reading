# Vocal-to-accompaniment: dataset, model, reward map

Scope: preserve the supplied dry singing voice and generate a separate instrumental accompaniment. A model that regenerates the singer, or a text-to-song prompt set without input vocals, is a different task. This note supports the public [main experiment page](../platform/site/demos/index.html); it is not an internal Huawei experiment receipt.

## Dataset Protocol

| Source | Native unit and usable signal | Decision for this task |
| --- | --- | --- |
| Current Mock | One ACE Studio Emma recording. Online training uses its first 6 s; fixed-seed 12 s replay contains those 6 s. The other phrase is from the same recording. | Measured mechanism/attack showcase only. No independent-song or singer test. |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 150 full songs with vocals, drums, bass, other stems; official 100 train / 50 test. | First reproducible real-pair candidate: condition on vocals, target the non-vocal stems. Split train/validation by whole song, keep official test sealed. These stems are separated studio sources, not always a clean consumer-recorded solo vocal. |
| [MoisesDB](https://github.com/moises-ai/moises-db) | 240 multitrack songs with more detailed source taxonomy. | Additional arrangements and independent-source evaluation after selecting vocal-bearing tracks and confirming access/license; avoid song/artist overlap with MUSDB. |
| [MedleyDB](https://medleydb.weebly.com/description.html) | Multitracks; many but not all tracks have a vocal stem. | Optional expansion after vocal filtering and overlap audit. MUSDB contains MedleyDB material, so do not combine blindly. |
| [CMI-Pref](https://huggingface.co/datasets/HaiwenXia/cmi-pref) | 4,027 human preference votes over pairs of generated music; prompt modalities include text, lyrics, optional reference audio. | **Not a direct V2A training pair**: the reference audio is not documented as an isolated input singer, and the preferred generated audio is not a ground-truth accompaniment stem. The audio-only subset does not repair this mismatch. Can inform a separately validated preference reward. The name is CMI-Pref, not CMI-Perf. |
| [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench) | Text/style/lyrics prompts for song generation. | Lyrics-to-song external evaluation, **not V2A**: no native dry-vocal condition or target instrumental stem. |

The smallest honest next experiment uses a song-disjoint subset of MUSDB18-HQ train for online updates, a disjoint validation subset for checkpoint selection, the official test only once, and a separate dry-vocal domain-shift set. Report the selected songs, licenses, leakage checks, vocal-stem quality, sample durations, model checkpoint, seed, reward version, listening normalization and exact held-out counts before comparing models.

## Direct models

| Model | Availability | Role |
| --- | --- | --- |
| [LaDA-Band](https://github.com/Duoluoluos/LaDA-Band) | Public code, access-controlled checkpoint. | Primary GRPO study. The current four public arms share one Emma input and a frozen starting checkpoint. |
| [AnyAccomp](https://github.com/AmphionTeam/AnyAccomp) | Public inference code and pretrained checkpoint download instructions. | Direct second-model candidate, particularly valuable for clean-vocal generalization. Existing site output is an inference smoke on a different input, not a matched LaDA comparison. |
| [FastSAG](https://github.com/chenjianyi/fastsag) | Public code and checkpoint download instructions. | Short-segment direct baseline; not yet run in this repository. |
| [SingSong](https://arxiv.org/abs/2301.12662) | Published method based on source-separated vocal/instrumental pairs. | Historical method and separation-artifact reference; no public matched run here. |
| [AccompGen](https://arxiv.org/abs/2604.09054) | Direct vocal-accompaniment paper; public weights not verified here. | Research comparison, not an immediately reproducible post-training target. |

Do not turn this into a cross-paper SOTA ranking: inputs, data, duration, output stem and evaluation protocols differ. A same-input, same-mix, same-metric comparison requires rerunning the available checkpoints on one sealed set.

## Rewards and ablations

| Arm or metric | Provenance | What it tests | Status |
| --- | --- | --- | --- |
| Coverage, 40 ms RMS / separate STFT check | Self-developed; STFT implementation is an independent check | Whether the accompaniment is active where needed. Constant sound and noise can game it. | RMS-only GRPO measured; STFT offline check. |
| Beat-v2 F1 | Self-developed scorer from the earlier vocal2accomp project, built on open-source madmom | Vocal/accompaniment beat alignment. Sparse clicks can score well. | 300-step single-arm GRPO measured. |
| Proxy blend | Self-developed: `0.45 coverage + 0.35 energy-onset fit + 0.20 band occupancy - quality penalty` | Fast proxy combination. **Neither actual Beat-v2 nor validated Richness is included.** | 300-step GRPO measured. |
| Beat-v2 + Coverage guard | Self-developed combination of actual Beat-v2, saturated coverage, relative-loudness/peak/noise constraints | Whether combined constraints lessen specific attacks. Coverage can still compensate for worse beat alignment. | 300-step GRPO measured; not proof against hacking. |
| Beat-v5 | Self-developed scorer from the earlier project; onset-grid score with confidence/abstention | Version ablation against v2 under the same LaDA start, Emma input, seed, group size and 6 s updates. | 100-step matched run launched on lab5090; do not report a result until checkpoint replay and confidence audit finish. |
| Richness-v0 proxy | New self-developed experimental scorer: multi-band layer activity + temporal movement, gated by spectral tonality and vocal-relative loudness | Whether a layer-aware target differs from coverage/beat. It cannot judge useful harmony or instrument choice. | Constructed silence/sine/noise/click tests pass; no GRPO result yet. |
| [SongEval](https://github.com/ASLP-lab/SongEval), [MuseCritic](https://github.com/WuqnEl/MuseCritic), [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) | External open-source models | Auxiliary music/audio quality checks, only after validating score range and loudness sensitivity on V2A pairs. | Not used for this V2A GRPO. |

The four completed arms optimize different numerical objectives. Compare **audio and independent metrics** on matched replay, not raw reward heights across arms. For Beat-v2 versus v5, also report v5 abstention/confidence and v2's valid-beat count; a 6 s score can be high variance. Richness-v0 requires human-listening and adversarial checks before it can be described as perceptual richness.

## Formal track

`Shanghai Experiment (Formal)` is a presentation slot only. Before publication, obtain the permitted dataset description, model revision, train/validation/test partition, reward recipe, checkpoint/audio pairs, scores, and explicit disclosure boundary. No private training data or score is inferred from the public Mock.
