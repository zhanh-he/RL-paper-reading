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
| [LaDA-Band](https://github.com/Duoluoluos/LaDA-Band) | Public code, access-controlled checkpoint. | Primary GRPO study. The six completed public arms share one Emma input and a frozen starting checkpoint. |
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
| Beat-v5 | Self-developed scorer from the earlier project; onset-grid score with confidence/abstention | Version ablation against v2 under the same LaDA start, Emma input, seed, group size and 6 s updates. | 100-step matched run complete; 0/5/50/100 audio and independent v2/v5 replay scores published on the main demo. |
| Richness-v0 proxy | New self-developed experimental scorer: multi-band layer activity + temporal movement, gated by spectral tonality and vocal-relative loudness | Whether a layer-aware target differs from coverage/beat. It cannot judge useful harmony or instrument choice. | 100-step online run and 0/5/50/100 audio complete. Constructed silence/sine/noise/click tests pass; a deliberately detuned layer scores within 0.03 of the consonant layer, an explicit harmonic bad case. |
| [SongEval](https://github.com/ASLP-lab/SongEval), [MuseCritic](https://github.com/WuqnEl/MuseCritic), [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) | External open-source models | Auxiliary music/audio quality checks, only after validating score range and loudness sensitivity on V2A pairs. | Not used for this V2A GRPO. |

The six completed arms optimize different numerical objectives. Compare **audio and fixed-replay cross-metrics** on matched replay, not raw reward heights across arms. A scorer used to optimize an arm is not an independent model when rerun on that arm's fixed WAV. For Beat-v2 versus v5, also report v5 abstention/confidence and v2's valid-beat count; a 6 s score can be high variance. Richness-v0 requires human-listening and adversarial checks before it can be described as perceptual richness.

The new v5 arm uses the existing `mir.reward_function.beat_v5.BeatV5Scorer` from `vocal2accomp-muse` commit `0a725f0f5d05ae8aa7a1902c9703fb397bcf71e8`, with the `madmom` backend, 6 s online windows and 12 s fixed replays. The LaDA code checkout is commit `1177fafd43a5f9b5166aa938c52f73180fdff46f`. Both version scorers are rerun on the **same original 16 s Emma vocal file**, truncated internally to the first 12 s for fixed replay; replacing the source with a separately cut 12 s WAV changes v2's detected reference beats and invalidates a direct comparison.

The v2/v5 run manifests agree on checkpoint, text, duration, sampler, group size, LR, seeds, LoRA rank/alpha and KL coefficient. Their vocal paths differ only in relative versus absolute spelling of the same file. Their frozen step-0 accompaniment WAVs have the identical SHA-256 `d499c6f1cab19ff3c8a8388337b6abeb540498762fd59ae4a55362c2c5d5bcae`.

The lab5090 RTX 5090 online runs update only the LaDA output-projection LoRA with group size 2, eight denoising steps, learning rate `5e-4`, 6 s candidate windows and fixed-seed 12 s checkpoint replays. The first 6 s of a single Emma vocal are the training input; the 12 s replay overlaps them. The [cross-scorer receipt](../music-gen/vocal2accomp/rewards/beat_v2_v5_replay_comparison_2026-10-06.json) records all available fixed WAVs, scorer settings and original-vocal SHA. Independent 12 s v2/v5 scores were `0.296/0.209` at the shared frozen start, `0.400/0.453` after 100 v2-only updates, and `0.357/0.276` after 100 v5-only updates. V5 confidence was `1.0` on these replays. This is one prompt, seed and recording; v5 did not beat v2 here on either cross-score at step 100, and no general superiority claim follows.

The Richness-v0 arm uses those same settings and the same step-0 WAV SHA. At 100 updates, its fixed 12 s Richness proxy moved `0.345 → 0.539`, RMS coverage `0.528 → 0.847`, Beat-v2 F1 `0.296 → 0.552`, and Beat-v5 `0.209 → 0.331`. At 50 updates Beat-v5 was `0.526`, so the cross-metric trajectory is not monotone. See the [offline richness cross-score receipt](../music-gen/vocal2accomp/rewards/richness_v0_replay_diagnostic_2026-10-06.json); it runs the same Richness-v0 scorer on fixed WAVs from every arm, not a human preference judge. The 5-step WAV remains byte-identical to the frozen start. These numbers are mechanistic single-recording observations, not evidence of more useful harmony or generalization.

Constructed bad case: the existing 6 s oracle-aligned click track scores `1.000` with the original Beat-v2 worker and `0.617` with the madmom-backend Beat-v5 worker. The frozen LaDA accompaniment scores about `0.226` under the same 6 s v5 setup. Thus v5 still rewards a sparse, non-musical click attack; confidence `1.0` on the click is not a musical-quality guarantee. This is a scorer stress test, not evidence that the trained policy learned clicks.

## Formal track

`Shanghai Experiment (Formal)` is a presentation slot only. Before publication, obtain the permitted dataset description, model revision, train/validation/test partition, reward recipe, checkpoint/audio pairs, scores, and explicit disclosure boundary. No private training data or score is inferred from the public Mock.
