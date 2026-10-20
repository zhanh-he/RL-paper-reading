# Model Parameters and Generation Cost

Audited 2026-10-15. [Online tables](https://zhanh-he.github.io/RL-paper-reading/demos/#model-costs) / [downloadable JSON](https://zhanh-he.github.io/RL-paper-reading/demos/data/model-costs.json).

2026-10-20: [timing audit and proposed experiment plan](EXPERIMENT_PLAN_2026-10-20.md). Online navigation is now **Models > Lyrics2song > Model Cost / Experiment Plan**; the old `#model-costs` link remains compatible. The plan is source-controlled in `experiment-plan.json` and included in the public download. No new benchmark or training is launched by this update.

## True Parameters, Not Model Names

The core column counts unique native `nn.Parameters`, including frozen weights, excluding fixed buffers. Shared weights count once. These are exact local checkpoint configurations, not estimates inferred from file sizes or names. `B = 10^9`, `M = 10^6`.

| Model / checkpoint | Generator/policy core | New-plan LoRA | Modules |
| --- | ---: | ---: | ---: |
| YuE2-3B MoT | 3,580,352,576 | 8,716,288 | 196 |
| Muse-0.6b | 612,555,776 | 5,046,272 | 196 |
| DiffRhythm2 | 1,136,249,664 | 6,029,312 | 112 |
| HeartMuLa oss-3B-happy-new-year | 3,938,113,536 | 13,484,032 | 217 |
| SongBloom recovered full-150s (36 AR / 12 NAR layers) | 1,872,372,929 | 3,096,576 | 60 |
| ACE-Step1.5 v15-base, not Turbo or XL | 2,393,872,518 | 10,223,616 | 264 |

All new adapters are rank 8 / alpha 16. For a selected linear weight `[out,in]`, LoRA adds `8*(in+out)` parameters; alpha changes scaling, not parameter count. This is not full-weight fine-tuning, and excludes old head-only adapters.

- YuE2: semantic AR attention + MLP; NAR and VAE frozen. New TRL migration not submitted. Old q/v rank-4 adapter had 802,816 parameters and is not the new plan.
- Muse: all attention + MLP projections. Embedding and LM head are tied: explicit `tie_weights()` is required after empty construction. Counting both creates a false 0.785B model.
- DiffRhythm2: DiT attention + MLP, already verified by the accepted native worker. The decoder and MuQ-MuLan condition encoder are separate from the core.
- HeartMuLa: both global backbone and local codebook decoder attention + MLP. Its marketed 3B label is not the full native generator count.
- SongBloom: all selected NAR DiT attention + MLP weights; AR planner and VAE frozen. This is the full-150s checkpoint, not tiny/DPO. New complete Miles worker not submitted.
- ACE-Step: base decoder self-attention, cross-attention and MLP. Shape-verified plan only; RL not submitted.

### Additional Components

- YuE2 VAE: 132,616,130. MoT + VAE: **3,712,968,706**. The MoT checkpoint also contains 50,331,648 positional-buffer elements that are not parameters.
- Muse MuCodec: **1,487,044,418**, counted in a separately loaded native service: VAE 413,817,458 + learned model 1,073,226,960. LM + loaded codec: **2,099,600,194**. Loaded components may include encoding-only modules; total parameters are not the same as parameters exercised in decoding.
- HeartCodec: 1,655,308,877. Generator + codec: **5,593,422,413**. Codec includes a learned flow network. Fixed buffers excluded.
- SongBloom VAE: 168,695,426. Unique whole pipeline: **2,041,068,355**. Duplicate checkpoint VAE aliases and 38,064 buffer elements excluded.
- ACE text encoder: 595,776,512; VAE: 168,695,426. Without optional planner: **3,158,344,456**. Optional named 1.7B planner has 1,854,243,840 parameters; combined total: **5,012,588,296**.
- DiffRhythm2 auxiliary components have not yet been reconciled to a unique whole-pipeline total. The table explicitly does not claim 1.136B is the whole system.

## Timing Protocol

`RTF = generation wall seconds / actual generated audio seconds`, lower is faster. Batch size 1. Warm timing excludes initial model load, queue, download, reward scoring, optimization and MP3 encoding. Different models, hardware, precision, sampler steps and timing boundaries do **not** form a controlled speed leaderboard.

1. **YuE2 and Muse:** separate Gadi interactive allocation `181300445.gadi-pbs`, `gadi-dgx-a100-0002`, A100-SXM4-80GB. One fixed original CMI validation condition per model, seed 5101. Native decoding, 4-second warmup, then actual 60/300-second outputs. EOS is suppressed to a 25-Hz token budget; no waveform looping. This is a throughput benchmark, not a test of natural musical endings or perceptual quality. Timings can differ from optimized production implementations.
2. **DiffRhythm2:** recover all 30 completed frozen-baseline validation timings on RTX5090, nominal32 time points / CFG2 / FP16. This is now classified as **worker/rollout timing, not optimized resident-GPU inference**: it includes CPU/GPU offload, condition encoding/cache, decoding, private lossless save and cleanup. The rollout retains block KV cache; the native loader disables FlexAttention and uses SDPA. No individual stage wall times were recorded. Hash bindings were checked against all scored audio. One output ended at31.8s; the60s median and300s extrapolation use only29 full-length outputs. All30 receipts are retained unchanged. 300s is a **linear estimate**, not measured.
3. **HeartMuLa:** all 72 archived 60-second-budget H200 rollouts from job `181262219.gadi-pbs` on `gadi-gpu-h200-0029`. BF16 generator / FP32 HeartCodec. Includes action/log-prob collection and policy-record save, so explicitly a **rollout proxy**, not pure inference. Three outputs ended at 48.8/56.32/56.96 seconds; only the 69 outputs within 60 +/- 0.1 seconds contribute to the 60-second median and extrapolation. All 72 remain archived. 300 seconds is only arithmetic extrapolation beyond the default 240-second limit, not validated native output.
4. **SongBloom:** same independent A100 allocation, native BF16 / 36 Euler steps / CFG1.5, one CMI valid audio+lyrics condition, 10-second reference audio, frozen full-150s checkpoint. Actual 60.16-second output took 93.081 seconds (RTF 1.54723). Includes condition read/resampling/encoding, AR/NAR and VAE; excludes model initialization and file saving. First call, no kernel warmup. Direct host execution hit a GLIBC/FlashAttention compatibility error; reran successfully in the existing hash-verified Bookworm container, without altering model dependencies or training jobs. See `songbloom-timing.json`. The [paper Table 2](https://arxiv.org/html/2506.07634v1#S6) RTF 1.649 has unspecified timing hardware and is only background reference, not the measured row. This checkpoint supports up to 150 seconds; 300-second native output is **n/a**.
5. **ACE-Step1.5 base:** no valid local60/300s end-to-end receipt; report **n/a**. Earlier staging was stopped because transfer was too slow; partial assets are `.incomplete` and were never loaded. The official240s/2s declaration remains a source claim, but its previous0.5s/2.5s duration extrapolations have now been **withdrawn**, not retained as usable full-chain speed estimates. LM identity and timing boundaries were never bound to that claim.

No training was stopped or moved. On 5090, only metadata/CPU audits and existing receipts were read; new GPU timing ran in the separate A100 allocation. A Kaya immediate interactive request failed because nodes were busy. No scheduled monitoring was created.

## Reproduction and Evidence

### Published Speed Research: MusicGen / LeVo2 / ACE-Step1.5

Added 2026-10-15, **not newly measured on our GPUs**. Structured facts, original RTF definitions, pinned README sources and normalized calculations are in `published-speed-sources.json`. They extend the latency table only, not the six locally audited parameter/LoRA rows.

| Version | Hardware | Wall/audio RTF | Wait for 60s | Wait for 300s | Evidence |
| --- | --- | ---: | --- | --- | --- |
| MusicGen 3.3B /32kHz mono | A100 + EPYC7443P; VRAM unspecified | 2.5 | 2m30s estimate | 12m30s estimate | Diff-A-Riff Table2, 95s output benchmark |
| LeVo2 v2-medium | H20 | 0.69 | 41.4s estimate | n/a, published maximum270s | Official version table |
| LeVo2 v2-large | H20 | 0.82 | 49.2s estimate | n/a, published maximum270s | Official version table |
| ACE-Step1.5 official declaration | A100 | n/a for full chain | n/a | n/a | Prior extrapolation withdrawn; original240s/2s statement retained only as evidence |

- **MusicGen inverse convention:** [Diff-A-Riff Table2](https://arxiv.org/html/2406.08384v1#S5.T2) reports `audio/wall=0.4`, not `wall/audio=0.4`. Our normalized RTF is `1/0.4=2.5`. The [official AudioCraft implementation](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/musicgen.py) handles durations over30s with rolling continuation (default stride18s, retaining12s context). Do not call these60/300s single-window measurements. Other sizes and optimized backends can differ. This is a comparison-paper baseline, not a Meta-standard speed guarantee.
- **LeVo2 length boundary:** [pinned official README](https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/README.md#model-versions) lists max4m30s for both v2 versions, excluding initialization from RTF. Pure multiplication for300s would yield207s/246s, but neither is a supported300s inference time. At the supported270s maximum the same rate converts to186.3s/221.4s; still estimates, not measured per-length results. Continuation would need separate implementation and timing.
- **ACE configuration boundary:** no planner model was selected or timed for the original extrapolation. The official profiler can time a complete blocking generation, but the webpage claim is not linked to that profiler's receipt. It is neither a verified full-chain value nor verified TTFA. The parameter audit is2B-class Base, not4B XL; the optional1.7B planner and0.6B caption encoder are separate components. See the correction below.

No concurrent training was stopped or shared-GPU inference launched for this literature research. Cross-hardware/model-version figures are not a controlled speed leaderboard. The selected online LeVo2 version is now **v2-medium**; large remains in the source archive only.

### ACE / DiffRhythm2 Timing Boundary Correction

`inference-boundaries.json` records pinned code evidence and a **proposed, unexecuted** matched benchmark contract. The corrections are published in the table and its source foldout.

- ACE's `backend=vllm` imports **nano-vLLM**, for the optional5Hz planner only. With thinking and all CoT switches off, the planner can be skipped entirely. For codes-enabled generation,60/300s require approximately300/1500 audio codes plus metadata, so LM time cannot be silently omitted. This is separate from the always-used text encoder.
- The official profiler synchronizes around the blocking `generate_music` call, which returns completed audio and saves files. That is not TTFA. Internal `pipeline_total_time` sums only its component timers and need not equal the complete outer wall time. Neither fact proves the unspecified webpage claim included the planner, VAE and saving.
- Distilled Turbo uses approximately8 steps without CFG, whereas Base/SFT normally use50. Diff's5Hz latent with block_size10 generates2s blocks sequentially. At60s,30 blocks times31 Euler intervals times2 CFG branches gives approximately1,860 short-block model calls, plus cache updates. This is serial-call accounting, **not a FLOPs ratio against8 full-sequence ACE passes**.
- The [Diff native loader](https://github.com/ASLP-lab/DiffRhythm2/blob/7804f821b797b4f276090e1a9dcd37e97d9915d5/inference.py#L84) overrides the checkpoint's FlexAttention flag to false; the executed DiT uses SDPA and block KV cache. The paper's FlexAttention/4090/120s benchmark is a different configuration. There is no stage profile proving which overhead accounts for how many seconds, so no fabricated optimization speedup is reported.
- The fair follow-up separates Base/Turbo, LM-off/LM1.7B-on, same hardware, actual60/300s, cached versus resident models, complete CPU-waveform return, and MP3 export. Cold load/compile and TTFA are separate. No new benchmark or training was launched for this correction.

### vLLM Compatibility and Acceleration

`vllm-support.json` records eight model-specific routes. The [pinned upstream Omni implementation](https://github.com/vllm-project/vllm-omni/tree/8201f614b4c213494e276f5eecd099d2b79b0090) includes a native YuE2 pipeline; our installed older runtimes do not automatically inherit it. Muse has an [official vLLM route](https://github.com/yuhui1038/Muse/tree/main/infer), already used in this project. ACE's [backend implementation](https://github.com/ace-step/ACE-Step-1.5/blob/8ac1be141c6d0c818755e8c4306dc8f72438022c/acestep/llm_inference.py) imports **nano-vLLM**, not the upstream vLLM package; it handles the planner, not the DiT/VAE.

LeVo2's upstream integration request [#3390](https://github.com/vllm-project/vllm-omni/issues/3390) remains open at this check. No ready full-song integration was identified in the pinned official Omni registry for LeVo2, HeartMuLa, SongBloom or DiffRhythm2. MusicGen's multi-codebook delay/continuation path also needs validation or adaptation. These are bounded compatibility findings, not claims that porting is impossible. Standard Transformer layers alone are insufficient evidence of an end-to-end compatible generator.

**There is no verified matched vLLM speedup receipt yet.** The independent Gadi interactive request `181310624.gadi-pbs` remained queued during this update and was withdrawn without executing a GPU benchmark. Kaya's immediate interactive request `146358` failed because nodes were busy. No training was interrupted and no benchmark was left queued. `bench_muse_vllm.py` and `run_muse_vllm_bench.sh` are prepared but **not executed/accepted**. They use exact audio-code stopping, batch1, the same prompt/seed/sampling/decoder settings and separate initialization; both engine and codec must finish before publishing an end-to-end timing. A future validated `muse-vllm-timing.json` can populate the measured table; a warmup-only receipt cannot.

The separate **conditional**, not measured or predicted-vLLM, table uses:

`T_new = T_AR / assumed_AR_speedup + T_other`

| Model / duration | Original measured wait | If AR were2x faster | If AR were4x faster |
| --- | ---: | ---: | ---: |
| YuE2 /60s | 84.51s | 46.24s /1.83x total | 27.10s /3.12x total |
| YuE2 /300s | 271.02s | 149.14s /1.82x total | 88.19s /3.07x total |
| Muse /60s | 58.80s | 38.48s /1.53x total | 28.32s /2.08x total |
| Muse /300s | 291.70s | 190.63s /1.53x total | 140.10s /2.08x total |

Only the AR stage changes in these scenarios; native codec/NAR/transfer time is held constant. Their hypothetical infinite-AR bounds are not global vLLM-Omni bounds: Omni may also optimize NAR. These scenarios are not a2x/4x promise, nor do they establish audible-quality parity. Batch throughput, cold compilation, concurrency and single-song latency must be reported separately. In particular, the nano-vLLM bundled README's256-sequence Qwen benchmark must not be reused as a music batch1 speedup.

- `audit_weights.py`: trusted local safetensors headers and meta/mmap native checkpoints. Header SHA256 is a header identity, **not a full weight-file hash**. Raw state element counts are not final unique parameter counts.
- `audit_native.py`: native model construction and unique parameter/buffer shapes; explicit HF weight tying. `native-hf-final-counts.json` is the finalized HF receipt.
- `native-heart-counts.json`, `native-songbloom-counts.json`: native construction receipts.
- `collect_timings.py`, `timing-5090.json`, `timing-heart.json`: read-only archived timing extraction; no reward averages or training claims.
- `bench_yue2.py`, `bench_muse.py`, `bench_songbloom.py`: independent duration-controlled native end-to-end measurements. SongBloom uses the existing `container_command.py` / `container-runtime.json` wrapper, image SHA256 `84e1b3c8f8187dffb8f0d0de223fe07336bf211fad2786de2338730bb3e4cb24`, because Gadi host GLIBC cannot import its existing FlashAttention binary.
- `profile_codec_server.py`, `run_muse_bench.sh`: isolated MuCodec service on port 48890, unique loaded parameters, cleanup of only its own process.
- `yue2-timing.json`, `muse-timing.json`, `songbloom-timing.json`: complete direct timing receipts. Each records actual output duration and separate setup time. Muse includes native codec HTTP transfer/base64 overhead; its codec weights are FP32, diffusion autocasts FP16, and VAE decode is FP32. The interactive allocation was released after the finite benchmarks completed; no profiler was left running.
- Public generated metadata: `platform/site/demos/data/model-costs.json`. No audio is added by this audit; public audio remains MP3-only.

Build with `cd platform && npm run build && npm run check`. Browser and data QA: `node platform/scripts/test-model-costs.mjs` from the repository root, with Playwright installed or `PLAYWRIGHT_MODULE` pointing to it.
