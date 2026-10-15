# Model Parameters and Generation Cost

Audited 2026-10-15. [Online tables](https://zhanh-he.github.io/RL-paper-reading/demos/#model-costs) / [downloadable JSON](https://zhanh-he.github.io/RL-paper-reading/demos/data/model-costs.json).

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
2. **DiffRhythm2:** recover all 30 completed frozen-baseline validation timings on RTX5090, native 32-step ODE / CFG2 / FP16. Includes style encoding, transfers, decoding, and private lossless save. Hash bindings were checked against all scored audio. One output ended at 31.8 seconds; the displayed 60-second median and 300-second extrapolation use only the 29 actual 60-second outputs. All 30 raw timings remain available. 300 seconds is a **linear estimate**, not measured.
3. **HeartMuLa:** all 72 archived 60-second-budget H200 rollouts from job `181262219.gadi-pbs` on `gadi-gpu-h200-0029`. BF16 generator / FP32 HeartCodec. Includes action/log-prob collection and policy-record save, so explicitly a **rollout proxy**, not pure inference. Three outputs ended at 48.8/56.32/56.96 seconds; only the 69 outputs within 60 +/- 0.1 seconds contribute to the 60-second median and extrapolation. All 72 remain archived. 300 seconds is only arithmetic extrapolation beyond the default 240-second limit, not validated native output.
4. **SongBloom:** same independent A100 allocation, native BF16 / 36 Euler steps / CFG1.5, one CMI valid audio+lyrics condition, 10-second reference audio, frozen full-150s checkpoint. Actual 60.16-second output took 93.081 seconds (RTF 1.54723). Includes condition read/resampling/encoding, AR/NAR and VAE; excludes model initialization and file saving. First call, no kernel warmup. Direct host execution hit a GLIBC/FlashAttention compatibility error; reran successfully in the existing hash-verified Bookworm container, without altering model dependencies or training jobs. See `songbloom-timing.json`. The [paper Table 2](https://arxiv.org/html/2506.07634v1#S6) RTF 1.649 has unspecified timing hardware and is only background reference, not the measured row. This checkpoint supports up to 150 seconds; 300-second native output is **n/a**.
5. **ACE-Step1.5 base:** no valid local 60/300-second end-to-end receipt yet; report **n/a**. Staging a separate Gadi copy was stopped because transfer was too slow; partial assets are marked `.incomplete` and were never loaded. The [official project page](https://ace-step.github.io/ace-step-v1.5.github.io/) separately reports a four-minute song in two seconds on A100. Exact configuration and timing scope are not disclosed: this is a separate author-reference row, not a prediction for our base/50-step model. Its 60/300-second figures are linear estimates only.

No training was stopped or moved. On 5090, only metadata/CPU audits and existing receipts were read; new GPU timing ran in the separate A100 allocation. A Kaya immediate interactive request failed because nodes were busy. No scheduled monitoring was created.

## Reproduction and Evidence

### Published Speed Research: MusicGen / LeVo2 / ACE-Step1.5

Added 2026-10-15, **not newly measured on our GPUs**. Structured facts, original RTF definitions, pinned README sources and normalized calculations are in `published-speed-sources.json`. They extend the latency table only, not the six locally audited parameter/LoRA rows.

| Version | Hardware | Wall/audio RTF | Wait for 60s | Wait for 300s | Evidence |
| --- | --- | ---: | --- | --- | --- |
| MusicGen 3.3B /32kHz mono | A100 + EPYC7443P; VRAM unspecified | 2.5 | 2m30s estimate | 12m30s estimate | Diff-A-Riff Table2, 95s output benchmark |
| LeVo2 v2-medium | H20 | 0.69 | 41.4s estimate | n/a, published maximum270s | Official version table |
| LeVo2 v2-large | H20 | 0.82 | 49.2s estimate | n/a, published maximum270s | Official version table |
| ACE-Step1.5 official fast configuration | A100 | 0.00833 derived | 0.50s estimate | 2.50s estimate | Official240s/2s graphic, exact pipeline boundary undisclosed |

- **MusicGen inverse convention:** [Diff-A-Riff Table2](https://arxiv.org/html/2406.08384v1#S5.T2) reports `audio/wall=0.4`, not `wall/audio=0.4`. Our normalized RTF is `1/0.4=2.5`. The [official AudioCraft implementation](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/musicgen.py) handles durations over30s with rolling continuation (default stride18s, retaining12s context). Do not call these60/300s single-window measurements. Other sizes and optimized backends can differ. This is a comparison-paper baseline, not a Meta-standard speed guarantee.
- **LeVo2 length boundary:** [pinned official README](https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/README.md#model-versions) lists max4m30s for both v2 versions, excluding initialization from RTF. Pure multiplication for300s would yield207s/246s, but neither is a supported300s inference time. At the supported270s maximum the same rate converts to186.3s/221.4s; still estimates, not measured per-length results. Continuation would need separate implementation and timing.
- **ACE configuration boundary:** [official project page](https://ace-step.github.io/ace-step-v1.5.github.io/) gives240s/2s/A100. The [pinned model zoo](https://github.com/ace-step/ACE-Step-1.5/blob/8ac1be141c6d0c818755e8c4306dc8f72438022c/README.md) distinguishes Base/SFT50steps and Turbo8steps. The fast figure cannot fill the Base timing gap. Planner-on/off and diffusion steps affect total wait; neither a2-second SLA nor multiplying latency by50/8 is justified. The benchmark guide's example output is illustrative, not a measured result.

No concurrent training was stopped or shared-GPU inference launched for this literature research. Cross-hardware/model-version figures are not a controlled speed leaderboard.

- `audit_weights.py`: trusted local safetensors headers and meta/mmap native checkpoints. Header SHA256 is a header identity, **not a full weight-file hash**. Raw state element counts are not final unique parameter counts.
- `audit_native.py`: native model construction and unique parameter/buffer shapes; explicit HF weight tying. `native-hf-final-counts.json` is the finalized HF receipt.
- `native-heart-counts.json`, `native-songbloom-counts.json`: native construction receipts.
- `collect_timings.py`, `timing-5090.json`, `timing-heart.json`: read-only archived timing extraction; no reward averages or training claims.
- `bench_yue2.py`, `bench_muse.py`, `bench_songbloom.py`: independent duration-controlled native end-to-end measurements. SongBloom uses the existing `container_command.py` / `container-runtime.json` wrapper, image SHA256 `84e1b3c8f8187dffb8f0d0de223fe07336bf211fad2786de2338730bb3e4cb24`, because Gadi host GLIBC cannot import its existing FlashAttention binary.
- `profile_codec_server.py`, `run_muse_bench.sh`: isolated MuCodec service on port 48890, unique loaded parameters, cleanup of only its own process.
- `yue2-timing.json`, `muse-timing.json`, `songbloom-timing.json`: complete direct timing receipts. Each records actual output duration and separate setup time. Muse includes native codec HTTP transfer/base64 overhead; its codec weights are FP32, diffusion autocasts FP16, and VAE decode is FP32. The interactive allocation was released after the finite benchmarks completed; no profiler was left running.
- Public generated metadata: `platform/site/demos/data/model-costs.json`. No audio is added by this audit; public audio remains MP3-only.

Build with `cd platform && npm run build && npm run check`. Browser and data QA: `node platform/scripts/test-model-costs.mjs` from the repository root, with Playwright installed or `PLAYWRIGHT_MODULE` pointing to it.
