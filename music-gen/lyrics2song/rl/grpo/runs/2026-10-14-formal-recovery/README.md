# Formal Muse Recovery, 2026-10-14

Five aligned-LR arms reached step 270: SongEval 2e-5 / 1e-4 / 3e-4 and
MuseCritic 2e-5 / 1e-4. Recovered JSON files verify the 270-point reward/KL trace,
available adapter checkpoints and rollout audio hashes. Explicit rejected outputs
keep their recorded zero reward and rejection receipt, not an invented audio hash.
These are **training receipts**, not validation/test results or audible improvements.
Independent Muse inference remains paused; no new listening audio is fabricated.

MuseCritic 3e-4 stopped at logged step 200; its last saved state is step 180.
MuCodec's padding guard compared token lengths against 128 latent overlap frames
instead of 256 overlap tokens. At 1664 tokens the last window was 896 tokens
(448 latent frames), incompatible with the 512-frame input. The narrow
[patch](../../patches/mucodec-window-alignment.patch) aligns all windows to 1024
tokens. A 1..8192 length regression passes; an actual 1664-token GPU decode
regression runs within the resumed allocation before optimizer updates.

Resume job **181197557.gadi-pbs**, iv96, priority 1023, 4 hours, restores
checkpoint-180 adapter/optimizer/scheduler/RNG using ms-swift's
`--resume_from_checkpoint` with `--resume_only_model false`. Total max steps stays
270 (90 new steps). It writes a separate tagged recovery run and leaves all old
logs/checkpoints untouched. Submission is not successful continuation; on recovery
verify the regression receipt, restored step, new 181..270 trace and final adapter.
The collector joins only original 1..180 with recovered 181..270, discarding
superseded original updates 181..200 rather than counting them twice.
Trainer RNG state is restored; freshly started codec/vLLM service processes are
not claimed to reproduce their old sampling streams bit-for-bit.

Q3O is deferred by user request. Existing Gadi evaluation scripts now skip its
worker by default; frozen weights and private archived MP3s remain available for
later authorization. SongBench/SongEval scoring continues over the full cohorts.

Both wa66 and iv96 evaluations use the same gpuhopper-exec pool, not independent
GPU inventories. At the inspected snapshot, all 13 evaluation jobs were priority
1023 and queued for insufficient GPUs. They entered the queue at 2026-10-13
13:55:47 AWST. `qstat -T` supplied no estimated start. No numerical queue ETA is
asserted; the 4-hour resume may backfill differently from 16-hour evaluation jobs,
but that is a possibility, not a start-time promise.
