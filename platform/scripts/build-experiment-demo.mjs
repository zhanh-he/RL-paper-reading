import { access, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const output = resolve(root, 'platform/site/demos/results.json');
const musePath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-muse/receipt.json';
const muse = await readJson(musePath);
const replaysPath = 'platform/site/demos/replays.json';
const replays = await readJson(replaysPath);
const choralDemoPath = 'music-trans/choral-singing/rl/grpo/runs/2026-09-30-public-synthetic';
for (const name of ['input', 'reference', 'baseline', 'grpo']) {
  await access(resolve(root, `platform/site/demos/audio/choral_synth_${name}.wav`));
  if (name !== 'input') await access(resolve(root, `platform/site/demos/midi/choral_synth_${name}.mid`));
}
await access(resolve(root, 'platform/site/demos/audio/muscriptor_synth_baseline.wav'));
await access(resolve(root, 'platform/site/demos/midi/muscriptor_synth_baseline.mid'));
const replayReceipts = new Set();
const cases = [...Object.values(replays.lyrics), replays.vocal, replays.choral];
for (const entry of cases) {
  const steps = entry.stages.map((stage) => stage.step);
  if (new Set(steps).size !== steps.length || steps.some((step, index) => !Number.isInteger(step) || step < 0 || (index > 0 && step <= steps[index - 1]))) {
    throw new Error('Replay stages need distinct, ascending nonnegative integer step counts');
  }
  for (const stage of [entry.input, entry.reference, ...entry.stages, ...(entry.outputs || [])].filter(Boolean)) {
    if (['pending', 'running'].includes(stage.status) && (stage.audio || stage.wave || stage.spectrum)) {
      throw new Error(`Unmeasured step ${stage.step} must not claim public media`);
    }
    if (stage.status === 'measured' && (!stage.audio || !stage.wave || !stage.spectrum)) {
      throw new Error(`Measured step ${stage.step} needs audio, waveform and spectrogram`);
    }
    for (const key of ['audio', 'wave', 'spectrum']) {
      if (!stage[key]) continue;
      if (!/^(audio|visuals)\/[a-z0-9_-]+\.(flac|wav|png)$/.test(stage[key])) {
        throw new Error(`Invalid public replay asset: ${stage[key]}`);
      }
      await access(resolve(root, 'platform/site/demos', stage[key]));
    }
  }
}
for (const entry of Object.values(replays.lyrics)) {
  for (const stage of entry.stages.filter((item) => item.status === 'measured')) {
    if (!/^music-gen\/lyrics2song\/rl\/grpo\/runs\/[a-z0-9-]+\/(?:step_[0-9]{6}\/)?receipt\.json$/.test(stage.receipt) || !['before', 'after', 'heldout:0'].includes(stage.receipt_key)) {
      throw new Error(`Measured lyrics step ${stage.step} needs a valid receipt and before/after key`);
    }
    const receipt = await readJson(stage.receipt);
    const metrics = stage.receipt_key === 'heldout:0' ? receipt.heldout?.[0] : receipt[stage.receipt_key];
    if (!['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].every((key) => Number.isFinite(metrics?.reward?.[key])) ||
      !['peak', 'rms', 'near_full_scale_fraction'].every((key) => Number.isFinite(metrics?.signal?.[key]))) {
      throw new Error(`Measured lyrics step ${stage.step} has incomplete reward or signal metrics`);
    }
    stage.metrics = metrics;
    if (Number.isFinite(receipt.mean_reward)) stage.mean_reward = receipt.mean_reward;
    replayReceipts.add(stage.receipt);
  }
}

const result = {
  generated_from: [...new Set([
    replaysPath,
    `${choralDemoPath}/notes.json`,
    `${choralDemoPath}/receipt.json`,
    'music-trans/multi-inst/models/muscriptor/muscriptor_receipt.json',
    'music-gen/vocal2accomp/models/anyaccomp/receipt.json',
    ...replayReceipts,
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed30.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed31.json',
    'music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json',
    'music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json',
    musePath,
  ])],
  choral: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json'),
  choral_seed_repeats: await Promise.all([30, 31].map((seed) => readJson(`music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed${seed}.json`))).then((receipts) => receipts.map((receipt) => ({ seed: receipt.seed, frame_f1: receipt.after_test.frame.f1, onset_f1: receipt.after_test.onset.f1, offset_f1: receipt.after_test.offset.f1 }))),
  reward_audit: await readJson('music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json'),
  choral_note: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json'),
  choral_demo: {
    notes: await readJson(`${choralDemoPath}/notes.json`),
    receipt: await readJson(`${choralDemoPath}/receipt.json`),
    muscriptor: await readJson('music-trans/multi-inst/models/muscriptor/muscriptor_receipt.json'),
  },
  songeval_audit: await readJson('music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json'),
  vocal_reward: await readJson('music-gen/vocal2accomp/models/anyaccomp/receipt.json'),
  yue2: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json'),
  yue2_verification: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json'),
  muse,
  musecritic: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-30-musecritic-heldout/receipt.json'),
  replays,
};
const serialized = `${JSON.stringify(result, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== serialized) {
    throw new Error('Experiment demo data is stale. Run npm --prefix platform run build.');
  }
} else {
  await writeFile(output, serialized);
  console.log('Updated experiment demo data');
}
