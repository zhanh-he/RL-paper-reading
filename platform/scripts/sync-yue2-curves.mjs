import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const runs = {};
const dimensions = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'];
const oldKl = JSON.parse(await readFile(resolve(root,
  'music-gen/lyrics2song/rl/grpo/runs/2026-09-30-yue2-kl-audit/kl.json'), 'utf8'));
const criticKl = JSON.parse(await readFile(resolve(root,
  'music-gen/lyrics2song/rl/grpo/runs/2026-10-01-yue2-musecritic-pcm24/kl.json'), 'utf8'));
if ([oldKl, criticKl].some((audit) => audit.status !== 'offline_fixed_reference_conditional_kl' ||
  audit.not_training_kl !== true || audit.identity_kl.some((value) => Math.abs(value) > 1e-5))) {
  throw new Error('YuE2 fixed-reference KL audit failed identity or provenance checks');
}
for (const [arm, directory, klAudit, klArm, baselineDirectory] of [
  ['yue2', '2026-09-30-yue2-longrun', oldKl, '2e-5', '2026-09-30-yue2-longrun'],
  ['yue2_stress', '2026-09-30-yue2-lr-stress', oldKl, '1e-4', '2026-09-30-yue2-longrun'],
  ['yue2_lr1e3', '2026-09-30-yue2-high-lr/lr1e-3', oldKl, '1e-3', '2026-09-30-yue2-longrun'],
  ['yue2_lr1e2', '2026-09-30-yue2-high-lr/lr1e-2', oldKl, '1e-2', '2026-09-30-yue2-longrun'],
  ['yue2_musecritic', '2026-10-01-yue2-musecritic-pcm24', criticKl, 'MuseCritic', '2026-10-01-yue2-musecritic-pcm24'],
]) {
  const source = resolve(root, `music-gen/lyrics2song/rl/grpo/runs/${directory}/steps.jsonl`);
  const rows = (await readFile(source, 'utf8')).trim().split('\n').map((line) => JSON.parse(line))
    .filter((row) => row.optimizer_step >= 2 && row.optimizer_step <= 100);
  if (rows.length !== 99 || rows.some((row, index) => row.optimizer_step !== index + 2 ||
    row.rewards.length !== 2 || row.rewards.some((reward) => !Number.isFinite(reward)))) {
    throw new Error(`Incomplete YuE2 training trace: ${arm}`);
  }
  const points = [];
  for (let end = 5; end <= 100; end += 5) {
    const window = rows.filter((row) => row.optimizer_step > end - 5 && row.optimizer_step <= end);
    const rewards = window.flatMap((row) => row.rewards);
    points.push({ step: end, reward: rewards.reduce((sum, reward) => sum + reward, 0) / rewards.length,
      updates: window.length });
  }
  const heldout = [];
  for (const step of [0, 1, 5, 25, 50, 100]) {
    const receiptDirectory = step <= 1 ? baselineDirectory : directory;
    const receipt = JSON.parse(await readFile(resolve(root,
      `music-gen/lyrics2song/rl/grpo/runs/${receiptDirectory}/step_${String(step).padStart(6, '0')}/receipt.json`), 'utf8'));
    const songs = receipt.heldout;
    if (receipt.optimizer_step !== step || songs?.length !== 3 ||
      songs.some((song, index) => song.seed !== 5101 + index ||
        dimensions.some((dimension) => !Number.isFinite(song.reward?.[dimension])))) {
      throw new Error(`Incomplete YuE2 held-out dimensions: ${arm} step ${step}`);
    }
    const scores = Object.fromEntries(dimensions.map((dimension) =>
      [dimension, songs.reduce((sum, song) => sum + song.reward[dimension], 0) / songs.length]));
    const mean = dimensions.reduce((sum, dimension) => sum + scores[dimension], 0) / dimensions.length;
    if (Math.abs(mean - receipt.mean_reward) > 1e-6) {
      throw new Error(`Inconsistent YuE2 held-out mean: ${arm} step ${step}`);
    }
    heldout.push({ step, mean, ...scores });
  }
  const kl = [1, 5, 25, 50, 100].map((step) => ({ step, kl: klAudit.arms[klArm]?.[String(step)]?.kl }));
  if (kl.some((point) => !Number.isFinite(point.kl))) throw new Error(`Missing audited KL: ${arm}`);
  runs[arm] = { points, heldout, kl, updates: 99,
    reward_model: arm === 'yue2_musecritic' ? 'MuseCritic' : 'SongEval',
    kl_status: 'offline_fixed_reference_conditional_kl' };
}
const output = resolve(root, 'platform/site/demos/yue2-training-curves.json');
const contents = `${JSON.stringify({ status: 'measured_training_reward_and_offline_kl', bin_size: 5, runs }, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== contents) throw new Error('YuE2 public curve data is stale');
} else {
  await writeFile(output, contents);
}
