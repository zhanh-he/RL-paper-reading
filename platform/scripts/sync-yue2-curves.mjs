import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const runs = {};
const dimensions = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'];
for (const [arm, directory] of [
  ['yue2', '2026-09-30-yue2-longrun'],
  ['yue2_stress', '2026-09-30-yue2-lr-stress'],
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
    const receiptDirectory = step <= 1 ? '2026-09-30-yue2-longrun' : directory;
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
    heldout.push({ step, ...scores });
  }
  runs[arm] = { points, heldout, updates: 99, kl_status: 'not_recorded' };
}
const output = resolve(root, 'platform/site/demos/yue2-training-curves.json');
const contents = `${JSON.stringify({ status: 'measured_training_reward_only', bin_size: 5, runs }, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== contents) throw new Error('YuE2 public curve data is stale');
} else {
  await writeFile(output, contents);
}
