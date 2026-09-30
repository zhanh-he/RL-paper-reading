import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const runs = {};
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
  runs[arm] = { points, updates: 99, kl_status: 'not_recorded' };
}
const output = resolve(root, 'platform/site/demos/yue2-training-curves.json');
const contents = `${JSON.stringify({ status: 'measured_training_reward_only', bin_size: 5, runs }, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== contents) throw new Error('YuE2 public curve data is stale');
} else {
  await writeFile(output, contents);
}
