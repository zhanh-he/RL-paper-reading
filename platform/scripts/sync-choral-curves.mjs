import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const source = resolve(root, 'music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head/training-curves.json');
const output = resolve(root, 'platform/site/demos/choral-training-curves.json');
const aggregate = JSON.parse(await readFile(resolve(root,
  'music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head/aggregate.json'), 'utf8'));
const contents = await readFile(source, 'utf8');
const data = JSON.parse(contents);
const arms = ['combined', 'onset', 'onset_offset', 'frame', 'coverage', 'continuity'];
if (data.status !== 'measured_training_trace_window_means' || data.bin_size !== 25 ||
    Object.keys(data.runs).sort().join(',') !== [...arms].sort().join(',') ||
    arms.some((arm) => data.runs[arm].steps !== 1000 ||
      data.runs[arm].updated_steps !== aggregate.arms[arm].milestones['1000'].updated_steps ||
      data.runs[arm].first_step?.step !== 1 ||
      (arm === 'combined' && (!Number.isFinite(data.runs[arm].first_step.reward) ||
        !Number.isFinite(data.runs[arm].first_step.kl))) ||
      data.runs[arm].points.length !== 40 ||
      data.runs[arm].points.reduce((sum, point) => sum + point.updates, 0) !== data.runs[arm].updated_steps ||
      data.runs[arm].points.some((point, index) => point.step !== (index + 1) * 25 ||
        !Number.isInteger(point.updates) || point.updates < 0 || point.updates > 25 ||
        (point.updates > 0 && (!Number.isFinite(point.reward) || !Number.isFinite(point.kl)))))) {
  throw new Error('Invalid ChoralGRPO training-curve export');
}
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== contents) throw new Error('ChoralGRPO public curve data is stale');
} else {
  await writeFile(output, contents);
}
