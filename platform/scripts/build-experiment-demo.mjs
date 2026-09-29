import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const tryReadJson = async (path) => {
  try { return await readJson(path); } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
};
const output = resolve(root, 'platform/site/demos/results.json');
const musePath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-muse/receipt.json';
const muse = await tryReadJson(musePath);

const result = {
  generated_from: [
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed30.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed31.json',
    'music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json',
    'music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json',
    ...(muse ? [musePath] : []),
  ],
  choral: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json'),
  choral_seed_repeats: await Promise.all([30, 31].map((seed) => readJson(`music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed${seed}.json`))).then((receipts) => receipts.map((receipt) => ({ seed: receipt.seed, frame_f1: receipt.after_test.frame.f1, onset_f1: receipt.after_test.onset.f1, offset_f1: receipt.after_test.offset.f1 }))),
  reward_audit: await readJson('music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json'),
  choral_note: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json'),
  songeval_audit: await readJson('music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json'),
  yue2: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json'),
  yue2_verification: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json'),
  muse,
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
