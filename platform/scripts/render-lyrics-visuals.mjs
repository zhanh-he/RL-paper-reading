import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const demos = resolve(import.meta.dirname, '../site/demos');
const replays = JSON.parse(await readFile(resolve(demos, 'replays.json'), 'utf8'));
const assets = new Map();
for (const model of Object.values(replays.lyrics)) {
  for (const stage of model.stages) {
    for (const asset of [stage, stage.paired_baseline].filter(Boolean)) {
      if (asset.status === 'pending' || !asset.audio) continue;
      assets.set(asset.audio, asset);
    }
  }
}

const filters = {
  wave: 'aformat=channel_layouts=mono,showwavespic=s=1200x140:colors=0x31bba6:scale=lin:filter=peak',
};
for (const asset of assets.values()) {
  for (const kind of ['wave']) {
    const result = spawnSync('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-y', '-i', resolve(demos, asset.audio),
      '-filter_complex', filters[kind], '-frames:v', '1', resolve(demos, asset[kind]),
    ], { stdio: 'inherit' });
    if (result.status !== 0) throw new Error(`ffmpeg failed for ${asset.audio} ${kind}`);
  }
  process.stdout.write(`${asset.audio}\n`);
}
const spectra = spawnSync('uv', [
  'run', '--with', 'librosa', '--with', 'matplotlib', '--with', 'pillow',
  'python', resolve(import.meta.dirname, 'render-mel-visuals.py'),
  ...[...assets.values()].map(asset => resolve(demos, asset.audio)),
], { stdio: 'inherit' });
if (spectra.status !== 0) throw new Error('mel spectrogram rendering failed');
