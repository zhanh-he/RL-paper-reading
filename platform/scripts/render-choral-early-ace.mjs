import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const cli = '/Applications/ACE Studio.app/Contents/Helpers/acestudio-cli';
const root = resolve(import.meta.dirname, '../..');
const template = resolve(root, '../ACE-Choral-Event-0001/ACE-Choral-Event-0001.acep');
const notes = JSON.parse(readFileSync(resolve(root,
  'music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head/public-synthetic-notes.json')));
const arms = ['onset', 'onset_offset', 'frame', 'coverage', 'continuity'];
const stages = process.argv.slice(2).length ? process.argv.slice(2) :
  [1, 100].flatMap((step) => arms.map((arm) => `arm_${arm}_${step}`));

function call(...args) {
  return execFileSync(cli, args, { encoding: 'utf8', timeout: 240_000, maxBuffer: 8 * 1024 * 1024 });
}

function singingNotes(stage, voice) {
  const selected = notes[stage].filter((note) => note.voice === voice)
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  const output = [];
  for (let index = 0; index < selected.length; index++) {
    const note = selected[index];
    const pos = Math.round(note.start * 960);
    const next = selected[index + 1];
    const limit = next ? Math.round(next.start * 960) - 1 : Infinity;
    const end = Math.min(Math.round(note.end * 960), limit);
    if (end > pos) output.push({ pos, dur: end - pos, pitch: note.pitch, lyric: 'la' });
  }
  return output;
}

for (const stage of stages) {
  if (!/^arm_(onset|onset_offset|frame|coverage|continuity)_(1|100)$/.test(stage) || !notes[stage]) {
    throw new Error(`Unknown early Choral stage: ${stage}`);
  }
  call('project', 'open', template);
  for (let voice = 0; voice < 4; voice++) {
    const clip = JSON.parse(call('clip', 'list', '--track-index', String(voice), '--json')).clips[0];
    const current = JSON.parse(call('clip', 'note-content', '--track-index', String(voice),
      '--clip-index', '0', '--json'));
    call('clip', 'replace-content', '--clip-uuid', clip.clipUuid,
      '--if-match', current.fingerprint, '--notes', JSON.stringify(singingNotes(stage, voice)));
  }
  const project = resolve(root, `../ACE-Choral-Early-${stage}-la.acep`);
  call('project', 'save-as', project);
  const audio = resolve(root, `platform/site/demos/audio/choral_event_${stage}.wav`);
  call('export', 'audio', '--path', audio, '--to', '11.04s', '--sample-rate', '44100',
    '--bit-depth', '16', '--wait');
  process.stdout.write(`${stage}: ${audio}\n`);
}
