import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const base = resolve(root, 'music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head');
const read = (path) => JSON.parse(readFileSync(path));
const write = (path, value) => writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
const aggregatePath = resolve(base, 'aggregate.json');
const replayPath = resolve(base, 'public-synthetic-replay.json');
const notesPath = resolve(base, 'public-synthetic-notes.json');
const aggregate = read(aggregatePath);
const replay = read(replayPath);
const notes = read(notesPath);
const earlyReplays = Object.fromEntries([1, 100].map((step) =>
  [step, read(resolve(base, `early-steps/public-synthetic-replay-${step}.json`))]));
const earlyNotes = read(resolve(base, 'early-steps/public-synthetic-notes.json'));
const lastReplay = read(resolve(base, 'early-steps/public-synthetic-replay-1000.json'));
const lastNotes = read(resolve(base, 'early-steps/public-synthetic-notes-1000.json'));

for (const arm of ['onset', 'onset_offset', 'frame', 'coverage', 'continuity']) {
  for (const step of [1, 100]) {
    const key = `arm_${arm}_${step}`;
    const receipt = read(resolve(base, `early-steps/${key}.json`));
    const publicResult = earlyReplays[step].steps[key];
    const publicNotes = earlyNotes[key];
    if (receipt.reward_arm !== arm || receipt.optimizer_step !== step || receipt.test_segments !== 8 ||
        receipt.updated_steps < 0 || receipt.updated_steps > step ||
        !['frame', 'onset', 'onset_offset'].every((metric) => Number.isFinite(receipt.metrics?.macro?.[metric])) ||
        publicResult?.midi !== `choral_event_${key}.mid` ||
        publicResult.note_count !== publicNotes?.length ||
        publicNotes.some((note) => ![0, 1, 2, 3].includes(note.voice) ||
          !Number.isInteger(note.pitch) || note.start < 0 || note.end <= note.start || note.end > 10.3)) {
      throw new Error(`Invalid early Choral stage: ${key}`);
    }
    aggregate.arms[arm].milestones[String(step)] = receipt;
    replay.steps[key] = publicResult;
    notes[key] = publicNotes;
  }
  const key = `arm_${arm}_1000`;
  const publicResult = lastReplay.steps[key];
  const publicNotes = lastNotes[key];
  if (!aggregate.arms[arm].milestones['1000'] ||
      publicResult?.midi !== `choral_event_${key}.mid` ||
      publicResult.note_count !== publicNotes?.length ||
      publicNotes.some((note) => ![0, 1, 2, 3].includes(note.voice) ||
        !Number.isInteger(note.pitch) || note.start < 0 || note.end <= note.start || note.end > 10.3)) {
    throw new Error(`Invalid 1000-step Choral replay: ${key}`);
  }
  replay.steps[key] = publicResult;
  notes[key] = publicNotes;
}

write(aggregatePath, aggregate);
write(replayPath, replay);
write(notesPath, notes);
