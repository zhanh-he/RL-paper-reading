// Test-only records. This file is outside the public site and is never copied there.
export function amtFixture() {
  const notes = [72, 65, 60, 48].map((pitch, track) => ({ track, pitch, start: 0, end: 2 }));
  const asset = { notes, tracks: ['Soprano', 'Alto', 'Tenor', 'Bass'] };
  return {
    schema_version: 1, updated_at: '2026-10-29',
    datasets: [{ name: 'TEST ONLY', labels: 'SATB', split: 'test fixture', source: 'https://example.org/test' }], models: [], rewards: [],
    experiments: [{
      id: 'test-only', title: 'TEST FIXTURE · never publish', status: 'running', task: 'SATB fixture', model: 'Test model', method: 'GRPO', reward: 'Test reward',
      receipt: 'notes/EXPERIMENT_RECORD.md', evaluation_protocol: 'Test fixture: 50 ms onset, max(50 ms, 20%) offset, macro SATB F1.',
      samples: [{ id: 'sample-a', title: 'Test sample A', duration: 3, input: { audio: 'demos/audio/choral_synth_input.mp3' }, reference: structuredClone(asset) }, { id: 'sample-b', title: 'Test sample B', duration: 3 }],
      stages: [
        { step: 0, status: 'measured', outputs: { 'sample-a': structuredClone(asset) }, evaluation: { cohort: 'fixture', n: 2, frame: .5, onset: .4, onset_offset: .3 } },
        { step: 100, status: 'measured', outputs: { 'sample-a': { ...structuredClone(asset), notes: [{ track: 0, pitch: 74, start: 1, end: 2 }] } }, evaluation: { cohort: 'fixture', n: 2, frame: .6, onset: .5, onset_offset: .4 } },
        { step: 300, status: 'planned' },
      ],
      trace: [{ step: 1, reward: .1, kl: .01 }, { step: 50, reward: .5, kl: .02 }, { step: 100, reward: .6, kl: .03 }],
      trace_source: 'notes/EXPERIMENT_RECORD.md', trace_note: 'TEST ONLY: raw logged points',
    }],
  };
}
