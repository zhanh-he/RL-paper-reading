import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const runPath = 'music-trans/score-bench/runs/2026-10-30';
const run = path.join(root, runPath);
const load = name => JSON.parse(fs.readFileSync(path.join(run, name), 'utf8'));
const baseline = load('duet-baseline-muster.json'), after = load('duet-sft50-muster.json');
const sft = load('duet-sft50.json'), sheet = load('sheet-gtzan10.json');
const grpo = load('duet-grpo50.json'), grpoScores = load('duet-grpo50-muster.json');
if ([baseline,after,grpoScores].some(item => item.n_complete !== 8) || sheet.samples.length !== 10 || sheet.samples.some(row => row.status !== 'complete')) throw new Error('Incomplete benchmark cohort; do not publish a final mean');
const testIds = load('asap-manifest.json').samples.filter(row => row.split === 'test').map(row => row.id).sort();
for (const receipt of [baseline,after,grpoScores]) {
  if (JSON.stringify(receipt.samples.map(row => row.id).sort()) !== JSON.stringify(testIds) || receipt.samples.some(row => row.status !== 'complete')) throw new Error('Mismatched test identities or failed score pair');
  for (const metric of ['PitchER','MissRate','ExtraRate','OnsetER','OffsetER','MeanER']) {
    const mean = receipt.samples.reduce((sum,row) => sum+row.metrics[metric],0)/testIds.length;
    if (!Number.isFinite(mean) || Math.abs(mean-receipt.mean[metric]) > 1e-10) throw new Error(`Invalid complete-cohort ${metric} mean`);
  }
}
const summary = {
  date: '2026-10-30', sheet: { n: 10, beat_f1: sheet.beat_f1, downbeat_f1: sheet.downbeat_f1, precision: sheet.precision, gpu: sheet.gpu, samples: sheet.samples },
  duet: { n: 8, baseline: baseline.mean, sft50: after.mean, grpo50: grpoScores.mean,
    delta_mean_er_pp: after.mean.MeanER - baseline.mean.MeanER, grpo_delta_mean_er_pp: grpoScores.mean.MeanER - baseline.mean.MeanER,
    samples: baseline.samples.map(row => {
      const paired = after.samples.find(item => item.id === row.id);
      const rl = grpoScores.samples.find(item => item.id === row.id);
      return { id: row.id, work: row.work, before: row.metrics.MeanER, after: paired.metrics.MeanER, delta: paired.metrics.MeanER-row.metrics.MeanER,
        grpo: rl.metrics.MeanER, grpo_delta: rl.metrics.MeanER-row.metrics.MeanER };
    }) },
  sft: { optimizer_steps: sft.optimizer_steps, trainable_parameters: sft.trainable_parameters, head_l2_change: sft.head_l2_change,
    train_works: sft.alignment.filter(row => row.status === 'cached' && row.id.startsWith('train')).length,
    valid_works: sft.alignment.filter(row => row.status === 'cached' && row.id.startsWith('validation')).length,
    before: sft.before, after: sft.after, limitations: sft.limitations },
  grpo: { rollouts: grpo.trace.length, optimizer_steps: grpo.optimizer_steps, trainable_parameters: grpo.trainable_parameters,
    head_l2_change: grpo.head_l2_change, before: grpo.before, after: grpo.after, trace: grpo.trace, limitations: grpo.limitations },
  sources: ['ASAP pinned commit 8cba199e15931975542010a7ea2ff94a6fc9cbee', 'DuET commit 6d2276c6d3b289584ffe4f602bc0fe1e4740f643',
    'MUSTER fork commit 4ef1165edda1f719068c4c699bd8ab2076e4d7ec', 'GTZAN-mini a61439e86c13037011fde8e0f0743ec55c50bce3', 'Beat This annotations v1.0 c3c47fd37d3074d9f8119f18bbf460f909609f22'],
};
fs.writeFileSync(path.join(run, 'summary.json'), JSON.stringify(summary, null, 2)+'\n');
const studyFile = path.join(root, 'platform/data/amt/score-study.json');
const study = JSON.parse(fs.readFileSync(studyFile));
study.results = summary;
study.experiments[0].status = `Completed 10/10 · beat ${(sheet.beat_f1*100).toFixed(2)}%, downbeat ${(sheet.downbeat_f1*100).toFixed(2)}% · 5090 BF16`;
study.experiments[1].status = `Completed 8/8 · MUSTER Mean ER ${baseline.mean.MeanER.toFixed(3)}%`;
study.experiments[2].status = `Completed 50 updates · ${summary.sft.train_works} usable train works / ${sft.before.train.n_notes} notes; test ER ${after.mean.MeanER.toFixed(3)}% (higher is worse)`;
study.experiments[3] = { id: 'duet-duration-grpo', title: 'E4 · Duration-head note-action GRPO',
  data: 'Same 15 usable train works / 1651 labels, 4 valid and 8 full test works as SFT',
  protocol: '8 sampled duration actions per note; exact forward KL to frozen head, beta .01, lr 1e-4; frozen baseline-history contexts',
  action: '50 rollout batches / 40 optimizer updates; zero-variance groups give no policy gradient. Not full-sequence GRPO; SFT has 50 updates, so update budgets differ.',
  status: `Completed 8/8 test · Mean ER ${grpoScores.mean.MeanER.toFixed(3)}%; duration-only did not improve complete-score error` };
study.models.find(model => model.name === 'DuET').priority = 'Measured · frozen + duration-head SFT / note-action GRPO';
study.rewards[0].status = 'Measured: SFT and singleton-duration GRPO; full-sequence RL not run';
fs.writeFileSync(studyFile, JSON.stringify(study, null, 2)+'\n');
const manifestFile = path.join(root, 'platform/data/amt/experiments.json');
const manifest = JSON.parse(fs.readFileSync(manifestFile));
const examples = JSON.parse(fs.readFileSync(path.join(root, 'platform/site/amt/demos/score-examples/examples.json')));
const benchmarkSamples = examples.map(item => ({ id: item.id, title: item.work,
  duration: item.id === 'test-02' ? 155.6156249999992 : 278.94140625000136,
  input: { midi: `amt/demos/score-examples/${item.id}/performance.mid` },
  reference: { score_image: `amt/demos/score-examples/${item.id}/reference.svg`, musicxml: `amt/demos/score-examples/${item.id}/reference-full.musicxml` },
  note: `${item.note} ASAP (Foscarin et al.), CC BY-NC-SA 4.0. Score images rendered by Verovio 6.3.0; no acoustic input or audio quality score is implied by this MIDI-to-score experiment.` }));
const scoreMetrics = ['PitchER', 'MissRate', 'ExtraRate', 'OnsetER', 'OffsetER', 'MeanER'];
const conditions = [ ['baseline','Frozen DuET',baseline], ['sft50','Duration-head SFT · 50 updates',after], ['grpo50','Duration-head GRPO · 50 rollouts / 40 updates',grpoScores] ].map(([id,label,receipt]) => ({
  id, label, status: 'measured', evaluation: { cohort: 'asap-acpas-test8-20261030', n: 8, values: Object.fromEntries(scoreMetrics.map(metric => [metric, receipt.mean[metric]])) },
  outputs: Object.fromEntries(benchmarkSamples.map(sample => [sample.id, { score_image: `amt/demos/score-examples/${sample.id}/${id}.svg`, musicxml: `amt/demos/score-examples/${sample.id}/${id}-full.musicxml` }]))
}));
const duetRun = { id: 'asap-duet-benchmark', title: 'ASAP · 8 real works · Frozen / SFT / note-action GRPO', kind: 'evaluation', status: 'measured',
  task: 'Performance MIDI → musical score', model: 'Wayne42/DuET · official epoch83 bundle', method: 'Duration-head adaptation: frozen vs SFT vs note-action GRPO', reward: 'SFT: duration CE; GRPO: exact duration match + 0.01 forward KL',
  receipt: `${runPath}/README.md`, evaluation_protocol: `${baseline.protocol} ACPAS source train/valid/test: 16/4/8 selected works. 15 train works / 1651 confident note labels used. Duration-head only, frozen baseline-history contexts, 176985 trainable parameters. SFT: 50 optimizer updates. GRPO: 50 rollout batches, group 8 per note, 40 effective optimizer updates; update budgets differ. This is conditional note-action GRPO, not full autoregressive sequence-policy training. Test uses original full-piece decoder. Not the complete paper split; base-checkpoint training overlap unverified. Error units are percent, lower is better.`,
  stages: [], trace: [], samples: benchmarkSamples, conditions, metric_definitions: scoreMetrics.map(id => ({ id, label: `${id} (%) ↓`, unit: 'number' })) };
const sheetRun = { id: 'sheet-gtzan-benchmark', title: 'SheetSage2 · real GTZAN 10-track benchmark', kind: 'evaluation', status: 'measured', task: 'Audio → lead-sheet / beat and downbeat component',
  model: 'm-a-p/SheetSage2 · b83c3d79 release', method: 'Frozen inference · paper preset · RTX 5090 BF16', reward: 'Evaluation only; no optimizer updates', receipt: `${runPath}/README.md`,
  evaluation_protocol: `${sheet.protocol} ${sheet.scope} Audio and reference hashes are fixed in the GTZAN manifest. First attempt failed on missing pretty_midi; dependency installed in isolated environment, all ten rerun. Duplicate pending Kaya job 256819 cancelled after 5090 completion. No GTZAN audio republished.`,
  stages: [], trace: [], samples: [], metric_definitions: [{ id: 'beat', label: 'Beat F1 · 70ms', unit: 'fraction' }, { id: 'downbeat', label: 'Downbeat F1 · 70ms', unit: 'fraction' }],
  conditions: [{ id: 'paper', label: 'Frozen · paper preset', status: 'measured', evaluation: { cohort: 'gtzan-ten-genres-20261030', n: 10, values: { beat: sheet.beat_f1, downbeat: sheet.downbeat_f1 } } }] };
manifest.experiments = [duetRun, sheetRun, ...manifest.experiments.filter(item => ![sheetRun.id, duetRun.id].includes(item.id))];
manifest.updated_at = '2026-10-30';
fs.writeFileSync(manifestFile, JSON.stringify(manifest,null,2)+'\n');
console.log('Published actual complete-cohort score benchmarks and paired score examples; no fabricated training reward/KL.');
