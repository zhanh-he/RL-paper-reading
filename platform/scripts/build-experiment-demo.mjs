import { access, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const output = resolve(root, 'platform/site/demos/results.json');
const musePath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-30-muse-shared-songeval/step_000001/receipt.json';
const muse = await readJson(musePath);
const sharedMusePath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-30-muse-shared-replay/receipt.json';
const sharedMuse = await readJson(sharedMusePath);
const replaysPath = 'platform/site/demos/replays.json';
const replays = await readJson(replaysPath);
const yue2KlPath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-30-yue2-kl-audit/kl.json';
const criticKlPath = 'music-gen/lyrics2song/rl/grpo/runs/2026-10-01-yue2-musecritic-pcm24/kl.json';
const yue2Kl = await readJson(yue2KlPath);
const criticKl = await readJson(criticKlPath);
const klArms = { yue2: '2e-5', yue2_stress: '1e-4', yue2_lr1e3: '1e-3', yue2_lr1e2: '1e-2', yue2_musecritic: 'MuseCritic' };
if (replays.lyrics.muse.style !== replays.lyrics.yue2.style ||
    replays.lyrics.muse.lyrics !== replays.lyrics.yue2.lyrics ||
    replays.lyrics.muse.seed !== replays.lyrics.yue2.seed ||
    replays.lyrics.musecritic.style !== replays.lyrics.yue2.style ||
    replays.lyrics.musecritic.lyrics !== replays.lyrics.yue2.lyrics ||
    replays.lyrics.musecritic.seed !== replays.lyrics.yue2.seed ||
    muse.heldout_prompt !== sharedMuse.prompt ||
    muse.heldout_seed !== sharedMuse.seed ||
    replays.lyrics.muse.stages[0].audio !== replays.lyrics.musecritic.stages[0].audio) {
  throw new Error('Muse arms must share YuE2 prompt metadata and the same frozen Muse replay');
}
const choralDemoPath = 'music-trans/choral-singing/rl/grpo/runs/2026-09-30-public-synthetic';
const eventPath = 'music-trans/choral-singing/rl/grpo/runs/2026-09-30-event-head';
const muscriptorSmokePath = 'music-trans/multi-inst/rl/grpo/runs/2026-09-30-muscriptor-medium-smoke';
const eventReceipt = await readJson(`${eventPath}/public-synthetic-replay.json`);
const eventNotes = await readJson(`${eventPath}/public-synthetic-notes.json`);
const muscriptorTrain = await readJson(`${muscriptorSmokePath}/receipt.json`);
const muscriptorDisjoint = await readJson(`${muscriptorSmokePath}/disjoint8.json`);
if (muscriptorTrain.status !== 'single_public_song_muscriptor_medium_grpo_smoke_not_heldout' ||
  muscriptorTrain.steps_requested !== 50 || muscriptorTrain.steps_updated !== 50 ||
  muscriptorDisjoint.status !== 'exploratory_disjoint_muscriptor_medium_5s_paired_evaluation' ||
  muscriptorDisjoint.recording_count !== 8 || muscriptorDisjoint.paired_anonymous?.length !== 8 ||
  ![muscriptorTrain.before.metrics, muscriptorTrain.after.metrics,
    muscriptorDisjoint.before_macro, muscriptorDisjoint.after_macro].every((metrics) =>
    ['frame', 'onset', 'offset'].every((name) => Number.isFinite(metrics?.[name])))) {
  throw new Error('MuScriptor medium smoke receipts are incomplete');
}
const activeChoralArms = ['onset', 'onset_offset', 'frame', 'coverage', 'continuity'];
for (const key of ['0', '1', '100', '300', '1000',
  ...activeChoralArms.flatMap((arm) => [1, 100, 300, 1000].map((step) => `arm_${arm}_${step}`)),
  'arm_weak_voice_300', 'arm_precision_300']) {
  if (!Number.isInteger(eventReceipt.steps[key]?.note_count) || eventReceipt.steps[key].note_count !== eventNotes[key]?.length) {
    throw new Error(`Public event MIDI note count mismatch at ${key}`);
  }
}
for (const name of ['input', 'reference', 'baseline', 'grpo']) {
  await access(resolve(root, `platform/site/demos/audio/choral_synth_${name}.wav`));
  if (name !== 'input') await access(resolve(root, `platform/site/demos/midi/choral_synth_${name}.mid`));
}
await access(resolve(root, 'platform/site/demos/audio/muscriptor_synth_baseline.wav'));
await access(resolve(root, 'platform/site/demos/midi/muscriptor_synth_baseline.mid'));
for (const name of ['choral_ace_reference_short', 'choral_ace_baseline',
  'choral_audit_onset_only_short_notes', 'choral_audit_overfill_octave',
  'choral_audit_fragment_sustained']) {
  await access(resolve(root, `platform/site/demos/audio/${name}.wav`));
  for (const kind of ['wave', 'spectrum']) {
    await access(resolve(root, `platform/site/demos/visuals/${name}_${kind}.png`));
  }
  if (name.startsWith('choral_audit_')) {
    await access(resolve(root, `platform/site/demos/midi/${name}.mid`));
  }
}
for (const step of [0, 1, 100, 300, 1000]) {
  const id = String(step).padStart(4, '0');
  await access(resolve(root, `platform/site/demos/midi/choral_event_${id}.mid`));
  await access(resolve(root, `platform/site/demos/audio/choral_event_${id}.wav`));
  for (const kind of step === 1 ? [] : ['wave', 'spectrum']) {
    await access(resolve(root, `platform/site/demos/visuals/choral_event_${id}_${kind}.png`));
  }
}
for (const arm of ['onset', 'onset_offset', 'frame', 'coverage', 'continuity', 'precision']) {
  const name = `choral_event_arm_${arm}_300`;
  await access(resolve(root, `platform/site/demos/midi/${name}.mid`));
  await access(resolve(root, `platform/site/demos/audio/${name}.wav`));
  for (const kind of ['wave', 'spectrum']) {
    await access(resolve(root, `platform/site/demos/visuals/${name}_${kind}.png`));
  }
}
for (const arm of activeChoralArms) {
  for (const step of [1, 100, 1000]) {
    const name = `choral_event_arm_${arm}_${step}`;
    await access(resolve(root, `platform/site/demos/midi/${name}.mid`));
    await access(resolve(root, `platform/site/demos/audio/${name}.wav`));
  }
}
await access(resolve(root, 'platform/site/demos/midi/choral_event_arm_weak_voice_300.mid'));
const replayReceipts = new Set();
const cases = [...Object.values(replays.lyrics), replays.vocal, replays.choral];
for (const entry of cases) {
  const steps = entry.stages.map((stage) => stage.step);
  if (new Set(steps).size !== steps.length || steps.some((step, index) => !Number.isInteger(step) || step < 0 || (index > 0 && step <= steps[index - 1]))) {
    throw new Error('Replay stages need distinct, ascending nonnegative integer step counts');
  }
  for (const stage of [entry.input, entry.reference, ...entry.stages, ...(entry.outputs || []),
    ...entry.stages.map((item) => item.paired_baseline).filter(Boolean)].filter(Boolean)) {
    if (['pending', 'queued', 'running'].includes(stage.status) && (stage.audio || stage.wave || stage.spectrum)) {
      throw new Error(`Unmeasured step ${stage.step} must not claim public media`);
    }
    if (stage.status === 'measured' && (!stage.audio || !stage.wave || !stage.spectrum)) {
      throw new Error(`Measured step ${stage.step} needs audio, waveform and spectrogram`);
    }
    for (const key of ['audio', 'wave', 'spectrum']) {
      if (!stage[key]) continue;
      if (!/^(audio|visuals)\/[a-z0-9_-]+\.(flac|wav|png)$/.test(stage[key])) {
        throw new Error(`Invalid public replay asset: ${stage[key]}`);
      }
      await access(resolve(root, 'platform/site/demos', stage[key]));
    }
  }
}
for (const [modelKey, entry] of Object.entries(replays.lyrics)) {
  for (const stage of entry.stages.filter((item) => item.status === 'measured')) {
    if (!/^music-gen\/lyrics2song\/rl\/grpo\/runs\/[a-z0-9-]+\/(?:lr1e-[23]\/)?(?:step_[0-9]{6}\/)?receipt\.json$/.test(stage.receipt) || !['before', 'after', 'heldout:0'].includes(stage.receipt_key)) {
      throw new Error(`Measured lyrics step ${stage.step} needs a valid receipt and before/after key`);
    }
    const receipt = await readJson(stage.receipt);
    const metrics = stage.receipt_key === 'heldout:0' ? receipt.heldout?.[0] : receipt[stage.receipt_key];
    if (!['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].every((key) => Number.isFinite(metrics?.reward?.[key])) ||
      !['peak', 'rms', 'near_full_scale_fraction'].every((key) => Number.isFinite(metrics?.signal?.[key]))) {
      throw new Error(`Measured lyrics step ${stage.step} has incomplete reward or signal metrics`);
    }
    stage.metrics = metrics;
    if (receipt.cross_reward) stage.cross_reward = receipt.cross_reward;
    if (receipt.shared_replay_receipt === sharedMusePath) {
      const expectedHash = receipt[stage.receipt_key]?.audio_sha256;
      const actualHash = createHash('sha256').update(await readFile(resolve(root, 'platform/site/demos', stage.audio))).digest('hex');
      if (actualHash !== expectedHash) throw new Error(`Muse ${stage.step} audio differs from scored receipt`);
    }
    if (stage.paired_baseline) {
      const paired = receipt.before;
      if (!['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness', 'mean'].every((key) => Number.isFinite(paired?.reward?.[key])) ||
        !['peak', 'rms', 'near_full_scale_fraction'].every((key) => Number.isFinite(paired?.signal?.[key]))) {
        throw new Error(`Measured lyrics step ${stage.step} has incomplete paired baseline`);
      }
      stage.paired_baseline.metrics = paired;
    }
    if (stage.receipt_key === 'heldout:0') {
      const klAudit = modelKey === 'yue2_musecritic' ? criticKl : yue2Kl;
      if (stage.step > 0) {
        stage.offline_kl = klAudit.arms[klArms[modelKey]]?.[String(stage.step)]?.kl;
        if (!Number.isFinite(stage.offline_kl)) throw new Error(`Missing audited offline KL for ${modelKey} step ${stage.step}`);
      }
      if (receipt.optimizer_step !== stage.step || receipt.heldout?.length !== 3) {
        throw new Error(`YuE2 stage ${stage.step} needs three matching held-out prompts`);
      }
      const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
      const dimensions = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'];
      const scores = Object.fromEntries(dimensions.map((key) =>
        [key, mean(receipt.heldout.map((item) => item.reward[key]))]));
      const recomputed = mean(receipt.heldout.map((item) => item.reward.mean));
      if (!Number.isFinite(receipt.mean_reward) || Math.abs(recomputed - receipt.mean_reward) > 1e-6) {
        throw new Error(`YuE2 stage ${stage.step} has an inconsistent held-out mean`);
      }
      stage.mean_reward = recomputed;
      stage.heldout_summary = {
        n: 3,
        scores,
        max_peak: Math.max(...receipt.heldout.map((item) => item.signal.peak)),
        near_full_scale_clips: receipt.heldout.filter((item) => item.signal.near_full_scale_fraction > 0).length,
        truncated: receipt.heldout.filter((item) => item.truncated?.semantic).length,
      };
      const baseline = replays.lyrics.yue2.stages[0];
      stage.examples = [];
      for (const [index, item] of receipt.heldout.entries()) {
        const stem = index === 0 ? stage.audio.replace(/^audio\//, '').replace(/\.flac$/, '') :
          stage.step === 0 ? `yue2_baseline_heldout${index}` :
            `${stage.audio.replace(/^audio\//, '').replace(/\.flac$/, '')}_heldout${index}`;
        const example = {
          index, seed: item.seed, prompt: item.prompt, metrics: item,
          audio: index === 0 ? stage.audio : `audio/${stem}.flac`,
          wave: index === 0 ? stage.wave : `visuals/${stem}_wave.png`,
          spectrum: index === 0 ? stage.spectrum : `visuals/${stem}_spectrum.png`,
        };
        const reference = (await readJson(baseline.receipt)).heldout[index];
        if (item.index !== index || item.seed !== reference.seed ||
            JSON.stringify(item.prompt) !== JSON.stringify(reference.prompt)) {
          throw new Error(`${modelKey} step ${stage.step} example ${index} does not match the frozen prompt and seed`);
        }
        for (const key of ['audio', 'wave', 'spectrum']) {
          await access(resolve(root, 'platform/site/demos', example[key]));
        }
        if (item.audio_sha256) {
          const actualHash = createHash('sha256').update(await readFile(resolve(root, 'platform/site/demos', example.audio))).digest('hex');
          if (actualHash !== item.audio_sha256) throw new Error(`${modelKey} step ${stage.step} example ${index} differs from scored FLAC`);
        }
        stage.examples.push(example);
      }
    } else if (Number.isFinite(receipt.mean_reward)) stage.mean_reward = receipt.mean_reward;
    replayReceipts.add(stage.receipt);
  }
}
const yue2Baseline = replays.lyrics.yue2.stages.find((stage) => stage.step === 0);
const yue2Stress100 = replays.lyrics.yue2_stress.stages.find((stage) => stage.step === 100);
const baselineHeldout = (await readJson(yue2Baseline.receipt)).heldout;
const stressHeldout = (await readJson(yue2Stress100.receipt)).heldout;
const pairs100 = replays.lyrics.yue2_stress.pairs_100;
if (pairs100?.length !== 3 || pairs100.some((pair, index) => pair.index !== index)) {
  throw new Error('YuE2 100-step A/B needs exactly three ordered held-out pairs');
}
for (const pair of pairs100) {
  for (const key of ['baseline', 'after']) {
    if (!/^audio\/[a-z0-9_-]+\.flac$/.test(pair[key])) throw new Error(`Invalid YuE2 pair audio: ${pair[key]}`);
    await access(resolve(root, 'platform/site/demos', pair[key]));
  }
  const before = baselineHeldout[pair.index];
  const after = stressHeldout[pair.index];
  if (before.index !== pair.index || after.index !== pair.index ||
      before.seed !== after.seed || JSON.stringify(before.prompt) !== JSON.stringify(after.prompt) ||
      !Number.isFinite(before.reward.mean) || !Number.isFinite(after.reward.mean)) {
    throw new Error(`YuE2 held-out pair ${pair.index} is not a matched replay`);
  }
  pair.seed = before.seed;
  pair.prompt = before.prompt;
  pair.baseline_score = before.reward.mean;
  pair.after_score = after.reward.mean;
}
const yue2ProbePath = 'music-gen/lyrics2song/rl/grpo/runs/2026-09-30-yue2-lr-stress/probes/step_000100_heldout0/receipt.json';
const yue2Probe = await readJson(yue2ProbePath);
if (!yue2Probe.identical_to_reference_flac || !yue2Probe.pipeline_matches_clamped_vae ||
    yue2Probe.heldout_index !== 0 || !Number.isFinite(yue2Probe.vae_preclamp_float?.peak) ||
    !Number.isInteger(yue2Probe.samples_changed_by_pipeline_clamp)) {
  throw new Error('YuE2 100-step VAE probe is incomplete or does not match the public audio');
}
yue2Stress100.probe = {
  vae_preclamp_peak: yue2Probe.vae_preclamp_float.peak,
  clamped_samples: yue2Probe.samples_changed_by_pipeline_clamp,
};

const result = {
  generated_from: [...new Set([
    replaysPath,
    yue2KlPath,
    criticKlPath,
    sharedMusePath,
    yue2ProbePath,
    `${choralDemoPath}/notes.json`,
    `${choralDemoPath}/receipt.json`,
    'music-trans/multi-inst/models/muscriptor/muscriptor_receipt.json',
    'music-gen/vocal2accomp/models/anyaccomp/receipt.json',
    ...replayReceipts,
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed30.json',
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed31.json',
    'music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json',
    'music-trans/choral-singing/rewards/audits/2026-09-30-satb-reward-counterexamples.json',
    'music-trans/choral-singing/benchmarks/icaspp2027_pawct_table2.json',
    `${eventPath}/aggregate.json`,
    `${eventPath}/step_000001/receipt.json`,
    ...activeChoralArms.flatMap((arm) => [1, 100].map((step) =>
      `${eventPath}/early-steps/arm_${arm}_${step}.json`)),
    `${eventPath}/early-steps/public-synthetic-replay-1.json`,
    `${eventPath}/early-steps/public-synthetic-replay-100.json`,
    `${eventPath}/early-steps/public-synthetic-replay-1000.json`,
    `${eventPath}/early-steps/public-synthetic-notes.json`,
    `${eventPath}/early-steps/public-synthetic-notes-1000.json`,
    `${eventPath}/public-synthetic-replay.json`,
    `${eventPath}/public-synthetic-notes.json`,
    `${muscriptorSmokePath}/receipt.json`,
    `${muscriptorSmokePath}/disjoint8.json`,
    'music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json',
    'music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json',
    'music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json',
    musePath,
  ])],
  choral: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/bootstrap_test.json'),
  choral_seed_repeats: await Promise.all([30, 31].map((seed) => readJson(`music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/frame_grpo_seed${seed}.json`))).then((receipts) => receipts.map((receipt) => ({ seed: receipt.seed, frame_f1: receipt.after_test.frame.f1, onset_f1: receipt.after_test.onset.f1, offset_f1: receipt.after_test.offset.f1 }))),
  reward_audit: await readJson('music-trans/choral-singing/rewards/audits/2026-09-29/receipt.json'),
  satb_reward_counterexamples: await readJson('music-trans/choral-singing/rewards/audits/2026-09-30-satb-reward-counterexamples.json'),
  pawct_paper: await readJson('music-trans/choral-singing/benchmarks/icaspp2027_pawct_table2.json'),
  event_pilot: await readJson(`${eventPath}/aggregate.json`),
  event_replay: {
    receipt: eventReceipt,
    notes: eventNotes,
  },
  muscriptor_medium_smoke: {
    train: {
      steps_requested: muscriptorTrain.steps_requested,
      steps_updated: muscriptorTrain.steps_updated,
      before: muscriptorTrain.before,
      after: muscriptorTrain.after,
    },
    disjoint: {
      recording_count: muscriptorDisjoint.recording_count,
      before_macro: muscriptorDisjoint.before_macro,
      after_macro: muscriptorDisjoint.after_macro,
    },
  },
  choral_note: await readJson('music-trans/choral-singing/rl/grpo/runs/2026-09-29-frame-head/note_metrics_50ms.json'),
  choral_demo: {
    notes: await readJson(`${choralDemoPath}/notes.json`),
    receipt: await readJson(`${choralDemoPath}/receipt.json`),
    muscriptor: await readJson('music-trans/multi-inst/models/muscriptor/muscriptor_receipt.json'),
  },
  songeval_audit: await readJson('music-gen/lyrics2song/rewards/audits/2026-09-29/perturbation_47clips.json'),
  vocal_reward: await readJson('music-gen/vocal2accomp/models/anyaccomp/receipt.json'),
  yue2: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/receipt.json'),
  yue2_verification: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-29-yue2/verification.json'),
  muse,
  musecritic: await readJson('music-gen/lyrics2song/rl/grpo/runs/2026-09-30-muse-shared-musecritic/step_000001/receipt.json'),
  replays,
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
