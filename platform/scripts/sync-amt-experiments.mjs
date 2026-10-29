import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const platform = fileURLToPath(new URL('../', import.meta.url));
const site = path.join(platform, 'site');
const repo = path.dirname(platform);
const input = path.join(platform, 'data/amt/experiments.json');
const output = path.join(site, 'amt/demos/data/experiments.json');
const statuses = new Set(['planned', 'running', 'measured']);

export function validateExperiments(data, exists = () => true) {
  const require = (condition, message) => { if (!condition) throw new Error(`AMT experiments: ${message}`); };
  const string = (value, name) => require(typeof value === 'string' && value.trim(), `${name} must be nonempty`);
  const array = (value, name) => require(Array.isArray(value), `${name} must be an array`);
  const unique = (items, name) => require(new Set(items).size === items.length, `duplicate ${name}`);
  const local = (value, name, scope) => {
    string(value, name);
    require(!value.startsWith('/') && !/[\\:?#]/.test(value) && value.split('/').every(part => part && part !== '..' && part !== '.'), `${name} must be a plain relative path`);
    require(exists(value, scope), `${name}: file not found: ${value}`);
  };
  require(data.schema_version === 1, 'unsupported schema_version');
  require(/^\d{4}-\d{2}-\d{2}$/.test(data.updated_at), 'updated_at must be YYYY-MM-DD');
  for (const kind of ['experiments', 'datasets', 'models', 'rewards']) array(data[kind], kind);
  unique(data.experiments.map(run => run.id), 'experiment ID');
  for (const kind of ['datasets', 'models', 'rewards']) for (const record of data[kind]) {
    string(record.name, `${kind}.name`);
    require(/^https:\/\//.test(record.source) && !/\s/.test(record.source), `${kind}.source must use HTTPS`);
  }
  for (const run of data.experiments) {
    for (const field of ['id', 'title', 'task', 'model', 'method', 'reward']) string(run[field], field);
    require(/^[a-z0-9][a-z0-9-]*$/.test(run.id), 'id must use lowercase ASCII letters, digits and hyphens');
    require(statuses.has(run.status), `${run.id}: invalid status`);
    array(run.samples, 'samples'); array(run.stages, 'stages'); array(run.trace, 'trace');
    if (run.receipt) local(run.receipt, 'receipt', 'repo');
    unique(run.samples.map(sample => sample.id), 'sample ID');
    const sampleIds = new Set(run.samples.map(sample => sample.id));
    const asset = (value, sample) => {
      require(value && typeof value === 'object', 'asset must be an object');
      for (const key of ['audio', 'midi']) if (value[key]) {
        local(value[key], key, 'site');
        require(key === 'audio' ? /\.mp3$/i.test(value[key]) : /\.midi?$/i.test(value[key]), `${key}: unsupported public format`);
      }
      require(value.audio || value.midi || Array.isArray(value.notes), 'asset needs audio, MIDI or notes');
      if (value.tracks) { array(value.tracks, 'tracks'); require(value.tracks.every(track => typeof track === 'string'), 'track names must be strings'); }
      if (value.notes) {
        array(value.notes, 'notes');
        for (const note of value.notes) {
          require(Number.isInteger(note.track) && note.track >= 0 && (!value.tracks || note.track < value.tracks.length), 'invalid note track');
          require(Number.isInteger(note.pitch) && note.pitch >= 0 && note.pitch <= 127, 'invalid MIDI pitch');
          require(Number.isFinite(note.start) && Number.isFinite(note.end) && note.start >= 0 && note.end > note.start && note.end <= sample.duration, 'invalid note timing');
        }
      }
    };
    for (const sample of run.samples) {
      string(sample.id, 'sample.id'); string(sample.title, 'sample.title');
      require(Number.isFinite(sample.duration) && sample.duration > 0, 'sample.duration must be positive seconds');
      if (sample.input) asset(sample.input, sample);
      if (sample.reference) asset(sample.reference, sample);
    }
    unique(run.stages.map(stage => stage.step), 'optimizer step');
    let previous = -1;
    for (const stage of run.stages) {
      require(Number.isInteger(stage.step) && stage.step > previous, 'stages must be sorted nonnegative optimizer steps'); previous = stage.step;
      require(statuses.has(stage.status), 'invalid stage status');
      if (stage.status === 'planned') require(!stage.evaluation && !Object.keys(stage.outputs || {}).length, 'planned stages cannot claim results');
      if (stage.status === 'measured') require(run.receipt && (stage.evaluation || Object.keys(stage.outputs || {}).length), 'measured stage needs a receipt and real results');
      if (stage.evaluation) {
        require(stage.status === 'measured', 'scores require measured stage');
        string(run.evaluation_protocol, 'evaluation_protocol'); string(stage.evaluation.cohort, 'evaluation.cohort');
        require(Number.isInteger(stage.evaluation.n) && stage.evaluation.n > 0, 'evaluation.n must be a positive count');
        for (const metric of ['frame', 'onset', 'onset_offset']) if (stage.evaluation[metric] != null) require(Number.isFinite(stage.evaluation[metric]) && stage.evaluation[metric] >= 0 && stage.evaluation[metric] <= 1, `${metric} must be a fraction in [0, 1]`);
      }
      for (const [id, value] of Object.entries(stage.outputs || {})) {
        require(sampleIds.has(id), `unknown sample ${id}`);
        require(stage.status === 'measured', 'public outputs require measured stage');
        asset(value, run.samples.find(sample => sample.id === id));
      }
    }
    if (run.stages.length) require(run.stages[0].step === 0, 'first stage must be frozen baseline step 0');
    if (run.status === 'planned') require(run.stages.every(stage => stage.status === 'planned'), 'planned run cannot claim completed checkpoints');
    if (run.trace.length) {
      require(run.status !== 'planned', 'planned run cannot claim training traces');
      local(run.trace_source, 'trace_source', 'repo'); string(run.trace_note, 'trace_note');
      let prior = -1;
      for (const point of run.trace) {
        require(Number.isInteger(point.step) && point.step > prior, 'trace steps must increase'); prior = point.step;
        require(['reward', 'kl'].some(key => Number.isFinite(point[key])), 'trace needs a finite reward or KL');
        for (const key of ['reward', 'kl']) if (point[key] != null) require(Number.isFinite(point[key]), 'non-finite trace value');
      }
    }
    if (run.status === 'measured') require(run.stages.some(stage => stage.status === 'measured'), 'measured run needs measured checkpoints');
  }
  return data;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const data = validateExperiments(JSON.parse(fs.readFileSync(input, 'utf8')), (value, scope) => fs.existsSync(path.join(scope === 'site' ? site : repo, value)));
  const next = JSON.stringify(data, null, 2) + '\n';
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(output) || fs.readFileSync(output, 'utf8') !== next) throw new Error('AMT experiment manifest is stale; run npm run build');
  } else { fs.mkdirSync(path.dirname(output), { recursive: true }); fs.writeFileSync(output, next); }
  console.log(`AMT experiments verified: ${data.experiments.length} runs (${data.updated_at})`);
}
