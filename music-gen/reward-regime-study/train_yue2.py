"""Full frozen CMI train/valid GRPO for the three preregistered rewards."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from yue2 import YuE2Pipeline
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END
from rewards import score_remote, sha

DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
SAMPLING = {'max_tokens': 600, 'min_tokens': 200, 'temperature': 1.,
            'top_p': 1., 'top_k': CODEC_SIZE+1, 'repetition_penalty': 1.}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def log_probs(model, rollout):
    actions = [CODEC_OFFSET+n for n in rollout['semantic']]
    if not rollout['truncated']['semantic']: actions.append(MUSIC_END)
    ids = rollout['prefix']+actions
    tensor = torch.tensor(ids, device='cuda')[None]
    logits = model(input_ids=tensor, use_cache=False).logits[0, len(rollout['prefix'])-1:-1]
    # Generation uses only codec tokens + EOS, with EOS masked for the first 200.
    # Disable top-p, repetition and top-k truncation so train/sample support agrees.
    scores = torch.cat([logits[:, CODEC_OFFSET:CODEC_OFFSET+CODEC_SIZE], logits[:, MUSIC_END:MUSIC_END+1]], -1)
    mask = torch.arange(len(actions), device='cuda') < SAMPLING['min_tokens']
    scores = scores.clone(); scores[mask, -1] = -torch.inf
    targets = torch.tensor([n-CODEC_OFFSET if n != MUSIC_END else CODEC_SIZE for n in actions], device='cuda')
    return scores.float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def sample(pipe, row, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        song = pipe(style=row['style'], lyrics=row['lyrics'], cot='off', cfg_scale=1.,
                    seed=seed, semantic_sampling=SAMPLING)
    sf.write(folder/'audio.flac', song.audio, song.sample_rate, subtype='PCM_24')
    rollout = {'prefix': list(map(int, song.semantic.plan.prefix)), 'semantic': list(map(int, song.semantic.tokens)),
               'seconds': len(song.audio)/song.sample_rate, 'truncated': song.truncated,
               'audio_sha256': sha(folder/'audio.flac'), 'seed': seed}
    write(folder/'rollout.json', rollout)
    return rollout


def offload(pipe):
    pipe._model.cpu()
    if pipe._vae is not None: pipe._vae.cpu()
    torch.cuda.empty_cache()


def checkpoint(model, optimizer, step, output):
    destination = output/'checkpoints'/f'step_{step:06d}'
    if destination.exists(): return
    temporary = destination.with_name('.'+destination.name+'.tmp')
    temporary.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temporary/'adapter')
    torch.save({'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all()}, temporary/'state.pt')
    write(temporary/'meta.json', {'step': step})
    temporary.rename(destination)


def evaluate(pipe, args, data, step):
    root = args.output/f'valid_{step:06d}'
    if (root/'receipt.json').exists(): return
    pending = []
    for i, row in enumerate(data):
        folder = root/f'{i:03d}'
        if not (folder/'rollout.json').exists(): sample(pipe, row, 5101+i, folder)
        pending.append(folder/'audio.flac')
        write(args.output/'progress.json', {'state': 'valid-generation', 'step': step,
                                           'completed_conditions': i+1, 'target_conditions': 30})
    offload(pipe)
    results = score_remote(args.scorer, pending, args.reward)
    for i, result in enumerate(results): write(root/f'{i:03d}'/'score.json', result)
    write(root/'receipt.json', {'step': step, 'n': len(data), 'scope': 'all 30 validation conditions, not test',
         'mean_reward': float(np.mean([r['mean'] for r in results])), 'rows': results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scorer', required=True); p.add_argument('--reward', choices=['richness','instrument','arrangement'], required=True)
    p.add_argument('--steps', type=int, default=270); p.add_argument('--lr', type=float, default=3e-4)
    a = p.parse_args()
    if sha(a.data) != DATA_SHA: raise ValueError('Frozen CMI projection changed')
    data = json.loads(a.data.read_text())
    if len(data['train']) != 270 or len(data['valid']) != 30: raise ValueError('Not full frozen split')
    a.output.mkdir(parents=True, exist_ok=True)
    config = {'model': 'YuE2-3B', 'reward': a.reward, 'lr': a.lr, 'seed': 42, 'beta': .01,
              'steps': a.steps, 'train_n': 270, 'valid_n': 30, 'data_sha256': DATA_SHA,
              'sampling': SAMPLING, 'source_code_sha256': sha(Path(__file__)), 'scored_audio': 'PCM24-FLAC',
              'policy': 'codec+EOS support; unfiltered sampling; single-update group-relative objective',
              'kl': 'k3 estimate against frozen base on the same masked support; training proxy, not exact full-policy KL'}
    if (a.output/'experiment.json').exists() and json.loads((a.output/'experiment.json').read_text()) != config:
        raise ValueError('Configuration changed')
    write(a.output/'experiment.json', config)
    torch.manual_seed(42)
    pipe = YuE2Pipeline.from_pretrained(a.model, vae=a.vae, device='cuda', backend='torch-eager', memory_budget_gib=32, progress=False)
    base = pipe._load_model().eval()
    ready = sorted((a.output/'checkpoints').glob('step_*'))
    if ready:
        last = ready[-1]; start = json.loads((last/'meta.json').read_text())['step']
        model = PeftModel.from_pretrained(base, last/'adapter', is_trainable=True).eval()
    else:
        start = 0
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=8,
                lora_dropout=0, target_modules=['q_proj','v_proj'], bias='none')).eval()
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    if ready:
        state = torch.load(last/'state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        checkpoint(model, optimizer, 0, a.output)
    # Full validation is independent and uses fixed seeds, not selected demo prompts.
    if start == 0: evaluate(pipe, a, data['valid'], 0)
    for step in range(start+1, a.steps+1):
        tick = time.monotonic(); row = data['train'][(step-1)%270]
        folder = a.output/'train_rollouts'/f'{step:06d}'
        if folder.exists(): raise FileExistsError('Incomplete prior rollout requires inspection')
        rolls = [sample(pipe, row, 700000+2*step+i, folder/f'{i}') for i in range(2)]
        offload(pipe)
        scores = score_remote(a.scorer, [folder/f'{i}'/'audio.flac' for i in range(2)], a.reward)
        model.cuda()
        rewards = torch.tensor([s['mean'] for s in scores], device='cuda')
        advantages = (rewards-rewards.mean())/rewards.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            old = [log_probs(model, r).detach() for r in rolls]
            with model.disable_adapter(): reference = [log_probs(model, r).detach() for r in rolls]
        optimizer.zero_grad(set_to_none=True); kls = []
        for i, rollout in enumerate(rolls):
            current = log_probs(model, rollout)
            log_ratio = (reference[i]-current).clamp(-20,20)
            kl = (log_ratio.exp()-log_ratio-1).mean(); kls.append(float(kl.detach()))
            ratio = (current-old[i]).exp()
            loss = -torch.minimum(ratio*advantages[i], ratio.clamp(.8,1.2)*advantages[i]).mean()/2+.01*kl/2
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite policy loss')
            loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.))
        if not np.isfinite(grad): raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError('Nonfinite adapter parameters')
        record = {'step': step, 'reward': float(rewards.mean()), 'sampled_kl': float(np.mean(kls)),
                  'rewards': rewards.tolist(), 'components': scores, 'gradient_norm': grad, 'seconds': time.monotonic()-tick}
        write(folder/'receipt.json', record)
        with (a.output/'steps.jsonl').open('a') as target: target.write(json.dumps(record, allow_nan=False)+'\n')
        checkpoint(model, optimizer, step, a.output)
        write(a.output/'progress.json', {'state': 'training', 'step': step, 'target': a.steps, 'latest': record})
        print(f'step={step} reward={record["reward"]:.5f} kl={record["sampled_kl"]:.5f} seconds={record["seconds"]:.1f}', flush=True)
        if step in [50,100,a.steps]: evaluate(pipe, a, data['valid'], step)
    write(a.output/'completion.json', {'steps': a.steps, 'full_train': 270, 'full_valid': 30, 'test_run': False})


if __name__ == '__main__': main()
