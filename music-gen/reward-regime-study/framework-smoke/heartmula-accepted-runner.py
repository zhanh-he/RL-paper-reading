"""Real HeartMuLa global/local backbone LoRA through TRL 0.24 GRPOTrainer.

Bounded integration smoke, not a formal experiment. All eight sampled codebooks
are teacher-forced under the current backbone; no cached actor features are used
for the gradient. The real audio reward must come from the frozen scorer.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--trl', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scorer', required=True)
    p.add_argument('--seconds', type=float, default=4.)
    p.add_argument('--dtype', choices=['float32','float16'], default='float32'); a = p.parse_args()
    sys.path[:0] = [str(a.source/'src'), str(a.trl), str(Path(__file__).resolve().parent.parent)]
    from datasets import Dataset
    from peft import get_peft_model, LoraConfig
    from transformers import PreTrainedTokenizerFast
    import importlib
    # Native music rollout is used, not the unrelated shared vLLM installation.
    importlib.import_module('trl.import_utils')._vllm_available = False
    from trl import GRPOConfig, GRPOTrainer
    from torchtune.modules.common_utils import disable_kv_cache
    from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
    import heartlib.heartmula.modeling_heartmula as native
    from rewards import score_remote, sha
    DATA_SHA = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'

    assert sha(a.data) == DATA_SHA
    row = json.loads(a.data.read_text())['train'][0]
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    dtype = getattr(torch, a.dtype)
    HeartMuLaGenPipeline._apply_compile = lambda self, model: None
    pipe = HeartMuLaGenPipeline.from_pretrained(str(a.assets), device=torch.device('cuda'),
        dtype={'mula': dtype, 'codec': torch.float32}, version='3B', lazy_load=True)
    base = pipe.mula.requires_grad_(False).eval()
    # Enumerating full paths prevents accidental adaptation of codec/output heads.
    targets = [name for name, m in base.named_modules() if isinstance(m, torch.nn.Linear)
               and (name.startswith('backbone.layers.') or name.startswith('decoder.layers.'))
               and any(part in name for part in ['.attn.', '.mlp.'])]
    assert targets and any(n.startswith('decoder.') for n in targets)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=targets, bias='none'))
    from torch.utils.checkpoint import checkpoint
    for layer in [*base.backbone.layers, *base.decoder.layers]:
        original = layer.forward
        def memory_forward(*args, _forward=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)
        layer.forward = memory_forward
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(a.assets/'tokenizer.json'))
    tokenizer.add_special_tokens({'pad_token': '[SMOKE_PAD]', 'eos_token': '[SMOKE_EOS]'})
    inputs = pipe.preprocess({'tags': row['style'], 'lyrics': row['lyrics']}, cfg_scale=1.5)
    sampled, records, audits = [], [], []

    def full_support(logits, topk, temperature):
        probs = (logits.float()/temperature).softmax(-1)
        token = torch.multinomial(probs, 1)
        sampled.append({'token': token.detach().cpu(), 'logp': probs.log().gather(-1, token).detach().cpu()})
        return token.int()
    native.sample_topk = full_support

    def teacher(root, actions):
        frames = actions.reshape(-1, 8)
        previous = torch.zeros((len(frames)-1, 9), dtype=torch.long, device='cuda')
        previous[:, :8] = frames[:-1]
        token = torch.cat([inputs['tokens'].to('cuda'), previous[None].repeat(2, 1, 1)], 1)
        mask = torch.cat([inputs['tokens_mask'].to('cuda'),
                          torch.tensor([True]*8+[False], device='cuda')[None, None].expand(2, len(previous), 9)], 1)
        global_cache = disable_kv_cache(root.backbone) if root.backbone.caches_are_setup() else nullcontext()
        local_cache = disable_kv_cache(root.decoder) if root.decoder.caches_are_setup() else nullcontext()
        with global_cache, local_cache, torch.autocast('cuda', dtype=dtype, enabled=dtype!=torch.float32):
            embeddings = root._embed_tokens(token, torch.tensor([False, True], device='cuda'))
            h = (embeddings*mask[..., None]).sum(2, dtype=embeddings.dtype)
            continuous = root.muq_linear(inputs['muq_embed'].to('cuda'))
            continuous[1] = root.unconditional_text_embedding(torch.zeros(1, device='cuda', dtype=torch.long))[0]
            h[torch.arange(2, device='cuda'), inputs['muq_idx']] = continuous.to(h.dtype)
            positions = torch.arange(h.shape[1], device='cuda')[None].expand(2, -1)
            causal = torch.ones(h.shape[1], h.shape[1], device='cuda', dtype=torch.bool).tril()[None]
            h = root.backbone(h, input_pos=positions, mask=causal)[:, -len(frames):]
            global_logits = root.codebook0_head(h)
            guided = (global_logits[1] + 1.5*(global_logits[0]-global_logits[1])).float()
            logps = [guided.log_softmax(-1).gather(-1, frames[:, 0:1]).squeeze(-1)]
            entropies = [-(guided.softmax(-1)*guided.log_softmax(-1)).sum(-1)]
            # Local causal sequence: [global state, codebook0,...,codebook6].
            # Position j+1 predicts codebook j+1. Codebook7 is never an input.
            local = torch.cat([h[:, :, None], torch.stack([
                root._embed_audio(i, frames[:, i])[None].expand(2, -1, -1) for i in range(7)], 2)], 2)
            local = root.projection(local).reshape(2*len(frames), 8, -1)
            lp = torch.arange(8, device='cuda')[None].expand(len(local), -1)
            local_mask = torch.ones(8, 8, dtype=torch.bool, device='cuda').tril()[None]
            decoded = root.decoder(local, input_pos=lp, mask=local_mask).reshape(2, len(frames), 8, -1)
            for i in range(1, 8):
                logits = decoded[:, :, i] @ root.audio_head[i-1]
                guided = (logits[1]+1.5*(logits[0]-logits[1])).float()
                dist = guided.log_softmax(-1)
                logps.append(dist.gather(-1, frames[:, i:i+1]).squeeze(-1))
                entropies.append(-(dist.exp()*dist).sum(-1))
        return torch.stack(logps, 1).flatten(), torch.stack(entropies, 1).flatten()

    class AudioTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            lookup = {tuple(r['action'].tolist()): r['old_logp'] for r in records}
            old = []
            for ids, mask in zip(result['completion_ids'], result['completion_mask']):
                values = lookup[tuple(ids[mask.bool()].tolist())].to(ids.device)
                old.append(torch.nn.functional.pad(values, (0, len(ids)-len(values))))
            # Use actual cached-sampling probabilities, not a detach of the
            # full-sequence recomputation with slightly different FP16 rounding.
            result['old_per_token_logps'] = torch.stack(old)
            return result

        def _generate_single_turn(self, prompts, images):
            records.clear()
            for i in range(len(prompts)):
                sampled.clear(); value = 420000 + self.state.global_step*2 + i
                torch.manual_seed(value); torch.cuda.manual_seed_all(value)
                pipe.lazy_load = False
                pipe._mula = base
                with torch.no_grad():
                    output = pipe._forward(inputs, max_audio_length_ms=int(a.seconds*1000), temperature=1., topk=8194, cfg_scale=1.5)
                # Cached rollout is over. Drop KV buffers before teacher forcing
                # and checkpointed backward; generation recreates them next time.
                for module in base.modules():
                    if hasattr(module, 'kv_cache'):
                        module.kv_cache = None
                        module.cache_enabled = False
                assert len(sampled) % 8 == 0
                action = torch.cat([r['token'].flatten() for r in sampled])
                old = torch.cat([r['logp'].flatten() for r in sampled])
                with torch.no_grad():
                    recomputed, _ = teacher(base, action.to('cuda'))
                error = float((recomputed.cpu()-old).abs().max())
                if error > (.002 if dtype==torch.float32 else .04):
                    raise ValueError(f'Native cache/full teacher likelihood mismatch: {error}')
                folder = a.output/f'group_{self.state.global_step:03d}'/str(i)
                folder.mkdir(parents=True, exist_ok=True)
                # Codec is frozen and loaded only for audio scoring, not the loss.
                base.cpu()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    wav = pipe.codec.to('cuda').detokenize(output['frames'].to('cuda')).float().cpu().numpy().T
                pipe._codec.cpu()
                torch.cuda.empty_cache()
                base.cuda()
                path = folder/'audio.flac'; sf.write(path, wav, 48000, subtype='PCM_24')
                records.append({'action': action, 'old_logp': old, 'path': path, 'sampling_error': error})
            # The scorer owns the GPU only after generator+codec are parked.
            base.cpu()
            if pipe._codec is not None:
                pipe._codec.cpu()
            torch.cuda.empty_cache()
            scores = score_remote(a.scorer, [r['path'] for r in records], 'richness')
            base.cuda()
            for record, score in zip(records, scores):
                record['score'] = score
                assert sha(record['path']) == score['audio_sha256']
            (a.output/f'group_{self.state.global_step:03d}'/'receipt.json').write_text(json.dumps({
                'scope': 'Short real-model integration smoke, not full validation',
                'rows': [dict(score=r['score'], sampling_recompute_max_error=r['sampling_error']) for r in records]}, indent=2)+'\n')
            return [[1] for _ in records], [r['action'].tolist() for r in records], None, {}

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                              batch_size=None, compute_entropy=False, **kwargs):
            root = self.accelerator.unwrap_model(model).get_base_model()
            result, entropy = [], []
            for ids, mask in zip(input_ids, attention_mask):
                action = ids[-logits_to_keep:][mask[-logits_to_keep:].bool()]
                logp, ent = teacher(root, action)
                pad = logits_to_keep-len(logp)
                result.append(torch.nn.functional.pad(logp, (0, pad)))
                entropy.append(torch.nn.functional.pad(ent, (0, pad)))
            return torch.stack(result), torch.stack(entropy) if compute_entropy else None

    def reward(completions, completion_ids, **kwargs):
        lookup = {tuple(r['action'].tolist()): r['score']['mean'] for r in records}
        return [lookup[tuple(ids)] for ids in completion_ids]

    from transformers import TrainerCallback
    class Receipt(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = {name: float(p.grad.float().norm()) for name, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
            if not grads or not all(np.isfinite(v) for v in grads.values()):
                raise FloatingPointError('Nonfinite or absent backbone gradients')
            audits.append({'step': state.global_step+1, 'nonzero_global': sum(v>0 for n,v in grads.items() if 'backbone.' in n),
                           'nonzero_local': sum(v>0 for n,v in grads.items() if 'decoder.' in n),
                           'gradient_norms': grads})
            assert audits[-1]['nonzero_global'] and audits[-1]['nonzero_local']

    cfg = GRPOConfig(output_dir=str(a.output/'trainer'), learning_rate=3e-4, max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, num_generations=2,
        beta=.01, max_completion_length=8*(int(a.seconds/.08)+2), max_prompt_length=None,
        bf16=dtype==torch.bfloat16, fp16=dtype==torch.float16, gradient_checkpointing=False,
        save_strategy='no', logging_steps=1, report_to='none', num_iterations=1,
        loss_type='grpo', remove_unused_columns=False, mask_truncated_completions=False, seed=42)
    tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    trainer = AudioTrainer(model=model, reward_funcs=reward, args=cfg,
        train_dataset=Dataset.from_list([{'prompt': row['style']+'\n'+row['lyrics']}]),
        processing_class=tokenizer, callbacks=[Receipt()])
    trainer.train()
    model.save_pretrained(a.output/'adapter')
    receipt = {'status': 'passed-real-model-TRL-GRPOTrainer-two-updates', 'formal_training': False,
        'model': 'HeartMuLa-oss-3B-happy-new-year', 'scope': 'All global/local attention+MLP backbone LoRA; frozen heads/embeddings/codec',
        'rank': 8, 'alpha': 16, 'lr': 3e-4, 'beta': .01, 'codebooks_in_loss': 8, 'targets': targets,
        'seconds_budget': a.seconds, 'activation_checkpointing': 'all global/local transformer layers; rollout KV buffers released before backward',
        'train_condition_id': row['condition_id'], 'data_sha256': DATA_SHA,
        'device': torch.cuda.get_device_name(), 'dtype': str(dtype), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'elapsed_seconds': time.monotonic()-tick, 'audits': audits, 'logs': trainer.state.log_history,
        'limitations': ['One real train condition and short audio for compatibility only; not formal270/30 results',
                       'Single GPU adapter uses native forward under TRL; distributed FSDP integration not yet validated',
                       'Sampled k3 reference KL logged by TRL, not exact full-policy KL']}
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['targets','audits','logs']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
