"""Native DiffRhythm2 DiT backbone LoRA + isolated upstream Flow-GRPO kernels.

Bounded compatibility check only. Rebuild text, clean block history and noisy
block under the current actor on every gradient forward. Never train on a stale
rollout KV cache. Full distributed framework integration is not claimed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import soundfile as sf
import torch
from backbone_lora import install, reference
from audit_flow_kernels import load, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['assets','source','data','miles','verl','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scorer', required=True)
    p.add_argument('--reward', choices=['richness','instrument','arrangement'], default='instrument'); a = p.parse_args()
    for name in ['assets','source','data','miles','verl','output']:
        setattr(a,name,getattr(a,name).resolve())
    os.chdir(a.assets.parent.parent)
    os.environ['HF_HOME']=str(a.assets.parent.parent/'hf-cache')
    os.environ['HF_HUB_OFFLINE']='1'
    import espeakng_loader
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(espeakng_loader.get_library_path())
    os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
    sys.path[:0] = [str(a.source), str(Path(__file__).resolve().parent.parent)]
    from rewards import score_remote, sha
    spec = importlib.util.spec_from_file_location('native_diff', a.source/'inference.py')
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    native.hf_hub_download = lambda repo_id, filename, **kwargs: str(a.assets/filename)
    model, mulan, tokenizer, decoder = native.prepare_model('ASLP-lab/DiffRhythm2', 'cuda')
    native.lrc_tokenizer = tokenizer
    model.requires_grad_(False).eval().half(); decoder.requires_grad_(False).eval().half()
    targets = install(model.transformer, ['transformer_blocks.'])
    assert any('self_attn' in n for n in targets) and any('mlp' in n for n in targets)
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    initial = {n:p.detach().cpu().clone() for n,p in parameters.items()}
    data_sha = 'c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
    assert sha(a.data) == data_sha
    row = json.loads(a.data.read_text())['train'][0]
    text = torch.tensor(sum(native.parse_lyrics(row['lyrics']), []), device='cuda').long()[None]
    with torch.no_grad():
        style = mulan(texts=[row['style']]).detach().half()
    ns = load(a.miles, a.verl); backend = ns['FlowSdeStepBackend']()
    records, errors = [], []
    history = None

    def predict(x, sigma, clean):
        length, past, block = text.shape[1], clean.shape[1], x.shape[1]
        n = length+past+block
        mask = torch.zeros((1,1,n,n), device='cuda', dtype=torch.bool)
        mask[:, :, :length, :length] = True
        for i in range(0, past, block):
            mask[:, :, length+i:length+i+block, :length+i+block] = True
        mask[:, :, length+past:, :] = True
        positions = torch.cat([torch.arange(length,device='cuda'), torch.arange(past+block,device='cuda')])[None]
        times = torch.cat([torch.full((1,length),-1.,device='cuda'), torch.ones((1,past),device='cuda'),
                           (1-sigma)[:,None].expand(1,block)], 1).half()
        latents = model.transformer.latent_embed(torch.cat([clean, x.half()], 1))
        outputs = []
        for tokens, conditioning in [(text,style),(torch.zeros_like(text),torch.zeros_like(style))]:
            emb = torch.cat([model.transformer.text_embed(tokens),latents],1)
            pred, *_ = model.transformer(x=emb, time=times, position_ids=positions,
                style_prompt=conditioning, attn_mask=mask, use_cache=False)
            outputs.append(pred[:, -block:].float())
        return -(outputs[0]+2*(outputs[0]-outputs[1]))

    def odeint(fn, noise, times, **kwargs):
        nonlocal history
        x = noise.float(); schedule = torch.linspace(1,0,len(times),device='cuda')
        clean = history.clone()
        for i in range(len(times)-1):
            sigma, sn = schedule[i:i+1], schedule[i+1:i+2]
            velocity = predict(x, sigma, clean)
            native_pred = -fn(1-sigma[0].half(), x.half()).float()
            error = float((velocity-native_pred).abs().max())
            errors.append(error)
            if error > .15:
                raise ValueError(f'Native-cache/full-context velocity mismatch {error}')
            mu, std, _ = backend.prev_sample_mean_and_std(velocity,x,sigma[:,None,None],sn[:,None,None],
                sigma_max=torch.full((1,1,1),.875,device='cuda'),noise_level=.1)
            nxt = mu+std*torch.randn_like(x)
            records.append({'x':x.cpu(),'next':nxt.cpu(),'sigma':sigma.cpu(),'next_sigma':sn.cpu(),
                            'history':clean.cpu(),'old':backend.log_prob(nxt,mu,std).cpu()})
            x = nxt
        history = torch.cat([history,x.half()],1)
        return torch.stack([noise.float(),x]).half()
    import diffrhythm2.cfm
    diffrhythm2.cfm.odeint = odeint
    a.output.mkdir(parents=True,exist_ok=True); tick=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    rolls=[]
    for i in range(2):
        records.clear(); history=torch.zeros((1,0,64),device='cuda',dtype=torch.float16)
        torch.manual_seed(420000+i)
        with torch.no_grad():
            latent=model.sample_block_cache(text,20,style,steps=9,cfg_strength=2.,process_bar=False)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(42)
                wav=decoder.decode_audio(latent.transpose(1,2),overlap=5,chunk_size=20)
        path=a.output/f'{i}.flac';sf.write(path,wav.float().cpu().numpy().reshape(-1),int(decoder.h.sampling_rate),subtype='PCM_24')
        rolls.append({'path':path,'transitions':list(records)})
    for m in [model,mulan,decoder]:m.cpu()
    torch.cuda.empty_cache()
    scores=score_remote(a.scorer,[r['path'] for r in rolls],a.reward)
    for r,s in zip(rolls,scores):assert sha(r['path'])==s['audio_sha256']
    model.cuda(); style=style.cuda()
    rewards=torch.tensor([r['mean'] for r in scores],device='cuda')
    assert float(rewards.std(unbiased=False))>0,'Real reward tie cannot demonstrate a policy update'
    adv=(rewards-rewards.mean())/rewards.std(unbiased=False)
    # Select a second-block transition: this specifically exercises clean-history
    # recomputation, the part a stale-output-head adaptation cannot validate.
    pairs=[r['transitions'][8] for r in rolls]
    audits=[]
    for framework in ['miles','verl-omni']:
        with torch.no_grad():
            for n,p in parameters.items():p.copy_(initial[n])
        optimizer=torch.optim.AdamW(parameters.values(),lr=3e-4)
        for step in [1,2]:
            optimizer.zero_grad(set_to_none=True); pred=[]; ref=[]
            for r in pairs:
                x,s,clean=(r[k].cuda() for k in ['x','sigma','history'])
                with torch.no_grad(),reference(model):ref.append(predict(x,s,clean))
                pred.append(predict(x,s,clean))
            tensors=[torch.cat([r[k] for r in pairs]).cuda() for k in ['x','next','old','sigma','next_sigma']]
            x,nxt,old,s,sn=tensors
            ml,vl,metrics=compare(ns,torch.cat(pred),x,nxt,old,adv,s,sn,torch.cat(ref),.875)
            loss=ml if framework=='miles' else vl
            assert torch.isfinite(loss);loss.backward()
            nonzero=[n for n,p in parameters.items() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0]
            assert any('self_attn' in n for n in nonzero) and any('mlp' in n for n in nonzero)
            grad=float(torch.nn.utils.clip_grad_norm_(parameters.values(),1.));assert np_isfinite(grad)
            optimizer.step()
            delta=sum(float((p.detach().cpu()-initial[n]).square().sum()) for n,p in parameters.items())**.5
            assert delta>0
            audits.append({'framework_kernel':framework,'step':step,'gradient_norm':grad,'adapter_delta_l2':delta,**metrics})
    receipt={'status':'passed-real-DiffRhythm2-backbone-two-updates-per-isolated-kernel','full_framework_e2e':False,
        'formal_training':False,'reward':a.reward,'data_sha256':data_sha,'condition_id':row['condition_id'],'targets':targets,
        'device':torch.cuda.get_device_name(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'seconds':time.monotonic()-tick,'native_cache_full_context_max_error':max(errors),'scores':scores,'audits':audits,
        'limitations':['Short native real train condition; no validation/test/MOS claim','Only isolated upstream loss kernels, not distributed framework trainers',
                       'Two updates on one second-block transition pair; not a full formal run','Miles raw scale-normalized KL and VeRL policy plus exact conditional KL reported separately']}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['scores','audits','targets']},indent=2),flush=True)


def np_isfinite(value):
    return __import__('math').isfinite(value)


if __name__=='__main__':main()
