"""TRL GRPO with native YuE2 rollouts, masked codec likelihood and pure Mean7."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from yue2 import YuE2Pipeline
from yue2 import sampling as native_sampling
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, EOD, VOCAB_SIZE
from yue2.tokenization_yue2 import YuE2TextTokenizer
from rewards import score_remote, sha
from yue2_checkpointing import install as install_checkpointing

DATA_SHA='c39d43e81793bee3f312c7028a0483b5da41068db4544acd258115043101e46f'
MUQ_SHA='273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6'
HEAD_SHA='ca27966a6d2b3312587b9e1d53ee3adf6bde8ac68a7d339e6f962206cbfde312'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


class NativeHFTokenizer(PreTrainedTokenizer):
    """HF lifecycle facade over the actual checkpoint tiktoken, not a substitute BPE."""
    model_input_names=['input_ids','attention_mask']
    def __init__(self,model):
        self.native=YuE2TextTokenizer(Path(model)/'qwen.tiktoken')
        super().__init__(pad_token=f'id_{EOD}',eos_token=f'id_{MUSIC_END}',unk_token=f'id_{EOD}',padding_side='left')
    @property
    def vocab_size(self): return VOCAB_SIZE
    def get_vocab(self): return {f'id_{i}':i for i in range(VOCAB_SIZE)}
    def _tokenize(self,text): return [f'id_{i}' for i in self.native.encode(text)]
    def _convert_token_to_id(self,token): return int(token[3:]) if token.startswith('id_') else EOD
    def _convert_id_to_token(self,index): return f'id_{index}'
    def convert_tokens_to_string(self,tokens): return self.native.decode([self._convert_token_to_id(t) for t in tokens])
    def save_vocabulary(self,save_directory,filename_prefix=None): return self.native.save_pretrained(save_directory)


class NativePolicy:
    def __init__(self,args):
        self.args=args
        self.pipe=YuE2Pipeline.from_pretrained(args.model,vae=args.vae,device='cuda',backend='torch-eager',memory_budget_gib=32,progress=False)
        self.base=self.pipe._load_model().eval()
        install_checkpointing(self.base)
        self.base.config.pad_token_id=EOD;self.base.config.eos_token_id=MUSIC_END
        self.tokenizer=NativeHFTokenizer(args.model)
        self.model=get_peft_model(self.base,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,
            target_modules=r'.*\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',bias='none'))
        self.pipe._model=self.model
        self.trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.trainable!=8716288: raise ValueError(f'Unexpected AR backbone adapter: {self.trainable}')
        self.sampling=dict(max_tokens=args.max_tokens,min_tokens=0,temperature=1.,top_p=1.,top_k=CODEC_SIZE+1,repetition_penalty=1.)

    def offload(self):
        self.model.cpu()
        if self.pipe._vae is not None:self.pipe._vae.cpu()
        torch.cuda.empty_cache()

    def sample(self,row,seed,folder):
        folder.mkdir(parents=True,exist_ok=False)
        captured=[];latest={};original=native_sampling.distribution
        def distribution(*args,**kwargs):
            result=original(*args,**kwargs)
            phase=args[4] if len(args)>4 else kwargs['phase']
            if phase=='semantic':
                probabilities=result.softmax(-1).float()
                latest['logp']=(probabilities/probabilities.sum(-1,keepdim=True)).log()
            return result
        def on_token(phase,token):
            if phase=='semantic':captured.append((token,float(latest['logp'][0,token])))
        native_sampling.distribution=distribution
        try:
            self.model.cuda().eval()
            with torch.inference_mode():
                song=self.pipe(style=row['style'],lyrics=row['lyrics'],cot='off',cfg_scale=1.,seed=seed,
                    semantic_sampling=self.sampling,on_token=on_token)
        finally:native_sampling.distribution=original
        actions=[CODEC_OFFSET+int(n) for n in song.semantic.tokens]
        if not song.truncated['semantic']:actions.append(MUSIC_END)
        if not actions or [a for a,_ in captured]!=actions:raise ValueError('Native sampled actions/logps not bound')
        if not np.isfinite(song.audio).all():raise ValueError('Nonfinite native decode; no fallback reward')
        sf.write(folder/'audio.flac',song.audio,song.sample_rate,subtype='PCM_24')
        record=dict(prefix=list(map(int,song.semantic.plan.prefix)),actions=actions,
            sampling_logps=[p for _,p in captured],seconds=len(song.audio)/song.sample_rate,
            audio_sha256=sha(folder/'audio.flac'),seed=seed,truncated=song.truncated)
        write(folder/'rollout.json',record)
        return record

    def score(self,paths):
        results=score_remote(self.args.scorer,paths,'mean7')
        for r in results:
            identity=r['scorer_identity']
            if identity.get('muq_weights_sha256')!=MUQ_SHA or identity['songbench_weights_sha256']!=HEAD_SHA:
                raise ValueError('Frozen SongBench/MuQ identity changed')
            if not identity.get('mean7_only'):raise ValueError('Pure reward requires the Mean7-only scorer')
            if r.get('technical_invalid'):
                if r['mean']!=0 or any(v is not None for v in r['songbench'].values()):
                    raise ValueError('Technical invalid outcome must not fabricate head scores')
                continue
            expected=sum(r['songbench'][k] for k in ['Vocal','Instrumental','Melody','Structure','Arrangement','Mixing','Musicality'])/7
            if abs(expected-r['mean'])>1e-7:raise ValueError('Extra reward/penalty detected')
        return results


class NativeGRPO(GRPOTrainer):
    def __init__(self,policy,*args,**kwargs):
        self.native=policy;self.latest_group=None;self.gradient_receipt=None
        super().__init__(*args,**kwargs)

    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,**kwargs):
        logps=[];entropies=[]
        for i in range(input_ids.shape[0]):
            n=int(attention_mask[i,-logits_to_keep:].sum())
            ids=input_ids[i][attention_mask[i].bool()]
            logits=model(input_ids=ids[None],use_cache=False,logits_to_keep=n+1).logits[0,:-1]
            scores=torch.cat([logits[:,CODEC_OFFSET:CODEC_OFFSET+CODEC_SIZE],logits[:,MUSIC_END:MUSIC_END+1]],-1)
            probabilities=scores.softmax(-1).float()
            probabilities=probabilities/probabilities.sum(-1,keepdim=True)
            targets=torch.where(ids[-n:]==MUSIC_END,CODEC_SIZE,ids[-n:]-CODEC_OFFSET)
            values=probabilities.gather(-1,targets[:,None]).squeeze(-1).log()
            if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite native action logp')
            entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
            logps.append(F.pad(values,(0,logits_to_keep-n)));entropies.append(F.pad(entropy,(0,logits_to_keep-n)))
        return torch.stack(logps),torch.stack(entropies)

    def _generate_and_score_completions(self,inputs):
        if len(inputs)!=2 or inputs[0]['condition_id']!=inputs[1]['condition_id']:
            raise ValueError('GRPO group must contain two repeats of the same frozen condition')
        step=self.state.global_step+1;folder=self.native.args.output/'train_rollouts'/f'{step:06d}'
        model=self.accelerator.unwrap_model(self.model_wrapped)
        self.native.model=model;self.native.pipe._model=model
        rows=[self.native.sample(inputs[i],700000+2*(step-1)+i,folder/str(i)) for i in range(2)]
        self.native.offload()
        results=self.native.score([folder/str(i)/'audio.flac' for i in range(2)])
        model.cuda().eval();device=self.accelerator.device
        prefix_n=max(len(r['prefix']) for r in rows);completion_n=max(len(r['actions']) for r in rows)
        prompt_ids=torch.tensor([[EOD]*(prefix_n-len(r['prefix']))+r['prefix'] for r in rows],device=device)
        prompt_mask=torch.tensor([[0]*(prefix_n-len(r['prefix']))+[1]*len(r['prefix']) for r in rows],device=device)
        completion_ids=torch.tensor([r['actions']+[EOD]*(completion_n-len(r['actions'])) for r in rows],device=device)
        mask=torch.tensor([[1]*len(r['actions'])+[0]*(completion_n-len(r['actions'])) for r in rows],device=device)
        ids=torch.cat([prompt_ids,completion_ids],1);full_mask=torch.cat([prompt_mask,mask],1)
        with torch.no_grad():
            teacher,_=self._get_per_token_logps_and_entropies(model,ids,full_mask,completion_n)
            with model.disable_adapter():reference,_=self._get_per_token_logps_and_entropies(model,ids,full_mask,completion_n)
        behavior=torch.tensor([r['sampling_logps']+[0.]*(completion_n-len(r['actions'])) for r in rows],device=device)
        gap=(teacher-behavior).abs()[mask.bool()]
        if gap.mean()>.02 or gap.max()>.2:raise ValueError(f'Sampling/teacher mismatch exceeds acceptance: {gap.mean()}, {gap.max()}')
        reward=torch.tensor([r['mean'] for r in results],device=device)
        std=reward.std();advantages=(reward-reward.mean())/(std+1e-4)
        kl=((reference-teacher).exp()-(reference-teacher)-1)[mask.bool()].mean()
        if step==1 and abs(float(kl))>1e-8:raise ValueError('Frozen reference identity failed')
        self.latest_group=dict(step=step,reward=float(reward.mean()),rewards=reward.tolist(),
            sampled_kl=float(kl),zero_std=bool(std==0),seconds=[r['seconds'] for r in rows],
            sampling_teacher_logp_gap_mean=float(gap.mean()),sampling_teacher_logp_gap_max=float(gap.max()),
            condition_id=inputs[0]['condition_id'],scores=results,lr=3e-4,beta=.05)
        write(folder/'receipt.json',self.latest_group)
        self._metrics['train']['reward'].append(float(reward.mean()))
        self._metrics['train']['reward_std'].append(float(std))
        self._metrics['train']['frac_reward_zero_std'].append(float(std==0))
        model.train()
        return dict(prompt_ids=prompt_ids,prompt_mask=prompt_mask,completion_ids=completion_ids,completion_mask=mask,
            advantages=advantages,old_per_token_logps=behavior,ref_per_token_logps=reference,
            num_items_in_batch=mask.sum())

    def validate_native(self,step):
        folder=self.native.args.output/f'valid_{step:06d}'
        if (folder/'receipt.json').exists():return
        self.native.model=self.accelerator.unwrap_model(self.model_wrapped);self.native.pipe._model=self.native.model
        paths=[]
        for i,row in enumerate(self.native.args.valid_rows):
            self.native.sample(row,5101+i,folder/f'{i:03d}');paths.append(folder/f'{i:03d}'/'audio.flac')
        self.native.offload();scores=self.native.score(paths)
        write(folder/'receipt.json',dict(step=step,n=30,mean=sum(r['mean'] for r in scores)/30,rows=scores,scope='full CMI validation; not test'))
        self.native.model.cuda().train()


class AuditCallback(TrainerCallback):
    def __init__(self):self.owner=None
    def on_train_begin(self,args,state,control,**kwargs):
        if not self.owner.native.args.acceptance_only:self.owner.validate_native(0)
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        gradients={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if not gradients or not all(np.isfinite(v) for v in gradients.values()):raise FloatingPointError('Missing/nonfinite AR gradients')
        if not self.owner.latest_group['zero_std']:
            for branch in ['self_attn','mlp']:
                if not any(v>0 and f'.{branch}.' in n for n,v in gradients.items()):raise ValueError('Missing backbone branch gradient: '+branch)
        self.owner.gradient_receipt=dict(nonzero=sum(v>0 for v in gradients.values()),norm_sum=sum(gradients.values()))
    def on_step_end(self,args,state,control,model=None,**kwargs):
        record=dict(self.owner.latest_group,gradients=self.owner.gradient_receipt)
        if record['step']!=state.global_step:raise ValueError('Update/journal step mismatch')
        for p in model.parameters():
            if p.requires_grad and not torch.isfinite(p).all():raise FloatingPointError('Nonfinite adapter')
        with (self.owner.native.args.output/'steps.jsonl').open('a') as target:target.write(json.dumps(record,allow_nan=False)+'\n')
        write(self.owner.native.args.output/'progress.json',dict(state='training',step=state.global_step,target=args.max_steps,latest=record))
        if state.global_step in [50,100,270] and not self.owner.native.args.acceptance_only:self.owner.validate_native(state.global_step)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','vae','data','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scorer',required=True);p.add_argument('--steps',type=int,default=270)
    p.add_argument('--max-tokens',type=int,default=1500);p.add_argument('--acceptance-only',action='store_true')
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if sha(a.data)!=DATA_SHA:raise ValueError('Frozen projection SHA changed')
    data=json.loads(a.data.read_text())
    if (len(data['train']),len(data['valid']))!=(270,30):raise ValueError('Full frozen split required')
    if a.output.exists():raise FileExistsError('Fresh experiment required; do not silently restart/reuse')
    a.output.mkdir(parents=True);a.valid_rows=data['valid']
    torch.manual_seed(42);policy=NativePolicy(a)
    for row in data['train']+data['valid']:
        text=row['style']+'\n'+row['lyrics']
        if policy.tokenizer.encode(text,add_special_tokens=False)!=policy.tokenizer.native.encode(text):raise ValueError('Native HF tokenization mismatch')
    write(a.output/'experiment.json',dict(model='YuE2-3B',framework='TRL GRPOTrainer; project native rollout/codec likelihood adapter',
        reward='SongBench-Mean7 only; native seven-head arithmetic mean; no extra penalty',
        lr=3e-4,beta=.05,group_size=2,lora_rank=8,lora_alpha=16,lora_parameters=policy.trainable,
        train_n=270,valid_n=30,data_sha256=DATA_SHA,steps=a.steps,max_semantic_tokens=a.max_tokens,
        native_eos=True,seed=42,acceptance_only=a.acceptance_only,source_sha256=sha(Path(__file__)),
        kl_unit='sampled k3 / semantic action against adapter-disabled frozen base; not waveform KL',
        logp_policy='actual BF16-native sampled probabilities retained; teacher uses same codec+EOS support; gap logged and bounded'))
    train=Dataset.from_list([dict(r,prompt=r['style']+'\n'+r['lyrics']) for r in data['train']])
    def reward_audio(completions,**kwargs):return [r['mean'] for r in policy.score([Path(x) for x in completions])]
    callback=AuditCallback()
    config=GRPOConfig(output_dir=str(a.output/'checkpoints'),max_steps=a.steps,learning_rate=3e-4,lr_scheduler_type='constant',
        warmup_ratio=0.,beta=.05,num_generations=2,per_device_train_batch_size=2,gradient_accumulation_steps=1,
        max_completion_length=a.max_tokens,temperature=1.,top_p=1.,top_k=CODEC_SIZE+1,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        bf16=True,logging_steps=1,save_steps=50,report_to='none',
        seed=42,data_seed=42,remove_unused_columns=False,loss_type='grpo',num_iterations=1,
        weight_decay=0.,max_grad_norm=1.,log_completions=False)
    trainer=NativeGRPO(policy,model=policy.model,reward_funcs=reward_audio,args=config,train_dataset=train,
        processing_class=policy.tokenizer,callbacks=[callback])
    callback.owner=trainer
    resumed_step=json.loads((a.resume/'trainer_state.json').read_text())['global_step'] if a.resume else 0
    trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None);trainer.save_model();trainer.save_state()
    if a.acceptance_only:
        records=[json.loads(line) for line in (a.output/'steps.jsonl').open()]
        if len(records)!=a.steps-resumed_step or not any(r['gradients']['nonzero']>0 for r in records):raise ValueError('Acceptance missing real updates')
        write(a.output/'acceptance-complete.json',dict(steps=a.steps,formal_training=False,full_native_60s_budget=a.max_tokens==1500,
            first_kl=records[0]['sampled_kl'],last_kl=records[-1]['sampled_kl'],checkpoint_saved=True,resumed_from_step=resumed_step))
    else:
        if not (a.output/'valid_000270/receipt.json').exists():raise ValueError('Missing full final validation')
        write(a.output/'completion.json',dict(steps=270,full_valid_n=30,test_run=False))


if __name__=='__main__':main()
