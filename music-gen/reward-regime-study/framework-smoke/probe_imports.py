"""Bounded real framework import probes; never a training success receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
checks={
    'miles_actor':('miles','from miles.backends.fsdp_utils.actor import FSDPTrainRayActor'),
    'miles_loss':('miles','from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula'),
    'verl_omni_loss':('verl-omni','from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss'),
    'trl':('trl',"import importlib; importlib.import_module('trl.import_utils')._vllm_available=False; from trl import GRPOTrainer"),
}
rows=[]
for name,(folder,code) in checks.items():
    tick=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(a.root/'sources'/folder)+os.pathsep+os.environ.get('PYTHONPATH','')}
    try:
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=120)
        rows.append({'name':name,'returncode':result.returncode,'stdout':result.stdout[-6000:],'stderr':result.stderr[-6000:],
                     'seconds':time.monotonic()-tick})
    except subprocess.TimeoutExpired as error:
        rows.append({'name':name,'returncode':None,'error':'120-second import timeout','seconds':time.monotonic()-tick})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'scope':'Import tests only, no GPU/model/optimizer pass claimed','python':sys.executable,'rows':rows},indent=2)+'\n')
print(json.dumps([{'name':r['name'],'returncode':r['returncode'],'last_error':r.get('stderr',r.get('error','')).splitlines()[-1:]} for r in rows]),flush=True)
