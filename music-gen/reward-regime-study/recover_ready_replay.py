"""Temporarily yield a verified CPU-offloaded validation process for ready replay."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path('/storage/zhanh_storage/reward-regimes-20261014')
CODE=Path(__file__).resolve().parent
MODEL=Path('/home/mengh/research/YuE2-posttrain')
PID=3845665


def gpu_memory(pid):
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    return next((int(line.split(',')[1]) for line in lines if int(line.split(',')[0])==pid),0)


def main():
    command=subprocess.check_output(['ps','-p',str(PID),'-o','args='],text=True)
    if 'yue2_shared_baseline.py' not in command or '--reward instrument' not in command:
        raise ValueError('Process identity changed; refusing to pause')
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        state=json.loads((ROOT/'yue2-instrument-lr3e4-seed42'/'progress.json').read_text())
        if state.get('state')=='valid-generation' and state.get('step')==100 and state.get('completed_conditions')==30 and gpu_memory(PID)<1500:
            os.kill(PID,signal.SIGSTOP)
            time.sleep(.3)
            if gpu_memory(PID)<1500:break
            os.kill(PID,signal.SIGCONT)
        time.sleep(1)
    else:raise TimeoutError('No safe offloaded validation window; training not interrupted')
    try:
        print('Instrument safely offloaded and paused; checkpoint100 preserved',flush=True)
        subprocess.run([str(MODEL/'.venv/bin/python'),'-u',str(CODE/'render_yue2_replays.py'),
            '--model',str(MODEL/'models/YuE2-3B'),'--vae',str(MODEL/'models/YuE2-Vae'),
            '--data',str(MODEL/'examples/cmi-pref-triple-text-lyrics-9to1-v2.json'),
            '--root',str(ROOT),'--rewards','richness','--skip-scoring'],check=True,timeout=1500)
    finally:
        os.kill(PID,signal.SIGCONT)
        print('Instrument continued; finite training queue preserved',flush=True)


if __name__=='__main__':main()
