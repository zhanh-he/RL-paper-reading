"""Frozen SongBench model/weights, Blackwell-compatible metrics Torch runtime.

Official eval Torch 2.7.0+cu126 has no sm_120 kernels. This is explicitly
a runtime extension, not a successful frozen-environment acceptance test.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.bundle), str(a.bundle / 'src'), str(a.bundle / 'vendor')]
    import torch
    import wsb_eval as frozen
    from mot_infer.evaluation import SongBench
    from mot_infer.extended_eval import save_result
    from mot_infer.eval_assets import digest, object_hash
    frozen.verify_release()
    frozen.verify_assets(['songbench'])
    # Verify the unchanged Blackwell-capable metric environment; never rewrite
    # the incompatible eval lock to manufacture a passing acceptance receipt.
    frozen.verify_environment('mulan', a.jobs.parent / 'songbench.compat-environment.json')
    data = json.loads(a.jobs.read_text())
    torch.manual_seed(data['config']['seed'])
    scorer = SongBench(a.bundle)
    failures = 0
    for row in data['jobs']:
        try:
            if digest(row['audio_path']) != row['audio_sha256']:
                raise ValueError('Scoring MP3 changed')
            result = scorer.score(Path(row['audio_path']))
            raw = {name: result['songbench_raw'][name] for name in frozen.DIMENSIONS}
            result.update(status='complete', scores={**raw, 'Global Avg': sum(raw.values()) / 7},
                          runtime_extension={'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
                              'shim_sha256': digest(Path(__file__)), 'reason': 'Official cu126 eval build lacks sm_120'})
            object_hash(result)
        except Exception as exc:
            result = {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}
            failures += 1
        save_result(Path(row['result_path']), row['identity'], result)
        print(f'SongBench {row["id"]}: {result["status"]}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
