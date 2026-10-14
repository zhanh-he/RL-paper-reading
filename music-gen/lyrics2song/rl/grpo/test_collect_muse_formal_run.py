import json
import tempfile
import unittest
from pathlib import Path
from collect_muse_formal_run import collect, TRAIN_SHA, SOURCE_SHA


def fixture(root, start, end, resume=None):
    root.mkdir()
    experiment = {'muse_train_jsonl_sha256': TRAIN_SHA, 'source_split_sha256': SOURCE_SHA,
                  'train_conditions': 270, 'valid_conditions': 30, 'input_modalities': ['text', 'lyrics'],
                  'reference_audio_used': False, 'reward': 'musecritic', 'learning_rate': '3e-4'}
    if resume:
        experiment.update(resume_step=180, resume_checkpoint=str(resume), restores_optimizer_scheduler_rng=True)
    (root / 'experiment.json').write_text(json.dumps(experiment))
    logdir = root / 'checkpoint/v0'
    logdir.mkdir(parents=True)
    (logdir / 'logging.jsonl').write_text(''.join(json.dumps({'global_step/max_steps': f'{s}/270', 'reward': 0.0, 'kl': .01})+'\n' for s in range(start, end+1)))
    rows = [{'audio_path_0': '', 'audio_path_1': '', 'reward_0': 0., 'reward_1': 0.,
             'completion_0_request_id': f'{s}-0', 'completion_1_request_id': f'{s}-1'} for s in range(start, end+1)]
    (root / 'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root / 'invalid_outputs.jsonl').write_text(''.join(json.dumps({'request_id': f'{s}-{i}', 'reward': 0., 'reason': 'no AUDIO tokens'})+'\n' for s in range(start, end+1) for i in range(2)))
    checkpoint = logdir / f'checkpoint-{180 if end == 200 else end}'
    checkpoint.mkdir()
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'unit-test fixture')
    (root / 'train.log').write_text('RuntimeError: decoder failure' if end == 200 else '')
    return checkpoint


class CollectMuseTests(unittest.TestCase):
    def test_resume_discards_superseded_steps_and_keeps_zero_penalties(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = fixture(Path(tmp)/'base', 1, 200)
            resumed = Path(tmp)/'resumed'
            fixture(resumed, 181, 270, base)
            result = collect(resumed)
            self.assertEqual(result['verified_train_steps'], 270)
            self.assertEqual([r['step'] for r in result['train_curve']], list(range(1, 271)))
            self.assertEqual(result['invalid_output_count'], 540)
            self.assertEqual(result['checkpoints'], [180, 270])
            self.assertEqual(result['status'], 'verified-training-complete')

    def test_missing_audio_needs_explicit_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'base'
            fixture(root, 1, 200)
            (root/'invalid_outputs.jsonl').unlink()
            with self.assertRaisesRegex(ValueError, 'Unexplained missing audio'):
                collect(root)


if __name__ == '__main__':
    unittest.main()
