"""CPU checks for exact categorical rollout/recompute and frozen reference."""
import unittest
import torch
from open_head_policy import HeadPolicy


class PolicyTests(unittest.TestCase):
    def test_identity_gradient_and_reference(self):
        torch.manual_seed(42)
        head = HeadPolicy(torch.nn.Linear(12, 7))
        head.capture = True
        features = torch.randn(2, 12)
        head(features)
        _, record = head.discrete_action(1.5)
        logp, kl = head.discrete_terms(record)
        self.assertEqual(float(kl.detach()), 0.)
        self.assertEqual(float(logp.detach()), float(record['old_logprob']))
        (-logp + .01*kl).backward()
        self.assertIsNone(head.base.weight.grad)
        self.assertGreater(float(head.b.grad.abs().sum()), 0.)
        with torch.no_grad(): head.b.add_(.05*torch.randn_like(head.b))
        self.assertGreater(float(head.discrete_terms(record)[1].detach()), 0.)
        head.active = False
        self.assertTrue(torch.equal(head.logits(features), head.logits(features, reference=True)))

    def test_frozen_audio_shared_across_rewards(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from train_open_head_grpo import frozen_sample
        class Backend:
            head = SimpleNamespace(active=True)
            calls = 0
            def metadata(self): return {'policy': 'test-only'}
            def sample(self, row, seed):
                self.calls += 1
                return np.zeros(24000, dtype=np.float32), 24000, [{'test': True}]
        backend = Backend(); row={'condition_id':'test-only','style':'piano','lyrics':'test-only'}
        def scored(backend, args, paths):
            from rewards import sha
            return [{'audio_sha256':sha(paths[0]),'reward':args.reward,'mean':.3,
                     'guard':{'invalid':False,'penalty':0},'richness':{'raw':.3},
                     'songbench':{'Instrumental':5,'Arrangement':4}}]
        with tempfile.TemporaryDirectory() as directory, patch('train_open_head_grpo.score', side_effect=scored) as scorer:
            args=SimpleNamespace(output=Path(directory)/'arm-a',model='test-only',reward='richness')
            a,meta,first=frozen_sample(backend,args,row,5101)
            args.output=Path(directory)/'arm-b';args.reward='instrument'
            b,_,second=frozen_sample(backend,args,row,5101)
            self.assertEqual(a,b);self.assertEqual(backend.calls,1);self.assertEqual(scorer.call_count,1)
            self.assertEqual(first['audio_sha256'],second['audio_sha256'])
            self.assertEqual(first['mean'],.3);self.assertEqual(second['mean'],.5)
            self.assertTrue(backend.head.active)


if __name__ == '__main__': unittest.main()
