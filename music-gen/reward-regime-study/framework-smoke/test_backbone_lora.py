"""Small CPU unit tests for the native-backbone adapter, not model evaluation."""
import unittest
import torch
from torch import nn
from backbone_lora import install, reference


class AdapterTest(unittest.TestCase):
    def test_scope_reference_and_update(self):
        torch.manual_seed(42)
        net=nn.ModuleDict({'backbone':nn.Sequential(nn.Linear(8,12),nn.SiLU(),nn.Linear(12,8)),
                           'head':nn.Linear(8,4)})
        x=torch.randn(2,8)
        before=net['head'](net['backbone'](x)).detach()
        self.assertEqual(install(net,['backbone.']),['backbone.0','backbone.2'])
        self.assertFalse(any(p.requires_grad for p in net['head'].parameters()))
        torch.testing.assert_close(net['head'](net['backbone'](x)),before)
        params=[p for p in net.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=3e-4)
        net['head'](net['backbone'](x)).square().mean().backward();optimizer.step()
        changed=net['head'](net['backbone'](x)).detach()
        self.assertGreater(float((before-changed).abs().max()),0)
        with reference(net):
            torch.testing.assert_close(net['head'](net['backbone'](x)),before)
        torch.testing.assert_close(net['head'](net['backbone'](x)),changed)
        try:
            with reference(net):
                raise RuntimeError('test restoration')
        except RuntimeError:
            pass
        torch.testing.assert_close(net['head'](net['backbone'](x)),changed)

    def test_empty_scope_fails(self):
        with self.assertRaises(ValueError):
            install(nn.Linear(8,8),['not-a-backbone.'])


if __name__=='__main__':unittest.main()
