"""CPU tests using the installed native YuE2, including cache and gradients."""
import copy
import unittest
import torch
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2_checkpointing import install
torch.set_num_threads(1)


class CheckpointingTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(42)
        config = YuE2Config(vocab_size=512,hidden_size=64,intermediate_size=128,
                           num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                           head_dim=16,latent_dim=16,max_latent_frames=100)
        base = YuE2ForCausalLM(config).float()
        checked = copy.deepcopy(base)
        keys = list(checked.state_dict())
        install(checked);install(checked)
        self.assertEqual(keys,list(checked.state_dict()))
        checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.assertTrue(checked.model.gradient_checkpointing)
        return base,checked

    def test_real_native_gradients(self):
        base,checked=self.models()
        ids=torch.randint(0,512,(1,9))
        losses=[]
        for model in [base,checked]:
            model.train()
            out=model(ids,use_cache=False).logits.square().mean()
            out.backward();losses.append(out.detach())
        torch.testing.assert_close(*losses,rtol=1e-5,atol=1e-7)
        for (name,a),(other,b) in zip(base.named_parameters(),checked.named_parameters()):
            self.assertEqual(name,other)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
        checked.gradient_checkpointing_disable()
        self.assertFalse(checked.model.gradient_checkpointing)

    def test_eval_cache_unchanged(self):
        base,checked=self.models()
        ids=torch.randint(0,512,(1,5))
        values=[]
        for model in [base,checked]:
            model.eval()
            with torch.no_grad():
                first=model(ids,use_cache=True)
                second=model(ids[:,:1],past_key_values=first.past_key_values,use_cache=True)
            self.assertEqual(second.past_key_values.get_seq_length(),6)
            values.append(second.logits)
        torch.testing.assert_close(*values,rtol=1e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
