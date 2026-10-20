import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_muse_base
from runtime_entry import enter


class StartupTests(unittest.TestCase):
    def test_container_entry_has_no_evaluation_module_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(enter(Path(folder)))

    def test_exact_original_base_and_adapter_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'model.safetensors').write_bytes(b'base')
            (root/'config.json').write_text(json.dumps({'architectures':['Qwen3ForCausalLM']}))
            with patch.object(check_muse_base,'PUBLISHED_SHA',hashlib.sha256(b'base').hexdigest()):
                self.assertFalse(check_muse_base.check(root)['starting_from_grpo'])
                (root/'adapter_config.json').write_text('{}')
                with self.assertRaises(ValueError): check_muse_base.check(root)
                (root/'adapter_config.json').unlink()
                (root/'model.safetensors').write_bytes(b'merged-grpo')
                with self.assertRaises(ValueError): check_muse_base.check(root)


if __name__=='__main__': unittest.main()
