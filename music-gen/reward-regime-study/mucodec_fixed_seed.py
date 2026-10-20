"""Run the native codec API with reproducible decoding, without editing shared code."""
import argparse
import importlib.util
import sys
from pathlib import Path
import torch


def install_fixed_seed(server_class, seed=42):
    original=server_class._decode_tokens

    def decode(self, request):
        # The native endpoint already holds its inference lock around this call.
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed)
                return original(self, request)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    server_class._decode_tokens=decode


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    spec=importlib.util.spec_from_file_location('regime_native_mucodec',args.source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    install_fixed_seed(module.MuCodecServer)
    sys.argv=[str(args.source),*remaining]
    module.main()
