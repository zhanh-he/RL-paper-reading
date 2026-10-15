"""Start an isolated native codec server and report unique loaded parameters."""
import importlib.util
import json
import os
from pathlib import Path
import sys

source=Path(os.environ['COSTS_CODEC_SOURCE'])
spec=importlib.util.spec_from_file_location('native_cost_codec',source)
native=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=native
spec.loader.exec_module(native)
classes=[value for value in vars(native).values() if isinstance(value,type)
         and value.__module__==native.__name__ and hasattr(value,'load_model')]
if len(classes)!=1:
    raise ValueError('Expected exactly one native codec service')
service=classes[0]
original=service.load_model


def load(self):
    original(self)
    import torch
    seen=set(); components={}
    for name,value in vars(self.model).items():
        if isinstance(value,torch.nn.Module):
            parameters=[p for p in value.parameters() if id(p) not in seen]
            seen.update(id(p) for p in parameters)
            components[name]=sum(p.numel() for p in parameters)
    Path(os.environ['COSTS_CODEC_COUNT']).write_text(json.dumps(dict(
        unique_parameters=sum(components.values()),components=components,
        device=self.device,scope='Native loaded MuCodec; shared nn.Parameters counted once, fixed buffers excluded'),indent=2)+'\n')


service.load_model=load
native.main()
