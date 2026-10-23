"""Study-local activation checkpointing without changing native weight names."""
from functools import wraps
import torch


def install(model):
    backbone = model.model
    if getattr(backbone, '_study_checkpointing_installed', False):
        return
    if not getattr(backbone, 'layers', None):
        raise ValueError('Expected the native YuE2 decoder backbone')
    backbone.gradient_checkpointing = False

    def wrap(layer):
        original = layer.forward

        @wraps(original)
        def forward(*args, **kwargs):
            if backbone.gradient_checkpointing and layer.training and torch.is_grad_enabled():
                cache = args[3] if len(args) > 3 else kwargs.get('past_key_value')
                if cache is not None:
                    raise ValueError('Training activation checkpointing requires use_cache=False')
                return backbone._gradient_checkpointing_func(original, *args, **kwargs)
            return original(*args, **kwargs)
        layer.forward = forward

    for layer in backbone.layers:
        wrap(layer)
    backbone._study_checkpointing_installed = True
