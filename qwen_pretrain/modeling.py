"""Shared model construction for training, generation and evaluation."""

PARAM_DTYPES = ('float32', 'bfloat16')


def require_device(device, *, training=False):
    import torch
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unsupported on this GPU')
    elif training:
        raise ValueError('Training currently requires a CUDA GPU')


def build_model(model_dir, device, *, training=False, gradient_checkpointing=False, weights=None,
                param_dtype=None):
    """Inference defaults to BF16 on CUDA. Training passes its configured parameter dtype:
    FP32 master weights under BF16 autocast keep small late-stage updates from rounding away."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    require_device(device, training=training)
    config = AutoConfig.from_pretrained(model_dir)
    config.use_cache = not training
    model = AutoModelForCausalLM.from_config(config)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    if param_dtype is None:
        param_dtype = 'bfloat16' if device == 'cuda' else 'float32'
    if param_dtype not in PARAM_DTYPES:
        raise ValueError(f'param_dtype must be one of {", ".join(PARAM_DTYPES)}')
    model.to(device=device, dtype=getattr(torch, param_dtype))
    model.train(training)
    if training and gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
