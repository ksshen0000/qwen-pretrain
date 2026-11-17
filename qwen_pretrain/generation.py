"""Generation from full checkpoints or permanent model-only milestones."""
import argparse
import math
from pathlib import Path

from .checkpoints import load_model_weights
from .modeling import build_model
from .prompts import PROMPTS


def generate_text(model, tokenizer, prompt, device, max_new_tokens=80, *, sample=False,
                  temperature=.8, top_p=.9):
    import torch
    batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    options = dict(max_new_tokens=max_new_tokens, do_sample=sample,
                   pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                   use_cache=True)
    if sample:
        options.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        output = model.generate(**batch, **options)
    return tokenizer.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='.pt or .safetensors')
    parser.add_argument('--model-dir', default='model')
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--output', type=Path, default=Path('logs/generation_samples.txt'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--prompt', action='append', help='Repeat for multiple prompts; defaults to fixed English probes')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-p', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if (args.max_new_tokens < 1 or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        parser.error('Require positive max-new-tokens/temperature and 0 < top-p <= 1')
    import torch
    from transformers import AutoTokenizer
    from .io import atomic_write
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    weights, info = load_model_weights(args.checkpoint)
    model = build_model(args.model_dir, args.device, weights=weights)
    del weights
    step = info.get('step', 'unknown')
    lines = [f'Checkpoint: {args.checkpoint}', f'Step: {step}',
             f'Decoding: {"sampling" if args.sample else "greedy"}', '']
    for index, prompt in enumerate(args.prompt or PROMPTS, 1):
        continuation = generate_text(model, tokenizer, prompt, args.device, args.max_new_tokens,
                                     sample=args.sample, temperature=args.temperature, top_p=args.top_p)
        block = f'[{index}] Prompt: {prompt}\nContinuation: {continuation}\n'
        print(block, flush=True)
        lines.append(block)
    atomic_write(args.output, lambda temporary: temporary.write_text('\n'.join(lines), encoding='utf-8'))
    print(f'Saved results to {args.output}', flush=True)
