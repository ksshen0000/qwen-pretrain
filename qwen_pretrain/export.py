"""Export a full checkpoint or milestone as a standard Hugging Face model directory."""
import argparse
import json
import shutil
from pathlib import Path

from .checkpoints import load_model_weights


def export(checkpoint, model_dir, output_dir, dtype='bfloat16'):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory is not empty: {output_dir}')
    weights, info = load_model_weights(checkpoint)
    config = AutoConfig.from_pretrained(model_dir)
    config.use_cache = True
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(weights, strict=True)
    del weights
    model.to(getattr(torch, dtype)).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Write beside the target and rename, so an interrupted export never looks complete.
    temporary = output_dir.with_name(output_dir.name + '.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    (temporary / 'training_info.json').write_text(json.dumps({
        'source_checkpoint': str(checkpoint),
        'step': int(info['step']) if 'step' in info else None,
        'tokens_processed': int(info['tokens_processed']) if 'tokens_processed' in info else None,
        'dtype': dtype,
    }, indent=2) + '\n', encoding='utf-8')

    # Reload what was written and require identical tensors before publishing it.
    # Cast after loading: the dtype keyword was renamed across transformers versions.
    reloaded = AutoModelForCausalLM.from_pretrained(temporary).to(getattr(torch, dtype))
    expected = model.state_dict()
    for key, value in reloaded.state_dict().items():
        if not torch.equal(value, expected[key]):
            raise RuntimeError(f'Exported tensor differs after reload: {key}')
    if output_dir.exists():
        output_dir.rmdir()
    temporary.rename(output_dir)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(prog='qwen-pretrain export', description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path, help='.pt or milestone .safetensors')
    parser.add_argument('--model-dir', default='model', help='Original model dir (config + tokenizer)')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    info = export(args.checkpoint, args.model_dir, args.output_dir, args.dtype)
    print(f'Exported step {info.get("step", "unknown")} to {args.output_dir} ({args.dtype}); '
          f'load with AutoModelForCausalLM.from_pretrained("{args.output_dir}")', flush=True)
