#!/usr/bin/env python3
"""Supervised fine-tuning (chat format) of an exported model with TRL.

Input is a Hugging Face model directory from `python -m qwen_pretrain export`.
Datasets need a `messages` column of {"role", "content"} turns, either a Hub
name (default: HuggingFaceTB/smol-smoltalk) or a local .json/.jsonl file.

    python posttrain/sft.py --base exported/5b --output-dir sft/5b --limit 2000   # trial
    python posttrain/sft.py --base exported/5b --output-dir sft/5b                # full, 1 epoch

Rerunning the same command after an interruption resumes from the latest
checkpoint in --output-dir.
"""
import argparse
from pathlib import Path

# Plain ChatML without Qwen3's <think> blocks: the official template adds an empty
# think block to the final reply only, so training and generation formats would
# disagree. {% generation %} marks reply tokens for assistant-only loss.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' }}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}{{ '\\n' }}"
    "{%- else %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)


def load_splits(name, limit, eval_samples, seed):
    from datasets import load_dataset
    data = load_dataset('json', data_files=name) if Path(name).is_file() else load_dataset(name)
    if 'test' not in data:
        # Hold out at most 5% when the dataset ships without a test split.
        size = min(eval_samples, max(1, len(data['train']) // 20))
        data = data['train'].train_test_split(test_size=size, seed=seed)
    train, test = data['train'], data['test']
    if 'messages' not in train.column_names:
        raise ValueError(f'Dataset needs a "messages" column; found {train.column_names}')
    # Drop unrelated columns so the trainer sees only conversations.
    train = train.select_columns(['messages'])
    test = test.select_columns(['messages'])
    if limit:
        train = train.shuffle(seed=seed).select(range(min(limit, len(train))))
    return train, test.select(range(min(eval_samples, len(test))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Exported HF model directory')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset', default='HuggingFaceTB/smol-smoltalk')
    parser.add_argument('--limit', type=int, help='Use only this many training conversations')
    parser.add_argument('--eval-samples', type=int, default=500)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    if not cuda:
        print('WARNING: no CUDA GPU; running in FP32 on CPU (only useful for smoke tests)', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if '<|im_end|>' not in tokenizer.get_vocab():
        raise SystemExit(f'{args.base} tokenizer lacks ChatML tokens; export with the Qwen tokenizer')
    tokenizer.chat_template = CHAT_TEMPLATE  # Saved with the final model, so chat.py matches.
    tokenizer.eos_token = '<|im_end|>'        # Generation stops at the end of a reply.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # FP32 master weights; on CUDA, bf16=True computes under BF16 autocast.
    model = AutoModelForCausalLM.from_pretrained(args.base).float()
    train, test = load_splits(args.dataset, args.limit, args.eval_samples, args.seed)
    print(f'train={len(train):,} eval={len(test):,} conversations', flush=True)

    # transformers 5 folded warmup_ratio into a fractional warmup_steps.
    from dataclasses import fields
    warmup = ({'warmup_ratio': 0.03} if 'warmup_ratio' in {f.name for f in fields(SFTConfig)}
              else {'warmup_steps': 0.03})
    config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        assistant_only_loss=True,  # Learn the replies, not the user turns.
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        num_train_epochs=args.epochs,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=cuda,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to='none',
        **warmup,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, eval_dataset=test,
                         processing_class=tokenizer)
    last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
    if last:
        print(f'[resume] {last}', flush=True)
    trainer.train(resume_from_checkpoint=last)
    final = Path(args.output_dir) / 'final'
    trainer.model.to(torch.bfloat16)  # Inference precision; halves the file size.
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f'Saved {final}', flush=True)


if __name__ == '__main__':
    main()
