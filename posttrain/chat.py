#!/usr/bin/env python3
"""Interactive multi-turn chat with an SFT model.

    python posttrain/chat.py sft/5b/final

Type /reset to clear the history, /quit (or Ctrl-D) to exit.
"""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--greedy', action='store_true')
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device, dtype).eval()
    options = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                   pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if not args.greedy:
        options.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    history = []
    while True:
        try:
            text = input('\nyou> ').strip()
        except EOFError:
            break
        if text == '/quit':
            break
        if text == '/reset':
            history = []
            print('(history cleared)')
            continue
        if not text:
            continue
        history.append({'role': 'user', 'content': text})
        inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(args.device)
        with torch.inference_mode():
            output = model.generate(**inputs, **options)
        reply = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f'model> {reply}')
        history.append({'role': 'assistant', 'content': reply})


if __name__ == '__main__':
    main()
