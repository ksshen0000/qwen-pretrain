"""Fixed small probes; scoring is shared across milestone evaluations."""


class ProbeScorer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def continuation(self, prefix, suffix):
        import torch
        tokenizer = self.tokenizer
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        all_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
        if len(all_ids) <= len(prefix_ids) or all_ids[:len(prefix_ids)] != prefix_ids:
            prefix = prefix.rstrip() + '\n'
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            all_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
        if (not prefix_ids or len(all_ids) <= len(prefix_ids)
                or all_ids[:len(prefix_ids)] != prefix_ids):
            raise ValueError('Tokenizer boundary mismatch; adjust candidate prompt')
        with torch.inference_mode():
            ids = torch.tensor([all_ids], device=self.device)
            logits = self.model(input_ids=ids, use_cache=False).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            start = len(prefix_ids)
            positions = torch.arange(start - 1, len(all_ids) - 1, device=self.device)
            scores = log_probs[positions, ids[0, start:]]
        return scores.sum().item()

    def sentence(self, sentence):
        import torch
        ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        if len(ids) < 2:
            return float('-inf')
        with torch.inference_mode():
            tokens = torch.tensor([ids], device=self.device)
            logits = self.model(input_ids=tokens, use_cache=False).logits[0, :-1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            return log_probs.gather(-1, tokens[0, 1:].unsqueeze(-1)).mean().item()
