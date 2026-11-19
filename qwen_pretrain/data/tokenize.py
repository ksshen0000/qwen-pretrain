"""Worker-local tokenizers and ordered Parquet document batches."""
import os

_TOKENIZER = None
_EOS = None


def init_worker(model_dir):
    global _TOKENIZER, _EOS
    from transformers import AutoTokenizer
    # Avoid Rust tokenizers spawning extra threads inside each process.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    _EOS = _TOKENIZER.eos_token_id
    if _EOS is None:
        raise ValueError('Tokenizer has no eos_token_id')


def tokenize_docs(docs):
    # A batch job, serialized back to the parent in the original order.
    texts = [text if isinstance(text, str) else '' for text in docs]
    # Use the native fast tokenizer's batched API, without padding/truncation.
    encoded = _TOKENIZER(texts, add_special_tokens=False, truncation=False,
                         padding=False, return_attention_mask=False,
                         return_token_type_ids=False, verbose=False)['input_ids']
    return [ids + [_EOS] if text else [] for text, ids in zip(texts, encoded)]


def doc_batches(paths, rows):
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=rows, columns=['text']):
            yield batch.column('text').to_pylist()

