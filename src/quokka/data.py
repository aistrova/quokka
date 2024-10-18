# src/quokka/data.py

import torch
from torch.utils.data import IterableDataset, DataLoader
from quokka.config import PretrainConfig
import tiktoken
from datasets import load_dataset
from typing import Literal

class ShuffledIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_ctx_len, shuffle_buffer_size, start_idx=0, end_idx=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.start_idx = start_idx  # starting index for this dataset
        self.end_idx = end_idx  # ending index (None means no limit)

    def __iter__(self):
        buffer = []
        token_buffer = []
        data_block_id = 0  # LFR data block ID
        for example in self.dataset:
            if data_block_id < self.start_idx:
                continue
            if self.end_idx is not None and data_block_id >= self.end_idx:
                break
            tokens = ([self.tokenizer.bos] + self.tokenizer.encode(example[PretrainConfig.text_col], disallowed_special=()))[:self.max_ctx_len - 1] + [self.tokenizer.eos]
            token_buffer.extend(tokens)
            while len(token_buffer) >= self.max_ctx_len:
                input_ids = token_buffer[:self.max_ctx_len]
                token_buffer = token_buffer[self.max_ctx_len:]
                buffer.append({'input_ids': torch.tensor(input_ids, dtype=torch.long), 'id': data_block_id})
                data_block_id += 1  # Increment data block ID
                if len(buffer) >= self.shuffle_buffer_size:
                    indices = torch.randperm(len(buffer))
                    for idx in indices:
                        item = buffer[idx]
                        labels = item['input_ids'].clone()
                        yield {'input_ids': item['input_ids'], 'labels': labels, 'id': item['id']}
                    buffer = []
        # yield any remaining items in the buffer
        if buffer:
            indices = torch.randperm(len(buffer))
            for idx in indices:
                item = buffer[idx]
                labels = item['input_ids'].clone()
                yield {'input_ids': item['input_ids'], 'labels': labels, 'id': item['id']}

class EmptyIterableDataset(IterableDataset):
    def __iter__(self):
        return iter([])

class FilteredIterableDataset(IterableDataset):
    def __init__(self, dataset, selected_ids):
        self.dataset = dataset
        self.selected_ids = set(selected_ids)

    def __iter__(self):
        for item in self.dataset:
            if item['id'] in self.selected_ids:
                yield item

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)
    ids = [item['id'] for item in batch]
    return {'input_ids': input_ids, 'labels': labels, 'ids': ids}

def get_tokenizer(
    encoding_type="p50k_base",
    bos=None,  # Allows for custom BOS token. Defaults set based on encoding_type.
    eos=None,  # Allows for custom EOS token. Defaults set based on encoding_type.
    custom_name="s"
):
    # p50k_base.max_token_value = 50280
    # o200k_base.max_token_value = 200018
    # cl100k_base.max_token_value = 100276
    enc = tiktoken.get_encoding(encoding_type)

    if bos is None:
        bos = {
            "p50k_base": 50281,
            "o200k_base": 200019,
            "cl100k_base": 100277
        }.get(encoding_type, 50281)  # default to p50k_base's value if encoding_type is unrecognized
    
    if eos is None:
        eos = {
            "p50k_base": 50282,
            "o200k_base": 200020,
            "cl100k_base": 100278
        }.get(encoding_type, 50282)  # default to p50k_base's value if encoding_type is unrecognized

    tokenizer = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name=f"{encoding_type}_{custom_name}",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={
            **enc._special_tokens,
            "<|im_start|>": bos,
            "<|im_end|>": eos,
        }
    )
    tokenizer.bos = bos
    tokenizer.eos = eos
    return tokenizer

def get_dataloaders(
        CACHE_DIR,
        DATASET_SIZE,
        PERCENT_TRAIN,
        openai_encoding: Literal['p50k_base', 'o200k_base', 'cl100k_base']="p50k_base"
    ):
    import os
    os.environ['HF_HOME'] = CACHE_DIR  # environment variable for transformers v5
    os.environ['HUGGINGFACE_HOME'] = CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
    os.environ['HF_DATASETS_DOWNLOADED_DATASETS_PATH'] = CACHE_DIR

    print("Loading dataset...")

    tokenizer = get_tokenizer(openai_encoding)

    dataset = load_dataset(PretrainConfig.hf_dataset, PretrainConfig.hf_subset, split='train', streaming=True, cache_dir=CACHE_DIR, trust_remote_code=PretrainConfig.trust_remote_code)

    use_partial_dataset = isinstance(DATASET_SIZE, int)
    train_size = int(PERCENT_TRAIN * DATASET_SIZE) if use_partial_dataset else None

    train_dataset = ShuffledIterableDataset(dataset, tokenizer, PretrainConfig.max_ctx_len, shuffle_buffer_size=1000, start_idx=0, end_idx=(train_size if use_partial_dataset else None))
    val_dataset = ShuffledIterableDataset(dataset, tokenizer, PretrainConfig.max_ctx_len, shuffle_buffer_size=1000, start_idx=train_size, end_idx=DATASET_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=PretrainConfig.batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=PretrainConfig.batch_size, collate_fn=collate_fn) if PERCENT_TRAIN < 1.0 else DataLoader(EmptyIterableDataset())

    return train_dataloader, val_dataloader, tokenizer