# src/quokka/utils.py

import torch

def clear_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()