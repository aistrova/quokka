# src/quokka/config.py


class Quokka230MConfig:
    def __init__(self, vocab_size=50257, n_layer=24, num_attention_heads=16,
                 num_key_value_heads=1, embd_dim=768, layer_dropout=0.0,
                 feedforward_dropout=0.0, max_seq_len=None, intermediate_size=None):
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_layer = n_layer
        self.embd_dim = embd_dim
        self.layer_dropout = layer_dropout
        self.feedforward_dropout = feedforward_dropout

class PretrainConfig:
    # effective batch size
    batch_size = 4  # batch size
    accumulation_steps = 128  # gradient accumulation

    # optimizer
    learning_rate = 1e-4  # SophiaG default
    weight_decay = 1e-1   # SophiaG default
    betas = (0.965, 0.99) # SophiaG default
    rho = 0.04            # SophiaG default
    max_ctx_len = 2048    # max context length
    max_grad_norm = 1.0
    total_steps = None    # set dynamically

    # hf dataset
    hf_dataset = "Skylion007/openwebtext"
    hf_subset = None
    text_col = 'text'
    trust_remote_code = True

    # LFR (https://arxiv.org/pdf/2409.06131)
    p1 = 1   # Learn phase epochs
    p2 = 1   # Focus phase epochs (Phase 2)
    p3 = 1   # Review phase epochs
    p4 = 1   # Focus phase epochs in Phase 4
    reps = 5 # Number of times to repeat Phase 4
    s1 = 50  # Percentage of data blocks to discard in Phase 2 (discard s1% easiest data blocks)
    s2 = 70  # Percentage of data blocks to discard in Phase 4 (discard s2% easiest data blocks)

    num_epochs = p1 + p2 + p3 + (reps * p4)


class GPT3Plus_125M_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 12
        self.num_attention_heads = 12
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 768
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_350M_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 24
        self.num_attention_heads = 16
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 1024
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_760M_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 24
        self.num_attention_heads = 16
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 1536
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_1B_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 24
        self.num_attention_heads = 24
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 2048
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_3B_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 2560
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_6B_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 4096
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_13B_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 40
        self.num_attention_heads = 40
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 5140
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0

class GPT3Plus_175B_Config:
    def __init__(self):
        self.intermediate_size = None
        self.vocab_size = 50257
        self.n_layer = 96
        self.num_attention_heads = 96
        self.num_key_value_heads = 1
        self.max_seq_len = 2048
        self.embd_dim = 12288
        self.layer_dropout = 0.0
        self.feedforward_dropout = 0.0
