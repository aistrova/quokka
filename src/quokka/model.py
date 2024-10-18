import torch.nn as nn
from quokka.transformer import (
    AttentionLayers,
    TokenEmbedding,
    RMSNorm
)
from quokka.config import Quokka230MConfig

class Quokka(nn.Module):
    def __init__(
        self,
        config: Quokka230MConfig,
        emb_dropout=0.0,
        emb_frac_gradient=1.0,
        tie_embedding=False,
        use_abs_pos_emb=False,  # Using RotaryEmbedding as default
        post_emb_norm=False,
        pretrained=False,
        **kwargs
    ):
        print("Initializing the model...")
        super().__init__()
        self.pretrained = pretrained
        self.max_seq_len = config.max_seq_len
        self.emb_dim = config.embd_dim
        self.num_tokens = config.vocab_size
        self.emb_frac_gradient = emb_frac_gradient

        # Token Embedding
        self.token_emb = TokenEmbedding(self.emb_dim, self.num_tokens)

        # Post-embedding LayerNorm
        self.ln_f = RMSNorm(self.emb_dim)

        # Embedding Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.post_emb_norm = (
            nn.LayerNorm(config.embd_dim) if post_emb_norm else nn.Identity()
        )

        # Project embedding to match model dimension if necessary
        self.project_emb = (
            nn.Linear(self.emb_dim, config.embd_dim) if self.emb_dim != config.embd_dim else nn.Identity()
        )

        kv_heads = config.num_key_value_heads

        self.decoder = AttentionLayers(
            dim=config.embd_dim,
            depth=config.n_layer,
            heads=config.num_attention_heads,
            attn_kv_heads=kv_heads if kv_heads > 1 else None,
            use_abs_pos_emb=use_abs_pos_emb,
            causal=True,
            rotary_pos_emb=True,
            attn_one_kv_head=True if kv_heads == 1 else None,  # multiquery attention
            attn_flash=True,
            qk_norm=True,
            qk_norm_dim_scale=True,
            layer_dropout=config.layer_dropout,
            ff_inner_dim=config.intermediate_size,
            **kwargs
        )

        # Vocabulary mapping
        self.vocab_proj = (
            nn.Linear(config.embd_dim, self.num_tokens, bias=False)
            if not tie_embedding
            else lambda t: t @ self.token_emb.emb.weight.t()
        )

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def train(self, mode=True):
        if not self.pretrained:
            # Randomize weights before training
            # Training = prediction -> error/loss -> optimization -> update weights -> repeat
            self._init_weights()
            self.pretrained = True
        
        return super().train(mode)

    def _init_weights(self):
        nn.init.normal_(self.token_emb.emb.weight, mean=0.0, std=0.02)
        if isinstance(self.vocab_proj, nn.Linear):
            nn.init.normal_(self.vocab_proj.weight, mean=0.0, std=0.02)
            if self.vocab_proj.bias is not None:
                nn.init.zeros_(self.vocab_proj.bias)
        self.decoder.apply(self._init_module_weights)

    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x,  # Token indices, same as in Transformer class
        targets=None,
        mask=None,
        mems=None,
        return_loss=False,
        return_embeddings=False,
        return_hiddens=False,
        **kwargs
    ):
        # Token Embeddings with optional positional embeddings (Rotary by default)
        x = self.token_emb(x)

        # Apply fractional gradient to embeddings if specified
        if self.emb_frac_gradient < 1.0:
            x = x * self.emb_frac_gradient + x.detach() * (1 - self.emb_frac_gradient)

        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        # feedforward layers
        decoder_output = self.decoder(x, mask=mask, mems=mems, return_hiddens=return_hiddens, **kwargs)

        if return_hiddens:
            x, intermediates = decoder_output
        else:
            x = decoder_output
            intermediates = None

        if return_embeddings:
            return x

        if isinstance(self.vocab_proj, nn.Linear):
            logits = self.vocab_proj(x)
        else:
            logits = self.vocab_proj(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        if return_loss:
            return logits, loss

        if return_hiddens:
            return logits, intermediates

        return logits

    @property
    def can_cache_kv(self):
        return False
