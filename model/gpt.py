import os
import math
import pickle
import inspect
import tiktoken
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size:   int   = 1024   
    n_layer:      int   = 12
    n_head:       int   = 12
    n_embd:       int   = 768
    dropout:      float = 0.1

    vocab_size:   int   = 100_277
    temperature:  float = 0.8
    top_k:        int   = 200
    bias:         bool  = True    


def load_config_from_meta(meta_path: str = "data/seccode/meta.pkl") -> GPTConfig:
    cfg = GPTConfig()
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        cfg.vocab_size = meta["vocab_size"]
        print(f"[gpt] Loaded vocab_size={cfg.vocab_size} from {meta_path}")
    else:
        print(f"[gpt] meta.pkl not found at {meta_path}, using default vocab_size={cfg.vocab_size}")
    return cfg


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.dropout = cfg.dropout

        # Fused Q, K, V projection
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att   = F.softmax(att, dim=-1)
        att   = self.attn_drop(att)

        out = att @ v                                       
        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        return self.resid_drop(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),   # positional embeddings
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias),
        ))
        # Language model head (no bias; weights tied to wte below)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying (reduces parameters, standard in GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 * n_layer) — GPT-2 paper §2.3
        for pname, p in self.named_parameters():
            if pname.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        total = sum(p.numel() for p in self.parameters())
        print(f"[GPT] Initialised — parameters: {total / 1e6:.2f}M  |  vocab_size: {cfg.vocab_size}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)        # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)        # (T,  n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float | None = None,
        top_k: int | None = None,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        """
        Auto-regressively generate `max_new_tokens` tokens.

        Args:
            idx:            (B, T) seed token indices.
            max_new_tokens: How many tokens to generate.
            temperature:    Sampling temperature (< 1 = sharper, > 1 = flatter).
            top_k:          Keep only the top-k logits before sampling.
            stop_token:     Stop early if this token is generated (e.g. EOT).

        Returns:
            (B, T + max_new_tokens) tensor.
        """
        temp  = temperature if temperature is not None else self.cfg.temperature
        top_k = top_k       if top_k       is not None else self.cfg.top_k

        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                            else idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp          # (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)

            if stop_token is not None and (idx_next == stop_token).all():
                break

        return idx

    def _encode_prefix(
        self,
        system_prompt: str,
        enc: tiktoken.Encoding,
        device: str,
    ) -> torch.Tensor:
        """Encode a system prompt string into a seed token tensor."""
        tokens = enc.encode(system_prompt)
        return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    @torch.no_grad()
    def generate_yaml(
        self,
        prompt: str,
        enc: tiktoken.Encoding,
        max_new_tokens: int = 512,
        device: str = "cpu",
    ) -> str:
        """
        Generate a Nuclei-style YAML template.

        A structured prefix is prepended so the model starts inside a
        well-formed YAML block.  The raw output is lightly post-processed
        to strip non-YAML tokens.
        """
        system_prefix = (
            "# Nuclei YAML Template\n"
            "id: generated-template\n"
            "info:\n"
            "  name: "
        )
        full_prompt  = system_prefix + prompt + "\n"
        idx          = self._encode_prefix(full_prompt, enc, device)
        output_ids   = self.generate(idx, max_new_tokens=max_new_tokens, temperature=0.6, top_k=100)
        output_text  = enc.decode(output_ids[0].tolist())

        # Trim to the YAML block only
        if "```" in output_text:
            output_text = output_text.split("```")[0]
        return output_text.strip()

    @torch.no_grad()
    def generate_json(
        self,
        prompt: str,
        enc: tiktoken.Encoding,
        max_new_tokens: int = 512,
        device: str = "cpu",
    ) -> str:
        """
        Generate a JSON-structured security payload / scan result.
        """
        system_prefix = '{\n  "scan_type": "' + prompt.strip() + '",\n  '
        idx           = self._encode_prefix(system_prefix, enc, device)
        output_ids    = self.generate(idx, max_new_tokens=max_new_tokens, temperature=0.5, top_k=80)
        output_text   = enc.decode(output_ids[0].tolist())

        # Attempt to extract the first complete JSON object
        brace_depth = 0
        end_pos     = -1
        for i, ch in enumerate(output_text):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_pos = i + 1
                    break

        if end_pos > 0:
            return output_text[:end_pos]
        return output_text.strip()

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Separate parameters into weight-decayed and non-decayed groups.
        (Biases and LayerNorm parameters are never decayed.)
        """
        decay    = {p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2}
        no_decay = {p for n, p in self.named_parameters() if p.requires_grad and p.dim() <  2}

        optim_groups = [
            {"params": list(decay),    "weight_decay": weight_decay},
            {"params": list(no_decay), "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        extra     = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)