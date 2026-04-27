import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from harddrop.layers.DropPath import DropPath

class MultiheadAttentionWithLogits(nn.Module):
    """
    Multi-head attention that exposes raw attention logits.
    
    This implementation manually computes multi-head attention and returns:
    1. The attention output (like nn.MultiheadAttention)
    2. The attention weights (if requested)
    3. The raw attention logits averaged over heads (for routing)
    
    Supports both self-attention and cross-attention scenarios.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights
        batch_first: If True, input/output tensors are (batch, seq, feature)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Head dimension
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Separate Q, K, V projections for cross-attention efficiency
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, need_weights=False):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [B, M, D]
            key: Key tensor [B, N, D]
            value: Value tensor [B, N, D]
            need_weights: Whether to return attention weights
            
        Returns:
            attn_output: [B, M, D] - the attention output
            attn_weights: [B, M, N] or None - attention probabilities averaged over heads
            attn_logits: [B, M, N] - raw attention scores averaged over heads
        """
        if not self.batch_first:
            # Convert to batch_first
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, M, D = query.shape
        N = key.shape[1]
        
        # Project Q, K, V separately for cross-attention
        q = self.q_proj(query)  # [B, M, D]
        k = self.k_proj(key)    # [B, N, D]
        v = self.v_proj(value)  # [B, N, D]
        
        # Reshape for multi-head: [B, L, D] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        q = q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, M, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        
        # Compute attention scores: [B, num_heads, M, N]
        attn_logits_heads = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, M, N]
           
        # Compute attention weights (softmax over key dimension)
        attn_weights_heads = F.softmax(attn_logits_heads, dim=-1)  # [B, num_heads, M, N]
        
        # Apply dropout
        attn_weights_heads = self.attn_dropout(attn_weights_heads)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights_heads, v)  # [B, num_heads, M, head_dim]
        
        # Concatenate heads: [B, num_heads, M, head_dim] -> [B, M, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, M, D)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)  # [B, M, D]
        
        # Average logits over heads for routing: [B, num_heads, M, N] -> [B, M, N]
        attn_logits = attn_logits_heads.mean(dim=1)  # [B, M, N]
        
        # Prepare attention weights output (averaged over heads to match PyTorch API)
        if need_weights:
            weights_out = attn_weights_heads.mean(dim=1)  # [B, M, N]
        else:
            weights_out = None
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, weights_out, attn_logits

class TokenCompressLayer(nn.Module):
    """
    Token compressive layer with attention-guided routing.
    
    Computationally efficient version that:
      - Computes explicit SMB↔token score tensor for routing (SMB = summary bank)
      - Uses hard top-1 routing to avoid full MxN residual computation
    
    Args:
        d_model: embedding dimension
        num_heads: number of attention heads
        num_smb: number of SMB (summary bank) tokens
        pos_dim: dimension of positional embedding per token
        mlp_ratio: expansion ratio for SMB MLP + decoder hidden size
        dropout: dropout probability in attention and MLPs
        residual_norm_type: "l2" or "l1" for residual norm computation
        smb_mhsa: if True, add an optional self-attention block on SMB only (smb<->smb) after MLP on SMB (default: False)
        drop_path: stochastic depth rate for compressive layer residual branches (default: 0.0)
        decoder_dim_multiplier: compression ratio in (0, 1]; SMB and pos are projected from d_model/pos_dim to
            decoder_dim_multiplier * d_model before concat. Default 1 uses nn.Identity() for backwards compatibility.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_smb: int,
        pos_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual_norm_type: str = "l2",
        smb_mhsa: bool = False,
        drop_path: float = 0.0,
        decoder_dim_multiplier: float = 1.0,
        return_residual_tokens: bool = True,
        recon_loss_dropped_only: bool = False,
    ):
        super().__init__()
        self.num_smb = num_smb
        self.d_model = d_model
        self.pos_dim = pos_dim
        self.return_residual_tokens = bool(return_residual_tokens)
        self.recon_loss_dropped_only = bool(recon_loss_dropped_only)

        self.residual_norm_type = str(residual_norm_type).lower()
        if self.residual_norm_type not in {"l2", "l1"}:
            raise ValueError(
                f"residual_norm_type must be 'l2' or 'l1', got {self.residual_norm_type}"
            )
        
        # Routing: fixed K=1 argmax routing (argmax over attention logits)
        self.smb_mhsa = bool(smb_mhsa)
        self._drop_path_rate = float(drop_path)

        # SMB-to-token attention (queries = SMB, keys/values = tokens)
        self.ln_smb_attn = nn.LayerNorm(d_model)
        self.ln_x_attn = nn.LayerNorm(d_model)
        self.attn = MultiheadAttentionWithLogits(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP on SMB bank
        hidden = int(d_model * mlp_ratio)
        self.ln_smb_mlp = nn.LayerNorm(d_model)
        self.mlp_smb = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # Optional self-attention on SMB only (smb<->smb), after MLP and before decoding
        if self.smb_mhsa:
            self.ln_smb_mhsa = nn.LayerNorm(d_model)
            self.smb_mhsa_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.ln_smb_mhsa = None
            self.smb_mhsa_attn = None

        # Projections of SMB and pos to decoder input (d_model/pos_dim -> decoder_dim_multiplier * d_model)
        # When decoder_dim_multiplier==1, use Identity for backwards compatibility
        # When decoder_dim_multiplier<1, compress dimension before decoder (e.g. c=0.5 -> d_model/2)
        self.decoder_dim_multiplier = float(decoder_dim_multiplier)
        if not (0.0 < self.decoder_dim_multiplier <= 1.0):
            raise ValueError("decoder_dim_multiplier must be in (0, 1]")
        if self.decoder_dim_multiplier == 1.0:
            self.smb_proj = nn.Identity()
            self.pos_proj = nn.Identity()
            decoder_input_dim = d_model + pos_dim
        else:
            proj_dim = max(1, int(round(self.decoder_dim_multiplier * d_model)))
            self.smb_proj = nn.Linear(d_model, proj_dim)
            self.pos_proj = nn.Linear(pos_dim, proj_dim)
            decoder_input_dim = 2 * proj_dim

        # Decoder from (projected SMB, projected pos) -> token embeddings
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        
        self.drop_path = DropPath(self._drop_path_rate) if self._drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        smb,             # [B, M, D] summary bank
        x,               # [B, N, D]
        pos,             # [1, N, pos_dim]
        keep_ratio: float,  # (0, 1]; fraction of tokens to keep (hard drop)
        current_epoch: int = 0,   # Current epoch for warmup control
        diagnostics: bool = False, # If True, compute and return diagnostic statistics
        cls_token: Optional[torch.Tensor] = None,  # [B, 1, D] optional; participates in smb_mlp and smb_mhsa only
    ):
        """
        Internal computation (attention, decoding, residual) uses all N tokens.
        Only the output is reduced: the layer returns the top k = round(N * keep_ratio)
        tokens with the highest residual norms (per sample). So tokens are actually
        dropped, not zero-masked.

        If cls_token is provided, it participates in smb_mlp and smb_mhsa only
        (not in x-attention or decoding). The updated cls_token is returned in the output.

        Returns dict with:
          smb: updated SMB (summary bank) [B, M, D]
          tokens: [B, k, D], k = round(N * keep_ratio); residuals (x-x_hat) if return_residual_tokens, else original x
          residual_norm: [B, N] residual norms for all N input tokens (for keep_mask construction)
          recon_loss: scalar, reconstruction loss over dropped tokens only
          cls_token: updated [B, 1, D] (only if cls_token was provided)
          diagnostics_info: dict (if diagnostics=True) with:
            - smb_usage_pct: [M] percentage of tokens using each SMB element
            - smb_usage_entropy: scalar, entropy of usage distribution
            - smb_mean_residual_norms: [M] mean residual norm for each SMB element
        """
        B, N, D = x.shape
        M = smb.size(1)
        device = x.device

        # Expand positions to batch
        pos_b = pos.expand(B, -1, -1)  # [B, N, pos_dim]

        # ---- SMB-to-token attention ----
        q = self.ln_smb_attn(smb)       # [B, M, D]
        k = self.ln_x_attn(x)          # [B, N, D]

        smb_attn, _, attn_logits = self.attn(
            q, k, k,
            need_weights=False,
        )  # [B, M, D], None, [B, M, N]
        smb = smb + self.drop_path(smb_attn)

        # Optionally prepend cls for MLP + MHSA only (cls does not participate in x-attn or decoding)
        if cls_token is not None:
            smb_for_mlp = torch.cat([cls_token, smb], dim=1)  # [B, 1+M, D]
        else:
            smb_for_mlp = smb

        # MLP on SMB bank (and cls if provided)
        h = self.ln_smb_mlp(smb_for_mlp)
        h = self.mlp_smb(h)
        smb_for_mlp = smb_for_mlp + self.drop_path(h)

        # Optional SMB self-attention (cls+smb if cls provided, else smb only) before decoding
        if self.smb_mhsa:
            h_mhsa = self.ln_smb_mhsa(smb_for_mlp)
            h_mhsa, _ = self.smb_mhsa_attn(h_mhsa, h_mhsa, h_mhsa, need_weights=False)
            smb_for_mlp = smb_for_mlp + self.drop_path(h_mhsa)

        # Split back: cls (if provided) and smb for decoding
        if cls_token is not None:
            cls_token = smb_for_mlp[:, :1, :]   # [B, 1, D]
            smb = smb_for_mlp[:, 1:, :]         # [B, M, D]
        else:
            smb = smb_for_mlp

        # Project SMB and pos for decoder input (Identity when decoder_dim_multiplier==1)
        smb_proj_out = self.smb_proj(smb)  # [B, M, D] or [B, M, c*D]
        pos_proj_out = self.pos_proj(pos_b)  # [B, N, pos_dim] or [B, N, c*D]
        D_proj = smb_proj_out.size(-1)

        # ---- Use attention logits directly for routing (K=1 argmax) ----
        # attn_logits already computed with learned Q/K projections: [B, M, N]
        best_idx = attn_logits.argmax(dim=1)  # [B, N]

        # Decode
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        smb_sel = smb_proj_out[b_idx, best_idx, :]  # [B, N, D_proj]
        dec_in = torch.cat([smb_sel, pos_proj_out], dim=-1)  # [B, N, 2*D_proj]
        x_hat = self.decoder(dec_in)  # [B, N, D]

        # ---- Residual computation (all N tokens) ----
        residual = x - x_hat                    # [B, N, D]
        if self.residual_norm_type == "l2":
            residual_norm_full = residual.norm(dim=-1)        # [B, N]
        else:  # "l1"
            residual_norm_full = residual.abs().sum(dim=-1)   # [B, N]

        # ---- Hard drop: keep only top k = round(N * keep_ratio) tokens by residual norm ----
        k = max(1, min(N, int(round(N * keep_ratio))))
        # Top-k indices (largest residual norm = keep); [B, k]
        _, keep_indices = torch.topk(residual_norm_full, k=k, dim=1, largest=True, sorted=False)
        keep_indices_exp = keep_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        if self.return_residual_tokens:
            # Return residuals (x - x_hat) at kept positions
            tokens_out = torch.gather(residual, dim=1, index=keep_indices_exp)  # [B, k, D]
        else:
            # Return original input tokens at kept positions
            tokens_out = torch.gather(x, dim=1, index=keep_indices_exp)  # [B, k, D]

        # Recon loss
        if self.recon_loss_dropped_only:
            # Only over dropped tokens (the ones we lose)
            drop_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            drop_mask.scatter_(1, keep_indices, False)          # True = dropped
            residual_norm_sel = residual_norm_full[drop_mask]   # [total_dropped]
        else:
            # Over all input tokens
            residual_norm_sel = residual_norm_full.reshape(-1)  # [B*N]
        if residual_norm_sel.numel() > 0:
            if self.residual_norm_type == "l2":
                recon_loss = ((residual_norm_sel ** 2) / D).mean()
            else:
                recon_loss = (residual_norm_sel / D).mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)

        out = {
            "smb": smb,                                    # [B, M, D]
            "tokens": tokens_out,                          # [B, k, D]  k = round(N * keep_ratio)
            "keep_indices": keep_indices,                  # [B, k]  indices into original N (for gathering pos)
            "residual_norm": residual_norm_full,           # [B, N]  full set, for keep_mask construction
            "recon_loss": recon_loss,                      # scalar
        }
        if cls_token is not None:
            out["cls_token"] = cls_token

        # Compute diagnostics if requested
        if diagnostics:
            # Determine which SMB element was actually used for each token (argmax routing)
            best_idx_diag = best_idx  # [B, N]

            # Count how many tokens use each SMB element (vectorized)
            best_idx_flat = best_idx_diag.view(-1)  # [B*N]
            smb_usage_counts = torch.bincount(best_idx_flat, minlength=M).float()  # [M]

            total_tokens = float(B * N)
            smb_usage_pct = (smb_usage_counts / total_tokens) * 100.0  # [M], percentage

            # Compute entropy of the usage distribution
            smb_usage_probs = smb_usage_counts / total_tokens  # [M], probabilities sum to 1
            eps = 1e-8
            smb_usage_entropy = -(smb_usage_probs * torch.log(smb_usage_probs + eps)).sum()

            # Compute mean residual norm for each SMB element via full residual scores
            smb_exp = smb_proj_out.unsqueeze(2).expand(-1, -1, N, -1)   # [B, M, N, D_proj]
            pos_exp = pos_proj_out.unsqueeze(1).expand(-1, M, -1, -1)   # [B, M, N, D_proj]
            dec_in_all = torch.cat([smb_exp, pos_exp], dim=-1)          # [B, M, N, 2*D_proj]
            x_hat_all = self.decoder(dec_in_all)                        # [B, M, N, D]

            x_exp = x.unsqueeze(1)  # [B, 1, N, D]
            residual_all = x_exp - x_hat_all  # [B, M, N, D]

            if self.residual_norm_type == "l2":
                residual_score_full = residual_all.pow(2).sum(dim=-1)  # [B, M, N]
                smb_mean_residual_norms = residual_score_full.mean(dim=(0, 2)).sqrt()  # [M]
            else:  # "l1"
                residual_score_full = residual_all.abs().sum(dim=-1)  # [B, M, N]
                smb_mean_residual_norms = residual_score_full.mean(dim=(0, 2))  # [M]

            out["diagnostics_info"] = {
                "smb_usage_pct": smb_usage_pct.detach().cpu(),
                "smb_usage_entropy": smb_usage_entropy.detach().cpu(),
                "smb_mean_residual_norms": smb_mean_residual_norms.detach().cpu(),
                "summary_bank": smb.detach().cpu(),  # [B, M, D] - SMB after this layer
            }

        return out

class TokenCompressLayerInfoNCE(nn.Module):
    """
    Token compressive layer with attention-guided routing + InfoNCE reconstruction.

    Key differences vs TokenCompressLayer:
      - No warmup / no full NxM decoding.
      - Forward takes a full positional-embedding table + per-token pos indices:
            pos_table: [1 or B, N_total, pos_dim]
            pos_idx:   [B, N]   (indices into pos_table for the N current tokens)
      - Reconstruction loss is InfoNCE in raw embedding space:
            anchor  = normalize(x)
            pos     = normalize(x_hat(pos_i, smb_sel))
            hard negs: same smb_sel, wrong positions (sampled among tokens in same image)
            easy negs: positives from other images in the batch
      - Drop is by orthogonal residual norm (raw space): u = normalize(x_hat), r = x - (x·u) u.

    Outputs (dict):
      smb:            [B, M, D]
      tokens:         [B, k, D]   orthogonal residuals if return_residual_tokens, else original x at kept positions
      keep_indices:   [B, k]      indices into the current N tokens
      keep_pos_idx:   [B, k]      original pos indices for kept tokens (indices into pos_table)
      residual_norm:  [B, N]      norms of orthogonal residuals (used for dropping)
      recon_loss:     scalar      InfoNCE loss (over dropped tokens only if recon_loss_dropped_only=True, else over all tokens used for InfoNCE; or None if eval + compute_recon_loss=False, no negatives, or no drops)
      cls_token:      [B, 1, D]   if provided
      diagnostics_info: optional  (if diagnostics=True)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_smb: int,
        pos_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        smb_mhsa: bool = False,
        drop_path: float = 0.0,
        decoder_dim_multiplier: float = 1.0,
        # InfoNCE / sampling
        T0: int = 0,                  # number of tokens per image to use for InfoNCE (0 => all N)
        nce_temperature: float = 0.07, # contrastive temperature
        num_hard_neg: int = 4,
        num_easy_neg: int = 16,
        reuse_decode_for_hardneg: bool = False,  # reuse already-decoded tokens with same SMB for hard negatives
        fill_decode_for_hardneg: bool = False,   # when reuse is on and fewer than K_h same-SMB tokens, decode extra ones
        eps: float = 1e-5,
        return_residual_tokens: bool = True,
        recon_loss_dropped_only: bool = False,
    ):
        super().__init__()
        self.num_smb = int(num_smb)
        self.d_model = int(d_model)
        self.pos_dim = int(pos_dim)
        self.return_residual_tokens = bool(return_residual_tokens)
        self.recon_loss_dropped_only = bool(recon_loss_dropped_only)

        # Routing (argmax only; no temperature parameter)
        self.smb_mhsa = bool(smb_mhsa)
        self._drop_path_rate = float(drop_path)

        # InfoNCE
        self.T0 = int(T0)
        self.nce_temperature = float(nce_temperature)
        self.num_hard_neg = int(num_hard_neg)
        self.num_easy_neg = int(num_easy_neg)
        self.reuse_decode_for_hardneg = bool(reuse_decode_for_hardneg)
        self.fill_decode_for_hardneg = bool(fill_decode_for_hardneg)
        self.eps = float(eps)

        # SMB-to-token attention (queries=SMB, keys/values=tokens)
        self.ln_smb_attn = nn.LayerNorm(d_model)
        self.ln_x_attn = nn.LayerNorm(d_model)
        self.attn = MultiheadAttentionWithLogits(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # MLP on SMB bank
        hidden = int(d_model * mlp_ratio)
        self.ln_smb_mlp = nn.LayerNorm(d_model)
        self.mlp_smb = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # Optional SMB self-attention (smb<->smb), after MLP and before decoding
        if self.smb_mhsa:
            self.ln_smb_mhsa = nn.LayerNorm(d_model)
            self.smb_mhsa_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.ln_smb_mhsa = None
            self.smb_mhsa_attn = None

        # Decoder input projections (SMB and pos -> concat -> decoder)
        self.decoder_dim_multiplier = float(decoder_dim_multiplier)
        if not (0.0 < self.decoder_dim_multiplier <= 1.0):
            raise ValueError("decoder_dim_multiplier must be in (0, 1]")

        if self.decoder_dim_multiplier == 1.0:
            self.smb_proj = nn.Identity()
            self.pos_proj = nn.Identity()
            decoder_input_dim = d_model + pos_dim
        else:
            proj_dim = max(1, int(round(self.decoder_dim_multiplier * d_model)))
            self.smb_proj = nn.Linear(d_model, proj_dim)
            self.pos_proj = nn.Linear(pos_dim, proj_dim)
            decoder_input_dim = 2 * proj_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        self.drop_path = (
            DropPath(self._drop_path_rate) if self._drop_path_rate > 0.0 else nn.Identity()
        )

    def _gather_pos(self, pos_table: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        """
        pos_table: [1 or B, N_total, pos_dim]
        pos_idx:   [B, N] indices into pos_table for each token

        returns:   [B, N, pos_dim]
        """
        if pos_table.dim() != 3:
            raise ValueError(f"pos_table must be [*, N_total, pos_dim], got {tuple(pos_table.shape)}")
        if pos_idx.dim() != 2:
            raise ValueError(f"pos_idx must be [B, N], got {tuple(pos_idx.shape)}")

        B, N = pos_idx.shape
        Bt, N_total, pos_dim = pos_table.shape

        if pos_dim != self.pos_dim:
            raise ValueError(f"pos_table last dim {pos_dim} != self.pos_dim {self.pos_dim}")
        if Bt == 1:
            pos_table_b = pos_table.expand(B, -1, -1)
        elif Bt == B:
            pos_table_b = pos_table
        else:
            raise ValueError(f"pos_table first dim must be 1 or B={B}, got {Bt}")

        if pos_idx.min() < 0 or pos_idx.max() >= N_total:
            raise ValueError(f"pos_idx out of range [0, {N_total-1}]")

        return torch.gather(
            pos_table_b,
            dim=1,
            index=pos_idx.unsqueeze(-1).expand(-1, -1, pos_dim),
        )

    def forward(
        self,
        smb: torch.Tensor,               # [B, M, D]
        x: torch.Tensor,                 # [B, N, D]
        pos_table: torch.Tensor,         # [1 or B, N_total, pos_dim]
        pos_idx: torch.Tensor,           # [B, N] indices into pos_table for current tokens
        keep_ratio: float,               # (0, 1]
        diagnostics: bool = False,
        cls_token: Optional[torch.Tensor] = None,  # [B, 1, D] optional
        compute_recon_loss: bool = False,  # if True, compute InfoNCE even when eval (e.g. for debug)
        skip_recon_loss: bool = False,     # if True, skip InfoNCE even when training (e.g. lambda_recon schedule warmup)
    ):
        B, N, D = x.shape
        M = smb.size(1)
        device = x.device

        # Gather positions for current tokens (per-sample subset)
        pos_b = self._gather_pos(pos_table, pos_idx)  # [B, N, pos_dim]

        # ---- SMB-to-token attention ----
        q = self.ln_smb_attn(smb)    # [B, M, D]
        k = self.ln_x_attn(x)        # [B, N, D]
        smb_attn, _, attn_logits = self.attn(q, k, k, need_weights=False)  # [B,M,D], None, [B,M,N]
        smb = smb + self.drop_path(smb_attn)

        # ---- Routing: argmax over attention logits (for decode) ----
        best_idx = attn_logits.argmax(dim=1) # [B, N]
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

        # Optionally prepend cls for MLP + SMB-MHSA only
        if cls_token is not None:
            smb_for_mlp = torch.cat([cls_token, smb], dim=1)  # [B, 1+M, D]
        else:
            smb_for_mlp = smb

        # MLP on SMB
        h = self.ln_smb_mlp(smb_for_mlp)
        h = self.mlp_smb(h)
        smb_for_mlp = smb_for_mlp + self.drop_path(h)

        # Optional SMB self-attention
        if self.smb_mhsa:
            h_mhsa = self.ln_smb_mhsa(smb_for_mlp)
            h_mhsa, _ = self.smb_mhsa_attn(h_mhsa, h_mhsa, h_mhsa, need_weights=False)
            smb_for_mlp = smb_for_mlp + self.drop_path(h_mhsa)

        # Split back
        if cls_token is not None:
            cls_token = smb_for_mlp[:, :1, :]
            smb = smb_for_mlp[:, 1:, :]
        else:
            smb = smb_for_mlp

        # Project SMB and pos for decoder, then decode using best_idx
        smb_proj_out = self.smb_proj(smb)   # [B, M, Dp]
        pos_proj_out = self.pos_proj(pos_b) # [B, N, Dp]
        Dp = pos_proj_out.size(-1)
        smb_sel = smb_proj_out[b_idx, best_idx, :]                   # [B, N, Dp]
        dec_in = torch.cat([smb_sel, pos_proj_out], dim=-1)          # [B, N, *]
        x_hat = self.decoder(dec_in)                                 # [B, N, D]
        x_hat = torch.clamp(x_hat, min=-1e2, max=1e2)               # NaN/explosion guard

        # ---- Orthogonal residual (raw space) ----
        u = F.normalize(x_hat, dim=-1, eps=self.eps)                 # [B, N, D]
        proj_coeff = (x * u).sum(dim=-1, keepdim=True)               # [B, N, 1]
        residual = x - proj_coeff * u                                # [B, N, D]
        residual_norm_full = residual.norm(dim=-1)                   # [B, N]

        # ---- Hard drop ----
        k_keep = max(1, min(N, int(round(N * float(keep_ratio)))))
        _, keep_indices = torch.topk(
            residual_norm_full, k=k_keep, dim=1, largest=True, sorted=False
        )  # [B, k_keep]
        keep_exp = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        if self.return_residual_tokens:
            # Return orthogonal residuals at kept positions
            tokens_out = torch.gather(residual, dim=1, index=keep_exp)  # [B, k_keep, D]
        else:
            # Return original input tokens at kept positions
            tokens_out = torch.gather(x, dim=1, index=keep_exp)  # [B, k_keep, D]

        keep_pos_idx = torch.gather(pos_idx, dim=1, index=keep_indices)  # [B, k_keep]

        out = {
            "smb": smb,
            "tokens": tokens_out,
            "keep_indices": keep_indices,
            "keep_pos_idx": keep_pos_idx,
            "residual_norm": residual_norm_full,
        }
        if cls_token is not None:
            out["cls_token"] = cls_token

        # ---- InfoNCE loss ----
        # Only compute when training OR explicitly requested (e.g. by debug scripts to report recon losses).
        # Skipped when skip_recon_loss=True (e.g. lambda_recon schedule warmup phase).
        nce_loss = None
        total_negs = max(0, self.num_hard_neg) + max(0, self.num_easy_neg)
        n_dropped = N - k_keep

        # Determine anchor pool: dropped-only or all tokens
        if self.recon_loss_dropped_only:
            anchor_pool_size = n_dropped
        else:
            anchor_pool_size = N

        if ((self.training and not skip_recon_loss) or compute_recon_loss) and total_negs > 0 and anchor_pool_size > 0:
            if self.recon_loss_dropped_only:
                # Build dropped-token indices [B, n_dropped]
                drop_mask = torch.ones(B, N, dtype=torch.bool, device=device)
                drop_mask.scatter_(1, keep_indices, False)
                pool_indices = torch.where(drop_mask)[1].view(B, n_dropped)  # [B, n_dropped]
            else:
                pool_indices = None  # will sample from all N

            # number of anchor tokens per image
            T = anchor_pool_size if self.T0 <= 0 else min(self.T0, anchor_pool_size)

            # sample anchors
            if pool_indices is not None:
                # dropped-only pool
                if T == anchor_pool_size:
                    anchor_idx = pool_indices  # [B, T]
                else:
                    rand = torch.rand(B, anchor_pool_size, device=device)
                    _, sub_idx = torch.topk(rand, k=T, dim=1, largest=True, sorted=False)
                    anchor_idx = torch.gather(pool_indices, dim=1, index=sub_idx)  # [B, T]
            else:
                # all-N pool
                if T == N:
                    anchor_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
                else:
                    rand = torch.rand(B, N, device=device)
                    _, anchor_idx = torch.topk(rand, k=T, dim=1, largest=True, sorted=False)  # [B, T]

            # normalize for contrastive similarity (raw space)
            x_n = F.normalize(x, dim=-1, eps=self.eps)                  # [B, N, D]
            x_hat_n = F.normalize(x_hat, dim=-1, eps=self.eps)         # [B, N, D]

            anchor_x = torch.gather(
                x_n, dim=1, index=anchor_idx.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, T, D]
            pos_vec = torch.gather(
                x_hat_n, dim=1, index=anchor_idx.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, T, D]

            temp = max(self.eps, self.nce_temperature)
            pos_logit = (anchor_x * pos_vec).sum(dim=-1) / temp  # [B, T]
            logits_parts = [pos_logit.unsqueeze(-1)]             # [B, T, 1]

            # Hard negatives: same SMB, wrong positions (sample other token positions within same image)
            if self.num_hard_neg > 0 and N > 1:
                K_h = min(self.num_hard_neg, N - 1)

                # SMB selected for each anchor token
                smb_idx_anchor = torch.gather(best_idx, dim=1, index=anchor_idx)  # [B, T]

                if self.reuse_decode_for_hardneg:
                    # Fully vectorized reuse path: no decoder calls, no Python loops.
                    # Reuse already-decoded & normalized x_hat_n for same-SMB tokens.

                    # 1. Build candidate mask [B, T, N]: same SMB as anchor, excluding anchor itself
                    best_idx_exp = best_idx.unsqueeze(1).expand(-1, T, -1)       # [B, T, N]
                    smb_anchor_exp = smb_idx_anchor.unsqueeze(-1)                # [B, T, 1]
                    same_smb_mask = (best_idx_exp == smb_anchor_exp)              # [B, T, N]
                    anchor_idx_exp = anchor_idx.unsqueeze(-1)                    # [B, T, 1]
                    n_idx = torch.arange(N, device=device).view(1, 1, N)         # [1, 1, N]
                    candidate_mask = same_smb_mask & (n_idx != anchor_idx_exp)   # [B, T, N]

                    # 2. Randomly sample K_h from candidates: random noise masked to candidates, then topk
                    rand_scores = torch.rand(B, T, N, device=device) * candidate_mask.float()  # [B, T, N]
                    _, topk_idx = torch.topk(rand_scores, k=K_h, dim=-1)  # [B, T, K_h]

                    # 3. Gather x_hat_n at selected positions (already normalized, no decoder needed)
                    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, T, K_h, D]
                    x_hat_sel = torch.gather(
                        x_hat_n.unsqueeze(1).expand(-1, T, -1, -1),               # [B, T, N, D]
                        dim=2,
                        index=topk_idx_exp,
                    )  # [B, T, K_h, D]

                    # 4. Compute logits: dot product of anchor with selected negatives
                    hard_logit = (anchor_x.unsqueeze(2) * x_hat_sel).sum(dim=-1) / temp  # [B, T, K_h]

                    # 5. Handle slots where we picked non-candidates (fewer than K_h available)
                    picked_is_candidate = torch.gather(candidate_mask, dim=-1, index=topk_idx)  # [B, T, K_h]
                    fill_mask = ~picked_is_candidate  # [B, T, K_h]

                    if self.fill_decode_for_hardneg and fill_mask.any():
                        # Decode extra hard negatives for empty slots (single batched call)
                        fill_b, fill_t, fill_k = torch.where(fill_mask)  # each [num_fill]
                        num_fill = fill_b.numel()

                        # Get anchor's SMB and a random position for each fill slot
                        fill_smb_idx = smb_idx_anchor[fill_b, fill_t]                          # [num_fill]
                        fill_anchor_pos = anchor_idx[fill_b, fill_t]                            # [num_fill]
                        r_fill = torch.randint(0, N - 1, (num_fill,), device=device)
                        fill_tok = r_fill + (r_fill >= fill_anchor_pos).long()                  # [num_fill]

                        # Gather projected smb + pos for decoder input
                        fill_smb = smb_proj_out[fill_b, fill_smb_idx]                           # [num_fill, Dp]
                        fill_pos = pos_proj_out[fill_b, fill_tok]                                # [num_fill, Dp]

                        # Single batched decoder call
                        fill_decoded = self.decoder(torch.cat([fill_smb, fill_pos], dim=-1))    # [num_fill, D]
                        fill_decoded = torch.clamp(fill_decoded, min=-1e2, max=1e2)           # NaN/explosion guard
                        fill_decoded_n = F.normalize(fill_decoded, dim=-1, eps=self.eps)       # [num_fill, D]

                        # Compute logits and scatter back
                        fill_anchor = anchor_x[fill_b, fill_t]                                  # [num_fill, D]
                        fill_logit = (fill_anchor * fill_decoded_n).sum(dim=-1) / temp           # [num_fill]
                        hard_logit[fill_b, fill_t, fill_k] = fill_logit
                    else:
                        # Zero out non-candidate logits
                        hard_logit = hard_logit * picked_is_candidate.float()
                else:
                    # Original logic: always decode new hard negatives
                    # sample token indices != anchor via "skip" trick
                    r = torch.randint(0, N - 1, size=(B, T, K_h), device=device)
                    neg_tok = r + (r >= anchor_idx.unsqueeze(-1)).long()  # [B, T, K_h]

                    # wrong positions (already projected for current tokens)
                    pos_proj_for_hard = torch.gather(
                        pos_proj_out.unsqueeze(1).expand(-1, T, -1, -1),  # [B, T, N, Dp]
                        dim=2,
                        index=neg_tok.unsqueeze(-1).expand(-1, -1, -1, Dp),
                    )  # [B, T, K_h, Dp]

                    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
                    smb_anchor = smb_proj_out[b_idx, smb_idx_anchor, :]               # [B, T, Dp]
                    smb_anchor = smb_anchor.unsqueeze(2).expand(-1, -1, K_h, -1)       # [B, T, K_h, Dp]

                    x_hat_hard = self.decoder(torch.cat([smb_anchor, pos_proj_for_hard], dim=-1))  # [B,T,K_h,D]
                    x_hat_hard = torch.clamp(x_hat_hard, min=-1e2, max=1e2)                         # NaN/explosion guard
                    x_hat_hard_n = F.normalize(x_hat_hard, dim=-1, eps=self.eps)                   # [B,T,K_h,D]
                    hard_logit = (anchor_x.unsqueeze(2) * x_hat_hard_n).sum(dim=-1) / temp         # [B,T,K_h]

                logits_parts.append(hard_logit)

            # Easy negatives: positives from other images in batch (SMB bank differs across images)
            if self.num_easy_neg > 0:
                K_e = self.num_easy_neg
                if B > 1:
                    r = torch.randint(0, B - 1, size=(B, T, K_e), device=device)
                    b_self = torch.arange(B, device=device).view(B, 1, 1)
                    b_other = r + (r >= b_self).long()  # [B, T, K_e] and b_other != b_self

                    tok_other = torch.randint(0, N, size=(B, T, K_e), device=device)
                    easy_vec = x_hat_n[b_other, tok_other]  # [B, T, K_e, D]
                    easy_logit = (anchor_x.unsqueeze(2) * easy_vec).sum(dim=-1) / temp  # [B,T,K_e]
                    logits_parts.append(easy_logit)
                else:
                    # degenerate fallback: no other images available
                    pass

            logits = torch.cat(logits_parts, dim=-1)  # [B, T, C]
            C = logits.size(-1)
            if C <= 1:
                nce_loss = logits.sum() * 0.0
            else:
                logits_flat = logits.reshape(B * T, C)
                logits_flat = torch.clamp(logits_flat, min=-50.0, max=50.0)   # numerical stability for cross_entropy
                target = torch.zeros(B * T, dtype=torch.long, device=device)  # positive is index 0
                nce_loss = F.cross_entropy(logits_flat, target)

        # Replace NaN/Inf recon loss with zero to avoid corrupting the step
        if nce_loss is not None and (torch.isnan(nce_loss).any() or torch.isinf(nce_loss).any()):
            nce_loss = torch.tensor(0.0, device=device)
        out["recon_loss"] = nce_loss

        # ---- Diagnostics ----
        if diagnostics:
            best_idx_flat = best_idx.reshape(-1)  # [B*N]
            counts = torch.zeros(M, device=device, dtype=torch.float32)
            counts.scatter_add_(0, best_idx_flat, torch.ones_like(best_idx_flat, dtype=torch.float32))

            total = float(B * N)
            smb_usage_pct = (counts / max(1.0, total)) * 100.0
            probs = counts / max(1.0, total)
            smb_usage_entropy = -(probs * torch.log(probs + 1e-8)).sum()

            # Mean residual norm per SMB
            sum_res = torch.zeros(M, device=device, dtype=torch.float32)
            sum_res.scatter_add_(0, best_idx_flat, residual_norm_full.reshape(-1).detach())
            smb_mean_residual_norms = sum_res / (counts + 1e-6)

            out["diagnostics_info"] = {
                "smb_usage_pct": smb_usage_pct.detach().cpu(),
                "smb_usage_entropy": smb_usage_entropy.detach().cpu(),
                "smb_mean_residual_norms": smb_mean_residual_norms.detach().cpu(),
                "summary_bank": smb.detach().cpu(),
            }

        return out