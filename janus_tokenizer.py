# coding=utf-8
from __future__ import annotations
from typing import Dict
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from contextlib import nullcontext

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from janus.models import MultiModalityCausalLM  # type: ignore
    HAS_JANUS = True
except Exception:
    HAS_JANUS = False


def _find_codebook_weight(vq: torch.nn.Module) -> torch.Tensor | None:
    for n, p in vq.named_parameters():
        if p.ndim == 2 and any(k in n.lower() for k in ["embed", "embedding", "codebook"]):
            return p
    for n, b in vq.named_buffers():
        if isinstance(b, torch.Tensor) and b.ndim == 2 and any(k in n.lower() for k in ["embed", "embedding", "codebook"]):
            return b
    return None

def _encode_to_feats_and_codebook(vq: torch.nn.Module, x: torch.Tensor):
    z = vq.encoder(x)
    if hasattr(vq, "quant_conv"):
        z = vq.quant_conv(z)
    B, D, h, w = z.shape
    feats = z.permute(0,2,3,1).reshape(B, h*w, D)  # [B,N,D]

    cb = _find_codebook_weight(vq)
    if cb is None:
        raise RuntimeError("codebook weight not found")
    if cb.shape[0] < cb.shape[1]:
        cb = cb.t().contiguous()     # [K,D]
    K, D_cb = cb.shape
    if D_cb != D:
        raise RuntimeError(f"embed dim mismatch: z {D} vs codebook {D_cb}")
    return feats, (h, w), cb


class JanusVQEncoder:
    """
    Returns code_vectors and four *distributional* continuous metrics (discrete->continuous):
      - feat_confidence:     [B, N]  (Top-1 prob)
      - feat_margin:         [B, N]  (Top-1 - Top-2 gap)
      - feat_topk_mass:      [B, N]  (adaptive top-k probability mass)
      - feat_tail_entropy:   [B, N]  (entropy over tail (excl. Top-1), normalized by log(K-1))
    Also returns:
      - probs:               [B, N, K]
      - logits:              [B, N, K]
      - code_vectors:        [B, N, D]  (soft assignment)
      - hw:                  [2]        (h, w)
    Calibration parameters:
      - similarity: "cosine" | "neg_l2" | "dot"
      - temperature (alpha), cosine_scale (a), label_smoothing_eps (epsilon)
    """
    def __init__(
        self,
        model: str = "deepseek-ai/Janus-Pro-7B",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # calibration
        similarity: str = "cosine",     # "cosine" | "neg_l2" | "dot"
        temperature: float = 1.0,       # alpha
        cosine_scale: float = 29.0381,  # a (cosine only)
        label_smoothing_eps: float = 0.01
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required to load the Janus models.")
        self.device = torch.device(device)
        self.dtype = dtype

        self.similarity = similarity
        self.temperature = float(temperature)
        self.cosine_scale = float(cosine_scale)
        if self.temperature <= 0.0:
            raise ValueError("temperature (alpha) must be positive")
        if self.cosine_scale <= 0.0:
            raise ValueError("cosine_scale (a) must be positive")
        self.label_smoothing_eps = float(label_smoothing_eps)

        if HAS_JANUS:
            self.mm = MultiModalityCausalLM.from_pretrained(
                model, trust_remote_code=True, torch_dtype=dtype
            ).to(self.device).eval()
        else:
            _ = AutoConfig.from_pretrained(model, trust_remote_code=True)
            self.mm = AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=True, torch_dtype=dtype
            ).to(self.device).eval()
        self.vq = self.mm.gen_vision_model
        self.codebook = None  # (K, D)

        self.pre = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            lambda x: x*2 - 1,  # [-1,1]
        ])

    def _compute_logits(self, feats32: torch.Tensor, cb32: torch.Tensor) -> torch.Tensor:
        # feats32:[B,N,D], cb32:[K,D]
        if self.similarity == "cosine":
            z = F.normalize(feats32, dim=-1)
            e = F.normalize(cb32, dim=-1)
            logits = (z @ e.t()) * self.cosine_scale
        elif self.similarity == "dot":
            logits = feats32 @ cb32.t()
        else:  # neg_l2
            z2 = (feats32**2).sum(-1, keepdim=True)
            e2 = (cb32**2).sum(-1).view(1,1,cb32.size(0))
            ze = feats32 @ cb32.t()
            logits = -(z2 + e2 - 2*ze)

        # Paper calibration: q = softmax((a * cosine_similarity) / alpha).
        if self.temperature != 1.0:
            logits = logits / self.temperature
        logits = logits - logits.amax(dim=-1, keepdim=True)
        return logits

    def _post_softmax(self, probs: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing_eps > 0.0:
            K = probs.size(-1)
            eps = self.label_smoothing_eps
            probs = (1 - eps) * probs + eps / K
        return probs

    @torch.no_grad()
    def encode_pil(self, images: list[Image.Image]) -> Dict[str, torch.Tensor]:
        x = torch.stack([self.pre(img.convert("RGB")) for img in images], dim=0).to(self.device).to(self.mm.dtype)
        return self.encode_tensor(x)  # default no_grad=True

    def encode_tensor(self, x: torch.Tensor, *, no_grad: bool = True) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,384,384] in [-1,1]
        """
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            x = x.to(self.device).to(self.mm.dtype)

            # ---------- features & codebook ----------
            feats, (h, w), codebook = _encode_to_feats_and_codebook(self.vq, x)  # feats:[B,N,D], cb:[K,D]
            self.codebook = codebook.detach()
            B, N, D = feats.shape
            K, _ = codebook.shape

            # ---------- soft logits & probs ----------
            feats32 = feats.to(torch.float32)        # [B,N,D]
            cb32 = codebook.to(torch.float32)        # [K,D]
            logits = self._compute_logits(feats32, cb32)             # [B,N,K]
            probs  = torch.softmax(logits, dim=-1).to(torch.float32) # [B,N,K]
            probs  = self._post_softmax(probs)

            # ---------- code vectors (soft) ----------
            code_vectors = probs @ codebook.to(probs.dtype)          # [B,N,D] D is codebook embedding's feature dimension. 

            # ===== 4 continuous metrics from distribution (perplexity-based top-k) =====
            eps = 1e-12

            # (1) entropy/perplexity → adaptive k
            H = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)        # [B,N]
            perplex = torch.exp(H)                                                       # [B,N]

            MIN_K = 8
            MAX_K_CAP = 64
            K_cap = min(K, MAX_K_CAP)
            k_adapt = perplex.round().clamp(min=MIN_K, max=K_cap).to(torch.long)        # [B,N]

            # (2) confidence, margin, top-k mass
            topk_vals, topk_idx = torch.topk(probs, k=K_cap, dim=-1, largest=True, sorted=True)  # [B,N,K_cap]
            p1 = topk_vals[..., 0]                                                      # [B,N]
            p2 = topk_vals[..., 1] if K_cap >= 2 else torch.zeros_like(p1)              # [B,N]
            idx = torch.arange(K_cap, device=probs.device).view(1,1,-1)                 # [1,1,K_cap]
            mask = (idx < k_adapt.unsqueeze(-1)).to(topk_vals.dtype)                    # [B,N,K_cap]
            topk_mass = (topk_vals * mask).sum(dim=-1)                                  # [B,N]

            margin = (p1 - p2).clamp_min(0.0)                                           # [B,N]

            # (3) tail-entropy (exclude Top-1)
            tail_mass = (1.0 - p1).clamp_min(eps)                                       # [B,N]
            probs_wo_top1 = probs.clone()
            oh = torch.nn.functional.one_hot(topk_idx[..., 0], num_classes=K).to(probs_wo_top1.dtype)  # [B,N,K]
            probs_wo_top1 = probs_wo_top1 * (1.0 - oh)
            tail = probs_wo_top1 / tail_mass.unsqueeze(-1)
            if K > 1:
                tail_entropy = -(tail.clamp_min(eps) * tail.clamp_min(eps).log()).sum(dim=-1) / math.log(K - 1)
                tail_entropy = torch.where(tail_mass < 1e-6, torch.zeros_like(tail_entropy), tail_entropy)
            else:
                tail_entropy = torch.zeros_like(p1)


            return {
                "probs": probs,                               # [B,N,K] float32 
                "logits": logits.to(torch.float32),           # [B,N,K]
                "code_vectors": code_vectors,                 # [B,N,D]
                "feat_confidence": p1,                        # [B,N]
                "feat_margin": margin,                        # [B,N]
                "feat_topk_mass": topk_mass,                  # [B,N]
                "feat_tail_entropy": tail_entropy,            # [B,N]
                "hw": torch.tensor([h, w], device=code_vectors.device),
            }
