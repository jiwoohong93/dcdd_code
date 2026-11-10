#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janus text-to-image generation with native VQ-VAE and DCDD diffusion decoding.
- Input metadata: GenEval's ``prompts/evaluation_metadata.jsonl``
  e.g., {"tag":"single_object","include":[{"class":"bench","count":1}],"prompt":"a photo of a bench"}
- Output structure:
  <OUT_ROOT>/vqvae/<00000>/{metadata.jsonl, grid.png, samples/0000.png..0003.png}
  <OUT_ROOT>/diff/<00000>/{metadata.jsonl, grid.png, samples/0000.png..0003.png}
- Category: use the "tag" field in the JSONL
"""


import os, json, argparse, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from transformers import AutoModelForCausalLM, PretrainedConfig, CLIPTokenizer, T5TokenizerFast
from janus.models import MultiModalityCausalLM, VLChatProcessor

# ---------- Speed-friendly backend toggles ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- Common: prompt builder ----------------
def build_prompt(vl_chat_processor: VLChatProcessor, text_prompt: str) -> str:
    conversation = [
        {"role": "User", "content": text_prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag

def find_codebook_matrix(janus_vision_model, vocab_size: int) -> torch.Tensor:
    cand = None
    for n, p in janus_vision_model.named_parameters():
        if p.ndim == 2 and vocab_size in p.shape and ("codebook" in n or "embed" in n or "embedding" in n):
            cand = p.detach(); break
    if cand is None:
        for n, p in janus_vision_model.named_parameters():
            if p.ndim == 2 and vocab_size in p.shape:
                cand = p.detach(); break
    if cand is None:
        raise RuntimeError("Could not find Janus codebook matrix.")
    if cand.shape[0] != vocab_size and cand.shape[1] == vocab_size:
        cand = cand.t().contiguous()
    if cand.shape[0] != vocab_size:
        raise RuntimeError(f"Codebook shape {tuple(cand.shape)} vs K={vocab_size}")
    return cand.to(torch.float32)

# ---------------- DCDD conditioning modules ----------------
class ConvTokenAdapter(nn.Module):
    """[B,N,D] + [B,H2*W2,4] -> [B, H2*W2, 20]."""
    def __init__(self, token_dim: int, base_ch: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, base_ch)
        self.stem = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GELU(),
        )
        self.to16 = nn.Conv2d(base_ch, 16, 1)
    def forward(self, code_vectors: torch.Tensor, mask_feat_tokens: torch.Tensor,
                out_height: int, out_width: int, vae_scale_factor: int):
        B, N, D = code_vectors.shape
        token_side = math.isqrt(N)
        if token_side * token_side != N:
            raise ValueError(f"Expected a square VQ token grid, received N={N}")
        H_lat, W_lat = out_height // vae_scale_factor, out_width // vae_scale_factor
        H2, W2 = H_lat // 2, W_lat // 2
        x = self.in_proj(code_vectors)
        x = x.view(B, token_side, token_side, -1).permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        x = F.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)
        x = self.to16(x)                                      # [B,16,H2,W2]
        x = x.permute(0, 2, 3, 1).reshape(B, H2*W2, 16).contiguous()
        assert mask_feat_tokens.shape[:2] == (B, H2*W2)
        return torch.cat([x, mask_feat_tokens.contiguous()], dim=-1).contiguous()

class ImageProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.proj(x)

class HintProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.proj(x)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str,
                                               revision: str | None,
                                               subfolder: str = "text_encoder"):
    te_cfg = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    arch = te_cfg.architectures[0]
    if arch == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif arch == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    elif arch == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"Unsupported text encoder arch: {arch}")

def pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int):
    x = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4).contiguous()
    return x

def unpack_packed(latents_packed: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int):
    up = latents_packed.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2).contiguous()
    up = up.permute(0, 3, 1, 4, 2, 5).contiguous()
    up = up.reshape(batch_size, num_channels, height, width).contiguous()
    return up

def prepare_latent_image_ids(batch_size: int, height: int, width: int, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2, device=device, dtype=dtype)[None, :]
    h, w, c = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(h * w, c).contiguous()
    return latent_image_ids

def prepare_img_ids(vae_scale_factor: int, batch_size: int, height: int, width: int, dtype, device):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width  = 2 * (int(width)  // (vae_scale_factor * 2))
    return prepare_latent_image_ids(batch_size, height, width, device, dtype)

# ---------------- Janus autoregressive generation ----------------
@torch.inference_mode()
def janus_generate_both_batch(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt_text: str,
    seeds: List[int],
    image_token_num_per_image: int = 576,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    args=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate multiple samples for the same prompt in one AR pass."""
    device = next(mmgpt.parameters()).device
    batch_size = len(seeds)
    generators = [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds
    ]

    prompts = (
        [prompt_text] * batch_size
        if isinstance(prompt_text, str)
        else list(prompt_text)
    )
    if len(prompts) != batch_size:
        raise ValueError("The prompt and seed batch sizes must match")
    token_lists = [
        vl_chat_processor.tokenizer.encode(
            build_prompt(vl_chat_processor, prompt)
        )
        for prompt in prompts
    ]
    max_length = max(len(ids) for ids in token_lists)
    tokens = torch.full(
        (2 * batch_size, max_length),
        vl_chat_processor.pad_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(tokens)
    for row, ids in enumerate(token_lists):
        ids_tensor = torch.as_tensor(ids, dtype=torch.long, device=device)
        start = max_length - len(ids)
        tokens[2 * row, start:] = ids_tensor
        tokens[2 * row + 1, start:] = ids_tensor
        if len(ids) > 2:
            tokens[2 * row + 1, start + 1:-1] = vl_chat_processor.pad_id
        attention_mask[2 * row:2 * row + 2, start:] = 1
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        batch_size, image_token_num_per_image, dtype=torch.long, device=device
    )
    code_vectors = None
    codebook = None
    mask_feats = torch.zeros(
        batch_size, image_token_num_per_image, 4, dtype=torch.float32, device="cpu"
    )
    outputs = None

    for token_idx in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=outputs.past_key_values if token_idx else None,
        )
        logits = mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
        conditional = logits[0::2]
        unconditional = logits[1::2]
        guided = unconditional + cfg_weight * (conditional - unconditional)

        sampling_logits = guided.float() / max(float(temperature), 1e-5)
        probs = torch.softmax(sampling_logits, dim=-1)
        probs = 0.99 * probs + 0.01 / probs.shape[-1]
        sample_probs = probs

        if codebook is None:
            vocab_size = logits.shape[-1]
            codebook = find_codebook_matrix(
                mmgpt.gen_vision_model, vocab_size=vocab_size
            ).to(device)
            code_vectors = torch.empty(
                batch_size,
                image_token_num_per_image,
                codebook.shape[-1],
                dtype=torch.float32,
                device="cpu",
            )

        code_vectors[:, token_idx] = (
            probs.float() @ codebook.float()
        ).cpu()

        probs_bnk = probs[:, None, :]
        vocab_size = probs.shape[-1]
        eps = 1e-12
        clamped = probs_bnk.clamp_min(eps)
        perplexity = torch.exp(-(clamped * clamped.log()).sum(dim=-1))
        k_cap = min(vocab_size, 64)
        adaptive_k = perplexity.round().clamp(min=8, max=k_cap).long()
        topk_values, topk_indices = torch.topk(
            probs_bnk, k=k_cap, dim=-1, largest=True, sorted=True
        )
        top1 = topk_values[..., 0]
        top2 = topk_values[..., 1] if k_cap >= 2 else torch.zeros_like(top1)
        rank_ids = torch.arange(k_cap, device=device).view(1, 1, -1)
        head_mask = (rank_ids < adaptive_k.unsqueeze(-1)).to(topk_values.dtype)
        head_mass = (topk_values * head_mask).sum(dim=-1)
        margin = (top1 - top2).clamp_min(0.0)

        tail_mass = (1.0 - top1).clamp_min(eps)
        without_top1 = probs_bnk * (
            1.0
            - F.one_hot(topk_indices[..., 0], num_classes=vocab_size).to(
                probs_bnk.dtype
            )
        )
        tail = without_top1 / tail_mass.unsqueeze(-1)
        tail_clamped = tail.clamp_min(eps)
        tail_entropy = -(
            tail_clamped * tail_clamped.log()
        ).sum(dim=-1) / math.log(vocab_size - 1)
        tail_entropy = torch.where(
            tail_mass < 1e-6, torch.zeros_like(tail_entropy), tail_entropy
        )

        mask_feats[:, token_idx, 0] = top1[:, 0].cpu()
        mask_feats[:, token_idx, 1] = margin[:, 0].cpu()
        mask_feats[:, token_idx, 2] = head_mass[:, 0].clamp_max(1.0).cpu()
        mask_feats[:, token_idx, 3] = tail_entropy[:, 0].clamp(0.0, 1.0).cpu()

        next_tokens = torch.stack(
            [
                torch.multinomial(
                    sample_probs[row], num_samples=1, generator=generators[row]
                )
                for row in range(batch_size)
            ],
            dim=0,
        )
        generated_tokens[:, token_idx] = next_tokens[:, 0]
        paired_tokens = next_tokens.repeat_interleave(2, dim=0).reshape(-1)
        inputs_embeds = mmgpt.prepare_gen_img_embeds(paired_tokens).unsqueeze(1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    return generated_tokens.to(torch.int32), code_vectors, mask_feats


# ---------------- VQ-VAE decode ----------------
def vqvae_decode_batch_to_arrays(
    mmgpt: MultiModalityCausalLM,
    generated_tokens: torch.Tensor,
    img_size: int,
    patch_size: int,
) -> np.ndarray:
    batch_size = generated_tokens.shape[0]
    model_device = next(mmgpt.gen_vision_model.parameters()).device
    with torch.no_grad():
        decoded = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(model_device),
            shape=[
                batch_size,
                8,
                img_size // patch_size,
                img_size // patch_size,
            ],
        )
    decoded = decoded.float().cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip((decoded + 1.0) / 2.0, 0.0, 1.0)

# ---------------- Diffusion Context: load once, reuse ----------------
class DiffusionCtx:
    def __init__(self, *, args, device, dtype):
        from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
        self.tokenizer_one = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
        self.tokenizer_two = T5TokenizerFast.from_pretrained(args.base_model, subfolder="tokenizer_2")

        te_cls_one = import_model_class_from_model_name_or_path(args.base_model, None, subfolder="text_encoder")
        te_cls_two = import_model_class_from_model_name_or_path(args.base_model, None, subfolder="text_encoder_2")
        self.text_encoder_one = te_cls_one.from_pretrained(args.base_model, subfolder="text_encoder").to(device, dtype=dtype).eval()
        self.text_encoder_two = te_cls_two.from_pretrained(args.base_model, subfolder="text_encoder_2").to(device, dtype=dtype).eval()

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, dtype=dtype).eval()
        self.transformer = FluxTransformer2DModel.from_pretrained(
            os.path.join(args.ckpt_dir, "transformer")
        ).eval()

        adapter_state = torch.load(
            os.path.join(args.ckpt_dir, "token_adapter.pt"), map_location="cpu"
        )
        token_dim = int(adapter_state["in_proj.weight"].shape[1])
        self.token_adapter = ConvTokenAdapter(token_dim=token_dim).to(device, dtype=dtype).eval()
        self.img_proj  = ImageProjector(in_dim=4 * getattr(self.vae.config, "latent_channels", 16),
                                        out_dim=int(getattr(self.transformer.config, "in_channels", 64))).to(device, dtype=dtype).eval()
        self.hint_proj = HintProjector(in_dim=20,
                                       out_dim=int(getattr(self.transformer.config, "in_channels", 64))).to(device, dtype=dtype).eval()

        self.token_adapter.load_state_dict(adapter_state)
        self.img_proj.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "img_proj.pt"), map_location="cpu"))
        self.hint_proj.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "hint_proj.pt"), map_location="cpu"))

        for m in [self.text_encoder_one, self.text_encoder_two,
                  self.vae, self.transformer,
                  self.token_adapter, self.img_proj, self.hint_proj]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        try:
            from diffusers.utils import is_xformers_available
            if is_xformers_available() and hasattr(self.transformer, "enable_xformers_memory_efficient_attention"):
                self.transformer.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        prompt_list = [""] * args.parallel_size
        with torch.no_grad():
            t_tok = self.tokenizer_two(prompt_list, padding="max_length", max_length=512, return_tensors="pt").input_ids.to(device)
            c_tok = self.tokenizer_one(prompt_list, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            self.prompt_embeds = self.text_encoder_two(t_tok)[0].to(dtype=dtype)
            self.pooled_prompt = self.text_encoder_one(c_tok).pooler_output.to(dtype=dtype)
            self.text_ids = torch.zeros(self.prompt_embeds.shape[1], 3, device=device, dtype=dtype).contiguous()

        self.text_encoder_one.to("cpu")
        self.text_encoder_two.to("cpu")
        torch.cuda.empty_cache()
        self.transformer.to(device=device, dtype=dtype)

        self.vae_scale_factor = 2 ** (len(getattr(self.vae.config, "block_out_channels", [0,0,0])) - 1) if getattr(self.vae.config, "block_out_channels", None) else 8
        self.latent_ch = getattr(self.vae.config, "latent_channels", 16)
        self.H_lat, self.W_lat = args.height // self.vae_scale_factor, args.width // self.vae_scale_factor
        self.guidance = torch.full([args.parallel_size], 1.0, device=device, dtype=dtype)
        self.latent_image_ids = prepare_img_ids(self.vae_scale_factor, args.parallel_size, args.height, args.width, dtype, device)

        try:
            mu = getattr(self.scheduler.config, "mu", 0.5)
            self.scheduler.set_timesteps(args.steps, device=device, mu=mu)
        except TypeError:
            if getattr(self.scheduler.config, "use_dynamic_shifting", False):
                self.scheduler.config.use_dynamic_shifting = False
            self.scheduler.set_timesteps(args.steps, device=device)

# ---------------- Diffusion decode ----------------
@torch.inference_mode()
def diffusion_decode_to_array(
    code_vectors_cpu: torch.Tensor,   # [1,576,D] float32 (cpu)
    mask_feats_cpu: torch.Tensor,     # [1,576,4] float32 (cpu)
    args, device, weight_dtype, ctx: DiffusionCtx, debug: bool = False,
) -> np.ndarray:
    vae, transformer = ctx.vae, ctx.transformer
    token_adapter, img_proj, hint_proj = ctx.token_adapter, ctx.img_proj, ctx.hint_proj
    scheduler = ctx.scheduler
    batch_size = code_vectors_cpu.shape[0]
    prompt_embeds = ctx.prompt_embeds[:batch_size]
    pooled_prompt = ctx.pooled_prompt[:batch_size]
    text_ids, latent_image_ids = ctx.text_ids, ctx.latent_image_ids
    H_lat, W_lat = ctx.H_lat, ctx.W_lat
    latent_ch = ctx.latent_ch
    guidance = ctx.guidance[:batch_size]
    vae_scale_factor = ctx.vae_scale_factor

    expected_dim = token_adapter.in_proj.in_features
    assert code_vectors_cpu.shape[-1] == expected_dim, (
        f"Expected code_vectors dim={expected_dim}, got {code_vectors_cpu.shape[-1]}"
    )
    B = batch_size
    H2, W2 = (H_lat // 2), (W_lat // 2)
    token_side = math.isqrt(code_vectors_cpu.shape[1])
    if token_side * token_side != code_vectors_cpu.shape[1]:
        raise ValueError("Code-vector token count must form a square grid")

    def up4(feat_1x_cpu: torch.Tensor) -> torch.Tensor:
        f = feat_1x_cpu.view(B, 1, token_side, token_side)
        f = F.interpolate(f, size=(H2, W2), mode="bilinear", align_corners=False)
        f = f.permute(0, 2, 3, 1).reshape(B, H2 * W2, 1).contiguous()
        return f

    code_vectors_gpu = code_vectors_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    feat_confidence_up   = up4(mask_feats_cpu[:, :, 0])
    feat_margin_up       = up4(mask_feats_cpu[:, :, 1])
    feat_topk_mass_up    = up4(mask_feats_cpu[:, :, 2])
    feat_tail_entropy_up = up4(mask_feats_cpu[:, :, 3])
    mask_feat_tokens = torch.cat(
        [feat_confidence_up, feat_margin_up, feat_topk_mass_up, feat_tail_entropy_up],
        dim=-1
    ).clamp(0, 1).to(device=device, dtype=weight_dtype)

    inpaint_cond = token_adapter(
        code_vectors_gpu.to(dtype=weight_dtype).contiguous(),
        mask_feat_tokens.contiguous(),
        args.height, args.width, vae_scale_factor
    ).contiguous()

    hs_hint = hint_proj(inpaint_cond).contiguous()
    timesteps = scheduler.timesteps

    latents = torch.stack(
        [
            torch.randn(
                latent_ch,
                H_lat,
                W_lat,
                device=device,
                dtype=weight_dtype,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )
            for _ in range(B)
        ],
        dim=0,
    ).contiguous()

    for si, t in enumerate(timesteps):
        for attr in ("_step_index", "step_index"):
            if hasattr(scheduler, attr):
                try: setattr(scheduler, attr, int(si))
                except Exception: pass

        packed_noisy = pack_latents(latents, B, latent_ch, H_lat, W_lat)
        hs_img  = img_proj(packed_noisy).contiguous()
        packed_input = (hs_img + hs_hint).contiguous()

        timestep = (t if t.dim() == 1 else t[None]).to(device=device, dtype=weight_dtype) / 1000.0
        if timestep.shape[0] == 1 and B > 1:
            timestep = timestep.expand(B).contiguous()

        model_out_packed = transformer(
            hidden_states=packed_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0].contiguous()
        model_out = unpack_packed(model_out_packed, B, latent_ch, H_lat, W_lat).contiguous()

        step_out = scheduler.step(model_out, t, latents)

        nxt = None
        for key in ("prev_sample", "sample", "denoised"):
            val = getattr(step_out, key, None)
            if val is not None:
                nxt = val
                break
        if nxt is None and isinstance(step_out, dict):
            for key in ("prev_sample", "sample", "denoised"):
                if key in step_out and step_out[key] is not None:
                    nxt = step_out[key]
                    break
        if nxt is None and isinstance(step_out, (tuple, list)) and len(step_out) > 0:
            nxt = step_out[0]
        if nxt is None:
            raise RuntimeError("scheduler.step() returned no usable sample (prev_sample/sample/denoised).")

        latents = nxt.contiguous()

    latents = (latents / getattr(vae.config, "scaling_factor", 0.18215)) + getattr(vae.config, "shift_factor", 0.0)
    images = vae.decode(latents.contiguous()).sample
    images = (images.to(torch.float32) / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return images  # [B,H,W,3] in [0,1]

# ---------------- utils: grid & I/O ----------------
def save_grid_2x2(imgs_0to1: List[np.ndarray], out_path: Path):
    assert len(imgs_0to1) == 4
    h, w = imgs_0to1[0].shape[:2]
    canvas = np.zeros((h*2, w*2, 3), dtype=np.float32)
    canvas[0:h,     0:w    ] = imgs_0to1[0]
    canvas[0:h,     w:2*w  ] = imgs_0to1[1]
    canvas[h:2*h,   0:w    ] = imgs_0to1[2]
    canvas[h:2*h,   w:2*w  ] = imgs_0to1[3]
    Image.fromarray((canvas*255).clip(0,255).astype("uint8")).save(out_path)

def ensure_prompt_folder(base: Path, idx5: str):
    d = base / idx5
    (d / "samples").mkdir(parents=True, exist_ok=True)
    return d

# ---------------- worker (per GPU) ----------------
def worker_proc(rank: int, world_size: int, visible_devices: List[str], args):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    # JSONL loading
    lines = []
    with open(args.meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            prompt = obj.get("prompt", "")
            tag = obj.get("tag", "unknown")
            include = obj.get("include", [])
            lines.append({"prompt": prompt, "tag": tag, "include": include, "raw": s})

    # tag filter
    if args.tags.strip():
        allow = {t.strip() for t in args.tags.split(",") if t.strip()}
        lines = [x for x in lines if x["tag"] in allow]

    total = len(lines)
    if args.max_items > 0:
        lines = lines[:args.max_items]
        total = len(lines)
    indices = [i for i in range(total) if (i % world_size) == rank]

    out_root = Path(args.out_root)
    out_vq = out_root / "vqvae"
    out_df = out_root / "diff"
    out_vq.mkdir(parents=True, exist_ok=True)
    out_df.mkdir(parents=True, exist_ok=True)

    # model load
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    _ = vl_chat_processor.tokenizer
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    )
    mmgpt = mmgpt.to(torch_dtype).to(device).eval()
    mmgpt.gen_vision_model.eval()

    ctx = DiffusionCtx(args=args, device=device, dtype=torch_dtype)

    def all_exist(base_dir: Path):
        if not (base_dir / "grid.png").exists():
            return False
        return all(
            (base_dir / "samples" / f"{sample_idx:04d}.png").exists()
            for sample_idx in range(4)
        )

    batch_starts = range(0, len(indices), args.prompt_batch_size)
    pbar = tqdm(
        batch_starts,
        total=math.ceil(len(indices) / args.prompt_batch_size),
        desc=f"[GPU{rank}] Gen {len(indices)}/{total}",
    )

    for start in pbar:
        pending = []
        for i in indices[start:start + args.prompt_batch_size]:
            item = lines[i]
            idx5 = f"{i:05d}"
            vq_dir = ensure_prompt_folder(out_vq, idx5)
            df_dir = ensure_prompt_folder(out_df, idx5)
            for metadata_path in (
                vq_dir / "metadata.jsonl",
                df_dir / "metadata.jsonl",
            ):
                if not metadata_path.exists():
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        f.write(item["raw"] + "\n")
            if args.skip_existing and all_exist(vq_dir) and all_exist(df_dir):
                continue
            pending.append((i, item, vq_dir, df_dir))

        if not pending:
            pbar.set_postfix_str(f"skip batch {start}")
            continue

        flat_prompts = [
            record[1]["prompt"]
            for record in pending
            for _ in range(4)
        ]
        sample_seeds = [
            int(args.seed) + sample_idx
            for _ in pending
            for sample_idx in range(4)
        ]
        gen_tokens, code_vectors, mask_feats = janus_generate_both_batch(
            mmgpt,
            vl_chat_processor,
            flat_prompts,
            seeds=sample_seeds,
            image_token_num_per_image=args.image_token_num_per_image,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            args=args,
        )
        all_vq_imgs = list(
            vqvae_decode_batch_to_arrays(
                mmgpt, gen_tokens, img_size=384, patch_size=16
            )
        )
        all_df_imgs = list(
            diffusion_decode_to_array(
                code_vectors_cpu=code_vectors,
                mask_feats_cpu=mask_feats,
                args=args,
                device=device,
                weight_dtype=torch_dtype,
                ctx=ctx,
                debug=args.debug,
            )
        )

        for prompt_offset, (i, item, vq_dir, df_dir) in enumerate(pending):
            begin = 4 * prompt_offset
            vq_imgs = all_vq_imgs[begin:begin + 4]
            df_imgs = all_df_imgs[begin:begin + 4]
            for sample_idx, (vq_arr, df_arr) in enumerate(
                zip(vq_imgs, df_imgs)
            ):
                Image.fromarray((vq_arr * 255).astype("uint8")).save(
                    vq_dir / "samples" / f"{sample_idx:04d}.png"
                )
                Image.fromarray((df_arr * 255).astype("uint8")).save(
                    df_dir / "samples" / f"{sample_idx:04d}.png"
                )
            save_grid_2x2(vq_imgs, vq_dir / "grid.png")
            save_grid_2x2(df_imgs, df_dir / "grid.png")
        pbar.set_postfix_str(f"done batch {start} ({len(pending)})")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Janus T2I: Geneval JSONL → VQ-VAE & Diffusion, 4-samples + 2x2 grids.")
    ap.add_argument("--meta_jsonl", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--tags", type=str, default="", help="comma-separated tag filters, e.g. single_object,two_objects")
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-7B")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--gpu", type=str, default="0", help='GPU ids string, e.g. "0", "0,1", "0,1,2,3"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max_items", type=int, default=0,
                    help="Process at most this many prompts; 0 means all")
    ap.add_argument("--prompt_batch_size", type=int, default=2,
                    help="Number of GenEval prompts batched per GPU")

    # Janus autoregressive sampling parameters
    ap.add_argument("--image_token_num_per_image", type=int, default=576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=5.0)

    # FLUX.1-Fill-dev decoder
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--parallel_size", type=int, default=1)

    args = ap.parse_args()

    # Respect a physical-GPU restriction supplied by the launcher. Child
    # processes address the restricted devices by logical rank.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible == "":
        visible = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if visible == "":
        visible_devices = ["0"]
    else:
        visible_devices = [x.strip() for x in visible.split(",") if x.strip() != ""]
    world_size = len(visible_devices)

    if world_size > 1:
        mp.set_start_method("spawn", force=True)
        procs = []
        for r in range(world_size):
            p = mp.Process(target=worker_proc, args=(r, world_size, visible_devices, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [(p.pid, p.exitcode) for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"GenEval worker failures: {failed}")
    else:
        worker_proc(0, 1, visible_devices, args)

if __name__ == "__main__":
    main()
