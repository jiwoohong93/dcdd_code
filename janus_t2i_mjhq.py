#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Janus text-to-image generation with native VQ-VAE and DCDD diffusion decoding.
- After filtering categories using meta_data.json, both methods generate images
  from the same prompts.
- Output structure:
    out_root/
      ├─ vqvae/<category>/<image_id>.jpg
      └─ diff/<category>/<image_id>.png
"""

import os
import json
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, List

import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, PretrainedConfig, CLIPTokenizer, T5TokenizerFast
from janus.models import MultiModalityCausalLM, VLChatProcessor

# ---------- Speed-friendly backend toggles ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Prefer Flash / memory-efficient SDPA kernels when available (PyTorch 2.x+ / Ampere+)
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

# High-throughput FP32 matmul path (has effect only for remaining FP32 ops)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- DCDD conditioning modules ----------------
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


class ConvTokenAdapter(nn.Module):
    """
    [B,576,D] + [B,H2*W2,4] -> [B, H2*W2, 20]
    """
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
        x = self.to16(x)                               # [B,16,H2,W2]
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H2 * W2, 16).contiguous()

        assert mask_feat_tokens.shape[:2] == (B, H2 * W2)
        return torch.cat([x, mask_feat_tokens.contiguous()], dim=-1).contiguous()  # [B,H2*W2,20]

class ImageProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

class HintProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

# ---------------- VQ-VAE decode ----------------
def vqvae_decode_and_save(mmgpt: MultiModalityCausalLM,
                          generated_tokens: torch.Tensor,
                          img_size: int,
                          patch_size: int,
                          save_path,
                          debug: bool = False):
    try:
        tokens_clean = generated_tokens.detach().to('cpu').clone().to(torch.int32)
        model_device = next(mmgpt.gen_vision_model.parameters()).device

        if debug:
            print(f"[VQ] tokens_clean: device={tokens_clean.device}, "
                  f"requires_grad={tokens_clean.requires_grad}, is_leaf={tokens_clean.is_leaf}, dtype={tokens_clean.dtype}",
                  flush=True)

        with torch.no_grad():
            batch_size = tokens_clean.shape[0]
            dec = mmgpt.gen_vision_model.decode_code(
                tokens_clean.to(model_device),
                shape=[batch_size, 8, img_size // patch_size, img_size // patch_size],
            )

        dec = dec.to(torch.float32).detach().to("cpu").numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        save_paths = (
            list(save_path)
            if isinstance(save_path, (list, tuple))
            else [save_path]
        )
        if len(save_paths) != batch_size:
            raise ValueError("Number of VQ output paths must match batch size")
        for image, output_path in zip(dec, save_paths):
            Image.fromarray(image).save(output_path)

    except Exception as e:
        print(f"[VQ][ERROR] save failed at {save_path}: {e}", flush=True)
        raise


# ---------------- Diffusion Context: load once, reuse ----------------
class DiffusionCtx:
    def __init__(self, *, args, device, dtype):
        from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler

        # 1) load model/tokenizer
        self.tokenizer_one = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
        self.tokenizer_two = T5TokenizerFast.from_pretrained(args.base_model, subfolder="tokenizer_2")

        te_cls_one = import_model_class_from_model_name_or_path(args.base_model, None, subfolder="text_encoder")
        te_cls_two = import_model_class_from_model_name_or_path(args.base_model, None, subfolder="text_encoder_2")
        self.text_encoder_one = te_cls_one.from_pretrained(args.base_model, subfolder="text_encoder").to(device, dtype=dtype).eval()
        self.text_encoder_two = te_cls_two.from_pretrained(args.base_model, subfolder="text_encoder_2").to(device, dtype=dtype).eval()

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, dtype=dtype).eval()
        # Stage the transformer on CPU until text embeddings are ready to reduce
        # peak device memory.
        self.transformer = FluxTransformer2DModel.from_pretrained(
            os.path.join(args.ckpt_dir, "transformer")
        ).eval()

        # 2) load adapter and projector
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

        # Ensure non-trainable & eval everywhere (safety for inference graph)
        for m in [self.text_encoder_one, self.text_encoder_two,
                  self.vae, self.transformer,
                  self.token_adapter, self.img_proj, self.hint_proj]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # (Optional) Enable xFormers memory-efficient attention if supported by model
        try:
            from diffusers.utils import is_xformers_available
            if is_xformers_available() and hasattr(self.transformer, "enable_xformers_memory_efficient_attention"):
                self.transformer.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # 3) Precompute static text embeddings (empty-string pair for parallel CFG)
        prompt_list = [""] * args.parallel_size
        with torch.no_grad():
            t_tok = self.tokenizer_two(prompt_list, padding="max_length", max_length=512, return_tensors="pt").input_ids.to(device)
            c_tok = self.tokenizer_one(prompt_list, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            self.prompt_embeds = self.text_encoder_two(t_tok)[0].to(dtype=dtype)            # [B,L,hd]
            self.pooled_prompt = self.text_encoder_one(c_tok).pooler_output.to(dtype=dtype) # [B,hd]
            self.text_ids = torch.zeros(self.prompt_embeds.shape[1], 3, device=device, dtype=dtype).contiguous()

        # Text embeddings are static for this workflow. Offload both encoders
        # before placing the trained FLUX transformer on the GPU.
        self.text_encoder_one.to("cpu")
        self.text_encoder_two.to("cpu")
        torch.cuda.empty_cache()
        self.transformer.to(device=device, dtype=dtype)

        # 4) Cache derived parameters
        self.vae_scale_factor = 2 ** (len(getattr(self.vae.config, "block_out_channels", [0,0,0])) - 1) if getattr(self.vae.config, "block_out_channels", None) else 8
        self.latent_ch = getattr(self.vae.config, "latent_channels", 16)
        self.H_lat, self.W_lat = args.height // self.vae_scale_factor, args.width // self.vae_scale_factor
        self.guidance = torch.full([args.parallel_size], 1.0, device=device, dtype=dtype)
        self.latent_image_ids = prepare_img_ids(self.vae_scale_factor, args.parallel_size, args.height, args.width, dtype, device)

        # Initialize scheduler timesteps (once at startup)
        try:
            mu = getattr(self.scheduler.config, "mu", 0.5)
            self.scheduler.set_timesteps(args.steps, device=device, mu=mu)
        except TypeError:
            if getattr(self.scheduler.config, "use_dynamic_shifting", False):
                self.scheduler.config.use_dynamic_shifting = False
            self.scheduler.set_timesteps(args.steps, device=device)


# ---------------- FLUX.1-Fill-dev decoding ----------------
@torch.inference_mode()
def diffusion_decode_and_save(
    code_vectors_cpu: torch.Tensor,   # [1,576,D] float32 (cpu)
    mask_feats_cpu: torch.Tensor,     # [1,576,4] float32 (cpu)
    args,
    device,
    weight_dtype,
    save_path: Path,
    ctx: DiffusionCtx,
    debug: bool = False,
):
    # Reuse the preloaded decoder context.
    vae = ctx.vae
    transformer = ctx.transformer
    token_adapter = ctx.token_adapter
    img_proj, hint_proj = ctx.img_proj, ctx.hint_proj
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

    # ---- CPU→GPU transfer (no clone; non_blocking) & condition preparation ----
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

    # Precompute loop-invariant items: hs_hint remains fixed.
    hs_hint = hint_proj(inpaint_cond).contiguous()

    timesteps = scheduler.timesteps
    num_steps = len(timesteps)

    sigmas_len = None
    if hasattr(scheduler, "sigmas"):
        try:
            sigmas_len = len(scheduler.sigmas)
        except Exception:
            sigmas_len = None

    if debug:
        print(f"[DIFF][DEBUG] len(timesteps)={num_steps}, len(sigmas)={sigmas_len}; using {len(timesteps)} steps",
              flush=True)

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
                try:
                    setattr(scheduler, attr, int(si))
                except Exception:
                    pass

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

        nxt = getattr(step_out, "prev_sample", None)
        if nxt is None:
            nxt = getattr(step_out, "sample", None)
        if nxt is None:
            nxt = getattr(step_out, "denoised", None)
        if nxt is not None:
            latents = nxt.contiguous()

        if debug and (si % 10 == 0 or si == len(timesteps) - 1):
            print(f"[DIFF] step {si}/{len(timesteps)} ok; "
                  f"latents[{latents.shape}] {latents.dtype} {latents.device}", flush=True)

    # Decode the final latents.
    latents = (latents / getattr(vae.config, "scaling_factor", 0.18215)) + getattr(vae.config, "shift_factor", 0.0)
    images = vae.decode(latents.contiguous()).sample
    images = (images.to(torch.float32) / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    save_paths = (
        list(save_path)
        if isinstance(save_path, (list, tuple))
        else [save_path]
    )
    if len(save_paths) != B:
        raise ValueError("Number of diffusion output paths must match batch size")
    for image, output_path in zip(images, save_paths):
        Image.fromarray((image * 255).astype("uint8")).save(output_path)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Janus T2I: VQ-VAE + Diffusion decoding, with category filter.")
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--categories", type=str, default="", help="comma-separated category filters, e.g. animals,art,fashion")
    ap.add_argument("--model", default="deepseek-ai/Janus-Pro-7B")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--debug", action="store_true", help="verbose tensor state prints")
    ap.add_argument("--max_items", type=int, default=0,
                    help="Process at most this many metadata entries; 0 means all")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split the selected metadata deterministically across workers")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Zero-based shard handled by this process")
    ap.add_argument("--batch_size_prompts", type=int, default=4,
                    help="Number of MJHQ prompts generated together")

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
    # Respect a physical-GPU restriction supplied by the launcher. Overwriting
    # CUDA_VISIBLE_DEVICES here would turn the launcher's logical ``--gpu 0``
    # back into physical GPU 0.
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_path = Path(args.meta_json)
    if not meta_path.exists():
        raise FileNotFoundError(f"meta json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Dict[str, Any]] = json.load(f)

    # category filter
    sel_set: List[str] = []
    if args.categories.strip():
        sel_set = [c.strip() for c in args.categories.split(",") if c.strip()]
    ids_all = list(meta.keys())
    if sel_set:
        ids = [i for i in ids_all if str(meta[i].get("category","")) in sel_set]
    else:
        ids = ids_all
    if args.max_items > 0:
        ids = ids[:args.max_items]
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    ids = ids[args.shard_id::args.num_shards]

    out_root = Path(args.out_root)
    out_vq = out_root / "vqvae"
    out_df = out_root / "diff"
    out_vq.mkdir(parents=True, exist_ok=True)
    out_df.mkdir(parents=True, exist_ok=True)

    # Load Janus once for the full shard.
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    _ = vl_chat_processor.tokenizer
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    )
    mmgpt = mmgpt.to(torch_dtype).to(device).eval()
    mmgpt.gen_vision_model.eval()

    # Load the diffusion decoder once for the full shard.
    ctx = DiffusionCtx(args=args, device=device, dtype=torch_dtype)

    if args.parallel_size < args.batch_size_prompts:
        raise ValueError("--parallel_size must be at least --batch_size_prompts")
    from janus_t2i_geneval import janus_generate_both_batch

    batch_starts = range(0, len(ids), args.batch_size_prompts)
    pbar = tqdm(
        batch_starts,
        total=math.ceil(len(ids) / args.batch_size_prompts),
        desc=f"Janus T2I batched (filtered {len(ids)}/{len(ids_all)})",
    )

    def attempt_batch(batch):
        """Run one batch in its own frame so OOM tensors are released on error."""
        prompts = [item[1] for item in batch]
        gen_tokens, code_vectors, mask_feats = janus_generate_both_batch(
            mmgpt,
            vl_chat_processor,
            prompts,
            seeds=[args.seed] * len(batch),
            image_token_num_per_image=args.image_token_num_per_image,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            args=args,
        )
        vqvae_decode_and_save(
            mmgpt,
            gen_tokens,
            img_size=384,
            patch_size=16,
            save_path=[item[3] for item in batch],
            debug=args.debug,
        )
        diffusion_decode_and_save(
            code_vectors_cpu=code_vectors,
            mask_feats_cpu=mask_feats,
            args=args,
            device=device,
            weight_dtype=torch_dtype,
            save_path=[item[4] for item in batch],
            ctx=ctx,
            debug=args.debug,
        )

    def process_batch(batch, label):
        """Retry failed batches as smaller halves instead of silently dropping them."""
        try:
            attempt_batch(batch)
            return len(batch), len(batch), 0
        except Exception as error:
            error_text = str(error)
            print(
                f"[RETRY] batch {label} size={len(batch)} failed: {error_text}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) > 1:
                midpoint = len(batch) // 2
                left = process_batch(batch[:midpoint], f"{label}L")
                right = process_batch(batch[midpoint:], f"{label}R")
                return tuple(a + b for a, b in zip(left, right))
            print(
                f"[FAIL] image_id={batch[0][0]}: {error_text}",
                flush=True,
            )
            return 0, 0, 1

    ok_vq, ok_df, fail = 0, 0, 0
    for start in pbar:
        batch = []
        for image_id in ids[start:start + args.batch_size_prompts]:
            entry = meta[image_id]
            category = str(entry.get("category", "unknown"))
            save_dir_vq = out_vq / category
            save_dir_df = out_df / category
            save_dir_vq.mkdir(parents=True, exist_ok=True)
            save_dir_df.mkdir(parents=True, exist_ok=True)
            vq_path = save_dir_vq / f"{image_id}.jpg"
            df_path = save_dir_df / f"{image_id}.png"
            if args.skip_existing and vq_path.exists() and df_path.exists():
                continue
            batch.append(
                (image_id, entry["prompt"], category, vq_path, df_path)
            )

        if not batch:
            pbar.set_postfix_str(f"skip batch {start}")
            continue

        batch_len = len(batch)
        batch_vq, batch_df, batch_fail = process_batch(batch, str(start))
        # Long MJHQ runs can retain cyclic Python objects and unused allocator
        # blocks across batches. Release both before processing the next batch.
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ok_vq += batch_vq
        ok_df += batch_df
        fail += batch_fail
        if batch_fail:
            pbar.set_postfix_str(
                f"partial batch {start}: ok={batch_vq}, fail={batch_fail}"
            )
        else:
            pbar.set_postfix_str(f"saved batch {start} ({batch_len})")

    print(f"[DONE] VQ success={ok_vq}, Diff success={ok_df}, fail={fail}", flush=True)
    if fail:
        raise RuntimeError(f"{fail} MJHQ images failed after adaptive retries")


if __name__ == "__main__":
    main()
