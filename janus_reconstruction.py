#!/usr/bin/env python
# coding: utf-8
"""
Batch reconstruction for FLUX.1-Fill-dev with Janus entropy-aligned mask features,
reading images via ImageDataset(phase="val") to reuse the exact transforms.

Notes:
- The dataset will scan <dataroot>/val/** for images.
- Output files preserve each input image's relative path and extension.
"""

import argparse, os
from pathlib import Path
from typing import Dict

from datetime import timedelta 

import torch
import torch.nn as nn
import math, torch.distributed as dist

from transformers import PretrainedConfig, CLIPTokenizer, T5TokenizerFast
from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler

from tqdm import tqdm

from dataset import ImageDataset
from torch.utils.data import DataLoader

from janus_tokenizer import JanusVQEncoder


# ---------- FLUX utilities ----------
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
    # (B, C, H_lat, W_lat) -> (B, (H_lat//2)*(W_lat//2), 4C)
    x = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return x


def unpack_packed(latents_packed: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int):
    up = latents_packed.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
    up = up.permute(0, 3, 1, 4, 2, 5)
    up = up.reshape(batch_size, num_channels, height, width)
    return up


def prepare_latent_image_ids(batch_size: int, height: int, width: int, device, dtype):
    # Per-token image IDs (shared across batch)
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    h, w, c = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(h * w, c)
    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_img_ids(vae_scale_factor: int, batch_size: int, height: int, width: int, dtype, device):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width  = 2 * (int(width)  // (vae_scale_factor * 2))
    return prepare_latent_image_ids(batch_size, height, width, device, dtype)


# ---------- ConvTokenAdapter ----------
class ConvTokenAdapter(nn.Module):
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
        x = x.view(B, token_side, token_side, -1).permute(0,3,1,2)
        x = self.stem(x)
        x = torch.nn.functional.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)
        x = self.to16(x)                            # [B,16,H2,W2]
        x = x.permute(0,2,3,1).contiguous().view(B, H2*W2, 16)  # [B,H2*W2,16]

        assert mask_feat_tokens.shape[:2] == (B, H2*W2), \
            f"mask_feat_tokens shape mismatch: {mask_feat_tokens.shape} vs {(B,H2*W2)}"
        return torch.cat([x, mask_feat_tokens], dim=-1)         # [B,H2*W2,20]


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


def build_mask_feat_tokens(metrics: Dict[str, torch.Tensor], hw: torch.Tensor, H2: int, W2: int, dtype):
    B = metrics["feat_confidence"].shape[0]
    h = int(hw[0].item())
    w = int(hw[1].item())

    def up(t: torch.Tensor):
        m = t.view(B, 1, h, w)
        m = torch.nn.functional.interpolate(m, size=(H2, W2), mode="bilinear", align_corners=False)
        return m  # [B,1,H2,W2]

    f_conf  = up(metrics["feat_confidence"])
    f_margin = up(metrics["feat_margin"])
    f_topk  = up(metrics["feat_topk_mass"])
    f_tail  = up(metrics["feat_tail_entropy"])

    feat = torch.cat([f_conf, f_margin, f_topk, f_tail], dim=1).clamp(0, 1)  # [B,4,H2,W2]
    feat = feat.permute(0,2,3,1).contiguous().view(B, H2*W2, 4)              # [B,H2*W2,4]
    return feat.to(dtype=dtype)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)

    # Input and output paths
    ap.add_argument("--dataroot", type=str, required=True, help="Root folder containing 'val' subfolder with images")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to save reconstructions (subfolders preserved)")

    # Sampling & sizes
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--max_sequence_length", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["no","fp16","bf16"])
    ap.add_argument("--global_prompt", type=str, default="")

    # Janus calibration params (match training)
    ap.add_argument("--janus_model", type=str, default="deepseek-ai/Janus-Pro-7B")
    ap.add_argument("--janus_similarity", type=str, default="cosine", choices=["cosine","neg_l2","dot"])
    ap.add_argument("--janus_temperature", type=float, default=1.0)
    ap.add_argument("--janus_cosine_scale", type=float, default=29.0381)
    ap.add_argument("--janus_label_smoothing_eps", type=float, default=0.01)

    # Dataloader options
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=0,
                    help="Process at most this many images; 0 means all")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images already present in --out_dir")
    args = ap.parse_args()

    # DDP init (works for both single and torchrun)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
    else:
        local_rank = 0; world_size = 1; rank = 0

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    # dtype
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    g = torch.Generator(device=device).manual_seed(args.seed + rank)

    # Enforce global batch semantics: per-rank = global // world_size
    if world_size > 1:
        assert args.batch_size % world_size == 0, \
            f"--batch_size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size})."
        per_rank_bs = args.batch_size // world_size
    else:
        per_rank_bs = args.batch_size


    # --- Dataset / DataLoader ---
    test_dataset = ImageDataset(
        dataroot_path=args.dataroot,
        phase="val",
        size=args.height,
    )
    if args.max_items > 0:
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(min(args.max_items, len(test_dataset)))
        )

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        sampler = None

    loader = DataLoader(
        test_dataset,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        batch_size=per_rank_bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Base components (loaded once) ---
    tokenizer_one = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.base_model, subfolder="tokenizer_2", revision=args.revision)
    te_cls_one = import_model_class_from_model_name_or_path(args.base_model, args.revision, subfolder="text_encoder")
    te_cls_two = import_model_class_from_model_name_or_path(args.base_model, args.revision, subfolder="text_encoder_2")
    text_encoder_one = te_cls_one.from_pretrained(args.base_model, subfolder="text_encoder",
                                                  revision=args.revision, variant=args.variant).to(device, dtype=weight_dtype)
    text_encoder_two = te_cls_two.from_pretrained(args.base_model, subfolder="text_encoder_2",
                                                  revision=args.revision, variant=args.variant).to(device, dtype=weight_dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae",
                                        revision=args.revision, variant=args.variant).to(device, dtype=weight_dtype)
    # Keep the denoiser on CPU until the one-time text embeddings are computed
    # to reduce peak device memory.
    transformer = FluxTransformer2DModel.from_pretrained(
        os.path.join(args.ckpt_dir, "transformer")
    ).eval()

    # VAE/latent sizes
    vae_scale_factor = 2 ** (len(getattr(vae.config, "block_out_channels", [0,0,0])) - 1) if getattr(vae.config, "block_out_channels", None) else 8
    latent_ch = getattr(vae.config, "latent_channels", 16)
    H_lat, W_lat = args.height // vae_scale_factor, args.width // vae_scale_factor
    in_ch = int(getattr(transformer.config, "in_channels", 64))

    # --- Adapters (load from ckpt_dir) ---
    janus = JanusVQEncoder(
        model=args.janus_model, device=device, dtype=weight_dtype,
        similarity=args.janus_similarity,
        temperature=args.janus_temperature,
        cosine_scale=args.janus_cosine_scale,
        label_smoothing_eps=args.janus_label_smoothing_eps,
    )
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 384, device=device, dtype=torch.float32)
        enc = janus.encode_tensor(dummy)  # to get token_dim
        token_dim = enc["code_vectors"].shape[-1]

    ta_path = os.path.join(args.ckpt_dir, "token_adapter.pt")
    ip_path = os.path.join(args.ckpt_dir, "img_proj.pt")
    hp_path = os.path.join(args.ckpt_dir, "hint_proj.pt")

    token_adapter = ConvTokenAdapter(token_dim).to(device, dtype=weight_dtype)
    token_adapter.load_state_dict(torch.load(ta_path, map_location="cpu"))
    token_adapter.eval()

    img_proj  = ImageProjector(in_dim=4*latent_ch, out_dim=in_ch).to(device, dtype=weight_dtype)
    hint_proj = HintProjector(in_dim=20,          out_dim=in_ch).to(device, dtype=weight_dtype)
    img_proj.load_state_dict(torch.load(ip_path, map_location="cpu")); img_proj.eval()
    hint_proj.load_state_dict(torch.load(hp_path, map_location="cpu")); hint_proj.eval()

    # --- Text embeddings (base, repeat per batch) ---
    prompt = [args.global_prompt]

    text_encoder_two = text_encoder_two.to("cpu").eval()
    with torch.no_grad():

        t_tok = tokenizer_two(
            prompt,
            padding="max_length",
            max_length=args.max_sequence_length,
            return_tensors="pt"
        ).input_ids.to("cpu")

        c_tok = tokenizer_one(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).input_ids.to(device)

        prompt_embeds_base_cpu = text_encoder_two(t_tok)[0].to(dtype=torch.float32)   # [1, L, D] on CPU

    prompt_embeds_base = prompt_embeds_base_cpu.to(device=device, dtype=weight_dtype, non_blocking=True)

    pooled_prompt_base = text_encoder_one(c_tok).pooler_output.to(device=device, dtype=weight_dtype)  # [1, D]
    text_ids = torch.zeros(prompt_embeds_base.shape[1], 3, device=device, dtype=weight_dtype)         # [L, 3]

    text_encoder_one = text_encoder_one.to("cpu")
    del text_encoder_one, text_encoder_two
    torch.cuda.empty_cache()
    transformer = transformer.to(device=device, dtype=weight_dtype)

    latent_image_ids = prepare_img_ids(
        vae_scale_factor, 1, args.height, args.width, weight_dtype, device
    )  # [HW, 3]

    # --- Ensure out_dir exists ---
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    show_bar = (rank == 0)

    with torch.inference_mode():
        total = 0
        for batch in tqdm(loader, desc="Reconstructing images", total=len(loader), disable=not show_bar):
            im_names = list(batch["im_name"])                            # list[str], len=B
            if args.skip_existing:
                keep = [i for i, name in enumerate(im_names)
                        if not (out_root / name).is_file()]
                if not keep:
                    continue
                im_names = [im_names[i] for i in keep]
                x_lr = batch["image"][keep].to(device=device, dtype=torch.float32)
            else:
                x_lr = batch["image"].to(device=device, dtype=torch.float32)
            B = x_lr.shape[0]

            # Repeat per-batch text/guidance
            with torch.no_grad():
                prompt_embeds = prompt_embeds_base.repeat(B, 1, 1)  # [B,L,D]
                pooled_prompt = pooled_prompt_base.repeat(B, 1)     # [B,D]
                guidance = torch.full((B,), args.guidance_scale, device=device, dtype=weight_dtype)  # [B]

                # Encode the batch with Janus.
                enc = janus.encode_tensor(x_lr)   # dict with batch dims
                code_vectors = enc["code_vectors"]  # [B,576,D]

                H2, W2 = (H_lat // 2), (W_lat // 2)
                mask_feat_tokens = build_mask_feat_tokens(
                    {
                        "feat_confidence":   enc["feat_confidence"],   # [B,h,w]
                        "feat_margin":       enc["feat_margin"],
                        "feat_topk_mass":    enc["feat_topk_mass"],
                        "feat_tail_entropy": enc["feat_tail_entropy"],
                    },
                    enc["hw"], H2, W2, dtype=weight_dtype
                )  # [B,H2*W2,4]

                inpaint_cond = token_adapter(
                    code_vectors.to(weight_dtype),  # [B,576,D]
                    mask_feat_tokens,               # [B,H2*W2,4]
                    args.height, args.width, vae_scale_factor
                )  # [B,H2*W2,20]

            # Initialize the batch of diffusion latents.
            latents = torch.randn(B, latent_ch, H_lat, W_lat, device=device, dtype=weight_dtype, generator=g)

            mu = getattr(scheduler.config, "mu", 0.5)
            try:
                scheduler.set_timesteps(args.steps, device=device, mu=mu)
            except TypeError:
                if getattr(scheduler.config, "use_dynamic_shifting", False):
                    scheduler.config.use_dynamic_shifting = False
                scheduler.set_timesteps(args.steps, device=device)
            timesteps = scheduler.timesteps

            # --- Batched sampling loop ---
            for t in tqdm(timesteps, desc="Denoising steps", leave=False):
                packed_noisy = pack_latents(latents, batch_size=B, num_channels_latents=latent_ch, height=H_lat, width=W_lat)  # [B,HW/4,4C]
                hs_img  = img_proj(packed_noisy)   # [B,HW/4,in_ch]
                hs_hint = hint_proj(inpaint_cond)  # [B,HW/4,in_ch]
                packed_input = hs_img + hs_hint

                tvec = torch.full((B,), float(t), device=device, dtype=weight_dtype) / 1000.0
                model_out_packed = transformer(
                    hidden_states=packed_input,
                    timestep=tvec,
                    guidance=guidance,
                    pooled_projections=pooled_prompt,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]  # [B,HW/4,4C]

                model_out = unpack_packed(model_out_packed, B, latent_ch, H_lat, W_lat)  # [B,C,H_lat,W_lat]
                latents = scheduler.step(model_out, t, latents).prev_sample

            # --- Decode and save the batch ---
            latents_dec = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            with torch.no_grad():
                images = vae.decode(latents_dec).sample  # [B,3,H,W]

            images = images.to(torch.float32)
            images = (images / 2 + 0.5).clamp(0, 1).cpu()  # [B,3,H,W], [0,1]

            from PIL import Image as PILImage
            for b, name in enumerate(im_names):
                out_path = out_root / name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                arr = (images[b].permute(1, 2, 0).numpy() * 255).astype("uint8")
                PILImage.fromarray(arr).save(str(out_path))
                total += 1


    # Avoid a final CUDA collective: a large final batch can leave too little
    # free memory even for NCCL's bookkeeping allocation. Ranks write disjoint
    # filenames, so no synchronization is needed here.
    print(f"[DONE rank={rank}] saved: {total} images → {out_root}")
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
