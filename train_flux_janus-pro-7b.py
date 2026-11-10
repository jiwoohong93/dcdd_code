#!/usr/bin/env python
# coding=utf-8

"""
Train FLUX Fill as a decoder for Janus-Pro-7B
"""

import json
import math
import os
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from tqdm.auto import tqdm

import transformers
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.optimization import get_scheduler

# Local modules
from dataset import ImageDataset
from janus_tokenizer import JanusVQEncoder

# --------------------- utils: pack/unpack latents ---------------------
def pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int):
    x = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return x

def unpack_latents(packed: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int):
    up = packed.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
    up = up.permute(0, 3, 1, 4, 2, 5)
    up = up.reshape(batch_size, num_channels, height, width)
    return up

def prepare_latent_image_ids(batch_size: int, height: int, width: int, device, dtype):
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

def encode_images_to_latents(vae: AutoencoderKL, pixel_values: torch.Tensor, weight_dtype, height: int, width: int):
    z = vae.encode(pixel_values).latent_dist.sample()
    z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    return z.to(dtype=weight_dtype)

# --------------------- token -> inpaint_cond adapter ---------------------
class ConvTokenAdapter(nn.Module):
    """
    Janus VQ code vectors [B,N,D] → per-token cond [B, (H_lat//2 * W_lat//2), 20]
    """
    def __init__(self, token_dim:int, base_ch:int=64):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, base_ch)
        self.stem = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GELU(),
        )
        self.to16 = nn.Conv2d(base_ch, 16, 1)

    def forward(self, code_vectors: torch.Tensor, mask_feat_tokens: torch.Tensor, out_height: int, out_width: int, vae_scale_factor: int):
        B, N, D = code_vectors.shape
        token_side = math.isqrt(N)
        if token_side * token_side != N:
            raise ValueError(f"Expected a square VQ token grid, received N={N}")
        H_lat, W_lat = out_height // vae_scale_factor, out_width // vae_scale_factor
        H2, W2 = H_lat // 2, W_lat // 2

        x = self.in_proj(code_vectors)                 # [B,N,C]
        x = x.view(B, token_side, token_side, -1).permute(0,3,1,2)
        x = self.stem(x)
        x = torch.nn.functional.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)
        x = self.to16(x)                               # [B,16,H2,W2]
        x = x.permute(0,2,3,1).contiguous().view(B, H2 * W2, 16)  # [B,H2*W2,16]

        assert mask_feat_tokens.shape[:2] == (B, H2 * W2)
        return torch.cat([x, mask_feat_tokens], dim=-1)           # [B, H2*W2, 20]

# -------------------- small projectors --------------------
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

# -------------------- wrapper --------------------
class CombinedModel(nn.Module):
    def __init__(self, transformer: nn.Module, token_adapter: nn.Module, img_proj: nn.Module, hint_proj: nn.Module):
        super().__init__()
        self.transformer = transformer
        self.token_adapter = token_adapter
        self.img_proj = img_proj
        self.hint_proj = hint_proj

    def forward(
        self,
        code_vectors: torch.Tensor,
        mask_feat_tokens: torch.Tensor,
        packed_noisy: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        out_height: int,
        out_width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        inpaint_cond = self.token_adapter(
            code_vectors,
            mask_feat_tokens,
            out_height,
            out_width,
            vae_scale_factor,
        )
        packed_input = self.img_proj(packed_noisy) + self.hint_proj(inpaint_cond)
        return self.transformer(
            hidden_states=packed_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

# --------------------- EMA (lightweight) ---------------------
class SimpleEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.data is not None:
                    self.shadow[n] = p.data.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad and p.data is not None:
                self.shadow[n].mul_(d).add_(p.data, alpha=(1.0 - d))

    def to(self, device, dtype=None):
        for k in list(self.shadow.keys()):
            t = self.shadow[k].to(device)
            if dtype is not None:
                t = t.to(dtype)
            self.shadow[k] = t

    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.decay = float(sd.get("decay", self.decay))
        sh = sd["shadow"]
        self.shadow = {k: v.clone() for k, v in sh.items()}

# --------------------- args ---------------------
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: Optional[str], subfolder: str = "text_encoder"):
    cfg = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    arch = cfg.architectures[0]
    if arch == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif arch == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{arch} is not supported.")

def build_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    # FLUX.1-Fill-dev configuration
    ap.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)

    ap.add_argument("--dataroot", type=str, required=True)

    ap.add_argument("--output_dir", type=str, default="trained-flux_janus-pro-7b")
    ap.add_argument("--logging_dir", type=str, default="logs")
    ap.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--use_8bit_adam", action="store_true")
    ap.add_argument("--lr_scheduler", type=str, default="constant")
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--max_train_steps", type=int, default=50000)
    ap.add_argument("--checkpointing_steps", type=int, default=10000)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_tf32", action="store_true")

    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max_sequence_length", type=int, default=512)

    ap.add_argument("--janus_model", type=str, default="deepseek-ai/Janus-Pro-7B")
    ap.add_argument("--guidance_scale", type=float, default=1.0)

    ap.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["uniform","logit_normal","cosine"])
    ap.add_argument("--logit_mean", type=float, default=0.0)
    ap.add_argument("--logit_std", type=float, default=1.0)
    ap.add_argument("--mode_scale", type=float, default=1.29)

    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Optional W&B tracking. Local JSONL metrics are always written.",
    )
    ap.add_argument("--tracker_project_name", type=str, default="dcdd")
    ap.add_argument("--tracker_run_name", type=str, default=None)
    ap.add_argument("--logging_steps", type=int, default=1)

    ap.add_argument("--grad_checkpointing", action="store_true", default=False)

    # ===== Janus calibration parameters =====
    ap.add_argument("--janus_similarity", type=str, default="cosine", choices=["cosine","neg_l2","dot"])
    ap.add_argument("--janus_temperature", type=float, default=1.0)
    ap.add_argument("--janus_cosine_scale", type=float, default=29.0381)
    ap.add_argument("--janus_label_smoothing_eps", type=float, default=0.01)

    return ap

# --------------------- Model-state resume ---------------------
def _find_transformer_weight_files(tr_dir: str):
    """Return every shard in a Diffusers checkpoint, in deterministic order."""
    if not os.path.isdir(tr_dir):
        return []

    index_files = sorted(
        fn for fn in os.listdir(tr_dir)
        if fn.endswith((".safetensors.index.json", ".bin.index.json"))
    )
    if index_files:
        with open(os.path.join(tr_dir, index_files[0]), "r") as f:
            weight_map = json.load(f).get("weight_map", {})
        return [
            os.path.join(tr_dir, fn)
            for fn in sorted(set(weight_map.values()))
        ]

    choices = [
        fn for fn in os.listdir(tr_dir)
        if fn.endswith((".safetensors", ".bin", ".pt"))
    ]
    return [os.path.join(tr_dir, fn) for fn in sorted(choices)]

def _load_state_dict_any(path: str):
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
            return safe_load(path)
        except Exception:
            return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu")

def resume_from_light_checkpoint(accelerator, model, optimizer, lr_scheduler, resume_dir, weight_dtype, ema_obj: Optional['SimpleEMA']):
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"[resume] checkpoint not found: {resume_dir}")
    unwrapped = accelerator.unwrap_model(model)

    tr_dir = os.path.join(resume_dir, "transformer")
    weight_files = _find_transformer_weight_files(tr_dir)
    if weight_files and all(os.path.isfile(p) for p in weight_files):
        expected = set(unwrapped.transformer.state_dict().keys())
        loaded = set()
        unexpected = set()
        for weight_file in weight_files:
            tr_sd = _load_state_dict_any(weight_file)
            loaded.update(expected.intersection(tr_sd.keys()))
            load_result = unwrapped.transformer.load_state_dict(tr_sd, strict=False)
            unexpected.update(load_result.unexpected_keys)
            del tr_sd
        missing = expected.difference(loaded)
        if missing or unexpected:
            raise RuntimeError(
                "Transformer checkpoint does not match the current model: "
                f"missing={sorted(missing)[:20]}, unexpected={sorted(unexpected)[:20]}"
            )
        if accelerator.is_main_process:
            accelerator.print(
                f"[resume] loaded transformer from {len(weight_files)} file(s) "
                f"| missing={len(missing)} unexpected={len(unexpected)}"
            )
    else:
        if accelerator.is_main_process:
            accelerator.print("[resume] WARNING: transformer weights not found; skipping.")

    for name, fn in [("token_adapter","token_adapter.pt"),("img_proj","img_proj.pt"),("hint_proj","hint_proj.pt")]:
        pth = os.path.join(resume_dir, fn)
        if os.path.exists(pth):
            getattr(unwrapped, name).load_state_dict(torch.load(pth, map_location="cpu"))

    sch_p = os.path.join(resume_dir, "scheduler.pt")
    if os.path.exists(sch_p): lr_scheduler.load_state_dict(torch.load(sch_p, map_location="cpu"))

    opt_p = os.path.join(resume_dir, "optimizer.pt")
    if os.path.exists(opt_p):
        optimizer.load_state_dict(torch.load(opt_p, map_location="cpu"))
    elif accelerator.is_main_process:
        accelerator.print(
            "[resume] optimizer.pt not found; optimizer state starts fresh"
        )

    ema_p = os.path.join(resume_dir, "ema.pt")
    if ema_obj is not None and os.path.exists(ema_p):
        ema_sd = torch.load(ema_p, map_location="cpu")
        ema_obj.load_state_dict(ema_sd)
        ema_obj.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    meta_p = os.path.join(resume_dir, "train_state.json")
    if os.path.exists(meta_p):
        with open(meta_p, "r") as f: meta = json.load(f)
        global_step = int(meta.get("global_step", 0))

    accelerator.wait_for_everyone()
    return global_step

# --------------------- main ---------------------
def main():
    args = build_argparser().parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1")
    if args.max_train_steps < 1:
        raise ValueError("--max_train_steps must be at least 1")
    if args.checkpointing_steps < 1:
        raise ValueError("--checkpointing_steps must be at least 1")
    if args.logging_steps < 1:
        raise ValueError("--logging_steps must be at least 1")
    if args.report_to == "wandb":
        try:
            __import__("wandb")
        except ImportError as exc:
            raise ImportError(
                "W&B logging was requested but wandb is not installed. "
                "Install requirements-wandb.txt or use --report_to=none."
            ) from exc

    if args.allow_tf32: torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        log_with=None if args.report_to == "none" else args.report_to,
    )
    if args.seed is not None: set_seed(args.seed)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        from diffusers.utils import logging as dlogging; dlogging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        from diffusers.utils import logging as dlogging; dlogging.set_verbosity_error()

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizers & encoders
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)

    te1_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    te2_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    text_encoder_one = te1_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = te2_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)

    # FLUX.1-Fill-dev components
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant)

    vae.requires_grad_(False).eval()
    text_encoder_one.requires_grad_(False).eval()
    text_encoder_two.requires_grad_(False).eval()

    # Gradient checkpointing
    if args.grad_checkpointing:
        ok = False
        for attr in ("enable_gradient_checkpointing","gradient_checkpointing_enable","set_gradient_checkpointing"):
            if hasattr(transformer, attr):
                try:
                    getattr(transformer, attr)(); ok=True; break
                except TypeError:
                    try:
                        getattr(transformer, attr)(True); ok=True; break
                    except Exception: pass
        if not ok:
            try: transformer.gradient_checkpointing = True
            except Exception: pass

    # Freeze the backbone except for the attention blocks used by DCDD.
    transformer.requires_grad_(False)
    for name, p in transformer.named_parameters():
        if ("attn" in name) and ("transformer_blocks" in name or "single_transformer_blocks" in name):
            p.requires_grad = True
    if not any(p.requires_grad for p in transformer.parameters()):
        raise RuntimeError("No FLUX attention parameters were selected for training")

    # Parameter dtypes
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    # The training prompt is always empty, so compute its embeddings once and
    # release both frozen text encoders before loading the remaining models.
    with torch.no_grad():
        empty_prompt = [""]
        t_tok = tokenizer_two(
            empty_prompt,
            padding="max_length",
            max_length=args.max_sequence_length,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        c_tok = tokenizer_one(
            empty_prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        prompt_embeds_base = text_encoder_two(t_tok)[0].to(
            device=accelerator.device, dtype=weight_dtype
        )
        pooled_prompt_base = text_encoder_one(c_tok).pooler_output.to(
            device=accelerator.device, dtype=weight_dtype
        )
        text_ids = torch.zeros(
            prompt_embeds_base.shape[1],
            3,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    text_encoder_one.to("cpu")
    text_encoder_two.to("cpu")
    del text_encoder_one, text_encoder_two
    torch.cuda.empty_cache()

    # VAE utils
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8

    # Dataset
    train_dataset = ImageDataset(
        dataroot_path=args.dataroot,
        phase="train",
        size=args.size,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    # Janus & token adapter
    janus = JanusVQEncoder(
        model=args.janus_model,
        device=accelerator.device,
        dtype=weight_dtype,
        similarity=args.janus_similarity,
        temperature=args.janus_temperature,
        cosine_scale=args.janus_cosine_scale,
        label_smoothing_eps=args.janus_label_smoothing_eps,
    )
    if hasattr(janus, "parameters"):
        for p in janus.parameters(): p.requires_grad = False

    with torch.no_grad():
        dummy = torch.zeros(1,3,384,384, device=accelerator.device, dtype=weight_dtype)
        enc = janus.encode_tensor(dummy.to(torch.float32))  # dict
        code_vec = enc["code_vectors"]  # [1,576,D]
    token_dim = code_vec.shape[-1]
    token_adapter = ConvTokenAdapter(token_dim).to(accelerator.device, dtype=weight_dtype)

    # latent channels
    latent_channels = getattr(vae.config, "latent_channels", None)
    if latent_channels is None:
        with torch.no_grad():
            dummy = torch.zeros(1,3,args.size,args.size, device=accelerator.device, dtype=weight_dtype)
            dummy_lat = vae.encode(dummy).latent_dist.sample()
            latent_channels = dummy_lat.shape[1]

    # projector dims
    cond_dim = 20
    H_lat = args.size // vae_scale_factor
    W_lat = args.size  // vae_scale_factor
    packed_noisy_dim = 4 * latent_channels
    in_ch = int(getattr(transformer.config, "in_channels", 64))

    img_proj = ImageProjector(packed_noisy_dim, in_ch).to(accelerator.device, dtype=weight_dtype)
    hint_proj = HintProjector(cond_dim, in_ch).to(accelerator.device, dtype=weight_dtype)

    # Optimizer
    params = [
        {"params": [p for p in transformer.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": token_adapter.parameters(), "lr": args.learning_rate},
        {"params": img_proj.parameters(), "lr": args.learning_rate},
        {"params": hint_proj.parameters(), "lr": args.learning_rate},
    ]
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2)

    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes
    )

    # Wrap & prepare
    model = CombinedModel(transformer, token_adapter, img_proj, hint_proj).to(accelerator.device, dtype=weight_dtype)
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    # Track exponential moving averages of trainable parameters.
    ema = SimpleEMA(accelerator.unwrap_model(model), decay=args.ema_decay)
    ema.to(accelerator.device, dtype=weight_dtype)

    # Logging
    if args.report_to != "none":
        init_kwargs = {}
        if args.tracker_run_name:
            init_kwargs["wandb"] = {"name": args.tracker_run_name}
        accelerator.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs=init_kwargs,
        )

    if accelerator.is_main_process:
        effective_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        run_config = dict(vars(args))
        run_config.update(
            {
                "num_processes": accelerator.num_processes,
                "effective_global_batch_size": effective_batch_size,
                "trainable_transformer_parameters": sum(
                    p.numel() for p in transformer.parameters() if p.requires_grad
                ),
            }
        )
        with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)

    # Restore training state when resuming from a checkpoint.
    global_step = 0
    if args.resume_from:
        global_step = resume_from_light_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            resume_dir=args.resume_from,
            weight_dtype=weight_dtype,
            ema_obj=ema,
        )
        if accelerator.is_main_process:
            accelerator.print(f"[resume] from {args.resume_from} -> step={global_step}")

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
        return sigma

    # ---------- helpers ----------
    def build_mask_feat_tokens(metrics: Dict[str, torch.Tensor], hw: torch.Tensor, H2: int, W2: int) -> torch.Tensor:
        """4 channels: confidence, margin, topk_mass, tail_entropy"""
        B = next(iter(metrics.values())).shape[0]
        h, w = int(hw[0].item()), int(hw[1].item())
        def up(t: torch.Tensor):
            m = t.view(B, 1, h, w)
            m = F.interpolate(m, size=(H_lat//2, W_lat//2), mode="bilinear", align_corners=False)
            return m  # [B,1,H2,W2]
        conf   = up(metrics["feat_confidence"])
        margin = up(metrics["feat_margin"])
        topk   = up(metrics["feat_topk_mass"])
        tent   = up(metrics["feat_tail_entropy"])
        feat = torch.cat([conf, margin, topk, tent], dim=1).clamp(0,1)  # [B,4,H2,W2]
        feat = feat.permute(0,2,3,1).contiguous().view(B, (H_lat//2)*(W_lat//2), 4)  # [B,H2*W2,4]
        return feat.to(dtype=weight_dtype)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    accumulated_loss = 0.0
    accumulated_microsteps = 0
    total_examples_seen = global_step * (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Train loop
    while global_step < args.max_train_steps:
        model.train()

        for batch in train_loader:
            with accelerator.accumulate(model):
                # 1) Encode conditioning tokens with the frozen Janus VQ model.
                images = batch["image"]
                image_for_janus = images.to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    enc = janus.encode_tensor(image_for_janus)
                code_vectors = enc["code_vectors"]      # [B,576,D]
                mask_feat_tokens = build_mask_feat_tokens(
                    {
                        "feat_confidence": enc["feat_confidence"],
                        "feat_margin": enc["feat_margin"],
                        "feat_topk_mass": enc["feat_topk_mass"],
                        "feat_tail_entropy": enc["feat_tail_entropy"],
                    },
                    enc["hw"],
                    H_lat // 2,
                    W_lat // 2,
                )
                del enc

                # 2) Encode the target image into FLUX latent space.
                target_image = images.to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    model_input = encode_images_to_latents(
                        vae, target_image, weight_dtype, args.size, args.size
                    )

                # 3) Add noise and sample timesteps.
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                ntt = noise_scheduler.config.num_train_timesteps
                indices = (u.clamp(0.0, (ntt-1)/float(ntt)) * ntt).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # 4) Pack latents into FLUX tokens.
                packed_noisy = pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=H_lat, width=W_lat,
                )  # [B, H2*W2, 4C]

                # 5) Reuse the precomputed empty-prompt embeddings.
                prompt_embeds = prompt_embeds_base.expand(bsz, -1, -1)
                pooled_prompt = pooled_prompt_base.expand(bsz, -1)
                guidance = torch.full([1], args.guidance_scale, device=accelerator.device).expand(bsz)

                # 6) Build image position IDs.
                latent_image_ids = prepare_img_ids(vae_scale_factor, bsz, args.size, args.size, weight_dtype, accelerator.device)

                # 7) Run the adapter, projector, and transformer through DDP.
                model_pred = model(
                    code_vectors=code_vectors.to(weight_dtype),
                    mask_feat_tokens=mask_feat_tokens,
                    packed_noisy=packed_noisy,
                    timestep=timesteps / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled_prompt,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    out_height=args.size,
                    out_width=args.size,
                    vae_scale_factor=vae_scale_factor,
                )  # [B,N,4C]

                # 8) Unpack the predicted latent tokens.
                model_pred = unpack_latents(model_pred, bsz, model_input.shape[1], H_lat, W_lat)

                # ---------- Losses ----------
                # Flow-matching prediction loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss_vpred = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                loss = loss_vpred

                reduced_loss = accelerator.gather(
                    loss.detach().float().reshape(1)
                ).mean().item()
                accumulated_loss += reduced_loss
                accumulated_microsteps += 1

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                total_examples_seen += (
                    args.train_batch_size
                    * accelerator.num_processes
                    * accumulated_microsteps
                )
                mean_loss = accumulated_loss / max(accumulated_microsteps, 1)
                log_record = {
                    "global_step": int(global_step),
                    "train/loss": float(mean_loss),
                    "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    "train/microsteps_per_update": int(accumulated_microsteps),
                    "train/examples_seen": int(total_examples_seen),
                }
                if grad_norm is not None:
                    log_record["train/grad_norm"] = (
                        float(grad_norm.detach().float().item())
                        if isinstance(grad_norm, torch.Tensor)
                        else float(grad_norm)
                    )
                if global_step % args.logging_steps == 0:
                    if args.report_to != "none":
                        accelerator.log(log_record, step=global_step)
                    if accelerator.is_main_process:
                        with open(metrics_path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(log_record, sort_keys=True) + "\n")
                    progress_bar.set_postfix(
                        loss=f"{mean_loss:.6f}",
                        lr=f"{log_record['train/lr']:.3e}",
                    )
                accumulated_loss = 0.0
                accumulated_microsteps = 0

                # ---------- Component checkpoint ----------
                if (global_step % args.checkpointing_steps == 0) and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.token_adapter.state_dict(), os.path.join(ckpt_dir, "token_adapter.pt"))
                    torch.save(unwrapped.img_proj.state_dict(),    os.path.join(ckpt_dir, "img_proj.pt"))
                    torch.save(unwrapped.hint_proj.state_dict(),   os.path.join(ckpt_dir, "hint_proj.pt"))
                    unwrapped.transformer.save_pretrained(os.path.join(ckpt_dir, "transformer"))
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                    torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                    torch.save(ema.state_dict(), os.path.join(ckpt_dir, "ema.pt"))
                    with open(os.path.join(ckpt_dir, "train_state.json"), "w") as f:
                        json.dump({"global_step": int(global_step)}, f)

                accelerator.wait_for_everyone()
                if global_step >= args.max_train_steps:
                    break

    accelerator.end_training()
    if accelerator.is_main_process:
        print("Training finished.")

if __name__ == "__main__":
    main()
