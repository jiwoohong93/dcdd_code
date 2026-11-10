#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Janus token distribution collection + automatic VQ Encoder calibration.

Notes:
 - This script does NOT perform FLUX decoding.
 - Calibration images must be specified via --calib_imgs_dir.
"""


import os
import math
import json
import glob
import argparse
import random
import numpy as np
from typing import List, Tuple, Dict, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus_tokenizer import JanusVQEncoder


# ---------------- Prompts ----------------
def build_prompts(num_prompts: int, prompts_file: Optional[str], seed: int = 0) -> List[str]:
    if not prompts_file or not os.path.isfile(prompts_file):
        raise FileNotFoundError(
            "A GenEval evaluation_metadata.jsonl file is required via --prompts_file"
        )
    pool = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                prompt = item.get("prompt") if isinstance(item, dict) else None
            except json.JSONDecodeError:
                prompt = line
            if prompt:
                pool.append(str(prompt))
    if len(pool) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts, but only found {len(pool)} in {prompts_file}"
        )
    random.Random(seed).shuffle(pool)
    return pool[:num_prompts]


# ---------------- image loader ----------------
def load_calib_images(img_dir: str,
                      max_n: int = 256,
                      seed: int = 0,
                      recursive: bool = True,
                      exts: Tuple[str, ...] = ("*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.JPEG")) -> Tuple[List[Image.Image], List[str]]:
    paths: List[str] = []
    if recursive:
        for e in exts:
            paths += glob.glob(os.path.join(img_dir, "**", e), recursive=True)
    else:
        for e in exts:
            paths += glob.glob(os.path.join(img_dir, e))

    paths = sorted(list(set(paths)))
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    random.seed(seed)
    random.shuffle(paths)
    paths = paths[:max_n]

    imgs: List[Image.Image] = []
    kept: List[str] = []
    for p in tqdm(paths, desc="Load images", leave=False, dynamic_ncols=True):
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
            kept.append(p)
        except Exception:
            continue
    if not imgs:
        raise RuntimeError("Failed to open any images.")
    return imgs, kept


# ---------------- Janus AR probability distribution collection ----------------
@torch.inference_mode()
def collect_janus_token_stats(model_path: str,
                              prompts: List[str],
                              image_token_num_per_image: int = 576,
                              temperature: float = 1.0,
                              cfg_weight: float = 5.0,
                              device: str | torch.device = "cuda",
                              heartbeat_interval: int = 50) -> Dict[str, float]:
    """
    Extract conditional and unconditional CFG probability distributions from the Janus
    generation head and collect token-level statistics:
    mean_norm_entropy, mean_conf, and p95_conf.
    """
    vl_proc: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_proc.tokenizer
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    mmgpt = mmgpt.to(torch.bfloat16).to(device).eval()

    conf_list, nent_list = [], []
    K = None
    step_counter = 0

    for pi, prompt in enumerate(tqdm(prompts, desc="Prompts", dynamic_ncols=True)):
        conversation = [{"role": "User", "content": prompt},
                        {"role": "Assistant", "content": ""}]
        sft = vl_proc.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation, sft_format=vl_proc.sft_format, system_prompt=""
        )
        sft = sft + vl_proc.image_start_tag

        input_ids = torch.LongTensor(tokenizer.encode(sft))
        tokens = torch.zeros((2, len(input_ids)), dtype=torch.int, device=device)
        tokens[0, :] = input_ids
        tokens[1, :] = tokens[0, :]
        tokens[1, 1:-1] = vl_proc.pad_id  # uncond

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)  # [2, L, Dh]
        outputs = None

        inner_bar = tqdm(range(image_token_num_per_image),
                         desc=f"Tokens [{pi+1}/{len(prompts)}]",
                         leave=False, dynamic_ncols=True)
        for _ in inner_bar:
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if outputs is not None else None
            )
            hs = outputs.last_hidden_state
            logits = mmgpt.gen_head(hs[:, -1, :])        # [2, K]
            if K is None:
                K = logits.shape[-1]

            logit_c = logits[0:1, :]
            logit_u = logits[1:2, :]
            guided = logit_u + cfg_weight * (logit_c - logit_u)  # [1,K]
            probs = torch.softmax(guided / temperature, dim=-1).float()

            conf = probs.max(dim=-1).values
            H = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            nent = H / math.log(K)

            conf_list.append(conf.item())
            nent_list.append(nent.item())

            step_counter += 1
            if heartbeat_interval > 0 and (step_counter % heartbeat_interval == 0):
                inner_bar.set_postfix_str(f"hb={step_counter}")

            next_token = torch.multinomial(probs, num_samples=1)
            next_pair = torch.cat([next_token, next_token], dim=0).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_pair)
            inputs_embeds = img_embeds.unsqueeze(1)

            del hs, logits, guided, probs, img_embeds, next_token, next_pair

    conf_arr = np.array(conf_list, dtype=np.float64)
    nent_arr = np.array(nent_list, dtype=np.float64)
    stats = {
        "K": int(K) if K is not None else 0,
        "mean_norm_entropy": float(nent_arr.mean()),
        "p95_norm_entropy": float(np.percentile(nent_arr, 95)),
        "mean_confidence": float(conf_arr.mean()),
        "p95_confidence": float(np.percentile(conf_arr, 95)),
        "num_samples": int(len(prompts)),
        "tokens_per_sample": int(image_token_num_per_image),
    }
    return stats


# ---------------- VQ: statistics computation ----------------
@torch.inference_mode()
def eval_vq_metrics(encoder: JanusVQEncoder, imgs: List[Image.Image]) -> dict:
    """
    Compute confidence/entropy statistics using the VQ encoder.
    - Robustly handle varying output dictionary keys (e.g., probs/logits/feat_confidence, etc.)
    - Compute mean and p95 at the token level
    """
    from torchvision import transforms
    pre = transforms.Compose([
        transforms.Resize((384,384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(), lambda x: x*2-1
    ])
    all_conf = []
    all_nent = []

    for im in tqdm(imgs, desc="VQ eval(images)", leave=False, dynamic_ncols=True):
        x = pre(im).unsqueeze(0).to(encoder.device).to(encoder.dtype)
        out = encoder.encode_tensor(x) 

        if "probs" in out:  # [B,N,K]
            probs = out["probs"].float()
            conf = probs.max(dim=-1).values
            K = probs.size(-1)
            H = -(probs.clamp_min(1e-12)*probs.clamp_min(1e-12).log()).sum(dim=-1)
            nent = H / math.log(K)
        elif "logits" in out:  # [B,N,K]
            logits = out["logits"].float()
            probs = torch.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values
            K = probs.size(-1)
            H = -(probs.clamp_min(1e-12)*probs.clamp_min(1e-12).log()).sum(dim=-1)
            nent = H / math.log(K)
        elif "conf" in out and "normalized_entropy" in out: 
            conf = out["conf"]
            nent = out["normalized_entropy"]
        elif "feat_confidence" in out and "normalized_entropy" in out:
            conf = out["feat_confidence"]
            nent = out["normalized_entropy"]
        else:
            raise KeyError(
                "encode_tensor output must contain one of: "
                "'probs', 'logits', ('conf' & 'normalized_entropy'), ('feat_confidence' & 'normalized_entropy'). "
                f"Got keys: {list(out.keys())}"
            )

        all_conf.append(conf.flatten().detach().cpu().numpy())
        all_nent.append(nent.flatten().detach().cpu().numpy())

    conf_arr = np.concatenate(all_conf, axis=0)
    nent_arr = np.concatenate(all_nent, axis=0)
    return {
        "mean_norm_entropy": float(nent_arr.mean()),
        "p95_norm_entropy": float(np.percentile(nent_arr, 95)),
        "mean_confidence": float(conf_arr.mean()),
        "p95_confidence": float(np.percentile(conf_arr, 95)),
    }


# ---------------- search helpers ----------------
def _objective(vq_stats: dict, tgt: dict, w=(1.0, 0.25, 0.25, 0.35)):
    """
    w = (w_meanEnt, w_meanConf, w_p95Conf, w_p95Ent)
    By default, mean entropy is weighted most strongly,
    followed by p95 entropy.
    """
    e_mean = (vq_stats["mean_norm_entropy"] - tgt["mean_norm_entropy"])**2
    c_mean = (vq_stats["mean_confidence"]    - tgt["mean_confidence"])**2
    c_p95  = (vq_stats["p95_confidence"]     - tgt["p95_confidence"])**2
    e_p95  = (vq_stats["p95_norm_entropy"]   - tgt["p95_norm_entropy"])**2
    return w[0]*e_mean + w[1]*c_mean + w[2]*c_p95 + w[3]*e_p95


@torch.inference_mode()
def search_params(encoder: JanusVQEncoder, imgs: List[Image.Image], tgt: dict,
                  similarity: str = "cosine"):
    """Search the paper parameters in order: a, alpha, then epsilon."""
    if similarity != "cosine":
        raise ValueError("Paper calibration requires cosine proxy logits")

    encoder.similarity = "cosine"
    encoder.cosine_scale = 1.0
    encoder.temperature = 1.0
    encoder.label_smoothing_eps = 0.0

    def select(name: str, values, desc: str):
        best = None
        for value in tqdm(values, desc=desc, dynamic_ncols=True, leave=False):
            setattr(encoder, name, float(value))
            stats = eval_vq_metrics(encoder, imgs)
            candidate = (_objective(stats, tgt), float(value), stats)
            if best is None or candidate[0] < best[0]:
                best = candidate
        setattr(encoder, name, best[1])
        return best[2]

    best = None
    lo, hi = 1.0, 60.0
    for _ in tqdm(range(12), desc="Search a", dynamic_ncols=True, leave=False):
        scale = 0.5 * (lo + hi)
        encoder.cosine_scale = scale
        stats = eval_vq_metrics(encoder, imgs)
        entropy_error = stats["mean_norm_entropy"] - tgt["mean_norm_entropy"]
        candidate = (abs(entropy_error), scale, stats)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if abs(entropy_error) <= 0.02:
            break
        if entropy_error > 0.0:
            lo = scale
        else:
            hi = scale
    encoder.cosine_scale = best[1]

    select("temperature", np.linspace(0.5, 2.0, 16), "Search alpha")
    stats = select(
        "label_smoothing_eps", np.linspace(0.0, 0.05, 11), "Search epsilon"
    )
    params = {
        "similarity": "cosine",
        "temperature": float(encoder.temperature),
        "cosine_scale": float(encoder.cosine_scale),
        "label_smoothing_eps": float(encoder.label_smoothing_eps),
    }
    return {"params": params, "stats": stats}


# ---------------- IO helpers ----------------
def write_env_txt(path_txt: str, janus_stats: dict, vq_suggest: dict):
    s = vq_suggest["params"]
    v = vq_suggest["stats"]
    lines = []
    lines.append("# Janus token distribution stats (targets)")
    lines.append(f"# vocab_size(K)         : {janus_stats['K']}")
    lines.append(f"# mean_norm_entropy     : {janus_stats['mean_norm_entropy']:.6f}")
    lines.append(f"# p95_norm_entropy      : {janus_stats['p95_norm_entropy']:.6f}")
    lines.append(f"# mean_confidence       : {janus_stats['mean_confidence']:.6f}")
    lines.append(f"# p95_confidence        : {janus_stats['p95_confidence']:.6f}")
    lines.append(f"# samples x tokens      : {janus_stats['num_samples']} x {janus_stats['tokens_per_sample']}")
    lines.append("# --- Suggested VQ calibration values (matched to above targets) ---")
    lines.append(f'export JANUS_SIMILARITY="{s["similarity"]}"')
    lines.append(f'export JANUS_TEMPERATURE="{s["temperature"]:.4f}"')
    lines.append(f'export JANUS_COSINE_SCALE="{s["cosine_scale"]:.4f}"')
    lines.append(f'export JANUS_LABEL_SMOOTHING_EPS="{s["label_smoothing_eps"]:.4f}"')
    lines.append("# --- VQ stats achieved (sanity check, token-wise) ---")
    lines.append(f"# VQ mean_norm_entropy  : {v['mean_norm_entropy']:.6f}")
    lines.append(f"# VQ p95_norm_entropy   : {v['p95_norm_entropy']:.6f}")
    lines.append(f"# VQ mean_confidence    : {v['mean_confidence']:.6f}")
    lines.append(f"# VQ p95_confidence     : {v['p95_confidence']:.6f}")
    txt = "\n".join(lines)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # Janus
    ap.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument(
        "--prompts_file", type=str, required=True,
        help="GenEval evaluation_metadata.jsonl used for the VLM target prompts",
    )
    ap.add_argument("--image_token_num_per_image", type=int, default=576)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--janus_temperature", type=float, default=1.0)
    ap.add_argument("--heartbeat_interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)

    # VQ calibration images
    ap.add_argument("--calib_imgs_dir", type=str, required=True)
    ap.add_argument("--calib_max_images", type=int, default=100)
    ap.add_argument(
        "--calib_recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--calib_exts", type=str, default="jpg,jpeg,png,webp,bmp,JPEG")

    # VQ similarity function
    ap.add_argument("--vq_similarity", type=str, default="cosine", choices=["cosine"])

    # Output
    ap.add_argument("--out_txt", type=str, default="janus_calibration.txt")
    ap.add_argument("--out_json", type=str, default=None)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # --------- 1) Janus stats ---------
    prompts = build_prompts(args.num_prompts, args.prompts_file, seed=args.seed)

    janus_stats = collect_janus_token_stats(
        model_path=args.model_path,
        prompts=prompts,
        image_token_num_per_image=args.image_token_num_per_image,
        temperature=args.janus_temperature,
        cfg_weight=args.cfg_weight,
        device=device,
        heartbeat_interval=args.heartbeat_interval,
    )

    # --------- 2) VQ search on real images ---------
    ext_tuple = tuple([f"*.{e.strip()}" for e in args.calib_exts.split(",") if e.strip()])
    imgs, _ = load_calib_images(
        args.calib_imgs_dir,
        max_n=args.calib_max_images,
        seed=args.seed,
        recursive=args.calib_recursive,
        exts=ext_tuple if ext_tuple else ("*.jpg","*.png")
    )

    encoder = JanusVQEncoder(
        model=args.model_path,
        device=device,
        dtype=torch.bfloat16,
        similarity="cosine",
        temperature=1.0,     
        cosine_scale=20.0,   
        label_smoothing_eps=0.0
    )

    vq_suggest = search_params(
        encoder=encoder,
        imgs=imgs,
        tgt={
            "mean_norm_entropy": janus_stats["mean_norm_entropy"],
            "p95_norm_entropy":  janus_stats["p95_norm_entropy"],
            "mean_confidence":   janus_stats["mean_confidence"],
            "p95_confidence":    janus_stats["p95_confidence"],
        },
        similarity=args.vq_similarity,
    )

    # --------- 3) Save / print ---------
    txt = write_env_txt(args.out_txt, janus_stats, vq_suggest)
    print(txt)

    out_json = args.out_json or (os.path.splitext(args.out_txt)[0] + ".json")
    payload = {
        "janus_stats": janus_stats,
        "vq_suggest": vq_suggest,
        "args": vars(args)
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
