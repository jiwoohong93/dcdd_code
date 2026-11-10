#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate two folders of images (GT vs. Generated) with FID, SSIM, LPIPS.

- FID requires non-empty folders with equal image counts; filenames and image
  extensions may differ.
- SSIM and LPIPS require matching relative filenames (flat or nested OK).
- Computes:
    * FID  (via clean-fid, folder-to-folder)
    * SSIM (torchmetrics)
    * LPIPS (torchmetrics, net=alex)

Usage:
python eval.py \
  --gt_folder /path/to/gt_images \
  --gen_folder /path/to/generated_images \
  --size 384 \
  --batch_size 16 \
  --workers 8

Notes:
- Images are resized to `--size x --size` for SSIM/LPIPS to match spatial size.
- FID is computed directly between the two folders (no resizing by us; clean-fid handles it).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cleanfid import fid
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])


def names_dict(paths: List[Path], base_root: Path) -> Dict[str, Path]:
    """
    Map 'relative path (posix)' -> absolute Path
    We preserve subfolder structure by using relative paths as keys.
    """
    out = {}
    for p in paths:
        rel = p.relative_to(base_root).as_posix()
        out[rel] = p
    return out


class PairFolderDataset(Dataset):
    """
    Loads (gt, gen) paired by relative path name.
    """
    def __init__(self, pairs: List[Tuple[Path, Path]], size: int):
        super().__init__()
        self.pairs = pairs
        # For SSIM/LPIPS we want [0,1] tensors; LPIPS(normalize=True) will internally map to [-1,1]
        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_p, gen_p = self.pairs[idx]
        gt_img = self.tf(PIL.Image.open(gt_p).convert("RGB"))
        gen_img = self.tf(PIL.Image.open(gen_p).convert("RGB"))
        return gt_img, gen_img


@torch.no_grad()
def compute_metrics_two_folders(
    gt_folder: str,
    gen_folder: str,
    size: int = 384,
    batch_size: int = 16,
    workers: int = 8,
    which_metrics: List[str] = ("fid", "ssim", "lpips"),
) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_root = Path(gt_folder)
    gen_root = Path(gen_folder)
    if not gt_root.exists() or not gen_root.exists():
        raise FileNotFoundError(f"Folder missing. GT: {gt_root.exists()} ({gt_root}), GEN: {gen_root.exists()} ({gen_root})")

    gt_list = list_images(gt_root)
    gen_list = list_images(gen_root)

    gt_map = names_dict(gt_list, gt_root)
    gen_map = names_dict(gen_list, gen_root)

    need_pair_metrics = any(m in which_metrics for m in ("ssim", "lpips"))

    # SSIM and LPIPS are paired metrics and therefore require exact relative
    # filenames. FID is distributional, so equal non-zero counts are enough;
    # this also permits reference JPGs to be compared with generated PNGs.
    gt_keys = set(gt_map.keys())
    gen_keys = set(gen_map.keys())
    if need_pair_metrics and gt_keys != gen_keys:
        missing_in_gen = sorted(list(gt_keys - gen_keys))
        missing_in_gt  = sorted(list(gen_keys - gt_keys))
        msg = [
            f"[Count mismatch] GT={len(gt_keys)} vs GEN={len(gen_keys)}",
        ]
        if missing_in_gen:
            msg.append(f"- Missing in GEN ({len(missing_in_gen)}): {missing_in_gen[:10]}{' ...' if len(missing_in_gen)>10 else ''}")
        if missing_in_gt:
            msg.append(f"- Missing in GT  ({len(missing_in_gt)}): {missing_in_gt[:10]}{' ...' if len(missing_in_gt)>10 else ''}")
        raise RuntimeError("\n".join(msg))
    if not need_pair_metrics:
        if not gt_keys or not gen_keys:
            raise RuntimeError(
                f"[Empty input] GT={len(gt_keys)} images, GEN={len(gen_keys)} images"
            )
        if len(gt_keys) != len(gen_keys):
            raise RuntimeError(f"[Count mismatch] GT={len(gt_keys)} vs GEN={len(gen_keys)}")

    # Paired list (sorted for stable order)
    keys_sorted = sorted(gt_keys) if need_pair_metrics else []
    pairs = [(gt_map[k], gen_map[k]) for k in keys_sorted]

    results: Dict[str, float] = {}

    # --- FID (folder-to-folder) ---
    if "fid" in which_metrics:
        # clean-fid can compute fid directly between two folders
        fid_score = fid.compute_fid(
            gt_root.as_posix(),
            fdir2=gen_root.as_posix(),
            mode="clean",
            dataset_split="custom",
            verbose=True,
            use_dataparallel=False,
        )
        results["fid"] = float(fid_score)

    # --- SSIM / LPIPS (paired, same spatial size) ---
    if need_pair_metrics:
        ds = PairFolderDataset(pairs, size=size)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
        )

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if "ssim" in which_metrics else None
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device) if "lpips" in which_metrics else None

        for gt_img, gen_img in tqdm(dl, desc="SSIM/LPIPS"):
            gt_img  = gt_img.to(device)
            gen_img = gen_img.to(device)

            if ssim_metric is not None:
                ssim_metric.update(gen_img, gt_img)

            if lpips_metric is not None:
                lpips_metric.update(gen_img, gt_img)

        if ssim_metric is not None:
            results["ssim"] = float(ssim_metric.compute().item())
        if lpips_metric is not None:
            results["lpips"] = float(lpips_metric.compute().item())

    return results


def main():
    ap = argparse.ArgumentParser(description="Compute FID / SSIM / LPIPS between two image folders (GT vs GEN)")
    ap.add_argument("--gt_folder",  type=str, required=True, help="Path to ground-truth images folder")
    ap.add_argument("--gen_folder", type=str, required=True, help="Path to generated images folder")
    ap.add_argument("--size",       type=int, default=384, help="Resize square side for SSIM/LPIPS")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--metrics",    type=str, default="fid,ssim,lpips", help="Comma-separated subset of {fid,ssim,lpips}")
    ap.add_argument("--save_json",  type=str, default="", help="Optional path to save JSON (default: gen_folder/metrics.json)")
    args = ap.parse_args()

    which = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    for m in which:
        if m not in ("fid", "ssim", "lpips"):
            raise ValueError(f"Unsupported metric: {m}")

    res = compute_metrics_two_folders(
        gt_folder=args.gt_folder,
        gen_folder=args.gen_folder,
        size=args.size,
        batch_size=args.batch_size,
        workers=args.workers,
        which_metrics=which,
    )

    for k, v in res.items():
        print(f"{k}: {v:.6f}")

    out_json = Path(args.save_json) if args.save_json else (Path(args.gen_folder) / "metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[DONE] saved metrics to: {out_json}")


if __name__ == "__main__":
    main()
