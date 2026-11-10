# coding=utf-8
from typing import Literal, List, Sequence, Union
import os
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

class ImageDataset(data.Dataset):
    """
    Recursively loads images from ``dataroot_path/phase``.

    Each item contains a square, normalized image tensor and its path relative
    to ``dataroot_path``.
    """
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "val", "test"] = "train",
        size: int = 384,
    ):
        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.size = size

        # ---------- Transforms ----------
        def build_center_crop_pipeline(target: Union[int, Sequence[int]]) -> transforms.Compose:
            if isinstance(target, int):
                resize_side = target
                crop_size = target
            else:
                target_h, target_w = int(target[0]), int(target[1])
                resize_side = max(target_h, target_w)
                crop_size = (target_h, target_w)

            return transforms.Compose([
                transforms.Resize(resize_side, interpolation=Image.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1,1]
            ])

        self.tf = build_center_crop_pipeline(self.size)

        # ---------- Collect all image file paths recursively ----------
        phase_root = os.path.join(dataroot_path, phase)
        self.im_paths: List[str] = []
        valid_exts = (".jpg", ".jpeg", ".png", ".webp")

        for root, _, files in os.walk(phase_root):
            for fn in files:
                if fn.lower().endswith(valid_exts):
                    full_path = os.path.join(root, fn)
                    self.im_paths.append(full_path)

        self.im_paths.sort()

        if len(self.im_paths) == 0:
            raise RuntimeError(f"No image files found under: {phase_root}")

    def __len__(self):
        return len(self.im_paths)

    def _load_image_with_transform(self, path: str, tfm: transforms.Compose) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return tfm(img)

    def __getitem__(self, index):
        path = self.im_paths[index]
        im_name = os.path.relpath(path, self.dataroot)  # e.g., "train/class1/img001.jpg"

        image = self._load_image_with_transform(path, self.tf)  # [3, H, W]

        return {
            "im_name": im_name,
            "image": image,
        }
