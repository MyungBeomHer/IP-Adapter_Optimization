#!/usr/bin/env python3
"""
CLIP‑I / CLIP‑T evaluator for COCO‑style pairs JSON.

Input assumptions
- pairs JSON: a list of {"image_file": "000000xxxxxx.jpg", "text": "caption ..."}
- image_root: directory containing the images (e.g., COCO val2017)
- gen_root: directory containing generated images. For each sample, all
  generated images should be named with the stem of the reference image file.
  By default this script looks for files matching {stem}_*.png under gen_root.
  (You can change with --gen_glob.)

Metrics (OpenAI CLIP ViT‑L/14 by default)
- CLIP‑I: cosine(clip_img(gen), clip_img(ref))
- CLIP‑T (CLIPScore): 100 * max( cosine(clip_img(gen), clip_txt(caption)), 0 )

Outputs
- Prints overall means and coverage stats
- Optional CSV with per‑sample aggregates and per‑image lists

Example
  python eval_clip_scores.py \
    --pairs_json /data/COCO2017/val_pairs.json \
    --image_root /data/COCO2017/val2017 \
    --gen_root /work/outputs/ipadapter_gen \
    --out_csv clip_eval_val5k.csv \
    --clip_model openai/clip-vit-large-patch14 \
    --batch_size 32

Notes
- If your generated files are organized differently, adjust --gen_glob.
- This script computes features on‑the‑fly with small caches; for 5k images it is fine on a single GPU.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


@torch.no_grad()
def get_image_features(model: CLIPModel, processor: CLIPProcessor, images: List[Image.Image], device: torch.device, batch_size: int = 32) -> torch.Tensor:
    feats = []
    for i in range(0, len(images), batch_size):
        b = images[i : i + batch_size]
        inputs = processor(images=b, return_tensors="pt").to(device)
        f = model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def get_text_features(model: CLIPModel, processor: CLIPProcessor, texts: List[str], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    feats = []
    for i in range(0, len(texts), batch_size):
        b = texts[i : i + batch_size]
        inputs = processor(text=b, padding=True, truncation=True, return_tensors="pt").to(device)
        f = model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_json", required=True, help="Path to pairs JSON: list of {image_file, text}")
    ap.add_argument("--image_root", required=True, help="Directory with reference images (e.g., val2017)")
    ap.add_argument("--gen_root", required=True, help="Directory with generated images")
    ap.add_argument("--gen_glob", default="{stem}_*.png", help="Glob pattern under gen_root; {stem} is replaced by image stem")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14", help="HF model id for CLIP")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for quick tests)")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    ap.add_argument("--out_csv", default=None, help="Optional CSV path for per‑sample outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load pairs
    pairs = json.load(open(args.pairs_json, "r"))
    if not isinstance(pairs, list):
        raise ValueError("pairs_json must be a list of {image_file, text}")
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]

    # Load CLIP
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()

    image_root = Path(args.image_root)
    gen_root = Path(args.gen_root)

    total_gen = 0
    used_samples = 0
    clip_i_vals: List[float] = []
    clip_t_vals: List[float] = []

    per_sample_rows: List[Tuple[str, int, float, float, str, str]] = []

    # Small caches
    ref_img_feat_cache: Dict[str, torch.Tensor] = {}
    text_feat_cache: Dict[str, torch.Tensor] = {}

    pbar = tqdm(pairs, desc="Evaluating")
    for rec in pbar:
        img_file = rec["image_file"]
        caption = rec.get("text", "")
        stem = Path(img_file).stem

        # Find generated images for this sample
        pattern = args.gen_glob.replace("{stem}", stem)
        gen_paths = sorted(glob.glob(str(gen_root / pattern)))
        if len(gen_paths) == 0:
            pbar.set_postfix_str(f"no gens for {stem}")
            continue

        # Reference image feature (cached)
        if stem in ref_img_feat_cache:
            ref_feat = ref_img_feat_cache[stem]
        else:
            ref_img_path = image_root / img_file
            if not ref_img_path.exists():
                pbar.set_postfix_str(f"missing ref {img_file}")
                continue
            ref_img = load_image(ref_img_path)
            ref_feat = get_image_features(model, processor, [ref_img], device, batch_size=args.batch_size)[0]
            ref_img_feat_cache[stem] = ref_feat

        # Text feature (cached)
        if caption in text_feat_cache:
            txt_feat = text_feat_cache[caption]
        else:
            txt_feat = get_text_features(model, processor, [caption], device, batch_size=max(8, args.batch_size))[0]
            text_feat_cache[caption] = txt_feat

        # Generated image features
        gen_imgs = [load_image(Path(p)) for p in gen_paths]
        gen_feats = get_image_features(model, processor, gen_imgs, device, batch_size=args.batch_size)

        # CLIP‑I: cos(gen, ref)
        ci = cosine_sim(gen_feats, ref_feat.unsqueeze(0)).cpu().tolist()
        # CLIP‑T: CLIPScore = 100 * max(cos(gen, txt), 0)
        ct_raw = cosine_sim(gen_feats, txt_feat.unsqueeze(0))
        ct = torch.clamp_min(ct_raw, 0.0) * 100.0
        ct = ct.cpu().tolist()

        # Aggregate per sample
        ci_mean = float(sum(ci) / len(ci))
        ct_mean = float(sum(ct) / len(ct))
        clip_i_vals.append(ci_mean)
        clip_t_vals.append(ct_mean)
        total_gen += len(gen_paths)
        used_samples += 1

        per_sample_rows.append(
            (
                stem,
                len(gen_paths),
                ci_mean,
                ct_mean,
                ";".join(f"{v:.5f}" for v in ci),
                ";".join(f"{v:.5f}" for v in ct),
            )
        )

    # Overall stats
    import math
    ci_overall = float(sum(clip_i_vals) / len(clip_i_vals)) if clip_i_vals else float("nan")
    ct_overall = float(sum(clip_t_vals) / len(clip_t_vals)) if clip_t_vals else float("nan")

    print("\n===== CLIP Evaluation Summary =====")
    print(f"Samples with generations: {used_samples} / {len(pairs)}")
    print(f"Total generated images:  {total_gen}")
    print(f"CLIP‑I mean:           {ci_overall:.5f}")
    print(f"CLIP‑T mean (score):   {ct_overall:.5f}")

    # Write CSV if requested
    if args.out_csv:
        header = [
            "stem",
            "num_gens",
            "clip_i_mean",
            "clip_t_mean",
            "clip_i_list",
            "clip_t_list",
        ]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in per_sample_rows:
                w.writerow(row)
        print(f"Saved per‑sample results to {args.out_csv}")


if __name__ == "__main__":
    main()
