#!/usr/bin/env python3
"""
Generate images from a trained IP‑Adapter (saved via accelerate.save_state)
using the same architecture as in tutorial_train.py, then save PNGs per sample.

This script loads:
- Base SD1.5 components (tokenizer/text_encoder/vae/unet/scheduler)
- CLIP image encoder (frozen)
- IP‑Adapter modules (image_proj_model + IPAttnProcessors) and then
  loads weights from an Accelerate checkpoint directory containing model.safetensors.

Outputs:
- For each record in pairs JSON, generate N images and save as
  {stem}_{k}.png so that eval_clip_scores.py can pick them up.

Example
  python gen_ipadapter_from_ckpt.py \
    --pretrained_model runwayml/stable-diffusion-v1-5 \
    --image_encoder_path /home/gpuadmin/MB/IP-Adapter-main/ip_adapter/models/image_encoder \
    --checkpoint_dir /home/gpuadmin/MB/IP-Adapter-main/output_dir_layerNorm/checkpoint-20 \
    --pairs_json /data1/coco2017/COCO2017/val_pairs.json \
    --image_root /data1/coco2017/COCO2017/val2017 \
    --out_dir /home/gpuadmin/MB/ipadapter_gen/ckpt20 \
    --num_images_per_sample 4 --steps 30 --seed 42 --fp16
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from safetensors.torch import load_file

# import IP‑Adapter bits exactly like training
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class IPAdapterModule(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

    def forward(self, latents, t, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        enc = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return self.unet(latents, t, enc).sample


def build_ipadapter(pretrained_model: str, image_encoder_path: str, device: torch.device, dtype: torch.dtype, resolution: int):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device, dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path,use_safetensors=True).to(device, dtype)
    clip_proc = CLIPImageProcessor()

    # init IP‑Adapter attn processors (same as training)
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            m = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            m.load_state_dict(weights)
            attn_procs[name] = m
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    ip_adapter = IPAdapterModule(unet, image_proj_model, adapter_modules)

    for m in [text_encoder, vae, image_encoder, ip_adapter]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    vae.enable_slicing()

    return tokenizer, text_encoder, vae, unet, image_encoder, clip_proc, ip_adapter


@torch.no_grad()
def sample_images(tokenizer, text_encoder, vae, scheduler, ip_adapter, image_encoder, clip_proc,
                  records, image_root: Path, out_dir: Path, device, dtype, steps: int, num_images_per_sample: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    for rec in tqdm(records, desc="Generating"):
        img_path = image_root / rec["image_file"]
        if not img_path.exists():
            continue
        raw = Image.open(img_path).convert("RGB")
        clip_img = clip_proc(images=raw, return_tensors="pt").pixel_values.to(device, dtype)
        image_embeds = image_encoder(clip_img).image_embeds

        text_ids = tokenizer(rec.get("text", ""), max_length=tokenizer.model_max_length,
                             padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        enc = text_encoder(text_ids)[0].to(dtype)

        # Prepare scheduler
        scheduler.set_timesteps(steps)
        H = W = 512
        latents = torch.randn((1, vae.config.latent_channels, H // 8, W // 8), generator=gen, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        for k in range(num_images_per_sample):
            z = latents.clone()
            for t in scheduler.timesteps:
                eps = ip_adapter(z, t, enc, image_embeds)
                step_out = scheduler.step(eps, t, z)
                z = step_out.prev_sample
            # decode
            x = (z / vae.config.scaling_factor)
            img = vae.decode(x).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img.mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()[0]
            Image.fromarray(img).save(out_dir / f"{Path(rec['image_file']).stem}_{k}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model", required=True)
    ap.add_argument("--image_encoder_path", required=True)
    ap.add_argument("--checkpoint_dir", required=True, help="Accelerate checkpoint dir (contains model.safetensors)")
    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images_per_sample", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    tokenizer, text_encoder, vae, unet, image_encoder, clip_proc, ip_adapter = build_ipadapter(
        args.pretrained_model, args.image_encoder_path, device, dtype, 512
    )

    # prepare accelerator and load checkpoint
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    # # ip_adapter = accelerator.prepare(ip_adapter)
    # # ...모델 빌드 후(토크나이저/텍스트/vae/unet/image_encoder/ip_adapter 생성 직후)...
    # # 2) dtype은 기존처럼, 이동은 accelerator.prepare로 한 번에
    # ip_adapter, image_encoder, text_encoder, vae = accelerator.prepare(
    #     ip_adapter, image_encoder, text_encoder, vae
    # )
    # fp16이면 UNet/IP-Adapter 쪽도 같은 dtype으로 맞추기
    for m in [unet, ip_adapter.image_proj_model, ip_adapter.adapter_modules]:
        m.to(device, dtype)

    # Accelerate 준비
    ip_adapter, image_encoder, text_encoder, vae = accelerator.prepare(
        ip_adapter, image_encoder, text_encoder, vae
    )

    # accelerator.load_state(args.checkpoint_dir)
    ckpt_file = os.path.join(args.checkpoint_dir, "model.safetensors")
    state = load_file(ckpt_file)  # safetensors 전용 로더
    accelerator.unwrap_model(ip_adapter).load_state_dict(state, strict=False)
    accelerator.wait_for_everyone()

    # records
    # records = json.load(open(args.pairs_json))
    records_all = json.load(open(args.pairs_json))
    rank = accelerator.process_index
    world = accelerator.num_processes
    records = [rec for i, rec in enumerate(records_all) if i % world == rank]
    if accelerator.is_main_process:
        print(f"[world={world}] total={len(records_all)}  per-rank≈{len(records)}")

    # scheduler for inference (reuse DDPM schedule)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    sample_images(tokenizer, text_encoder, vae, scheduler, ip_adapter, image_encoder, clip_proc,
                  records, Path(args.image_root), Path(args.out_dir), device, dtype, args.steps, args.num_images_per_sample, args.seed)


if __name__ == "__main__":
    main()
