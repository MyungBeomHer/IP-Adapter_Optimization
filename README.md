# IP-Adapter_Optimization
## Diffusion based image generator using image and text

팀원 : [허명범](https://github.com/MyungBeomHer)

### 프로젝트 주제 
이미지와 텍스트를 이용한 이미지 생성기 

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Dataset 
```bash
wget http://images.cocodataset.org/zips/train2017.zip   # train dataset
wget http://images.cocodataset.org/zips/val2017.zip     # validation dataset
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## ➡️ Data Preparation
```bash
python make_coco_pairs.py \
  --coco_captions /data1/coco2017/COCO2017/annotations/captions_train2017.json \
  --out_json /data1/coco2017/COCO2017/train_pairs.json \
  --pick random

python make_coco_pairs.py \
  --coco_captions /data1/coco2017/COCO2017/annotations/captions_val2017.json \
  --out_json /data1/coco2017/COCO2017/val_pairs.json \
  --pick random
```

### Requirements
```bash
pip install -r requirements.txt
```

### train
```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision "fp16" \
  tutorial_train.py \
  --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" \
  --image_encoder_path="/home/gpuadmin/MB/IP-Adapter-main/ip_adapter/models/image_encoder" \
  --data_json_file="/data1/coco2017/COCO2017/train_pairs.json" \
  --data_root_path="/data1/coco2017/COCO2017/train2017" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="output_dir_layerNorm" \
  --save_steps=2000
```

### Image generate 
```bash
  ### img generate by Tuning Img encoder 
  #1st output_dir_layerNorm/model_1.safetensors -move/change-> image_encoder/model.safetensors
  #2nd image_encoder/config.json -move-> image_encoder-20/config.json
accelerate launch --num_processes 4 --multi_gpu \
  gen_ipadapter_from_ckpt.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --image_encoder_path /home/gpuadmin/MB/IP-Adapter-main/output_dir_layerNorm/image_encoder-20 \
  --checkpoint_dir /home/gpuadmin/MB/IP-Adapter-main/output_dir_layerNorm/checkpoint-20 \
  --pairs_json /data1/coco2017/COCO2017/val_pairs.json \
  --image_root /data1/coco2017/COCO2017/val2017 \
  --out_dir /home/gpuadmin/MB/ipadapter_gen_tuning_LN/ckpt20 \
  --num_images_per_sample 4 \
  --steps 18 \
  --fp16
```


### preprocessing for tuning LayerNorm
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
    image_encoder.requires_grad_(False)
    for name, param in image_encoder.named_parameters():
        if 'norm' in name:
            param.requires_grad = True
    ...
    # optimizer
    # params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    #---Tuning for MB---#
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters(), image_encoder_layerNorm)
    ...
    # Prepare everything with our `accelerator`.
    # ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    ip_adapter, image_encoder, optimizer, train_dataloader = accelerator.prepare(
                                                                        ip_adapter, image_encoder, optimizer, train_dataloader
                                                                        )
    ...
    # with torch.no_grad():
    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
```
[tutorial_train.py](tutorial_train.py)

- Benchmark (COCO2017val)
For fair Comparison, we train the two cases in COCO2017train dataset and then evaluate COCO2017val dataset.
|Model|Clip Score|
|IP-Adapter|???||
|IP-Adapter tuning LayerNorm|???||

### Reference Repo
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file)
- [download COCO2017dataset](https://iambeginnerdeveloper.tistory.com/207)
- [Parameter-Efficient Tuning on Layer Normalization for Pre-trained Language Models](https://arxiv.org/abs/2211.08682)
