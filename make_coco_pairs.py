# save as: make_coco_pairs.py
import json, os, random, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--coco_captions", required=True)
parser.add_argument("--out_json", required=True)
parser.add_argument("--pick", choices=["first","random","all"], default="first")
args = parser.parse_args()

with open(args.coco_captions, "r") as f:
    coco = json.load(f)

id2name = {img["id"]: img["file_name"] for img in coco["images"]}

# image_id -> [caption,...]
caps = {}
for ann in coco["annotations"]:
    caps.setdefault(ann["image_id"], []).append(ann["caption"].strip())

pairs = []
for img_id, fn in id2name.items():
    if img_id not in caps: 
        continue
    if args.pick == "all":
        for c in caps[img_id]:
            pairs.append({"image_file": fn, "text": c})
    elif args.pick == "random":
        pairs.append({"image_file": fn, "text": random.choice(caps[img_id])})
    else:  # first
        pairs.append({"image_file": fn, "text": caps[img_id][0]})

with open(args.out_json, "w") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"wrote {len(pairs)} pairs to {args.out_json}")