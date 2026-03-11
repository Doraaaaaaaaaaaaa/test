"""
★ 预计算 CLIP 特征 — 最大的单项加速优化
由于 CLIP 是冻结的(freeze_clip=True)，每张图的 CLIP 特征是固定的。
预计算一次保存到磁盘，训练时直接读取，跳过 CLIP 前向传播。

用法:
  python precompute_clip.py \
    --csv /path/to/train.csv \
    --images_dir /path/to/images \
    --output_dir /root/autodl-tmp/clip_features \
    --batch_size 64

训练时使用:
  在 dataset_ava.py 中加载预计算特征 (见 AVACaptionsDatasetFast)
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd
import numpy as np
import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        clip_norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            clip_norm,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx]["image_id"])
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            if not image_id.lower().endswith(".jpg") and os.path.exists(img_path + ".jpg"):
                img_path = img_path + ".jpg"
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), image_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_name", type=str, default="ViT-B/16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(args.clip_name, device=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = ImageOnlyDataset(args.csv, args.images_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    all_features = {}
    with torch.no_grad():
        for i, (imgs, ids) in enumerate(loader):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=-1)
            for j, img_id in enumerate(ids):
                all_features[img_id] = feats[j].cpu()
            if (i + 1) % 100 == 0:
                print(f"Processed {(i+1) * args.batch_size} images...")

    save_path = os.path.join(args.output_dir, "clip_features.pt")
    torch.save(all_features, save_path)
    print(f"Saved {len(all_features)} CLIP features to {save_path}")


if __name__ == "__main__":
    main()
