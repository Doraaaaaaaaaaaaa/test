import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_transforms():
    # Swin / main branch (448 + ImageNet normalize)
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    transform_main = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    # CLIP attribute branch (224 + CLIP normalize)
    clip_norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    transform_clip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        clip_norm,
    ])
    return transform_main, transform_clip

class AVACaptionsDataset(Dataset):
    """
    CSV columns expected:
      image_id, comment, score2..score11
    """
    def __init__(self, csv_path: str, images_dir: str, tokenizer, max_len: int = 200):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform_main, self.transform_clip = build_transforms()

        self.score_cols = [f"score{i}" for i in range(2, 12)]
        need = ["image_id", "comment", *self.score_cols]
        for c in need:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["image_id"])
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            if (not image_id.lower().endswith(".jpg")) and os.path.exists(img_path + ".jpg"):
                img_path = img_path + ".jpg"
            else:
                raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        image = self.transform_main(img)
        image_att = self.transform_clip(img)

        comment = str(row["comment"])
        enc = self.tokenizer(
            comment,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        text_ids = enc["input_ids"].squeeze(0)          # (200,)
        # 如果你后面愿意改模型支持 mask，可以顺带返回 attention_mask
        # text_mask = enc["attention_mask"].squeeze(0)

        scores = torch.tensor([float(row[c]) for c in self.score_cols], dtype=torch.float32)
        y = scores / (scores.sum() + 1e-8)              # (10,) distribution

        return image, text_ids, image_att, y
