import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from scipy.stats import pearsonr, spearmanr

from dataset_ava import AVACaptionsDataset

# 你现有的 Test.py 里必须包含：catNet, emd_loss, binary_accuracy
# 并且 Test.py 必须是 import-safe（不能在 import 时就去读 TestSet 文件）
import Test as amm


def safe_corr(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)

    if len(pred) < 2:
        return 0.0, 0.0
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0, 0.0

    plcc = pearsonr(pred, target)[0]
    srcc = spearmanr(pred, target)[0]

    if np.isnan(plcc):
        plcc = 0.0
    if np.isnan(srcc):
        srcc = 0.0

    return float(plcc), float(srcc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows 稳
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--freeze_clip", action="store_true")   # 加这个，便于做消融
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")

    model = amm.catNet(bert, freeze_clip=args.freeze_clip).to(device)
    start_epoch = 1

    train_ds = AVACaptionsDataset(args.train_csv, args.images_dir, tokenizer)
    val_ds = AVACaptionsDataset(args.val_csv, args.images_dir, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = amm.emd_loss(dist_r=1)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            print(f"Loaded checkpoint from {args.checkpoint}, resume at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded raw state_dict from {args.checkpoint}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already at epoch {start_epoch - 1}, nothing to train.")
        return

    os.makedirs("checkpoints", exist_ok=True)

    bins_tensor = torch.arange(1, 11, dtype=torch.float32, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for step, (image, text_ids, image_att, y) in enumerate(train_loader, 1):
            image = image.to(device, non_blocking=True)
            text_ids = text_ids.to(device, non_blocking=True)
            image_att = image_att.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(image, text_ids, image_att)
            loss = criterion(out, y) / args.accum_steps
            loss.backward()

            if step % args.accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * args.accum_steps
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step}: loss={running/50:.4f}")
                running = 0.0

        # =========================
        # Validation
        # =========================
        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        val_acc_batches = 0

        pred_means_all = []
        gt_means_all = []
        pred_dist_all = []
        gt_dist_all = []

        with torch.no_grad():
            for image, text_ids, image_att, y in val_loader:
                image = image.to(device, non_blocking=True)
                text_ids = text_ids.to(device, non_blocking=True)
                image_att = image_att.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                out = model(image, text_ids, image_att)

                batch_loss = criterion(out, y).item()
                val_loss += batch_loss

                batch_acc = amm.binary_accuracy(out, y, bins=10).item()
                val_acc_sum += batch_acc
                val_acc_batches += 1

                pred_mean = torch.sum(out * bins_tensor, dim=1)
                gt_mean = torch.sum(y * bins_tensor, dim=1)

                pred_means_all.extend(pred_mean.cpu().numpy().tolist())
                gt_means_all.extend(gt_mean.cpu().numpy().tolist())
                pred_dist_all.append(out.cpu().numpy())
                gt_dist_all.append(y.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        val_acc = val_acc_sum / max(1, val_acc_batches)
        val_plcc, val_srcc = safe_corr(pred_means_all, gt_means_all)

        print(
            f"Epoch {epoch}: "
            f"val_emd={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_plcc={val_plcc:.4f} | "
            f"val_srcc={val_srcc:.4f}"
        )

        ckpt_path = f"checkpoints/ammnet_clipattr_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_emd": val_loss,
                "val_acc": val_acc,
                "val_plcc": val_plcc,
                "val_srcc": val_srcc,
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()