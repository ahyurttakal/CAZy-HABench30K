#!/usr/bin/env python3
import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

import esm


# ---------------------------
# utils
# ---------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_id(x: str) -> str:
    s = str(x).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            s = parts[1]
        else:
            s = parts[-1]
    s = s.split(".")[0]
    return s


def normalize_seq(seq: str) -> str:
    seq = seq.strip().upper().replace("*", "")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    return "".join([c if c in allowed else "X" for c in seq])


def read_fasta_dict(fasta_path: str) -> Dict[str, str]:
    d = {}
    n = 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        n += 1
        rid = norm_id(rec.id)
        if rid not in d:
            d[rid] = normalize_seq(str(rec.seq))
    if n == 0:
        raise RuntimeError(f"FASTA empty or unreadable: {fasta_path}")
    if len(d) == 0:
        raise RuntimeError(f"No usable sequences parsed: {fasta_path}")
    return d


def save_confusion_csv(cm, labels: List[str], out_csv: str):
    df = pd.DataFrame(cm, index=[f"true:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])
    df.to_csv(out_csv)


def plot_confusion(cm, labels: List[str], out_png: str, title: str, normalize: bool = True):
    cm = cm.astype("float")
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    n = len(labels)
    fig_w = min(24, max(8, 0.35 * n))
    fig_h = min(24, max(6, 0.35 * n))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title + (" (norm)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(n), labels, rotation=90, fontsize=8)
    plt.yticks(range(n), labels, fontsize=8)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# dataset
# ---------------------------
class CAZyDataset(Dataset):
    def __init__(self, fasta_path: str, labels_csv: str, cls2id=None, fam2id=None):
        self.seqs = read_fasta_dict(fasta_path)
        df = pd.read_csv(labels_csv)

        needed = {"id", "class", "family"}
        if not needed.issubset(df.columns):
            raise RuntimeError(f"{labels_csv} must contain columns {sorted(list(needed))}")

        df = df.copy()
        df["id"] = df["id"].astype(str).map(norm_id)
        df["class"] = df["class"].astype(str)
        df["family"] = df["family"].astype(str)

        # keep only ids present in fasta
        df = df[df["id"].isin(self.seqs.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(
                f"No overlap between {os.path.basename(fasta_path)} and {os.path.basename(labels_csv)} after ID normalization.\n"
                f"FASTA example: {list(self.seqs.keys())[:5]}\n"
                f"CSV example: {pd.read_csv(labels_csv)['id'].head(5).tolist()}"
            )

        if cls2id is None:
            # keep stable order
            class_order = ["GH", "GT", "PL", "CE", "AA", "CBM"]
            present = [c for c in class_order if c in set(df["class"])]
            if not present:
                present = sorted(df["class"].unique().tolist())
            cls2id = {c: i for i, c in enumerate(present)}

        if fam2id is None:
            fams = sorted(df["family"].unique().tolist())
            fam2id = {f: i for i, f in enumerate(fams)}

        # final filter
        df = df[df["class"].isin(cls2id.keys()) & df["family"].isin(fam2id.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("After mapping filter, dataset became empty. Check class/family values.")

        self.df = df
        self.cls2id = cls2id
        self.fam2id = fam2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        rid = r["id"]
        return {
            "id": rid,
            "seq": self.seqs[rid],
            "y_class": self.cls2id[r["class"]],
            "y_family": self.fam2id[r["family"]],
        }


class ESMCollator:
    def __init__(self, alphabet, max_len: int = 1024):
        self.batch_converter = alphabet.get_batch_converter()
        self.max_len = max_len

    def __call__(self, batch):
        items = []
        y_class = []
        y_family = []
        for b in batch:
            # truncate (token count includes special tokens; esm handles)
            seq = b["seq"]
            if self.max_len and len(seq) > self.max_len:
                seq = seq[: self.max_len]
            items.append((b["id"], seq))
            y_class.append(b["y_class"])
            y_family.append(b["y_family"])

        _, _, tokens = self.batch_converter(items)
        pad_idx = 1
        mask = (tokens != pad_idx).long()
        return tokens, mask, torch.tensor(y_class, dtype=torch.long), torch.tensor(y_family, dtype=torch.long)


# ---------------------------
# model
# ---------------------------
class MultiScaleAdapter(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, kernels=(3, 7, 15), dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, hidden, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden * len(kernels), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # H: [B,L,D], mask: [B,L]
        x = H.transpose(1, 2)  # [B,D,L]
        feats = [F.gelu(conv(x)) for conv in self.convs]  # each [B,hidden,L]
        y = torch.cat(feats, dim=1).transpose(1, 2)      # [B,L,hidden*k]
        y = self.proj(y)                                 # [B,L,D]
        y = y * mask.unsqueeze(-1).float()
        return self.norm(H + y)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (H * self.q.view(1, 1, -1)).sum(-1)     # [B,L]
        logits = logits.masked_fill(~mask.bool(), -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)   # [B,L,1]
        return (H * w).sum(dim=1)                        # [B,D]


class ESM2Hier(nn.Module):
    def __init__(self, esm_model, n_classes: int, n_families: int, freeze_esm: bool = True):
        super().__init__()
        self.esm = esm_model
        self.d_model = esm_model.embed_dim

        if freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False

        self.adapter = MultiScaleAdapter(self.d_model)
        self.pool = AttnPool(self.d_model)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.class_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, n_classes),
        )
        self.family_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, n_families),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        out = self.esm(tokens, repr_layers=[self.esm.num_layers], return_contacts=False)
        H = out["representations"][self.esm.num_layers]  # [B,L,D]
        H = self.adapter(H, mask)
        z = self.pool(H, mask)                            # [B,D]
        zc = self.proj(z)
        return self.class_head(z), self.family_head(z), zc


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    sim = (z @ z.t()) / temperature
    logits_mask = ~torch.eye(B, device=z.device).bool()
    sim = sim.masked_fill(~logits_mask, -1e9)

    y = y.view(-1, 1)
    pos = (y == y.t()) & logits_mask
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    num = (exp_sim * pos.float()).sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss = -torch.log(num / denom)

    has_pos = pos.sum(dim=1) > 0
    if has_pos.any():
        return loss[has_pos].mean()
    return loss.mean()


# ---------------------------
# train/eval
# ---------------------------
@dataclass
class TrainCfg:
    epochs: int = 10
    batch_size: int = 8
    lr: float = 2e-4
    wd: float = 0.01
    lambda_family: float = 1.0
    lambda_contrast: float = 0.2
    temperature: float = 0.07
    grad_clip: float = 1.0
    seed: int = 42
    freeze_esm: bool = True
    max_len: int = 1024
    cm_topn: int = 30


def train_one_epoch(model, loader, optim, cfg: TrainCfg, device):
    model.train()
    total = 0.0
    n = 0
    for tokens, mask, y_c, y_f in tqdm(loader, desc="train", leave=False):
        tokens = tokens.to(device)
        mask = mask.to(device)
        y_c = y_c.to(device)
        y_f = y_f.to(device)

        optim.zero_grad(set_to_none=True)
        logits_c, logits_f, z = model(tokens, mask)
        lc = F.cross_entropy(logits_c, y_c)
        lf = F.cross_entropy(logits_f, y_f)
        lcon = supcon_loss(z, y_f, temperature=cfg.temperature)

        loss = lc + cfg.lambda_family * lf + cfg.lambda_contrast * lcon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ytf, ypf = [], []
    ytc, ypc = [], []

    for tokens, mask, y_c, y_f in tqdm(loader, desc="eval", leave=False):
        tokens = tokens.to(device)
        mask = mask.to(device)
        logits_c, logits_f, _ = model(tokens, mask)
        pred_c = torch.argmax(logits_c, dim=1).cpu().tolist()
        pred_f = torch.argmax(logits_f, dim=1).cpu().tolist()

        ytc += y_c.tolist()
        ypc += pred_c
        ytf += y_f.tolist()
        ypf += pred_f

    return ytc, ypc, ytf, ypf


def topn_by_support(y_true: List[int], n: int) -> List[int]:
    s = pd.Series(y_true).value_counts()
    return [int(x) for x in s.index.tolist()[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder path")
    ap.add_argument("--out", default="run_q1_single", help="output folder")
    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no_freeze_esm", action="store_true")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--cm_topn", type=int, default=30)
    args = ap.parse_args()

    cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_esm=(not args.no_freeze_esm),
        max_len=args.max_len,
        cm_topn=args.cm_topn,
    )
    set_seed(cfg.seed)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_fa = os.path.join(args.data_dir, "train_homology.fasta")
    test_fa = os.path.join(args.data_dir, "test_homology.fasta")
    train_csv = os.path.join(args.data_dir, "labels_train.csv")
    test_csv = os.path.join(args.data_dir, "labels_test.csv")

    # Build mappings from train
    train_df = pd.read_csv(train_csv)
    train_df["class"] = train_df["class"].astype(str)
    train_df["family"] = train_df["family"].astype(str)

    class_order = ["GH", "GT", "PL", "CE", "AA", "CBM"]
    present = [c for c in class_order if c in set(train_df["class"])]
    if not present:
        present = sorted(train_df["class"].unique().tolist())
    cls2id = {c: i for i, c in enumerate(present)}
    fams = sorted(train_df["family"].unique().tolist())
    fam2id = {f: i for i, f in enumerate(fams)}
    id2cls = {v: k for k, v in cls2id.items()}
    id2fam = {v: k for k, v in fam2id.items()}

    print(f"[A] classes={len(cls2id)} families={len(fam2id)}")
    print(f"[A] device={device} freeze_esm={cfg.freeze_esm} max_len={cfg.max_len}")

    # Load ESM
    print(f"[B] Loading ESM2: {args.esm}")
    esm_model, alphabet = esm.pretrained.__dict__[args.esm]()
    collator = ESMCollator(alphabet, max_len=cfg.max_len)

    # Datasets/loaders
    train_ds = CAZyDataset(train_fa, train_csv, cls2id=cls2id, fam2id=fam2id)
    test_ds = CAZyDataset(test_fa, test_csv, cls2id=cls2id, fam2id=fam2id)
    print(f"[C] train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    model = ESM2Hier(esm_model, n_classes=len(cls2id), n_families=len(fam2id), freeze_esm=cfg.freeze_esm).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.wd)

    # Save mappings
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id, "esm": args.esm, "max_len": cfg.max_len}, f, indent=2)

    best = -1.0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, cfg, device)
        ytc, ypc, ytf, ypf = evaluate(model, test_loader, device)

        macro_f1_fam = f1_score(ytf, ypf, average="macro")
        macro_f1_cls = f1_score(ytc, ypc, average="macro")
        bal_acc_fam = balanced_accuracy_score(ytf, ypf)

        metrics = {
            "epoch": epoch,
            "loss": float(loss),
            "macro_f1_family": float(macro_f1_fam),
            "macro_f1_class": float(macro_f1_cls),
            "balanced_acc_family": float(bal_acc_fam),
            "n_test": len(ytf),
        }
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | F1_fam={macro_f1_fam:.4f} | F1_cls={macro_f1_cls:.4f} | bal_acc_fam={bal_acc_fam:.4f}")

        # reports each epoch (overwrite)
        fam_labels = list(range(len(id2fam)))
        fam_names = [id2fam[i] for i in fam_labels]
        cls_labels = list(range(len(id2cls)))
        cls_names = [id2cls[i] for i in cls_labels]

        rep_f = classification_report(ytf, ypf, labels=fam_labels, target_names=fam_names, output_dict=True, zero_division=0)
        rep_c = classification_report(ytc, ypc, labels=cls_labels, target_names=cls_names, output_dict=True, zero_division=0)
        pd.DataFrame(rep_f).transpose().to_csv(os.path.join(args.out, "per_family_report.csv"))
        pd.DataFrame(rep_c).transpose().to_csv(os.path.join(args.out, "per_class_report.csv"))

        # confusion matrices
        cm_f = confusion_matrix(ytf, ypf, labels=fam_labels)
        cm_c = confusion_matrix(ytc, ypc, labels=cls_labels)

        save_confusion_csv(cm_c, cls_names, os.path.join(args.out, "confusion_class.csv"))
        plot_confusion(cm_c, cls_names, os.path.join(args.out, "confusion_class_norm.png"), "Class confusion", normalize=True)

        save_confusion_csv(cm_f, fam_names, os.path.join(args.out, "confusion_family_full.csv"))

        keep_ids = topn_by_support(ytf, cfg.cm_topn)
        keep_names = [id2fam[i] for i in keep_ids]
        cm_f_top = confusion_matrix(ytf, ypf, labels=keep_ids)
        save_confusion_csv(cm_f_top, keep_names, os.path.join(args.out, f"confusion_family_top{cfg.cm_topn}.csv"))
        plot_confusion(cm_f_top, keep_names, os.path.join(args.out, f"confusion_family_top{cfg.cm_topn}_norm.png"),
                       f"Family confusion top{cfg.cm_topn}", normalize=True)

        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # save best
        if macro_f1_fam > best:
            best = macro_f1_fam
            ckpt = {
                "state_dict": model.state_dict(),
                "cls2id": cls2id,
                "fam2id": fam2id,
                "esm": args.esm,
                "cfg": cfg.__dict__,
                "best_macro_f1_family": float(best),
            }
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ✅ saved best.pt (best macro_f1_family={best:.4f})")

    print("\nDONE. Best macro_f1_family =", best)
    print("Outputs in:", args.out)
    print("- best.pt / metrics.json / per_family_report.csv / confusion_family_full.csv / confusion_class_norm.png")


if __name__ == "__main__":
    main()
