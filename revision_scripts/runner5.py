#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import esm


# =========================================================
# Style
# =========================================================
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(path_png: str, path_pdf: str):
    plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight", dpi=300)
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# FASTA / dataset helpers
# =========================================================
def norm_id(x: str) -> str:
    s = str(x).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            s = parts[1]
        else:
            s = parts[-1]
    return s.split(".")[0]


def normalize_seq(seq: str) -> str:
    seq = seq.strip().upper().replace("*", "")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    return "".join([c if c in allowed else "X" for c in seq])


def read_fasta_dict(path: str) -> Dict[str, str]:
    d = {}
    n = 0
    for rec in SeqIO.parse(path, "fasta"):
        n += 1
        rid = norm_id(rec.id)
        if rid not in d:
            d[rid] = normalize_seq(str(rec.seq))
    if n == 0:
        raise RuntimeError(f"FASTA empty/unreadable: {path}")
    if len(d) == 0:
        raise RuntimeError(f"No usable sequences parsed: {path}")
    return d


class CAZyDataset(Dataset):
    def __init__(self, fasta_path: str, labels_csv: str, cls2id: Dict[str, int], fam2id: Dict[str, int]):
        self.seqs = read_fasta_dict(fasta_path)
        df = pd.read_csv(labels_csv).copy()

        need = {"id", "class", "family"}
        if not need.issubset(df.columns):
            raise RuntimeError(f"{labels_csv} must contain columns {sorted(list(need))}")

        df["id"] = df["id"].astype(str).map(norm_id)
        df["class"] = df["class"].astype(str)
        df["family"] = df["family"].astype(str)

        df = df[df["id"].isin(self.seqs.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No overlap between FASTA and labels for {labels_csv}")

        df = df[df["class"].isin(cls2id.keys()) & df["family"].isin(fam2id.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("Dataset empty after mapping filter.")

        self.df = df
        self.cls2id = cls2id
        self.fam2id = fam2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rid = r["id"]
        return {
            "id": rid,
            "seq": self.seqs[rid],
            "y_class": self.cls2id[r["class"]],
            "y_family": self.fam2id[r["family"]],
        }


# =========================================================
# Collators
# =========================================================
class ESMCollator:
    def __init__(self, alphabet, max_len: int = 768):
        self.batch_converter = alphabet.get_batch_converter()
        self.max_len = max_len

    def __call__(self, batch):
        items, y_c, y_f = [], [], []
        for b in batch:
            s = b["seq"]
            if self.max_len and len(s) > self.max_len:
                s = s[: self.max_len]
            items.append((b["id"], s))
            y_c.append(b["y_class"])
            y_f.append(b["y_family"])
        _, _, tokens = self.batch_converter(items)
        pad_idx = 1
        mask = (tokens != pad_idx).long()
        return tokens, mask, torch.tensor(y_c, dtype=torch.long), torch.tensor(y_f, dtype=torch.long)


class CNNRawCollator:
    AA = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    AA2I = {a: i + 1 for i, a in enumerate(AA)}  # 0 PAD

    def __init__(self, max_len: int = 768):
        self.max_len = max_len

    def __call__(self, batch):
        y_c, y_f = [], []
        xs = []
        for b in batch:
            s = b["seq"]
            if self.max_len and len(s) > self.max_len:
                s = s[: self.max_len]
            ids = [self.AA2I.get(ch, self.AA2I["X"]) for ch in s]
            xs.append(ids)
            y_c.append(b["y_class"])
            y_f.append(b["y_family"])

        L = min(max(len(t) for t in xs), self.max_len) if self.max_len else max(len(t) for t in xs)
        X = torch.zeros(len(xs), L, dtype=torch.long)
        M = torch.zeros(len(xs), L, dtype=torch.long)
        for i, ids in enumerate(xs):
            ids = ids[:L]
            X[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            M[i, :len(ids)] = 1
        return X, M, torch.tensor(y_c, dtype=torch.long), torch.tensor(y_f, dtype=torch.long)


# =========================================================
# Models
# =========================================================
class MeanPool(nn.Module):
    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp_min(1.0)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (H * self.q.view(1, 1, -1)).sum(-1)
        logits = logits.masked_fill(~mask.bool(), -1e4)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (H * w).sum(dim=1)


class HiLoPool(nn.Module):
    """HiLo = attention pooled motif signal + mean pooled global context."""
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = AttnPool(d_model)
        self.mean = MeanPool()
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_hi = self.attn(H, mask)
        z_lo = self.mean(H, mask)
        return self.fuse(torch.cat([z_hi, z_lo], dim=-1))


class ESMBackbone(nn.Module):
    def __init__(self, esm_model, freeze: bool = True):
        super().__init__()
        self.esm = esm_model
        self.d = esm_model.embed_dim
        self.last_layer = int(getattr(esm_model, "num_layers"))
        self.set_freeze(freeze)

    def set_freeze(self, freeze: bool):
        for p in self.esm.parameters():
            p.requires_grad = (not freeze)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=[self.last_layer], return_contacts=False)
        return out["representations"][self.last_layer]


class Head(nn.Module):
    def __init__(self, d: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class ESMHierModel(nn.Module):
    def __init__(
        self,
        esm_model,
        n_cls: int,
        n_fam: int,
        pooling: str = "hilo",      # hilo | attn | mean
        hierarchical: bool = True,
        freeze_esm: bool = True,
    ):
        super().__init__()
        self.backbone = ESMBackbone(esm_model, freeze=freeze_esm)
        d = self.backbone.d
        if pooling == "hilo":
            self.pool = HiLoPool(d)
        elif pooling == "attn":
            self.pool = AttnPool(d)
        elif pooling == "mean":
            self.pool = MeanPool()
        else:
            raise ValueError(f"Unknown pooling={pooling}")

        self.pooling = pooling
        self.hierarchical = hierarchical
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_f = Head(d, n_fam)
        self.head_c = Head(d, n_cls) if hierarchical else None

    def set_freeze_esm(self, freeze: bool):
        self.backbone.set_freeze(freeze)

    def forward(self, tokens, mask):
        H = self.backbone(tokens)
        z = self.pool(H, mask)
        z = self.proj(z)
        logits_f = self.head_f(z)
        logits_c = self.head_c(z) if self.hierarchical else None
        return logits_c, logits_f, z


class CNNBaseline(nn.Module):
    def __init__(self, n_cls: int, n_fam: int, hierarchical: bool = True, emb: int = 64):
        super().__init__()
        self.hierarchical = hierarchical
        vocab = len(CNNRawCollator.AA) + 1
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        d = 128
        self.head_f = Head(d, n_fam)
        self.head_c = Head(d, n_cls) if hierarchical else None

    def forward(self, tokens, mask):
        x = self.emb(tokens).transpose(1, 2)
        z = self.conv(x).squeeze(-1)
        logits_f = self.head_f(z)
        logits_c = self.head_c(z) if self.hierarchical else None
        return logits_c, logits_f, z


# =========================================================
# Training / Eval
# =========================================================
@dataclass
class TrainCfg:
    batch_size: int = 16
    max_len: int = 768
    wd: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    grad_accum: int = 2
    lambda_class: float = 0.5
    family_reweight: bool = True


def _device_is_cuda(device: str) -> bool:
    return (device == "cuda") and torch.cuda.is_available()


def train_epoch(
    model,
    loader,
    opt,
    device,
    cfg: TrainCfg,
    scaler: GradScaler,
    hierarchical: bool,
    fam_ce_weight: Optional[torch.Tensor] = None,
):
    model.train()
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for step, (tokens, mask, y_c, y_f) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        tokens, mask = tokens.to(device), mask.to(device)
        y_c, y_f = y_c.to(device), y_f.to(device)

        with autocast("cuda", enabled=(cfg.amp and _device_is_cuda(device))):
            logits_c, logits_f, _ = model(tokens, mask)

            if cfg.family_reweight and fam_ce_weight is not None:
                loss_f = F.cross_entropy(logits_f, y_f, weight=fam_ce_weight)
            else:
                loss_f = F.cross_entropy(logits_f, y_f)
            loss = loss_f

            if hierarchical and (logits_c is not None):
                loss_c = F.cross_entropy(logits_c, y_c)
                loss = loss + cfg.lambda_class * loss_c

            loss = loss / float(cfg.grad_accum)

        if cfg.amp and _device_is_cuda(device):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % cfg.grad_accum == 0:
            if cfg.amp and _device_is_cuda(device):
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if cfg.amp and _device_is_cuda(device):
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        total += float(loss.detach().cpu()) * float(cfg.grad_accum)
        n += 1

    if (len(loader) % cfg.grad_accum) != 0:
        if cfg.amp and _device_is_cuda(device):
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.amp and _device_is_cuda(device):
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, hierarchical: bool):
    model.eval()
    ytf, ypf = [], []
    ytc, ypc = [], []

    for tokens, mask, y_c, y_f in tqdm(loader, desc="eval", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        logits_c, logits_f, _ = model(tokens, mask)

        pred_f = torch.argmax(logits_f, dim=1).cpu().tolist()
        ytf += y_f.tolist()
        ypf += pred_f

        if hierarchical and (logits_c is not None):
            pred_c = torch.argmax(logits_c, dim=1).cpu().tolist()
            ytc += y_c.tolist()
            ypc += pred_c

    out = {
        "macro_f1_family": float(f1_score(ytf, ypf, average="macro")),
        "balanced_acc_family": float(balanced_accuracy_score(ytf, ypf)),
        "y_f_true": ytf,
        "y_f_pred": ypf,
    }
    if hierarchical and len(ytc) > 0:
        out["macro_f1_class"] = float(f1_score(ytc, ypc, average="macro"))
        out["y_c_true"] = ytc
        out["y_c_pred"] = ypc
    return out


# =========================================================
# Reports / plots
# =========================================================
def plot_history(hist_df: pd.DataFrame, out_dir: str, title: str):
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(hist_df["epoch"], hist_df["macro_f1_family"], label="Macro-F1 (Family)")
    if "macro_f1_class" in hist_df.columns and hist_df["macro_f1_class"].notna().any():
        plt.plot(hist_df["epoch"], hist_df["macro_f1_class"], label="Macro-F1 (Class)")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(frameon=False)
    save_fig(os.path.join(out_dir, "training_curve.png"), os.path.join(out_dir, "training_curve.pdf"))


def normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float32)
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return cm / s


def plot_confusion(cm: np.ndarray, labels: List[str], title: str, out_png: str, out_pdf: str):
    n = len(labels)
    fig_w = min(24, max(8.0, 0.38 * n))
    fig_h = min(24, max(7.0, 0.38 * n))
    tick_fs = 10 if n <= 20 else 9 if n <= 35 else 8

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(cm, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(n), labels, rotation=90, fontsize=tick_fs)
    plt.yticks(range(n), labels, fontsize=tick_fs)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    save_fig(out_png, out_pdf)


def save_classification_report(y_true, y_pred, id2name, out_csv: str):
    labels = [id2name[i] for i in range(len(id2name))]
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    rows = []
    for k, v in rep.items():
        if isinstance(v, dict) and "f1-score" in v:
            rows.append({"label": k, **v})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def save_run_outputs(
    out_dir: str,
    metrics: Dict,
    id2cls: Dict[int, str],
    id2fam: Dict[int, str],
    cm_topn: int,
):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_classification_report(metrics["y_f_true"], metrics["y_f_pred"], id2fam, os.path.join(out_dir, "per_family_report.csv"))

    cm_f = confusion_matrix(metrics["y_f_true"], metrics["y_f_pred"], labels=list(range(len(id2fam))))
    pd.DataFrame(cm_f, index=[id2fam[i] for i in range(len(id2fam))], columns=[id2fam[i] for i in range(len(id2fam))])\
        .to_csv(os.path.join(out_dir, "confusion_family_full.csv"))
    cm_f_norm = normalize_cm(cm_f)

    y = np.array(metrics["y_f_true"])
    sup = np.bincount(y, minlength=len(id2fam))
    top = np.argsort(-sup)[:cm_topn]
    labels_top = [id2fam[i] for i in top]
    cm_top = cm_f_norm[np.ix_(top, top)]
    pd.DataFrame(cm_top, index=labels_top, columns=labels_top)\
        .to_csv(os.path.join(out_dir, f"confusion_family_top{cm_topn}.csv"))
    plot_confusion(
        cm_top,
        labels_top,
        f"Family confusion top{len(labels_top)} (row-normalized)",
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.png"),
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.pdf"),
    )

    if "y_c_true" in metrics and "y_c_pred" in metrics:
        save_classification_report(metrics["y_c_true"], metrics["y_c_pred"], id2cls, os.path.join(out_dir, "per_class_report.csv"))
        cm_c = confusion_matrix(metrics["y_c_true"], metrics["y_c_pred"], labels=list(range(len(id2cls))))
        pd.DataFrame(cm_c, index=[id2cls[i] for i in range(len(id2cls))], columns=[id2cls[i] for i in range(len(id2cls))])\
            .to_csv(os.path.join(out_dir, "confusion_class.csv"))
        plot_confusion(
            normalize_cm(cm_c),
            [id2cls[i] for i in range(len(id2cls))],
            "Class confusion (row-normalized)",
            os.path.join(out_dir, "confusion_class_norm.png"),
            os.path.join(out_dir, "confusion_class_norm.pdf"),
        )


# =========================================================
# Dataset stats
# =========================================================
def save_dataset_summary(train_ds: CAZyDataset, test_ds: CAZyDataset, out_dir: str):
    rows = []
    for split_name, ds in [("train", train_ds), ("test", test_ds)]:
        lens = [len(ds.seqs[rid]) for rid in ds.df["id"].tolist()]
        rows.append({
            "split": split_name,
            "n_sequences": len(ds),
            "n_classes": ds.df["class"].nunique(),
            "n_families": ds.df["family"].nunique(),
            "len_mean": float(np.mean(lens)),
            "len_std": float(np.std(lens)),
            "len_min": int(np.min(lens)),
            "len_median": float(np.median(lens)),
            "len_max": int(np.max(lens)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "dataset_summary.csv"), index=False)

    fam_train = train_ds.df["family"].value_counts().rename_axis("family").reset_index(name="train_count")
    fam_test = test_ds.df["family"].value_counts().rename_axis("family").reset_index(name="test_count")
    fam_dist = fam_train.merge(fam_test, on="family", how="outer").fillna(0)
    fam_dist["total"] = fam_dist["train_count"] + fam_dist["test_count"]
    fam_dist = fam_dist.sort_values("total", ascending=False).reset_index(drop=True)
    fam_dist.to_csv(os.path.join(out_dir, "family_distribution.csv"), index=False)

    topn = min(30, len(fam_dist))
    top_df = fam_dist.head(topn)
    plt.figure(figsize=(10, max(4.5, 0.28 * topn)))
    plt.barh(top_df["family"].tolist()[::-1], top_df["total"].tolist()[::-1])
    plt.xlabel("Count")
    plt.title(f"Top-{topn} family frequencies")
    save_fig(
        os.path.join(out_dir, "family_distribution_top30.png"),
        os.path.join(out_dir, "family_distribution_top30.pdf"),
    )


# =========================================================
# Experiment definitions
# =========================================================
def build_experiments():
    return [
        {"name": "Proposed_HiLo_Hier_InvSqrt", "key": "proposed"},
        {"name": "Abl_NoHiLo", "key": "ab_no_hilo"},
        {"name": "Abl_NoHier", "key": "ab_no_hier"},
        {"name": "Abl_NoFamWeight", "key": "ab_no_famw"},
        {"name": "Baseline_ESM2_MeanPool", "key": "base_mean"},
        {"name": "Baseline_CNN", "key": "base_cnn"},
    ]


def make_model(exp_key: str, esm_model, n_cls: int, n_fam: int, freeze_esm: bool):
    if exp_key == "proposed":
        return ESMHierModel(esm_model, n_cls, n_fam, pooling="hilo", hierarchical=True, freeze_esm=freeze_esm), True, "esm", True
    if exp_key == "ab_no_hilo":
        return ESMHierModel(esm_model, n_cls, n_fam, pooling="attn", hierarchical=True, freeze_esm=freeze_esm), True, "esm", True
    if exp_key == "ab_no_hier":
        return ESMHierModel(esm_model, n_cls, n_fam, pooling="hilo", hierarchical=False, freeze_esm=freeze_esm), False, "esm", True
    if exp_key == "ab_no_famw":
        return ESMHierModel(esm_model, n_cls, n_fam, pooling="hilo", hierarchical=True, freeze_esm=freeze_esm), True, "esm", False
    if exp_key == "base_mean":
        return ESMHierModel(esm_model, n_cls, n_fam, pooling="mean", hierarchical=True, freeze_esm=freeze_esm), True, "esm", True
    if exp_key == "base_cnn":
        return CNNBaseline(n_cls, n_fam, hierarchical=True), True, "cnn", True
    raise ValueError(exp_key)


def vram_safe_stage2(cfg: TrainCfg) -> TrainCfg:
    new_cfg = TrainCfg(**asdict(cfg))
    eff = int(max(1, new_cfg.batch_size * max(1, new_cfg.grad_accum)))

    if new_cfg.batch_size > 2:
        new_cfg.batch_size = 2
    elif new_cfg.batch_size > 1:
        new_cfg.batch_size = 1

    new_cfg.grad_accum = min(int(math.ceil(eff / float(new_cfg.batch_size))), 64)
    return new_cfg


# =========================================================
# Main
# =========================================================
def main():
    set_pub_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder")
    ap.add_argument("--out", default="ARCH_Q1_FINAL")
    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")

    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--lr_stage1", type=float, default=2e-4)
    ap.add_argument("--epochs_stage2", type=int, default=5)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--cm_topn", type=int, default=30)

    ap.add_argument("--lambda_class", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--only", default="", help="comma-separated experiment names to run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device}")

    cfg = TrainCfg(
        batch_size=args.batch_size,
        max_len=args.max_len,
        amp=(False if args.no_amp else True),
        grad_accum=max(1, args.grad_accum),
        lambda_class=args.lambda_class,
        family_reweight=True,
    )

    train_fa = os.path.join(args.data_dir, "train_homology.fasta")
    test_fa  = os.path.join(args.data_dir, "test_homology.fasta")
    train_csv = os.path.join(args.data_dir, "labels_train.csv")
    test_csv  = os.path.join(args.data_dir, "labels_test.csv")

    train_df = pd.read_csv(train_csv).copy()
    train_df["class"] = train_df["class"].astype(str)
    train_df["family"] = train_df["family"].astype(str)

    class_order = ["GH", "GT", "PL", "CE", "AA", "CBM"]
    present = [c for c in class_order if c in set(train_df["class"].unique())]
    cls2id = {c: i for i, c in enumerate(present)}
    id2cls = {i: c for c, i in cls2id.items()}

    fams = sorted(train_df["family"].unique().tolist())
    fam2id = {f: i for i, f in enumerate(fams)}
    id2fam = {i: f for f, i in fam2id.items()}

    train_ds = CAZyDataset(train_fa, train_csv, cls2id, fam2id)
    test_ds  = CAZyDataset(test_fa,  test_csv,  cls2id, fam2id)

    print(f"[DATA] classes={len(cls2id)} families={len(fam2id)}")
    print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id}, f, indent=2)

    save_dataset_summary(train_ds, test_ds, args.out)

    print(f"[LOAD] ESM2: {args.esm}")
    esm_model, alphabet = esm.pretrained.__dict__[args.esm]()
    esm_model = esm_model.to(device)

    scaler = GradScaler("cuda", enabled=(cfg.amp and _device_is_cuda(device)))

    exps = build_experiments()
    if args.only.strip():
        allow = set(x.strip() for x in args.only.split(",") if x.strip())
        exps = [e for e in exps if e["name"] in allow]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_rows = []

    for exp in exps:
        exp_name = exp["name"]
        exp_key = exp["key"]

        for seed in seeds:
            set_seed(seed)

            run_dir = os.path.join(args.out, exp_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            model, hierarchical, kind, use_family_weight = make_model(
                exp_key,
                esm_model,
                len(cls2id),
                len(fam2id),
                freeze_esm=True
            )
            model = model.to(device)

            fam_ce_weight = None
            if use_family_weight:
                fam_ids = train_ds.df["family"].map(fam2id).values
                counts = np.bincount(fam_ids, minlength=len(fam2id))
                w = np.zeros_like(counts, dtype=np.float32)
                nz = counts > 0
                w[nz] = 1.0 / np.sqrt(counts[nz].astype(np.float32))
                if nz.any():
                    w[nz] = w[nz] / (w[nz].mean() + 1e-8)
                fam_ce_weight = torch.tensor(w, dtype=torch.float32, device=device)

            run_cfg = TrainCfg(**asdict(cfg))

            if kind == "esm":
                collator = ESMCollator(alphabet, max_len=run_cfg.max_len)
            else:
                collator = CNNRawCollator(max_len=run_cfg.max_len)

            train_loader = DataLoader(train_ds, batch_size=run_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
            test_loader  = DataLoader(test_ds,  batch_size=run_cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

            history = []
            best_f1 = -1.0
            best_state = None

            # Stage-1
            print("\n==============================")
            print(f"[RUN] exp={exp_name} seed={seed}")
            print(f" Stage-1: epochs={args.epochs_stage1} lr={args.lr_stage1}")
            if kind == "esm":
                print(f" Stage-2: epochs={args.epochs_stage2} lr={args.lr_stage2}")
            print("==============================")

            opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_stage1, weight_decay=run_cfg.wd)

            for ep in range(1, args.epochs_stage1 + 1):
                loss = train_epoch(model, train_loader, opt1, device, run_cfg, scaler, hierarchical, fam_ce_weight=fam_ce_weight)
                ev = evaluate(model, test_loader, device, hierarchical)
                row = {
                    "epoch": ep,
                    "stage": "S1",
                    "loss": loss,
                    "macro_f1_family": ev["macro_f1_family"],
                    "macro_f1_class": ev.get("macro_f1_class", np.nan),
                }
                history.append(row)
                print(f"  [S1] ep={ep:02d} loss={loss:.4f} macroF1_fam={ev['macro_f1_family']:.4f}")

                if ev["macro_f1_family"] > best_f1:
                    best_f1 = ev["macro_f1_family"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            # Stage-2 for ESM only
            if kind == "esm" and args.epochs_stage2 > 0:
                if best_state is not None:
                    model.load_state_dict(best_state, strict=True)

                model.set_freeze_esm(False)
                gc.collect()
                if _device_is_cuda(device):
                    torch.cuda.empty_cache()

                run_cfg_s2 = vram_safe_stage2(run_cfg)
                collator2 = ESMCollator(alphabet, max_len=run_cfg_s2.max_len)
                train_loader2 = DataLoader(train_ds, batch_size=run_cfg_s2.batch_size, shuffle=True, num_workers=0, collate_fn=collator2)
                test_loader2  = DataLoader(test_ds,  batch_size=run_cfg_s2.batch_size, shuffle=False, num_workers=0, collate_fn=collator2)

                opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_stage2, weight_decay=run_cfg_s2.wd)

                for ep2 in range(1, args.epochs_stage2 + 1):
                    loss = train_epoch(model, train_loader2, opt2, device, run_cfg_s2, scaler, hierarchical, fam_ce_weight=fam_ce_weight)
                    ev = evaluate(model, test_loader2, device, hierarchical)
                    row = {
                        "epoch": args.epochs_stage1 + ep2,
                        "stage": "S2",
                        "loss": loss,
                        "macro_f1_family": ev["macro_f1_family"],
                        "macro_f1_class": ev.get("macro_f1_class", np.nan),
                    }
                    history.append(row)
                    print(f"  [S2] ep={ep2:02d} loss={loss:.4f} macroF1_fam={ev['macro_f1_family']:.4f}")

                    if ev["macro_f1_family"] > best_f1:
                        best_f1 = ev["macro_f1_family"]
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                final_loader = test_loader2
                final_cfg = run_cfg_s2
            else:
                final_loader = test_loader
                final_cfg = run_cfg

            # Final eval
            if best_state is not None:
                model.load_state_dict(best_state, strict=True)
            metrics = evaluate(model, final_loader, device, hierarchical)
            metrics.update({
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "lambda_class": final_cfg.lambda_class,
                "family_reweight": bool(use_family_weight),
            })

            save_run_outputs(run_dir, metrics, id2cls, id2fam, args.cm_topn)

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
            plot_history(hist_df, run_dir, f"{exp_name} (seed={seed})")

            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "metrics": metrics,
                "cfg": asdict(final_cfg),
                "cls2id": cls2id,
                "fam2id": fam2id,
            }, os.path.join(run_dir, "best.pt"))

            run_rows.append({
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "macro_f1_family": float(metrics["macro_f1_family"]),
                "balanced_acc_family": float(metrics["balanced_acc_family"]),
                "macro_f1_class": float(metrics.get("macro_f1_class", np.nan)),
                "lambda_class": float(final_cfg.lambda_class),
                "family_reweight": bool(use_family_weight),
            })

            del model
            gc.collect()
            if _device_is_cuda(device):
                torch.cuda.empty_cache()

    # Summary
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)

    fam_summ = runs_df.groupby("exp")["macro_f1_family"].agg(["mean", "std", "count"]).reset_index()
    fam_summ.columns = ["exp", "macro_f1_family_mean", "macro_f1_family_std", "n"]
    cls_summ = runs_df.groupby("exp")["macro_f1_class"].agg(["mean", "std"]).reset_index()
    cls_summ.columns = ["exp", "macro_f1_class_mean", "macro_f1_class_std"]
    summ = fam_summ.merge(cls_summ, on="exp", how="left")
    summ.to_csv(os.path.join(args.out, "summary_models.csv"), index=False)

    # Comparison plots
    order = summ.sort_values("macro_f1_family_mean", ascending=False)["exp"].tolist()
    xs = np.arange(len(order))
    vals = [summ[summ["exp"] == e]["macro_f1_family_mean"].values[0] for e in order]
    err = [summ[summ["exp"] == e]["macro_f1_family_std"].values[0] for e in order]

    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, vals, yerr=err, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (Family)")
    plt.title("Comparative Macro-F1 (Family): mean ± std across seeds")
    save_fig(
        os.path.join(args.out, "compare_macro_f1_family.png"),
        os.path.join(args.out, "compare_macro_f1_family.pdf"),
    )

    vals_c = [summ[summ["exp"] == e]["macro_f1_class_mean"].values[0] for e in order]
    err_c = [summ[summ["exp"] == e]["macro_f1_class_std"].values[0] for e in order]
    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, vals_c, yerr=err_c, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (Class)")
    plt.title("Comparative Macro-F1 (Class): mean ± std across seeds")
    save_fig(
        os.path.join(args.out, "compare_macro_f1_class.png"),
        os.path.join(args.out, "compare_macro_f1_class.pdf"),
    )

    print("\n✅ DONE")
    print("Output root:", args.out)
    print("- dataset_summary.csv / family_distribution.csv")
    print("- summary_runs.csv / summary_models.csv")
    print("- compare_macro_f1_family.(png|pdf)")
    print("- compare_macro_f1_class.(png|pdf)")
    print("- per-run outputs under out/EXP_NAME/seed_k/")


if __name__ == "__main__":
    main()
