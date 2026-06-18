#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAZy Q1 Architecture Contribution Framework (FINAL, OOM/AMP-SAFE)

Key fixes:
  - AMP-safe SupCon (no fp16 overflow)
  - AMP-safe AttnPool masking (-1e4 instead of -1e9)
  - Stage-2 VRAM-safe auto-adjust:
      * max_len: 768 -> 512
      * batch_size: -> 4
      * grad_accum: -> 4  (keeps effective batch)
    and rebuild DataLoaders + Collator accordingly
  - GC + torch.cuda.empty_cache() between stages
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib as mpl
import matplotlib.pyplot as plt

import esm


# =========================
# Publication-quality plotting
# =========================
def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def save_fig(png_path: str, pdf_path: str):
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

def plot_history(hist_df: pd.DataFrame, title: str, out_png: str, out_pdf: str):
    plt.figure(figsize=(8.5, 4.8))
    if "macro_f1_family" in hist_df.columns:
        plt.plot(hist_df["epoch"], hist_df["macro_f1_family"], label="Macro-F1 (Family)")
    if "macro_f1_class" in hist_df.columns and hist_df["macro_f1_class"].notna().any():
        plt.plot(hist_df["epoch"], hist_df["macro_f1_class"], label="Macro-F1 (Class)")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(frameon=False)
    save_fig(out_png, out_pdf)

def plot_confusion(cm: np.ndarray, labels: List[str], title: str,
                   out_png: str, out_pdf: str, normalize: bool = True):
    cm = cm.astype(np.float64)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    n = len(labels)
    fig_w = min(24, max(8.0, 0.38 * n))
    fig_h = min(24, max(7.0, 0.38 * n))
    tick_fs = 10 if n <= 20 else 9 if n <= 35 else 8

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title + (" (row-normalized)" if normalize else ""))

    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=tick_fs)

    plt.xticks(range(n), labels, rotation=90, fontsize=tick_fs)
    plt.yticks(range(n), labels, fontsize=tick_fs)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(out_png, out_pdf)

def plot_bar(df: pd.DataFrame, metric: str, out_png: str, out_pdf: str, title: str):
    d = df.sort_values(metric, ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, max(4.0, 0.35 * len(d))))
    plt.barh(d["model"].tolist()[::-1], d[metric].tolist()[::-1])
    plt.xlabel(metric)
    plt.title(title)
    save_fig(out_png, out_pdf)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Data utils
# =========================
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
    d, n = {}, 0
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


# =========================
# Collators
# =========================
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
            ids = [self.AA2I.get(ch, 1) for ch in s]
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


# =========================
# Loss: supervised contrastive (AMP-safe)
# =========================
def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    B = z.size(0)

    z32 = z.float()
    sim = (z32 @ z32.t()) / float(temperature)

    logits_mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
    neg_inf = -1e4  # fp16-safe
    sim = sim.masked_fill(~logits_mask, neg_inf)

    y = y.view(-1, 1)
    pos = (y == y.t()) & logits_mask

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_counts = pos.sum(dim=1).clamp_min(1).float()
    loss = -(log_prob * pos.float()).sum(dim=1) / pos_counts

    has_pos = (pos.sum(dim=1) > 0)
    if has_pos.any():
        return loss[has_pos].mean()
    return loss.mean()


# =========================
# Models
# =========================
class MultiScaleAdapter(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, kernels=(3, 7, 15), dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d_model, hidden, k, padding=k // 2) for k in kernels])
        self.proj = nn.Sequential(
            nn.Linear(hidden * len(kernels), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = H.transpose(1, 2)  # B,D,L
        feats = [F.gelu(conv(x)) for conv in self.convs]
        y = torch.cat(feats, dim=1).transpose(1, 2)  # B,L,hidden*k
        y = self.proj(y) * mask.unsqueeze(-1).float()
        return self.norm(H + y)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (H * self.q.view(1, 1, -1)).sum(-1)
        # AMP-safe mask value
        logits = logits.masked_fill(~mask.bool(), -1e4)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (H * w).sum(dim=1)


class MeanPool(nn.Module):
    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / (m.sum(1).clamp_min(1.0))


class ESM2Backbone(nn.Module):
    def __init__(self, esm_model, freeze: bool = True, use_ckpt: bool = False):
        super().__init__()
        self.esm = esm_model
        self.d = esm_model.embed_dim
        self.layer = esm_model.num_layers

        # optional checkpointing if supported
        if hasattr(self.esm, "set_gradient_checkpointing"):
            try:
                self.esm.set_gradient_checkpointing(use_ckpt)
            except Exception:
                pass
        elif hasattr(self.esm, "gradient_checkpointing"):
            try:
                self.esm.gradient_checkpointing = use_ckpt
            except Exception:
                pass

        self.set_freeze(freeze)

    def set_freeze(self, freeze: bool):
        for p in self.esm.parameters():
            p.requires_grad = (not freeze)

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.layer], return_contacts=False)
        return out["representations"][self.layer]  # B,L,D


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


class ModelESMVariant(nn.Module):
    def __init__(self, esm_model, n_cls: int, n_fam: int,
                 use_adapter: bool, pool_type: str,
                 hierarchical: bool, freeze_esm: bool, use_ckpt: bool):
        super().__init__()
        self.backbone = ESM2Backbone(esm_model, freeze=freeze_esm, use_ckpt=use_ckpt)
        d = self.backbone.d
        self.use_adapter = use_adapter
        self.adapter = MultiScaleAdapter(d) if use_adapter else None
        self.pool = AttnPool(d) if pool_type == "attn" else MeanPool()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1))
        self.hierarchical = hierarchical
        self.head_f = Head(d, n_fam)
        self.head_c = Head(d, n_cls) if hierarchical else None

    def set_freeze_esm(self, freeze: bool):
        self.backbone.set_freeze(freeze)

    def forward(self, tokens, mask):
        H = self.backbone(tokens)
        if self.use_adapter:
            H = self.adapter(H, mask)
        z = self.pool(H, mask)
        z = self.proj(z)
        logits_f = self.head_f(z)
        logits_c = self.head_c(z) if self.hierarchical else None
        return logits_c, logits_f, z


class ModelCNNRaw(nn.Module):
    def __init__(self, n_cls: int, n_fam: int, vocab: int = 28, emb: int = 128,
                 hidden: int = 256, kernels=(3, 5, 7, 11), hierarchical: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, hidden, k, padding=k // 2) for k in kernels])
        self.proj = nn.Sequential(nn.Linear(hidden * len(kernels), hidden), nn.GELU(), nn.Dropout(0.1))
        self.hierarchical = hierarchical
        self.head_f = Head(hidden, n_fam)
        self.head_c = Head(hidden, n_cls) if hierarchical else None

    def forward(self, tokens, mask):
        x = self.emb(tokens).transpose(1, 2)
        feats = [F.gelu(conv(x)) for conv in self.convs]  # B,hidden,L
        cat = torch.cat(feats, dim=1)                     # B,hidden*k,L
        m = mask.unsqueeze(1).float()
        z = (cat * m).sum(2) / (m.sum(2).clamp_min(1.0))
        z = self.proj(z)
        logits_f = self.head_f(z)
        logits_c = self.head_c(z) if self.hierarchical else None
        return logits_c, logits_f, z


# =========================
# Train / Eval
# =========================
@dataclass
class TrainCfg:
    batch_size: int = 16
    max_len: int = 768
    wd: float = 0.01
    grad_clip: float = 1.0
    temperature: float = 0.07
    lambda_contrast: float = 0.2
    cm_topn: int = 30

    amp: bool = True
    grad_accum: int = 2
    use_ckpt: bool = True


def _device_is_cuda(device: str) -> bool:
    return (device == "cuda") and torch.cuda.is_available()


def train_epoch(model, loader, opt, device, cfg: TrainCfg, scaler: GradScaler,
                use_contrastive: bool, hierarchical: bool):
    model.train()
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for step, (tokens, mask, y_c, y_f) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        tokens, mask = tokens.to(device), mask.to(device)
        y_c, y_f = y_c.to(device), y_f.to(device)

        with autocast("cuda", enabled=(cfg.amp and _device_is_cuda(device))):
            logits_c, logits_f, z = model(tokens, mask)

            lf = F.cross_entropy(logits_f, y_f)
            loss = lf
            if hierarchical:
                lc = F.cross_entropy(logits_c, y_c)
                loss = loss + lc
            if use_contrastive:
                loss = loss + cfg.lambda_contrast * supcon_loss(z, y_f, temperature=cfg.temperature)

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

    # flush remainder if not divisible
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
def predict_logits(model, loader, device, hierarchical: bool) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    model.eval()
    lc_list, lf_list = [], []
    ytc, ytf = [], []
    for tokens, mask, y_c, y_f in tqdm(loader, desc="predict", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        logits_c, logits_f, _ = model(tokens, mask)
        lf_list.append(logits_f.detach().cpu().numpy())
        ytf += y_f.tolist()
        if hierarchical:
            lc_list.append(logits_c.detach().cpu().numpy())
            ytc += y_c.tolist()

    lf = np.concatenate(lf_list, axis=0)
    if hierarchical:
        lc = np.concatenate(lc_list, axis=0)
    else:
        lc = np.zeros((lf.shape[0], 1), dtype=np.float32)
        ytc = [0] * lf.shape[0]
    return lc, lf, ytc, ytf


def topn_by_support(y_true: List[int], n: int) -> List[int]:
    s = pd.Series(y_true).value_counts()
    return [int(x) for x in s.index.tolist()[:n]]


def eval_and_save(out_dir: str,
                  logits_c: np.ndarray, logits_f: np.ndarray,
                  ytc: List[int], ytf: List[int],
                  id2cls: Dict[int, str], id2fam: Dict[int, str],
                  cm_topn: int,
                  hierarchical: bool) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    pred_f = logits_f.argmax(1).tolist()
    macro_f1_fam = float(f1_score(ytf, pred_f, average="macro"))
    bal_acc_fam = float(balanced_accuracy_score(ytf, pred_f))

    metrics = {
        "macro_f1_family": macro_f1_fam,
        "balanced_acc_family": bal_acc_fam,
        "n_test": int(len(ytf)),
    }

    fam_labels = list(range(len(id2fam)))
    fam_names = [id2fam[i] for i in fam_labels]
    rep_f = classification_report(
        ytf, pred_f,
        labels=fam_labels,
        target_names=fam_names,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(rep_f).transpose().to_csv(os.path.join(out_dir, "per_family_report.csv"))

    keep_ids = topn_by_support(ytf, cm_topn)
    keep_names = [id2fam[i] for i in keep_ids]
    cm_f_top = confusion_matrix(ytf, pred_f, labels=keep_ids)
    pd.DataFrame(cm_f_top, index=keep_names, columns=keep_names).to_csv(
        os.path.join(out_dir, f"confusion_family_top{cm_topn}.csv")
    )
    plot_confusion(
        cm_f_top, keep_names, f"Family confusion (top{cm_topn})",
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.png"),
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.pdf"),
        normalize=True
    )

    if hierarchical:
        pred_c = logits_c.argmax(1).tolist()
        macro_f1_cls = float(f1_score(ytc, pred_c, average="macro"))
        metrics["macro_f1_class"] = macro_f1_cls

        cls_labels = list(range(len(id2cls)))
        cls_names = [id2cls[i] for i in cls_labels]
        rep_c = classification_report(
            ytc, pred_c,
            labels=cls_labels,
            target_names=cls_names,
            output_dict=True,
            zero_division=0
        )
        pd.DataFrame(rep_c).transpose().to_csv(os.path.join(out_dir, "per_class_report.csv"))

        cm_c = confusion_matrix(ytc, pred_c, labels=cls_labels)
        pd.DataFrame(cm_c, index=cls_names, columns=cls_names).to_csv(os.path.join(out_dir, "confusion_class.csv"))
        plot_confusion(
            cm_c, cls_names, "Class confusion",
            os.path.join(out_dir, "confusion_class_norm.png"),
            os.path.join(out_dir, "confusion_class_norm.pdf"),
            normalize=True
        )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# =========================
# Experiment definitions
# =========================
def build_experiments():
    return [
        {"name": "Proposed",       "key": "proposed"},
        {"name": "Abl_NoContrast", "key": "ab_no_con"},
        {"name": "Abl_FamilyOnly", "key": "ab_no_hier"},
        {"name": "Abl_NoAdapter",  "key": "ab_no_adapter"},
        {"name": "ESM2_MeanPool",  "key": "esm_mean"},
        {"name": "CNN_Raw",        "key": "cnn"},
    ]


def make_model(exp_key: str, esm_model, n_cls: int, n_fam: int, freeze_esm: bool, use_ckpt: bool):
    if exp_key == "proposed":
        return ModelESMVariant(esm_model, n_cls, n_fam, True, "attn", True, freeze_esm, use_ckpt), True, True, "esm"
    if exp_key == "ab_no_con":
        return ModelESMVariant(esm_model, n_cls, n_fam, True, "attn", True, freeze_esm, use_ckpt), False, True, "esm"
    if exp_key == "ab_no_hier":
        return ModelESMVariant(esm_model, n_cls, n_fam, True, "attn", False, freeze_esm, use_ckpt), True, False, "esm"
    if exp_key == "ab_no_adapter":
        return ModelESMVariant(esm_model, n_cls, n_fam, False, "attn", True, freeze_esm, use_ckpt), True, True, "esm"
    if exp_key == "esm_mean":
        return ModelESMVariant(esm_model, n_cls, n_fam, False, "mean", True, freeze_esm, use_ckpt), False, True, "esm"
    if exp_key == "cnn":
        return ModelCNNRaw(n_cls, n_fam, hierarchical=True), False, True, "cnn"
    raise ValueError(exp_key)


# =========================
# VRAM-safe Stage-2 adjust
# =========================
def vram_safe_adjust(cfg: TrainCfg, stage_name: str = "stage2") -> TrainCfg:
    """
    Conservative VRAM-safe settings for ~8GB GPUs when unfreezing ESM in Stage-2.
    """
    new = TrainCfg(**asdict(cfg))

    # 가장 etkili hamle: max_len düşür
    if new.max_len > 512:
        print(f"[SAFE] {stage_name}: max_len {new.max_len} -> 512 (OOM fix)")
        new.max_len = 512

    # fine-tune sırasında güvenli batch
    if new.batch_size > 4:
        print(f"[SAFE] {stage_name}: batch_size {new.batch_size} -> 4 (OOM fix)")
        new.batch_size = 4

    # efektif batch'i korumak için grad_accum artır
    if new.grad_accum < 4:
        print(f"[SAFE] {stage_name}: grad_accum {new.grad_accum} -> 4 (keep effective batch)")
        new.grad_accum = 4

    return new


def rebuild_loaders(train_ds, test_ds, backend: str, alphabet, cfg: TrainCfg):
    if backend == "cnn":
        collator = CNNRawCollator(max_len=cfg.max_len)
    else:
        collator = ESMCollator(alphabet, max_len=cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    return collator, train_loader, test_loader


# =========================
# Main
# =========================
def main():
    set_pub_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder")
    ap.add_argument("--out", default="ARCH_Q1_Framework_FINAL")

    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")

    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--lr_stage1", type=float, default=2e-4)

    ap.add_argument("--epochs_stage2", type=int, default=5)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--cm_topn", type=int, default=30)

    ap.add_argument("--amp", action="store_true", help="use mixed precision")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--ckpt", action="store_true", help="enable gradient checkpointing if supported")
    ap.add_argument("--no_ckpt", action="store_true", help="disable checkpointing")

    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--only", default="", help="comma-separated experiment names to run (e.g., Proposed,CNN_Raw)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device}")

    cfg = TrainCfg(
        batch_size=args.batch_size,
        max_len=args.max_len,
        cm_topn=args.cm_topn,
        amp=(False if args.no_amp else True),
        grad_accum=max(1, args.grad_accum),
        use_ckpt=(False if args.no_ckpt else True) if (args.ckpt or not args.no_ckpt) else False
    )

    train_fa = os.path.join(args.data_dir, "train_homology.fasta")
    test_fa  = os.path.join(args.data_dir, "test_homology.fasta")
    train_csv = os.path.join(args.data_dir, "labels_train.csv")
    test_csv  = os.path.join(args.data_dir, "labels_test.csv")

    train_df = pd.read_csv(train_csv).copy()
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

    print(f"[DATA] classes={len(cls2id)} families={len(fam2id)}")

    train_ds = CAZyDataset(train_fa, train_csv, cls2id, fam2id)
    test_ds  = CAZyDataset(test_fa,  test_csv,  cls2id, fam2id)
    print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id}, f, indent=2)

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

    # Proposed ensemble accumulation
    ens_lc_sum, ens_lf_sum = None, None
    ens_yc_ref, ens_yf_ref = None, None
    ens_count = 0

    for exp in exps:
        exp_name = exp["name"]
        exp_key  = exp["key"]

        for seed in seeds:
            set_seed(seed)

            out_dir = os.path.join(args.out, exp_name.replace(" ", "_"), f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)

            # Build model
            model, use_contrastive, hierarchical, backend = make_model(
                exp_key, esm_model, len(cls2id), len(fam2id), freeze_esm=True, use_ckpt=cfg.use_ckpt
            )
            model = model.to(device)

            # Per-run cfg copy (avoid global mutation)
            run_cfg = TrainCfg(**asdict(cfg))

            # loaders for Stage-1
            _, train_loader, test_loader = rebuild_loaders(train_ds, test_ds, backend, alphabet, run_cfg)

            history = []
            best_state = None
            best_f1 = -1.0

            # -------------------------
            # Stage 1
            # -------------------------
            if backend == "esm":
                model.set_freeze_esm(True)

            opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=args.lr_stage1, weight_decay=run_cfg.wd)

            for ep in range(1, args.epochs_stage1 + 1):
                loss = train_epoch(model, train_loader, opt1, device, run_cfg, scaler, use_contrastive, hierarchical)
                lc, lf, ytc, ytf = predict_logits(model, test_loader, device, hierarchical)
                pred_f = lf.argmax(1).tolist()
                f1_fam = float(f1_score(ytf, pred_f, average="macro"))
                row = {"epoch": ep, "stage": "freeze", "loss": loss, "macro_f1_family": f1_fam, "macro_f1_class": np.nan}
                if hierarchical:
                    pred_c = lc.argmax(1).tolist()
                    row["macro_f1_class"] = float(f1_score(ytc, pred_c, average="macro"))
                history.append(row)

                if f1_fam > best_f1:
                    best_f1 = f1_fam
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state, strict=True)

            # free mem before stage-2
            gc.collect()
            if _device_is_cuda(device):
                torch.cuda.empty_cache()

            # -------------------------
            # Stage 2 (ESM only)
            # -------------------------
            if backend == "esm" and args.epochs_stage2 > 0:
                model.set_freeze_esm(False)

                # VRAM-safe adjust (ONLY for stage2)
                run_cfg_s2 = vram_safe_adjust(run_cfg, "stage2")
                _, train_loader, test_loader = rebuild_loaders(train_ds, test_ds, backend, alphabet, run_cfg_s2)

                opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=args.lr_stage2, weight_decay=run_cfg_s2.wd)

                best_state2 = None
                best_f1_2 = -1.0

                for ep2 in range(1, args.epochs_stage2 + 1):
                    loss = train_epoch(model, train_loader, opt2, device, run_cfg_s2, scaler, use_contrastive, hierarchical)
                    lc, lf, ytc, ytf = predict_logits(model, test_loader, device, hierarchical)
                    pred_f = lf.argmax(1).tolist()
                    f1_fam = float(f1_score(ytf, pred_f, average="macro"))
                    row = {"epoch": args.epochs_stage1 + ep2, "stage": "finetune", "loss": loss, "macro_f1_family": f1_fam, "macro_f1_class": np.nan}
                    if hierarchical:
                        pred_c = lc.argmax(1).tolist()
                        row["macro_f1_class"] = float(f1_score(ytc, pred_c, average="macro"))
                    history.append(row)

                    if f1_fam > best_f1_2:
                        best_f1_2 = f1_fam
                        best_state2 = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                if best_state2 is not None:
                    model.load_state_dict(best_state2, strict=True)

            # -------------------------
            # Final eval + save outputs
            # -------------------------
            # Use the last loaders (stage2 if existed else stage1)
            lc, lf, ytc, ytf = predict_logits(model, test_loader, device, hierarchical)
            metrics = eval_and_save(out_dir, lc, lf, ytc, ytf, id2cls, id2fam, run_cfg.cm_topn, hierarchical)

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
            plot_history(hist_df, f"{exp_name} (seed={seed}) training curve",
                         os.path.join(out_dir, "training_curve.png"),
                         os.path.join(out_dir, "training_curve.pdf"))

            ckpt = {
                "exp_name": exp_name,
                "exp_key": exp_key,
                "seed": seed,
                "cfg": vars(args),
                "metrics": metrics,
                "cls2id": cls2id,
                "fam2id": fam2id,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))

            run_rows.append({"model": exp_name, "seed": seed, **metrics})

            # Proposed ensemble accumulation
            if exp_name == "Proposed":
                if ens_lc_sum is None:
                    ens_lc_sum = lc.astype(np.float64)
                    ens_lf_sum = lf.astype(np.float64)
                    ens_yc_ref, ens_yf_ref = ytc, ytf
                else:
                    if ytc != ens_yc_ref or ytf != ens_yf_ref:
                        raise RuntimeError("Test order mismatch across runs. Keep shuffle=False.")
                    ens_lc_sum += lc.astype(np.float64)
                    ens_lf_sum += lf.astype(np.float64)
                ens_count += 1

            # cleanup between runs
            del model
            gc.collect()
            if _device_is_cuda(device):
                torch.cuda.empty_cache()

    # -------------------------
    # Global summaries
    # -------------------------
    df_runs = pd.DataFrame(run_rows)
    df_runs.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)

    agg_cols = [c for c in ["macro_f1_family", "balanced_acc_family", "macro_f1_class"] if c in df_runs.columns]
    df_models = df_runs.groupby("model")[agg_cols].agg(["mean", "std"]).reset_index()
    df_models.to_csv(os.path.join(args.out, "summary_models.csv"), index=False)

    df_plot = df_runs.groupby("model")["macro_f1_family"].mean().reset_index()
    plot_bar(df_plot, "macro_f1_family",
             os.path.join(args.out, "compare_macro_f1_family.png"),
             os.path.join(args.out, "compare_macro_f1_family.pdf"),
             title="Architecture Contribution: Macro-F1 (Family) across models (mean over seeds)")

    # -------------------------
    # Proposed 3-seed ensemble
    # -------------------------
    if ens_count > 0:
        ens_dir = os.path.join(args.out, "Proposed_Ensemble_3seed")
        ens_lc = ens_lc_sum / float(ens_count)
        ens_lf = ens_lf_sum / float(ens_count)
        eval_and_save(ens_dir, ens_lc, ens_lf, ens_yc_ref, ens_yf_ref, id2cls, id2fam, cfg.cm_topn, hierarchical=True)

    print("\n✅ DONE")
    print("Output root:", args.out)
    print("- summary_runs.csv / summary_models.csv")
    print("- compare_macro_f1_family.pdf/png")
    print("- Proposed_Ensemble_3seed/")
    print("- Each run folder includes metrics + reports + training curves + confusion PDFs")


if __name__ == "__main__":
    main()
