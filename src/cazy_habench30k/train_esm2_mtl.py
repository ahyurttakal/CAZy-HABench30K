#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAZy Enzyme Classification — Full benchmark Comparison Suite 
========================================================
Baselines : HBI | CNN | ESM-2 (frozen) | ProtBERT
Proposed  : ESM-2 + Mean Pooling + Hierarchical Multi-Task

Proposed components:
  ① Two-stage training  — Stage-1 frozen backbone, Stage-2 last-n unfreeze
  ② Mean pooling        — stable sequence summarization
  ③ Multi-task CE loss  — family CE + λ * class CE
  ④ Validation-based model selection

Analyses / outputs:
  • Homology-aware vs random split leakage analysis
  • Few-shot comparison
  • Per-family F1 reports
  • Head / Medium / Tail group analysis
  • Statistical tests
  • Calibration / ECE
  • UMAP / t-SNE embedding visualization (optional)
  • HBI vs Proposed per-family delta table
  • Publication-ready CSV / LaTeX tables

Example:
  python train_esm2_mtl.py \
      --train_fasta data/splits/train.fasta \
      --val_fasta data/splits/val.fasta \
      --test_fasta data/splits/test.fasta \
      --train_labels data/splits/train_labels.csv \
      --val_labels data/splits/val_labels.csv \
      --test_labels data/splits/test_labels.csv \
      --out results/comparison \
      --models hbi,cnn,esm_frozen,protbert,proposed \
      --epochs 5 \
      --epochs_stage1 10 \
      --epochs_stage2 5 \
      --batch_size 4 \
      --grad_accum 4 \
      --lr 3e-4 \
      --lr_stage2 5e-5 \
      --unfreeze_last_n 2 \
      --seeds 1,7,42 \
      --eval_few_shot \
      --k_shots 1,5,10,20 \
      --fewshot_seeds 3 \
      --eval_leakage \
      --eval_uncertainty \
      --eval_embedding
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    print("[WARN] pip install scipy")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL DEPS
# ──────────────────────────────────────────────────────────────────────────────

try:
    import esm as esm_lib
    ESM_OK = True
except Exception:
    ESM_OK = False
    print("[WARN] pip install fair-esm")

try:
    from transformers import BertModel, BertTokenizer
    BERT_OK = True
except Exception:
    BERT_OK = False
    print("[WARN] pip install transformers")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOT_OK = True
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
except Exception:
    PLOT_OK = False
    print("[WARN] pip install matplotlib seaborn")

try:
    import umap as umap_lib
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ESM_MODEL = "esm2_t12_35M_UR50D"
ESM_DIM = 480
ESM_MAXLEN = 256

PBERT_NAME = "Rostlab/prot_bert"
PBERT_DIM = 1024
PBERT_MAXLEN = 256

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
AA_VOCAB = {aa: i + 1 for i, aa in enumerate(sorted(AA_SET))}
CAZY_CLS_ORDER = ["GH", "GT", "PL", "CE", "AA", "CBM"]

ESM_ENCODE_CHUNK = 4

MODEL_LABEL = {
    "hbi": "HBI (MMseqs2 1-NN)",
    "cnn": "CNN",
    "esm_frozen": "ESM-2 (frozen)",
    "protbert": "ProtBERT",
    "proposed": "ESM-2+Mean+MTL (ours)",
}
PALETTE = {
    "HBI (MMseqs2 1-NN)": "#bcbd22",
    "CNN": "#7f7f7f",
    "ESM-2 (frozen)": "#1f77b4",
    "ProtBERT": "#2ca02c",
    "ESM-2+Mean+MTL (ours)": "#d62728",
}
MARKERS = {
    "HBI (MMseqs2 1-NN)": "v",
    "CNN": "s",
    "ESM-2 (frozen)": "p",
    "ProtBERT": "^",
    "ESM-2+Mean+MTL (ours)": "o",
}

FREQ_THRESHOLDS = {"head": 200, "tail": 50}

# ──────────────────────────────────────────────────────────────────────────────
# GENERAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def free_gpu(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def normalize_id(raw: str) -> str:
    s = str(raw).strip().lstrip(">")
    s = s.split()[0]
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1].strip()
    return s.strip()


def fam2cls(fam: str) -> str:
    fam_up = str(fam).upper()
    for c in sorted(CAZY_CLS_ORDER, key=len, reverse=True):
        if fam_up.startswith(c):
            return c
    return fam_up[:2]


def metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_ece(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    confs = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(correct[m].mean() - confs[m].mean())
    return float(ece / max(len(labels), 1))


def assign_freq_group(n: int) -> str:
    if n > FREQ_THRESHOLDS["head"]:
        return "Head (n>200)"
    if n > FREQ_THRESHOLDS["tail"]:
        return "Medium (50<n≤200)"
    return "Tail (n≤50)"


def _mean_std(x: List[float]) -> Tuple[float, float]:
    arr = np.array(x, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_split(fasta: str, labels_csv: str):
    df = pd.read_csv(labels_csv)

    if "id" not in df.columns:
        for cand in ("uniprot_id", "accession", "protein_id", "seq_id"):
            if cand in df.columns:
                df = df.rename(columns={cand: "id"})
                break
        else:
            raise ValueError(f"{labels_csv} içinde 'id' sütunu bulunamadı.")

    if "family" not in df.columns:
        for cand in ("cazy_family", "fam", "Family", "CAZy_family"):
            if cand in df.columns:
                df = df.rename(columns={cand: "family"})
                break
        else:
            raise ValueError(f"{labels_csv} içinde 'family' sütunu bulunamadı.")

    if "class" not in df.columns:
        df["class"] = df["family"].astype(str).map(fam2cls)

    df["id"] = df["id"].astype(str).apply(normalize_id)
    df["family"] = df["family"].astype(str)
    df["class"] = df["class"].astype(str)
    df = df.drop_duplicates(subset="id", keep="first").set_index("id")

    seqs, fams, clss = [], [], []
    skipped_id = 0
    skipped_short = 0

    for rec in SeqIO.parse(fasta, "fasta"):
        rid = normalize_id(rec.id)
        seq = "".join(a for a in str(rec.seq).upper() if a in AA_SET)
        if rid not in df.index:
            skipped_id += 1
            continue
        if len(seq) < 10:
            skipped_short += 1
            continue
        fam = str(df.loc[rid, "family"])
        cls = str(df.loc[rid, "class"])
        seqs.append(seq[:ESM_MAXLEN])
        fams.append(fam)
        clss.append(cls)

    if skipped_id > 0:
        print(f"  [UYARI] {skipped_id} FASTA kaydı labels içinde bulunamadı")
    if skipped_short > 0:
        print(f"  [UYARI] {skipped_short} çok kısa sekans atlandı")
    if len(seqs) == 0:
        raise ValueError(f"{fasta} ve {labels_csv} eşleşmesinden hiç örnek çıkmadı.")

    print(f"  {Path(fasta).name}: {len(seqs)} seq | {len(set(fams))} fam | {len(set(clss))} cls")
    return seqs, fams, clss


def load_split_ids(fasta: str, labels_csv: str):
    """
    load_split() ile aynı filtreleme sırasını kullanarak FASTA ID listesini döndürür.
    Bu sayede per_sample_predictions.csv içinde test protein ID'leri korunur.
    """
    df = pd.read_csv(labels_csv)

    if "id" not in df.columns:
        for cand in ("uniprot_id", "accession", "protein_id", "seq_id"):
            if cand in df.columns:
                df = df.rename(columns={cand: "id"})
                break
        else:
            raise ValueError(f"{labels_csv} içinde 'id' sütunu bulunamadı.")

    if "family" not in df.columns:
        for cand in ("cazy_family", "fam", "Family", "CAZy_family"):
            if cand in df.columns:
                df = df.rename(columns={cand: "family"})
                break
        else:
            raise ValueError(f"{labels_csv} içinde 'family' sütunu bulunamadı.")

    if "class" not in df.columns:
        df["class"] = df["family"].astype(str).map(fam2cls)

    df["id"] = df["id"].astype(str).apply(normalize_id)
    df["family"] = df["family"].astype(str)
    df["class"] = df["class"].astype(str)
    df = df.drop_duplicates(subset="id", keep="first").set_index("id")

    ids = []
    skipped_id = 0
    skipped_short = 0

    for rec in SeqIO.parse(fasta, "fasta"):
        rid = normalize_id(rec.id)
        seq = "".join(a for a in str(rec.seq).upper() if a in AA_SET)
        if rid not in df.index:
            skipped_id += 1
            continue
        if len(seq) < 10:
            skipped_short += 1
            continue
        ids.append(rid)

    if skipped_id > 0:
        print(f"  [UYARI] {skipped_id} FASTA ID kaydı labels içinde bulunamadı")
    if skipped_short > 0:
        print(f"  [UYARI] {skipped_short} çok kısa sekans ID listesinde atlandı")

    return ids


def few_shot_sample(seqs, lbls, k, seed):
    rng = random.Random(seed)
    c2i = defaultdict(list)
    for i, l in enumerate(lbls):
        c2i[l].append(i)
    idx = []
    for _, idxs in c2i.items():
        idx.extend(rng.sample(idxs, min(k, len(idxs))))
    return [seqs[i] for i in idx], [lbls[i] for i in idx]

# ──────────────────────────────────────────────────────────────────────────────
# LOSSES / SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        with torch.no_grad():
            denom = max(n - 1, 1)
            smooth = torch.full_like(logits, self.ls / denom)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=-1)
        ce = -(smooth * log_p).sum(-1)
        weight = (1 - torch.exp(-ce)) ** self.gamma
        return (weight * ce).mean()


def balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    weights = torch.tensor(1.0 / (counts[labels] + 1e-6), dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class SimpleHierLoss(nn.Module):
    def __init__(self, lambda_class: float = 0.3, fam_weight=None):
        super().__init__()
        self.lambda_class = lambda_class
        self.fam_weight = fam_weight

    def forward(self, fam_logits, cls_logits, fam_labels, cls_labels):
        l_fam = F.cross_entropy(fam_logits, fam_labels, weight=self.fam_weight)
        l_cls = F.cross_entropy(cls_logits, cls_labels)
        loss = l_fam + self.lambda_class * l_cls
        return loss, {
            "l_fam": float(l_fam.detach().cpu()),
            "l_cls": float(l_cls.detach().cpu()),
        }

# ──────────────────────────────────────────────────────────────────────────────
# DATASETS / COLLATE
# ──────────────────────────────────────────────────────────────────────────────

def seq2tok(seq, maxlen=ESM_MAXLEN):
    ids = [AA_VOCAB.get(a, 0) for a in seq[:maxlen]]
    ids += [0] * (maxlen - len(ids))
    return ids


class SeqDS(Dataset):
    def __init__(self, seqs, fids, cids):
        self.seqs = seqs
        self.fids = fids
        self.cids = cids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.fids[i], self.cids[i]


class CNNDataset(Dataset):
    def __init__(self, seqs, fids, cids):
        self.toks = [torch.tensor(seq2tok(s)) for s in seqs]
        self.fids = fids
        self.cids = cids

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        return self.toks[i], self.fids[i], self.cids[i]


def seq_collate(b):
    s, f, c = zip(*b)
    return list(s), torch.tensor(f), torch.tensor(c)


def cnn_collate(b):
    t, f, c = zip(*b)
    return torch.stack(t), torch.tensor(f), torch.tensor(c)


def _make_ds(seqs, fam, cls_, fam_le, cls_le):
    fids = fam_le.transform(fam).tolist()
    cids = cls_le.transform(cls_).tolist()
    return fids, cids


def get_loader(seqs, fam, cls_, fam_le, cls_le, bs, shuffle=False, balanced=False):
    fids, cids = _make_ds(seqs, fam, cls_, fam_le, cls_le)
    ds = SeqDS(seqs, fids, cids)
    sampler = balanced_sampler(fids) if balanced else None
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and not balanced),
        sampler=sampler,
        collate_fn=seq_collate,
        num_workers=0,
    )


def get_cnn_loader(seqs, fam, cls_, fam_le, cls_le, bs, shuffle=False, balanced=False):
    fids, cids = _make_ds(seqs, fam, cls_, fam_le, cls_le)
    ds = CNNDataset(seqs, fids, cids)
    sampler = balanced_sampler(fids) if balanced else None
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(shuffle and not balanced),
        sampler=sampler,
        collate_fn=cnn_collate,
        num_workers=0,
    )

# ──────────────────────────────────────────────────────────────────────────────
# ESM HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def esm_encode(
    esm_model,
    alphabet,
    seqs: List[str],
    repr_layer: int,
    device,
    chunk: int = ESM_ENCODE_CHUNK,
) -> torch.Tensor:
    bc = alphabet.get_batch_converter()
    eos_idx = alphabet.eos_idx
    outs = []
    for i in range(0, len(seqs), chunk):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i + chunk])]
        _, _, tok = bc(batch)
        tok = tok.to(device)
        with torch.no_grad() if not esm_model.training else torch.enable_grad():
            rep = esm_model(tok, repr_layers=[repr_layer], return_contacts=False)["representations"][repr_layer]
        mask = (tok != alphabet.padding_idx).float().to(device)
        mask = mask * (tok != eos_idx).float().to(device)
        mask[:, 0] = 0
        mean = (rep * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        outs.append(mean)
    return torch.cat(outs, dim=0)


def esm_encode_raw(
    esm_model,
    alphabet,
    seqs: List[str],
    repr_layer: int,
    device,
    chunk: int = ESM_ENCODE_CHUNK,
):
    bc = alphabet.get_batch_converter()
    eos_idx = alphabet.eos_idx
    reps, masks = [], []
    for i in range(0, len(seqs), chunk):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i + chunk])]
        _, _, tok = bc(batch)
        tok = tok.to(device)
        with torch.no_grad() if not esm_model.training else torch.enable_grad():
            rep = esm_model(tok, repr_layers=[repr_layer], return_contacts=False)["representations"][repr_layer]
        mask = (tok != alphabet.padding_idx).float()
        mask = mask * (tok != eos_idx).float()
        mask[:, 0] = 0.0
        mask = mask.to(device)
        reps.append(rep)
        masks.append(mask)

    max_len = max(r.size(1) for r in reps)
    B_total = sum(r.size(0) for r in reps)
    D = reps[0].size(2)
    out_rep = torch.zeros(B_total, max_len, D, device=device)
    out_mask = torch.zeros(B_total, max_len, device=device)
    idx = 0
    for r, m in zip(reps, masks):
        b, l, _ = r.shape
        out_rep[idx:idx + b, :l, :] = r
        out_mask[idx:idx + b, :l] = m
        idx += b
    return out_rep, out_mask

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────

class MeanPool(nn.Module):
    def forward(self, H, mask):
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp_min(1.0)


class SimpleHead(nn.Module):
    def __init__(self, d, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class ProposedModel(nn.Module):
    """
    Old-runner style:
    ESM-2 + Mean Pooling + two-head multi-task
    """
    def __init__(self, n_fam, n_cls, hidden=ESM_DIM, dropout=0.1):
        super().__init__()
        if not ESM_OK:
            raise ImportError("pip install fair-esm")

        self.esm, self.alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM_MODEL)
        self.rl = self.esm.num_layers
        self.embed_dim = self.esm.embed_dim

        for p in self.esm.parameters():
            p.requires_grad = False

        if hasattr(self.esm, "gradient_checkpointing_enable"):
            self.esm.gradient_checkpointing_enable()

        self.pool = MeanPool()
        self.proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = SimpleHead(hidden, n_cls, dropout=dropout)
        self.family_head = SimpleHead(hidden, n_fam, dropout=dropout)

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Proposed-OldStyle: {n_tr/1e3:.0f}K trainable (heads only, frozen backbone)")

    def freeze_backbone(self):
        for p in self.esm.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n=2):
        for p in self.esm.parameters():
            p.requires_grad = False

        layers = getattr(self.esm, "layers", None)
        if layers is None:
            print("  [WARN] ESM layers bulunamadı, tüm backbone açılıyor")
            for p in self.esm.parameters():
                p.requires_grad = True
            return

        n_layers = len(layers)
        start = max(0, n_layers - n)
        for i in range(start, n_layers):
            for p in layers[i].parameters():
                p.requires_grad = True

        extra_modules = ["emb_layer_norm_after", "contact_head"]
        for mod_name in extra_modules:
            mod = getattr(self.esm, mod_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Proposed-OldStyle: backbone last-{n} unfrozen | trainable={n_tr/1e6:.2f}M")

    def _encode(self, seqs):
        device = next(self.parameters()).device
        H, mask = esm_encode_raw(
            self.esm,
            self.alphabet,
            seqs,
            self.rl,
            device,
            chunk=ESM_ENCODE_CHUNK,
        )
        z = self.pool(H, mask)
        z = self.proj(z)
        return z

    def forward(self, seqs):
        z = self._encode(seqs)
        cls_logits = self.class_head(z)
        fam_logits = self.family_head(z)
        return cls_logits, fam_logits


class CNNModel(nn.Module):
    def __init__(self, n_fam, n_cls, emb_dim=64, hidden=256, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(len(AA_VOCAB) + 1, emb_dim, padding_idx=0)

        def cb(i, o, k):
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=k // 2),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.c1 = cb(emb_dim, hidden, 3)
        self.c2 = cb(hidden, hidden, 5)
        self.c3 = cb(hidden, hidden, 7)
        self.proj = nn.Conv1d(emb_dim, hidden, 1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, n_cls)
        self.fam_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_fam),
        )

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = self.c3(self.c2(self.c1(e))) + self.proj(e)
        z = self.drop(self.pool(h).squeeze(-1))
        return self.cls_head(z), self.fam_head(z)


class ESMFrozenModel(nn.Module):
    def __init__(self, n_fam, n_cls, dropout=0.1):
        super().__init__()
        if not ESM_OK:
            raise ImportError("pip install fair-esm")
        print("  ESM-2 (frozen) yükleniyor...")
        self.esm, self.alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM_MODEL)
        self.rl = self.esm.num_layers
        for p in self.esm.parameters():
            p.requires_grad = False
        if hasattr(self.esm, "gradient_checkpointing_enable"):
            self.esm.gradient_checkpointing_enable()
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(ESM_DIM, n_cls)
        self.fam_head = nn.Sequential(
            nn.Linear(ESM_DIM, ESM_DIM // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ESM_DIM // 2, n_fam),
        )
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ESM-2 (frozen): {n_tr/1e3:.0f}K trainable (heads only)")

    def forward(self, seqs):
        device = next(self.parameters()).device
        z = esm_encode(self.esm, self.alphabet, seqs, self.rl, device, chunk=ESM_ENCODE_CHUNK)
        z = self.drop(z)
        return self.cls_head(z), self.fam_head(z)


class ProtBERTModel(nn.Module):
    def __init__(self, n_fam, n_cls, dropout=0.1):
        super().__init__()
        if not BERT_OK:
            raise ImportError("pip install transformers")
        print("  ProtBERT yükleniyor...")
        try:
            self.tok = BertTokenizer.from_pretrained(PBERT_NAME, do_lower_case=False, local_files_only=True)
            self.bert = BertModel.from_pretrained(PBERT_NAME, local_files_only=True)
        except Exception:
            self.tok = BertTokenizer.from_pretrained(PBERT_NAME, do_lower_case=False)
            self.bert = BertModel.from_pretrained(PBERT_NAME)
        for p in self.bert.parameters():
            p.requires_grad = False
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(PBERT_DIM, n_cls)
        self.fam_head = nn.Linear(PBERT_DIM, n_fam)
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ProtBERT: {n_tr/1e3:.0f}K trainable (frozen BERT + linear heads)")

    def _encode_chunk(self, seqs):
        device = next(self.parameters()).device
        fmt = [" ".join(s[:PBERT_MAXLEN]) for s in seqs]
        enc = self.tok(fmt, return_tensors="pt", padding=True, truncation=True, max_length=PBERT_MAXLEN + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.set_grad_enabled(self.training):
            out = self.bert(**enc)
        return out.last_hidden_state[:, 0, :]

    def forward(self, seqs):
        chunks = []
        for i in range(0, len(seqs), ESM_ENCODE_CHUNK):
            chunks.append(self._encode_chunk(seqs[i:i + ESM_ENCODE_CHUNK]))
            if not self.training and torch.cuda.is_available():
                torch.cuda.empty_cache()
        z = self.drop(torch.cat(chunks, 0))
        return self.cls_head(z), self.fam_head(z)

# ──────────────────────────────────────────────────────────────────────────────
# SETFIT-LIKE FEWSHOT EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────

class SetFitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if not ESM_OK:
            raise ImportError("pip install fair-esm")
        self.esm, self.alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM_MODEL)
        self.rl = self.esm.num_layers
        for p in self.esm.parameters():
            p.requires_grad = False
        for layer in list(self.esm.layers)[-1:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, seqs):
        device = next(self.parameters()).device
        return esm_encode(self.esm, self.alphabet, seqs, self.rl, device, chunk=ESM_ENCODE_CHUNK)

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model, loader, device, is_cnn):
    model.eval()
    all_ft, all_fp = [], []
    for batch in loader:
        try:
            if is_cnn:
                toks, fids, _ = batch
                _, fl = model(toks.to(device))
            else:
                seqs, fids, _ = batch
                _, fl = model(seqs)
            all_ft.extend(fids.tolist())
            all_fp.extend(fl.argmax(-1).cpu().tolist())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                continue
            raise
    if not all_ft:
        return 0.0
    return f1_score(all_ft, all_fp, average="macro", zero_division=0)


@torch.no_grad()
def full_eval(model, loader, device, is_cnn):
    model.eval()
    all_ft, all_fp, all_ct, all_cp = [], [], [], []
    all_cls_logits = []

    for batch in loader:
        try:
            if is_cnn:
                toks, fids, cids = batch
                cl, fl = model(toks.to(device))
            else:
                seqs, fids, cids = batch
                cl, fl = model(seqs)

            all_ft.extend(fids.tolist())
            all_fp.extend(fl.argmax(-1).cpu().tolist())
            all_ct.extend(cids.tolist())
            all_cp.extend(cl.argmax(-1).cpu().tolist())
            all_cls_logits.append(cl.detach().cpu().float().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                continue
            raise

    if len(all_ft) == 0:
        empty_m = {"macro_f1": 0.0, "weighted_f1": 0.0, "mcc": 0.0, "bal_acc": 0.0}
        return empty_m.copy(), empty_m.copy(), [], np.empty((0, 1), dtype=np.float32), []

    fam_m = metrics(all_ft, all_fp)
    cls_m = metrics(all_ct, all_cp)

    n_labels = max(all_ft) + 1 if len(all_ft) > 0 else 0
    per_fam = (
        f1_score(all_ft, all_fp, average=None, labels=list(range(n_labels)), zero_division=0).tolist()
        if n_labels > 0 else []
    )
    cls_logits_np = (
        np.concatenate(all_cls_logits, axis=0)
        if len(all_cls_logits) > 0
        else np.empty((0, 1), dtype=np.float32)
    )
    return fam_m, cls_m, per_fam, cls_logits_np, all_ct


@torch.no_grad()
def predict_per_sample(
    model,
    loader,
    device,
    is_cnn,
    ids: List[str],
    fam_le: LabelEncoder,
    cls_le: LabelEncoder,
    model_key: str,
    model_label: str,
    seed,
) -> pd.DataFrame:
    """
    Test seti için ID bazlı tahmin CSV'si üretir.

    Çıktı sütunları:
      id, true_family, pred_family, true_class, pred_class,
      model_key, model_label, seed, confidence, class_confidence
    """
    model.eval()

    rows = []
    ptr = 0

    for batch in loader:
        try:
            if is_cnn:
                toks, fids, cids = batch
                toks = toks.to(device)
                cls_logits, fam_logits = model(toks)
            else:
                seqs, fids, cids = batch
                cls_logits, fam_logits = model(seqs)

            fam_probs = F.softmax(fam_logits.detach().cpu().float(), dim=-1).numpy()
            cls_probs = F.softmax(cls_logits.detach().cpu().float(), dim=-1).numpy()

            fam_pred_ids = fam_probs.argmax(axis=1)
            cls_pred_ids = cls_probs.argmax(axis=1)

            fam_true_ids = fids.detach().cpu().numpy()
            cls_true_ids = cids.detach().cpu().numpy()

            batch_size = len(fam_true_ids)

            for i in range(batch_size):
                sid = ids[ptr + i] if (ptr + i) < len(ids) else f"idx_{ptr+i}"

                rows.append({
                    "id": sid,
                    "true_family": fam_le.inverse_transform([int(fam_true_ids[i])])[0],
                    "pred_family": fam_le.inverse_transform([int(fam_pred_ids[i])])[0],
                    "true_class": cls_le.inverse_transform([int(cls_true_ids[i])])[0],
                    "pred_class": cls_le.inverse_transform([int(cls_pred_ids[i])])[0],
                    "model_key": model_key,
                    "model_label": model_label,
                    "seed": seed,
                    "confidence": float(fam_probs[i, fam_pred_ids[i]]),
                    "class_confidence": float(cls_probs[i, cls_pred_ids[i]]),
                })

            ptr += batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  [OOM] per-sample prediction batch skipped")
                free_gpu()
                continue
            raise

    return pd.DataFrame(rows)


def train_and_eval_model(
    model,
    name,
    tr_ld,
    va_ld,
    te_ld,
    device,
    epochs,
    lr,
    loss_fn=None,
    is_cnn=False,
    grad_accum: int = 1,
    ckpt_path: Optional[Path] = None,
):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(tr_ld) / max(grad_accum, 1))
    total_steps = max(steps_per_epoch * epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.06,
        anneal_strategy="cos",
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    focal = FocalLoss(gamma=2.0, label_smoothing=0.05)

    best_val = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(tr_ld, 1):
            try:
                if is_cnn:
                    toks, fids, cids = batch
                    toks = toks.to(device)
                    fids = fids.to(device)
                    cids = cids.to(device)
                    with autocast(enabled=torch.cuda.is_available()):
                        cl, fl = model(toks)
                        loss = (0.3 * focal(cl, cids) + focal(fl, fids)) / grad_accum
                else:
                    seqs, fids, cids = batch
                    fids = fids.to(device)
                    cids = cids.to(device)
                    with autocast(enabled=torch.cuda.is_available()):
                        cl, fl = model(seqs)
                        if loss_fn is not None:
                            raw, _ = loss_fn(fl, cl, fids, cids)
                        else:
                            raw = 0.3 * focal(cl, cids) + focal(fl, fids)
                        loss = raw / grad_accum

                scaler.scale(loss).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  [OOM] ep={ep} step={step}")
                    free_gpu()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            if step % grad_accum == 0 or step == len(tr_ld):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        val_f1 = quick_eval(model, va_ld, device, is_cnn)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    if ckpt_path:
        torch.save({"state_dict": best_state, "best_val_f1": best_val}, ckpt_path)

    fam_m, cls_m, per_fam, cls_logits_np, cls_labels_np = full_eval(model, te_ld, device, is_cnn)

    if cls_logits_np.shape[0] > 0:
        ece = compute_ece(cls_logits_np, np.array(cls_labels_np))
        fam_m["ece"] = ece
        cls_m["ece"] = ece
    else:
        fam_m["ece"] = float("nan")
        cls_m["ece"] = float("nan")

    print(
        f"  [{name:<30}] FamF1={fam_m['macro_f1']:.4f} "
        f"ClsF1={cls_m['macro_f1']:.4f} MCC={fam_m['mcc']:.4f} "
        f"ECE={fam_m['ece']:.4f} ({elapsed:.0f}s)"
    )
    return fam_m, cls_m, per_fam


def train_and_eval_proposed_oldstyle(
    model,
    tr_ld,
    va_ld,
    te_ld,
    device,
    epochs_stage1,
    epochs_stage2,
    lr_stage1,
    lr_stage2,
    lambda_class: float = 0.3,
    grad_accum=1,
    unfreeze_last_n=2,
    ckpt_path: Optional[Path] = None,
):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    fam_ids = []
    for _, fids, _ in tr_ld.dataset:
        fam_ids.append(fids)
    if len(fam_ids) > 0:
        fam_ids = np.array(fam_ids, dtype=np.int64)
        counts = np.bincount(fam_ids)
        w = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)
        if (w > 0).sum() > 0:
            w = w / (w[w > 0].mean() + 1e-8)
            fam_weight = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            fam_weight = None
    else:
        fam_weight = None

    loss_fn = SimpleHierLoss(lambda_class=lambda_class, fam_weight=fam_weight)

    # Stage 1
    model.freeze_backbone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr_stage1, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr_ld) / max(grad_accum, 1))
    total_steps = max(steps_per_epoch * epochs_stage1, 1)
    warmup_steps = max(1, int(total_steps * 0.10))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"  [Stage-1] frozen backbone | steps/epoch={steps_per_epoch} total_steps={total_steps}")

    for ep in range(1, epochs_stage1 + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_loss = 0.0
        ep_n = 0

        for step, batch in enumerate(tr_ld, 1):
            seqs, fids, cids = batch
            fids = fids.to(device)
            cids = cids.to(device)

            try:
                with autocast(enabled=torch.cuda.is_available()):
                    cl, fl = model(seqs)
                    raw, _ = loss_fn(fl, cl, fids, cids)
                    loss = raw / grad_accum
                scaler.scale(loss).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM S1] ep={ep} step={step}")
                    free_gpu()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            if step % grad_accum == 0 or step == len(tr_ld):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            ep_loss += float(loss.detach().cpu()) * grad_accum
            ep_n += 1

        val_f1 = quick_eval(model, va_ld, device, is_cnn=False)
        print(f"  [S1] ep={ep:02d} loss={ep_loss/max(ep_n,1):.4f} val_FamF1={val_f1:.4f}")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2
    if epochs_stage2 > 0:
        model.load_state_dict(best_state)
        free_gpu()
        model.unfreeze_last_n_layers(unfreeze_last_n)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr_stage2, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(tr_ld) / max(grad_accum, 1))
        total_steps = max(steps_per_epoch * epochs_stage2, 1)
        warmup_steps = max(1, int(total_steps * 0.10))

        def lr_lambda2(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda2)
        print(f"  [Stage-2] unfreeze last-{unfreeze_last_n} | steps/epoch={steps_per_epoch} total_steps={total_steps}")

        for ep in range(1, epochs_stage2 + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            ep_loss = 0.0
            ep_n = 0

            for step, batch in enumerate(tr_ld, 1):
                seqs, fids, cids = batch
                fids = fids.to(device)
                cids = cids.to(device)

                try:
                    with autocast(enabled=torch.cuda.is_available()):
                        cl, fl = model(seqs)
                        raw, _ = loss_fn(fl, cl, fids, cids)
                        loss = raw / grad_accum
                    scaler.scale(loss).backward()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  [OOM S2] ep={ep} step={step}")
                        free_gpu()
                        opt.zero_grad(set_to_none=True)
                        continue
                    raise

                if step % grad_accum == 0 or step == len(tr_ld):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

                ep_loss += float(loss.detach().cpu()) * grad_accum
                ep_n += 1

            val_f1 = quick_eval(model, va_ld, device, is_cnn=False)
            print(f"  [S2] ep={ep:02d} loss={ep_loss/max(ep_n,1):.4f} val_FamF1={val_f1:.4f}")
            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    if ckpt_path:
        torch.save({"state_dict": best_state, "best_val_f1": best_val}, ckpt_path)

    fam_m, cls_m, per_fam, cls_logits_np, cls_labels_np = full_eval(model, te_ld, device, is_cnn=False)
    if cls_logits_np.shape[0] > 0:
        ece = compute_ece(cls_logits_np, np.array(cls_labels_np))
        fam_m["ece"] = ece
        cls_m["ece"] = ece
    else:
        fam_m["ece"] = float("nan")
        cls_m["ece"] = float("nan")

    print(
        f"  [Proposed-OldStyle] FamF1={fam_m['macro_f1']:.4f} "
        f"ClsF1={cls_m['macro_f1']:.4f} MCC={fam_m['mcc']:.4f} "
        f"ECE={fam_m['ece']:.4f} ({elapsed:.0f}s)"
    )
    return fam_m, cls_m, per_fam

# ──────────────────────────────────────────────────────────────────────────────
# HBI
# ──────────────────────────────────────────────────────────────────────────────

def run_hbi(
    tr_s: List[str],
    tr_f: List[str],
    tr_c: List[str],
    te_s: List[str],
    te_f: List[str],
    te_c: List[str],
    fam_le: LabelEncoder,
    cls_le: LabelEncoder,
) -> Tuple[Dict, Dict, List]:
    mmseqs = shutil.which("mmseqs")
    if mmseqs:
        return _hbi_mmseqs(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le, mmseqs)
    print("  [HBI] mmseqs bulunamadı → TF-IDF cosine 1-NN fallback")
    return _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le)


def _hbi_mmseqs(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le, mmseqs_bin):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr_fa = tmp / "train.fasta"
        te_fa = tmp / "test.fasta"
        with open(tr_fa, "w", encoding="utf-8") as f:
            for i, (s, lbl) in enumerate(zip(tr_s, tr_f)):
                f.write(f">tr_{i}|{lbl}\n{s}\n")
        with open(te_fa, "w", encoding="utf-8") as f:
            for i, s in enumerate(te_s):
                f.write(f">te_{i}\n{s}\n")

        res_file = tmp / "hits.tsv"
        mmseqs_tmp = tmp / "mmseqs_tmp"
        mmseqs_tmp.mkdir()

        try:
            subprocess.check_call(
                [
                    mmseqs_bin,
                    "easy-search",
                    str(te_fa),
                    str(tr_fa),
                    str(res_file),
                    str(mmseqs_tmp),
                    "--format-output",
                    "query,target,evalue",
                    "--max-seqs",
                    "1",
                    "-v",
                    "0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print("  [HBI] mmseqs easy-search başarısız → cosine fallback")
            return _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le)

        te_idx2fam = {}
        if res_file.exists():
            with open(res_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    q_idx = int(parts[0].replace("te_", ""))
                    target = parts[1]
                    if "|" in target:
                        fam = target.split("|")[1]
                    else:
                        tr_idx = int(target.replace("tr_", ""))
                        fam = tr_f[tr_idx] if tr_idx < len(tr_f) else tr_f[0]
                    te_idx2fam[q_idx] = fam

        majority_fam = max(set(tr_f), key=tr_f.count)
        fam_pred = [te_idx2fam.get(i, majority_fam) for i in range(len(te_s))]

    return _hbi_metrics(fam_pred, te_f, te_c, fam_le, cls_le)


def _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def kmerize(seqs, k=3):
        return [" ".join(s[i:i + k] for i in range(len(s) - k + 1)) for s in seqs]

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1))
    tr_mat = vec.fit_transform(kmerize(tr_s))
    te_mat = vec.transform(kmerize(te_s))

    batch = 256
    fam_pred = []
    tr_f_arr = list(tr_f)
    for i in range(0, len(te_s), batch):
        sims = cosine_similarity(te_mat[i:i + batch], tr_mat)
        best = sims.argmax(axis=1)
        fam_pred.extend(tr_f_arr[int(j)] for j in best)

    return _hbi_metrics(fam_pred, te_f, te_c, fam_le, cls_le)


def _hbi_metrics(fam_pred, te_f, te_c, fam_le, cls_le):
    known = set(fam_le.classes_)
    majority_fam = max(set(te_f), key=te_f.count)
    fam_pred = [f if f in known else majority_fam for f in fam_pred]

    fam_true_ids = fam_le.transform(te_f).tolist()
    fam_pred_ids = fam_le.transform(fam_pred).tolist()

    cls_pred = [fam2cls(f) for f in fam_pred]
    cls_true = te_c
    known_c = set(cls_le.classes_)
    majority_cls = max(set(cls_true), key=cls_true.count)
    cls_pred = [c if c in known_c else majority_cls for c in cls_pred]
    cls_true_ids = cls_le.transform(cls_true).tolist()
    cls_pred_ids = cls_le.transform(cls_pred).tolist()

    fam_m = metrics(fam_true_ids, fam_pred_ids)
    cls_m = metrics(cls_true_ids, cls_pred_ids)
    fam_m["ece"] = float("nan")
    cls_m["ece"] = float("nan")

    n_fam = len(fam_le.classes_)
    per_fam = f1_score(
        fam_true_ids,
        fam_pred_ids,
        average=None,
        labels=list(range(n_fam)),
        zero_division=0,
    ).tolist()

    print(f"  [HBI] FamF1={fam_m['macro_f1']:.4f} ClsF1={cls_m['macro_f1']:.4f} MCC={fam_m['mcc']:.4f}")
    return fam_m, cls_m, per_fam

# ──────────────────────────────────────────────────────────────────────────────
# FEWSHOT
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings_generic(model, loader, device, mode: str):
    model.eval()
    Z, Y = [], []
    for batch in loader:
        try:
            if mode == "cnn":
                toks, fids, _ = batch
                cl, fl = model(toks.to(device))
                z = fl.detach().cpu().float().numpy()
                y = np.array(fids.tolist(), dtype=np.int64)
            else:
                seqs, fids, _ = batch
                if hasattr(model, "_encode"):
                    zt = model._encode(seqs)
                elif hasattr(model, "esm"):
                    zt = esm_encode(model.esm, model.alphabet, seqs, model.rl, device, chunk=ESM_ENCODE_CHUNK)
                else:
                    cl, _ = model(seqs)
                    zt = cl
                z = zt.detach().cpu().float().numpy()
                y = np.array(fids.tolist(), dtype=np.int64)
            Z.append(z)
            Y.append(y)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                continue
            raise

    if not Z:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(Z), np.concatenate(Y)


def prototype_predict(z_query, z_support, y_support, fam_ids):
    fam_ids = sorted(set(fam_ids))
    prototypes = []
    for f in fam_ids:
        zf = z_support[y_support == f]
        if len(zf) == 0:
            continue
        v = zf.mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        prototypes.append((f, v))
    fam_ids2 = [p[0] for p in prototypes]
    P = np.stack([p[1] for p in prototypes], axis=0)
    Z = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + 1e-8)
    sim = Z @ P.T
    idx = sim.argmax(axis=1)
    return np.array([fam_ids2[i] for i in idx], dtype=np.int64)


def fewshot_episode_eval(train_Z, train_Y, test_Z, test_Y, fam_ids, shots, episodes, seed):
    rng = np.random.default_rng(seed)
    fam_ids = [f for f in fam_ids if (train_Y == f).sum() >= shots and (test_Y == f).sum() >= 1]
    if len(fam_ids) < 2:
        return {"macro_f1": np.nan, "macro_f1_std": np.nan, "n_families": len(fam_ids), "shots": shots, "episodes": episodes}
    scores = []
    for _ in range(episodes):
        sup_i, q_i, used = [], [], []
        for f in fam_ids:
            tr_i = np.where(train_Y == f)[0]
            te_i = np.where(test_Y == f)[0]
            if len(tr_i) < shots or len(te_i) < 1:
                continue
            sup_i.extend(rng.choice(tr_i, shots, replace=False).tolist())
            q_i.extend(te_i.tolist())
            used.append(f)
        if len(set(used)) < 2:
            continue
        pred = prototype_predict(test_Z[q_i], train_Z[sup_i], train_Y[sup_i], used)
        scores.append(f1_score(test_Y[q_i], pred, average="macro", zero_division=0))
    return {
        "macro_f1": float(np.mean(scores)) if scores else np.nan,
        "macro_f1_std": float(np.std(scores)) if scores else np.nan,
        "n_families": len(fam_ids),
        "shots": shots,
        "episodes": episodes,
    }


def real_fewshot_run(model_name, tr_s, tr_f, te_s, te_f, fam_le, cls_le, device, k, seed, args):
    k_seqs, k_fam = few_shot_sample(tr_s, tr_f, k, seed)
    k_cls = [fam2cls(f) for f in k_fam]
    te_cls = [fam2cls(f) for f in te_f]
    n_fam, n_cls = len(fam_le.classes_), len(cls_le.classes_)

    if model_name == "hbi":
        fam_m, _, _ = run_hbi(k_seqs, k_fam, k_cls, te_s, te_f, te_cls, fam_le, cls_le)
        return fam_m["macro_f1"]

    if model_name == "cnn":
        model = CNNModel(n_fam, n_cls).to(device)
        tr_ld = get_cnn_loader(k_seqs, k_fam, k_cls, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
        te_ld = get_cnn_loader(te_s, te_f, te_cls, fam_le, cls_le, args.batch_size)
        fam_m, _, _ = train_and_eval_model(model, f"cnn-{k}shot", tr_ld, tr_ld, te_ld, device, args.fewshot_epochs, args.lr, is_cnn=True, grad_accum=args.grad_accum)
        free_gpu(model)
        return fam_m["macro_f1"]

    if model_name == "esm_frozen":
        if not ESM_OK:
            return 0.0
        model = ESMFrozenModel(n_fam, n_cls).to(device)
        tr_ld = get_loader(k_seqs, k_fam, k_cls, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
        te_ld = get_loader(te_s, te_f, te_cls, fam_le, cls_le, args.batch_size)
        fam_m, _, _ = train_and_eval_model(model, f"esmf-{k}shot", tr_ld, tr_ld, te_ld, device, args.fewshot_epochs, args.lr, grad_accum=args.grad_accum)
        free_gpu(model)
        return fam_m["macro_f1"]

    if model_name == "protbert":
        if not BERT_OK:
            return 0.0
        model = ProtBERTModel(n_fam, n_cls).to(device)
        tr_ld = get_loader(k_seqs, k_fam, k_cls, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
        te_ld = get_loader(te_s, te_f, te_cls, fam_le, cls_le, args.batch_size)
        fam_m, _, _ = train_and_eval_model(model, f"pbert-{k}shot", tr_ld, tr_ld, te_ld, device, args.fewshot_epochs, args.lr, grad_accum=args.grad_accum)
        free_gpu(model)
        return fam_m["macro_f1"]

    if model_name == "proposed":
        if not ESM_OK:
            return 0.0
        model = ProposedModel(n_fam, n_cls, hidden=ESM_DIM, dropout=0.1).to(device)
        tr_ld = get_loader(k_seqs, k_fam, k_cls, fam_le, cls_le, args.batch_size, shuffle=True, balanced=False)
        te_ld = get_loader(te_s, te_f, te_cls, fam_le, cls_le, args.batch_size)
        fam_m, _, _ = train_and_eval_proposed_oldstyle(
            model=model,
            tr_ld=tr_ld,
            va_ld=tr_ld,
            te_ld=te_ld,
            device=device,
            epochs_stage1=max(1, args.fewshot_epochs - 1),
            epochs_stage2=1,
            lr_stage1=args.lr,
            lr_stage2=max(args.lr * 0.25, 5e-5),
            lambda_class=args.lambda_class,
            grad_accum=args.grad_accum,
            unfreeze_last_n=args.unfreeze_last_n,
        )
        free_gpu(model)
        return fam_m["macro_f1"]

    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION / UNCERTAINTY / EMBEDDING
# ──────────────────────────────────────────────────────────────────────────────

def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_eval(model, loader, device, n_passes: int = 20):
    model.eval()
    _enable_dropout(model)
    all_conf, all_uncert, all_true, all_pred = [], [], [], []

    for batch in loader:
        if isinstance(batch[0], torch.Tensor):
            # CNN
            toks, _, _ = batch
            raise RuntimeError("MC dropout is implemented only for sequence models in this script.")
        seqs, _, yf = batch
        probs_list = []
        for _ in range(n_passes):
            _, lf = model(seqs)
            probs_list.append(F.softmax(lf, -1).cpu().float().numpy())

        probs_arr = np.stack(probs_list)
        probs = probs_arr.mean(0)
        std_arr = probs_arr.std(0)
        pred = probs.argmax(1)
        conf = probs.max(1)
        uncert = std_arr[np.arange(len(pred)), pred]

        all_conf.extend(conf.tolist())
        all_uncert.extend(uncert.tolist())
        all_true.extend(yf.tolist())
        all_pred.extend(pred.tolist())

    model.eval()
    confs = np.array(all_conf)
    correct = np.array([p == t for p, t in zip(all_pred, all_true)], dtype=bool)

    n_bins = 15
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf_m, bin_count = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask_b = (confs >= lo) & (confs < hi)
        if mask_b.sum() == 0:
            continue
        bin_acc.append(float(correct[mask_b].mean()))
        bin_conf_m.append(float(confs[mask_b].mean()))
        bin_count.append(int(mask_b.sum()))

    ece = float(sum(abs(a - c) * n for a, c, n in zip(bin_acc, bin_conf_m, bin_count)) / max(sum(bin_count), 1))
    return {
        "confidences": confs,
        "uncertainty": np.array(all_uncert),
        "correct": correct,
        "predictions": np.array(all_pred),
        "true": np.array(all_true),
        "ece": ece,
        "macro_f1_mc": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "mcc_mc": float(matthews_corrcoef(all_true, all_pred)),
        "bin_acc": bin_acc,
        "bin_conf": bin_conf_m,
        "bin_count": bin_count,
    }


def plot_calibration_curve(mc_result: Dict, out_dir: Path, tag: str = ""):
    if not PLOT_OK:
        return None
    ece = mc_result["ece"]
    bin_conf = mc_result["bin_conf"]
    bin_acc = mc_result["bin_acc"]
    bin_count = mc_result["bin_count"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    scatter = ax.scatter(bin_conf, bin_acc, c=bin_count, s=80, cmap="Blues", edgecolors="k", linewidths=0.5, zorder=3)
    plt.colorbar(scatter, ax=ax, label="Sample count per bin")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Calibration Curve{(' — ' + tag) if tag else ''}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    name = f"calibration_curve{'_' + tag if tag else ''}"
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    return ece


@torch.no_grad()
def collect_embeddings_with_meta(model, loader, device, mode: str):
    model.eval()
    Z, YF, YC = [], [], []
    for batch in loader:
        try:
            if mode == "cnn":
                toks, fids, cids = batch
                cl, fl = model(toks.to(device))
                z = fl.detach().cpu().float().numpy()
                YF.append(np.array(fids.tolist(), dtype=np.int64))
                YC.append(np.array(cids.tolist(), dtype=np.int64))
            else:
                seqs, fids, cids = batch
                if hasattr(model, "_encode"):
                    zt = model._encode(seqs)
                elif hasattr(model, "esm"):
                    zt = esm_encode(model.esm, model.alphabet, seqs, model.rl, device, chunk=ESM_ENCODE_CHUNK)
                else:
                    cl, _ = model(seqs)
                    zt = cl
                z = zt.detach().cpu().float().numpy()
                YF.append(np.array(fids.tolist(), dtype=np.int64))
                YC.append(np.array(cids.tolist(), dtype=np.int64))
            Z.append(z)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                continue
            raise
    if not Z:
        return {"Z": np.empty((0, 2), dtype=np.float32), "YF": np.empty((0,), dtype=np.int64), "YC": np.empty((0,), dtype=np.int64)}
    return {"Z": np.concatenate(Z), "YF": np.concatenate(YF), "YC": np.concatenate(YC)}


def plot_embedding_space(emb_data: Dict, id2cls: Dict, out_dir: Path, tag: str = ""):
    if not PLOT_OK:
        return
    Z = emb_data["Z"]
    YC = emb_data["YC"]
    if len(Z) == 0:
        return

    n = min(len(Z), 5000)
    if n < len(Z):
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(len(Z), n, replace=False)
        Z = Z[idx]
        YC = YC[idx]

    print(f"[EMBED] {len(Z)} nokta için boyut indirgeme...")
    if HAS_UMAP:
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        coords = reducer.fit_transform(Z)
        method = "UMAP"
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Z) - 1)).fit_transform(Z)
        method = "t-SNE"

    CLASS_COLORS = {
        "GH": "#E74C3C", "GT": "#3498DB", "PL": "#2ECC71",
        "CE": "#F39C12", "AA": "#9B59B6", "CBM": "#1ABC9C",
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_id, cls_name in id2cls.items():
        mask = YC == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLORS.get(cls_name, "#95A5A6"),
            s=5,
            alpha=0.5,
            label=cls_name,
            linewidths=0,
        )
    ax.set_xlabel(f"{method} dim-1")
    ax.set_ylabel(f"{method} dim-2")
    ax.set_title(f"Embedding Space — {method}{(' (' + tag + ')') if tag else ''}")
    ax.legend(markerscale=4, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    name = f"embedding_{method.lower()}{'_' + tag if tag else ''}"
    plt.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_main_bar(summary_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in zip(
        axes,
        ["fam_macro_f1", "cls_macro_f1"],
        ["(A) Family-level Classification", "(B) Class-level Classification"],
    ):
        names = summary_df["model_label"].tolist()
        means = summary_df[f"{metric}_mean"].tolist()
        stds = summary_df[f"{metric}_std"].tolist()
        colors = [PALETTE.get(n, "#1f77b4") for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.8, error_kw=dict(elinewidth=1.2, ecolor="#333"))
        ax.set_ylim(0.0, min(1.08, max(means) + max(stds) + 0.08))
        ax.set_ylabel("Macro-F1")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.013, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    plt.suptitle("CAZy Enzyme Classification — Model Comparison\n(homology-aware split · mean ± std)", y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_main_bar.pdf", bbox_inches="tight")
    plt.close()


def plot_fewshot_curve(fs_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or fs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for mkey in fs_df["model_key"].unique():
        sub = fs_df[fs_df["model_key"] == mkey].sort_values("k")
        lbl = MODEL_LABEL.get(mkey, mkey)
        col = PALETTE.get(lbl, "#1f77b4")
        mrk = MARKERS.get(lbl, "o")
        ax.plot(sub["k"], sub["f1_mean"], marker=mrk, color=col, linewidth=2, markersize=7, label=lbl)
        ax.fill_between(sub["k"], sub["f1_mean"] - sub["f1_std"], sub["f1_mean"] + sub["f1_std"], alpha=0.12, color=col)
    ax.set_xlabel("k examples per family")
    ax.set_ylabel("Family Macro-F1")
    ax.set_title("Real Few-Shot Learning", fontweight="bold")
    ax.set_xticks(sorted(fs_df["k"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_fewshot_curve.pdf", bbox_inches="tight")
    plt.close()


def plot_radar(summary_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or summary_df.empty:
        return
    cols = ["fam_macro_f1_mean", "cls_macro_f1_mean", "fam_mcc_mean", "fam_bal_acc_mean", "fam_weighted_f1_mean"]
    lbls = ["Fam Macro-F1", "Cls Macro-F1", "Fam MCC", "Bal. Acc", "Weighted F1"]
    N = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylim(0.0, 1.0)

    for _, row in summary_df.iterrows():
        lbl = row["model_label"]
        col = PALETTE.get(lbl, "#1f77b4")
        vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
        ax.plot(angles, vals, color=col, linewidth=2, label=lbl, marker="o", markersize=5)
        ax.fill(angles, vals, color=col, alpha=0.07)

    ax.set_title("Multi-metric Comparison", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_radar.pdf", bbox_inches="tight")
    plt.close()


def plot_family_heatmap(pf_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or pf_df.empty:
        return
    pivot = pf_df.pivot(index="family", columns="model_label", values="f1_mean")
    ours_label = MODEL_LABEL.get("proposed", "ESM-2+Mean+MTL (ours)")
    if ours_label in pivot.columns:
        pivot = pivot.sort_values(ours_label, ascending=False)

    fig_h = max(6, len(pivot) * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0, linewidths=0.3, cbar_kws={"label": "F1"})
    ax.set_title("Per-family Macro-F1 (test set)", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("CAZy Family")
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_family_heatmap.pdf", bbox_inches="tight")
    plt.close()


def plot_group_f1(grp_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or grp_df.empty:
        return

    groups = ["Head (n>200)", "Medium (50<n≤200)", "Tail (n≤50)"]
    models = grp_df["model_label"].unique().tolist()
    x = np.arange(len(groups))
    n_m = len(models)
    width = 0.72 / n_m

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mlbl in enumerate(models):
        sub = grp_df[grp_df["model_label"] == mlbl].set_index("freq_group")
        vals = [sub.loc[g, "f1_mean"] if g in sub.index else 0.0 for g in groups]
        errs = [sub.loc[g, "f1_std"] if g in sub.index else 0.0 for g in groups]
        col = PALETTE.get(mlbl, "#888888")
        bars = ax.bar(x + (i - n_m / 2 + 0.5) * width, vals, width * 0.88, yerr=errs, capsize=3, color=col, alpha=0.88, label=mlbl, error_kw={"elinewidth": 1.2, "alpha": 0.6})
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.013, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="0.25")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Family Macro-F1", fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Per-frequency-group Family Macro-F1", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.axvspan(1.5, 2.5, color="#FFF3CD", alpha=0.35, zorder=0)
    ax.text(2.0, 1.03, "Tail focus", ha="center", fontsize=7.5, color="#856404", style="italic")

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_group_f1.pdf", bbox_inches="tight")
    plt.close()


def plot_confusion(conf_data: Dict, cls_le: LabelEncoder, out_dir: Path):
    if not PLOT_OK or "proposed" not in conf_data:
        return
    cm = np.array(conf_data["proposed"])
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    lbls = cls_le.classes_
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_n, ax=ax, annot=True, fmt=".2f", cmap="Blues", xticklabels=lbls, yticklabels=lbls, vmin=0, vmax=1, linewidths=0.4)
    ax.set_title("Proposed: Class-level Confusion", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_confusion_cls.pdf", bbox_inches="tight")
    plt.close()


def plot_homology_leakage(homo_df: pd.DataFrame, rand_df: pd.DataFrame, metric: str, out_dir: Path):
    if not PLOT_OK or homo_df.empty or rand_df.empty:
        return
    homo_g = homo_df.groupby("model_label")[metric].agg(["mean", "std"]).reset_index()
    rand_g = rand_df.groupby("model_label")[metric].agg(["mean", "std"]).reset_index()
    merged = homo_g.merge(rand_g, on="model_label", suffixes=("_homo", "_rand"))
    merged["delta"] = merged["mean_rand"] - merged["mean_homo"]
    merged = merged.sort_values("delta", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    n = len(merged)
    xs = np.arange(n)
    w = 0.35
    ax.bar(xs - w / 2, merged["mean_homo"], w, yerr=merged["std_homo"], capsize=3, label="Homology-aware", color="#3498DB", alpha=0.85)
    ax.bar(xs + w / 2, merged["mean_rand"], w, yerr=merged["std_rand"], capsize=3, label="Random split", color="#E74C3C", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(merged["model_label"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("(A) Homology-aware vs Random")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colors = ["#E74C3C" if d > 0 else "#2ECC71" for d in merged["delta"]]
    ax2.barh(range(n), merged["delta"], color=colors, alpha=0.85)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(merged["model_label"], fontsize=9)
    ax2.set_xlabel("ΔF1 (random − homology)")
    ax2.set_title("(B) Homology Leakage")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig7_homology_leakage.pdf", bbox_inches="tight")
    plt.close()

    merged.to_csv(out_dir / "homology_leakage_table.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# REPORTS / TABLES
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_f1(per_fam_rows: List[Dict], fam_le: LabelEncoder, fam_counts: List[int]) -> pd.DataFrame:
    if not per_fam_rows:
        return pd.DataFrame()
    fam_count_map = {fam_le.classes_[i]: fam_counts[i] for i in range(len(fam_le.classes_))}
    df = pd.DataFrame(per_fam_rows)
    df["n_train"] = df["family"].map(fam_count_map).fillna(1).astype(int)
    df["freq_group"] = df["n_train"].apply(assign_freq_group)

    grp = (
        df.groupby(["model_key", "model_label", "seed", "freq_group"])["f1"]
        .mean()
        .reset_index()
        .groupby(["model_key", "model_label", "freq_group"])["f1"]
        .agg(f1_mean="mean", f1_std="std")
        .reset_index()
    )
    return grp


def run_statistical_tests(per_seed_df: pd.DataFrame) -> Dict:
    results = {}
    if not SCIPY_OK or per_seed_df.empty:
        return results

    prop_f1s = per_seed_df[per_seed_df["model_key"] == "proposed"]["fam_macro_f1"].values
    if len(prop_f1s) < 2:
        return results

    for mkey in per_seed_df["model_key"].unique():
        if mkey == "proposed":
            continue
        base_f1s = per_seed_df[per_seed_df["model_key"] == mkey]["fam_macro_f1"].values
        n = min(len(prop_f1s), len(base_f1s))
        if n < 3:
            results[mkey] = {"p_value": float("nan"), "symbol": "n/a", "note": f"only {n} seeds"}
            continue
        try:
            stat, p = wilcoxon(prop_f1s[:n], base_f1s[:n], alternative="two-sided")
            delta = float(np.mean(prop_f1s[:n]) - np.mean(base_f1s[:n]))
            sym = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            results[mkey] = {"statistic": float(stat), "p_value": float(p), "delta_f1": delta, "symbol": sym}
            print(f"  [Stats] proposed vs {mkey:<14}: ΔF1={delta:+.4f} p={p:.4f} {sym}")
        except Exception as e:
            results[mkey] = {"p_value": float("nan"), "symbol": "err", "note": str(e)}
    return results


def save_hbi_vs_proposed_delta(hbi_pf_df: pd.DataFrame, proposed_pf_df: pd.DataFrame, out_csv: Path):
    hbi = hbi_pf_df[["family", "f1_mean"]].rename(columns={"f1_mean": "hbi_f1"})
    pro = proposed_pf_df[["family", "f1_mean"]].rename(columns={"f1_mean": "proposed_f1"})
    df = hbi.merge(pro, on="family", how="inner")
    df["delta"] = df["proposed_f1"] - df["hbi_f1"]
    df = df.sort_values("delta", ascending=False)
    df.to_csv(out_csv, index=False)


def make_table1(summary_df: pd.DataFrame, stat_results: Dict = None) -> str:
    cols = [
        ("fam_macro_f1", "Fam Macro-F1"),
        ("fam_mcc", "Fam MCC"),
        ("fam_bal_acc", "Fam Bal.Acc"),
        ("cls_macro_f1", "Cls Macro-F1"),
        ("cls_mcc", "Cls MCC"),
        ("fam_ece", "ECE↓"),
    ]
    best = {
        c: summary_df[c + "_mean"].max() if "ece" not in c else summary_df[c + "_mean"].min()
        for c, _ in cols if c + "_mean" in summary_df.columns
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Main results (homology-aware test set, mean $\pm$ std). Best per column in bold.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Model & " + " & ".join(lbl for _, lbl in cols) + r" \\",
        r"\midrule",
    ]

    for _, row in summary_df.iterrows():
        parts = [row["model_label"]]
        for col, _ in cols:
            mn = col + "_mean"
            sd = col + "_std"
            if mn not in row:
                parts.append("—")
                continue
            m = float(row[mn])
            s = float(row.get(sd, 0))
            val = f"{m:.3f} $\\pm$ {s:.3f}"
            is_best = (m >= best.get(col, m) - 1e-4) if "ece" not in col else (m <= best.get(col, m) + 1e-4)
            if is_best:
                val = r"\textbf{" + val + r"}"
            parts.append(val)
        lines.append("  " + " & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


def make_table2(fs_df: pd.DataFrame) -> str:
    if fs_df.empty:
        return "% No few-shot data"
    k_vals = sorted(fs_df["k"].unique())
    hdr = " & ".join(f"$k={k}$" for k in k_vals)
    best = {k: fs_df[fs_df["k"] == k]["f1_mean"].max() for k in k_vals}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Real few-shot family Macro-F1 (train from scratch, mean $\pm$ std).}",
        r"\label{tab:fewshot}",
        r"\begin{tabular}{l" + "c" * len(k_vals) + "}",
        r"\toprule",
        f"Model & {hdr} \\\\",
        r"\midrule",
    ]
    for mkey in fs_df["model_key"].unique():
        sub = fs_df[fs_df["model_key"] == mkey]
        parts = [MODEL_LABEL.get(mkey, mkey)]
        for k in k_vals:
            r = sub[sub["k"] == k]
            if r.empty:
                parts.append("—")
                continue
            m = float(r["f1_mean"].iloc[0])
            s = float(r["f1_std"].iloc[0])
            val = f"{m:.3f} $\\pm$ {s:.3f}"
            if m >= best[k] - 1e-4:
                val = r"\textbf{" + val + r"}"
            parts.append(val)
        lines.append("  " + " & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_table3(leakage_df: pd.DataFrame) -> str:
    if leakage_df.empty:
        return "% No leakage data"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Homology leakage. $\Delta$F1 = F1(random) $-$ F1(homology-aware).}",
        r"\label{tab:leakage}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Homology-aware F1 & Random split F1 & $\Delta$F1 & Overestimation (\%) \\",
        r"\midrule",
    ]
    for _, row in leakage_df.iterrows():
        lines.append(
            f"  {row['model_label']} & {row['homology_f1']:.3f} & "
            f"{row['random_f1']:.3f} & {row['delta_f1']:+.3f} & "
            f"{row['overestimation_pct']:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_table4(grp_df: pd.DataFrame) -> str:
    if grp_df.empty:
        return "% No group F1 data"
    groups = ["Head (n>200)", "Medium (50<n≤200)", "Tail (n≤50)"]
    short = {"Head (n>200)": "Head", "Medium (50<n≤200)": "Medium", "Tail (n≤50)": "Tail"}
    models = grp_df["model_key"].unique().tolist()
    best = {g: grp_df[grp_df["freq_group"] == g]["f1_mean"].max() for g in groups}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Family Macro-F1 stratified by training frequency.}",
        r"\label{tab:group_f1}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        f"Model & {' & '.join(short[g] for g in groups)} \\\\",
        r"\midrule",
    ]
    for mkey in models:
        sub = grp_df[grp_df["model_key"] == mkey].set_index("freq_group")
        label = MODEL_LABEL.get(mkey, mkey)
        parts = [label]
        for g in groups:
            if g not in sub.index:
                parts.append("—")
                continue
            m = float(sub.loc[g, "f1_mean"])
            s = float(sub.loc[g, "f1_std"])
            val = f"{m:.3f} $\\pm$ {s:.3f}"
            if m >= best[g] - 1e-4:
                val = r"\textbf{" + val + r"}"
            parts.append(val)
        lines.append("  " + " & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# LEAKAGE
# ──────────────────────────────────────────────────────────────────────────────

def run_leakage_analysis(tr_s, tr_f, tr_c, te_s, te_f, te_c, fam_le, cls_le, device, args, seeds):
    all_s = tr_s + te_s
    all_f = tr_f + te_f
    all_c = tr_c + te_c

    sss = StratifiedShuffleSplit(1, test_size=len(te_s) / len(all_s), random_state=42)
    tr_idx, te_idx = next(sss.split(all_s, all_f))

    r_tr_s = [all_s[i] for i in tr_idx]
    r_tr_f = [all_f[i] for i in tr_idx]
    r_tr_c = [all_c[i] for i in tr_idx]
    r_te_s = [all_s[i] for i in te_idx]
    r_te_f = [all_f[i] for i in te_idx]
    r_te_c = [all_c[i] for i in te_idx]

    print(f"  [Leakage] Random split: {len(r_tr_s)} train / {len(r_te_s)} test")

    rows = []
    model_keys = ["hbi", "cnn", "esm_frozen", "protbert", "proposed"]
    n_fam_total = len(fam_le.classes_)
    n_cls_total = len(cls_le.classes_)

    for mkey in model_keys:
        if mkey == "esm_frozen" and not ESM_OK:
            continue
        if mkey == "protbert" and not BERT_OK:
            continue

        label = MODEL_LABEL[mkey]
        for split, ts, tf, tc, vs, vf, vc in [
            ("homology", tr_s, tr_f, tr_c, te_s, te_f, te_c),
            ("random", r_tr_s, r_tr_f, r_tr_c, r_te_s, r_te_f, r_te_c),
        ]:
            f1s = []
            for seed in seeds[:2]:
                set_seed(seed)

                if mkey == "hbi":
                    fam_m, _, _ = run_hbi(ts, tf, tc, vs, vf, vc, fam_le, cls_le)

                elif mkey == "cnn":
                    m = CNNModel(n_fam_total, n_cls_total).to(device)
                    tr_ld = get_cnn_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True)
                    te_ld = get_cnn_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_model(
                        m, f"{mkey}-{split}-s{seed}", tr_ld, tr_ld, te_ld, device, max(3, args.epochs // 2), args.lr, is_cnn=True, grad_accum=args.grad_accum
                    )
                    free_gpu(m)

                elif mkey == "esm_frozen":
                    m = ESMFrozenModel(n_fam_total, n_cls_total).to(device)
                    tr_ld = get_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True)
                    te_ld = get_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_model(
                        m, f"{mkey}-{split}-s{seed}", tr_ld, tr_ld, te_ld, device, max(3, args.epochs // 2), args.lr, grad_accum=args.grad_accum
                    )
                    free_gpu(m)

                elif mkey == "protbert":
                    m = ProtBERTModel(n_fam_total, n_cls_total).to(device)
                    tr_ld = get_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True)
                    te_ld = get_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_model(
                        m, f"{mkey}-{split}-s{seed}", tr_ld, tr_ld, te_ld, device, max(3, args.epochs // 2), args.lr, grad_accum=args.grad_accum
                    )
                    free_gpu(m)

                elif mkey == "proposed":
                    m = ProposedModel(n_fam_total, n_cls_total, hidden=ESM_DIM, dropout=0.1).to(device)
                    tr_ld = get_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True, balanced=False)
                    te_ld = get_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_proposed_oldstyle(
                        model=m,
                        tr_ld=tr_ld,
                        va_ld=tr_ld,
                        te_ld=te_ld,
                        device=device,
                        epochs_stage1=max(2, args.epochs_stage1 // 2),
                        epochs_stage2=max(1, args.epochs_stage2 // 2),
                        lr_stage1=args.lr,
                        lr_stage2=args.lr_stage2,
                        lambda_class=args.lambda_class,
                        grad_accum=args.grad_accum,
                        unfreeze_last_n=args.unfreeze_last_n,
                    )
                    free_gpu(m)

                else:
                    continue

                f1s.append(fam_m["macro_f1"])

            rows.append({
                "model_key": mkey,
                "model_label": label,
                "split": split,
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            })

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    homo_df = df[df["split"] == "homology"].copy()
    rand_df = df[df["split"] == "random"].copy()

    delta_rows = []
    for mkey in df["model_key"].unique():
        sub = df[df["model_key"] == mkey]
        hf = float(sub[sub["split"] == "homology"]["f1_mean"].iloc[0])
        rf = float(sub[sub["split"] == "random"]["f1_mean"].iloc[0])
        d = rf - hf
        delta_rows.append({
            "model_key": mkey,
            "model_label": MODEL_LABEL.get(mkey, mkey),
            "homology_f1": hf,
            "random_f1": rf,
            "delta_f1": d,
            "overestimation_pct": 100 * d / (hf + 1e-8),
        })
    leakage_df = pd.DataFrame(delta_rows)
    return leakage_df, homo_df, rand_df

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--train_fasta", required=True)
    ap.add_argument("--val_fasta", required=True)
    ap.add_argument("--test_fasta", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--test_labels", required=True)
    ap.add_argument("--out", default="results/comparison_v8")

    ap.add_argument("--models", default="hbi,cnn,esm_frozen,protbert,proposed")
    ap.add_argument("--seeds", default="1,7,42")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--epochs_stage2", type=int, default=5)

    ap.add_argument("--fewshot_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)
    ap.add_argument("--unfreeze_last_n", type=int, default=2)
    ap.add_argument(
        "--lambda_class",
        type=float,
        default=0.3,
        help="Weight of the auxiliary class-level loss in the multi-task objective."
    )

    ap.add_argument("--eval_few_shot", action="store_true")
    ap.add_argument("--k_shots", default="1,5,10,20")
    ap.add_argument("--fewshot_seeds", type=int, default=3)

    ap.add_argument("--eval_leakage", action="store_true")
    ap.add_argument("--eval_uncertainty", action="store_true")
    ap.add_argument("--eval_embedding", action="store_true")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    k_shots = [int(k) for k in args.k_shots.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seeds[0])

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {torch.cuda.get_device_name(0)} · {vram_gb:.1f} GB VRAM")
    print(f"  Models: {models} | Seeds: {seeds} | Out: {out_dir}\n")

    print("[1] Veri yükleniyor...")
    tr_s, tr_f, tr_c = load_split(args.train_fasta, args.train_labels)
    va_s, va_f, va_c = load_split(args.val_fasta, args.val_labels)
    te_s, te_f, te_c = load_split(args.test_fasta, args.test_labels)

    te_ids = load_split_ids(args.test_fasta, args.test_labels)
    if len(te_ids) != len(te_s):
        print(f"  [UYARI] test ID sayısı ile test sequence sayısı farklı: ids={len(te_ids)} seqs={len(te_s)}")

    if args.debug:
        tr_s, tr_f, tr_c = tr_s[:300], tr_f[:300], tr_c[:300]
        va_s, va_f, va_c = va_s[:80], va_f[:80], va_c[:80]
        te_s, te_f, te_c = te_s[:100], te_f[:100], te_c[:100]
        te_ids = te_ids[:100]
        args.epochs = 2
        args.epochs_stage1 = 2
        args.epochs_stage2 = 1
        args.fewshot_epochs = 2
        print("  [DEBUG] küçültüldü")

    all_fam = tr_f + va_f + te_f
    all_cls = tr_c + va_c + te_c
    fam_le = LabelEncoder().fit(all_fam)
    cls_le = LabelEncoder().fit(all_cls)

    n_fam_total = len(fam_le.classes_)
    n_cls_total = len(cls_le.classes_)

    tr_fam_ids = fam_le.transform(tr_f).tolist()
    fam_counts = (np.bincount(tr_fam_ids, minlength=n_fam_total) + 1).tolist()

    print(f"  {n_fam_total} fam | {n_cls_total} cls")

    all_rows = []
    per_fam_rows = []
    per_sample_rows = []
    conf_data = {}

    print("\n[2] Full-supervised eğitim...")

    for model_key in models:
        label = MODEL_LABEL.get(model_key, model_key)
        print(f"\n  ══ {label} ══")
        free_gpu()

        if model_key == "hbi":
            fam_m, cls_m, per_fam = run_hbi(tr_s, tr_f, tr_c, te_s, te_f, te_c, fam_le, cls_le)
            for seed in seeds:
                all_rows.append({
                    "model_key": model_key,
                    "model_label": label,
                    "seed": seed,
                    **{f"fam_{k}": v for k, v in fam_m.items()},
                    **{f"cls_{k}": v for k, v in cls_m.items()},
                })
                for fi, f1v in enumerate(per_fam):
                    if fi < n_fam_total:
                        per_fam_rows.append({
                            "model_key": model_key,
                            "model_label": label,
                            "seed": seed,
                            "family": fam_le.classes_[fi],
                            "f1": f1v,
                        })
            continue

        for seed in seeds:
            set_seed(seed)
            fam_m, cls_m, per_fam = {}, {}, []

            if model_key == "cnn":
                model = CNNModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_cnn_loader(tr_s, tr_f, tr_c, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
                va_ld = get_cnn_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_cnn_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"CNN s={seed}", tr_ld, va_ld, te_ld, device, args.epochs, args.lr, is_cnn=True, grad_accum=args.grad_accum, ckpt_path=out_dir / f"ckpt_cnn_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "esm_frozen":
                if not ESM_OK:
                    print("  ESM yok, atlandı")
                    continue
                model = ESMFrozenModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"ESM-frozen s={seed}", tr_ld, va_ld, te_ld, device, args.epochs, args.lr, grad_accum=args.grad_accum, ckpt_path=out_dir / f"ckpt_esm_frozen_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "protbert":
                if not BERT_OK:
                    print("  transformers yok, atlandı")
                    continue
                model = ProtBERTModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le, args.batch_size, shuffle=True, balanced=True)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"ProtBERT s={seed}", tr_ld, va_ld, te_ld, device, args.epochs, args.lr, grad_accum=args.grad_accum, ckpt_path=out_dir / f"ckpt_protbert_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "proposed":
                if not ESM_OK:
                    print("  ESM yok, atlandı")
                    continue

                model = ProposedModel(n_fam_total, n_cls_total, hidden=ESM_DIM, dropout=0.1).to(device)
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le, args.batch_size, shuffle=True, balanced=False)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)

                fam_m, cls_m, per_fam = train_and_eval_proposed_oldstyle(
                    model=model,
                    tr_ld=tr_ld,
                    va_ld=va_ld,
                    te_ld=te_ld,
                    device=device,
                    epochs_stage1=args.epochs_stage1,
                    epochs_stage2=args.epochs_stage2,
                    lr_stage1=args.lr,
                    lr_stage2=args.lr_stage2,
                    lambda_class=args.lambda_class,
                    grad_accum=args.grad_accum,
                    unfreeze_last_n=args.unfreeze_last_n,
                    ckpt_path=out_dir / f"ckpt_proposed_s{seed}.pt",
                )

                pred_df = predict_per_sample(
                    model=model,
                    loader=te_ld,
                    device=device,
                    is_cnn=False,
                    ids=te_ids,
                    fam_le=fam_le,
                    cls_le=cls_le,
                    model_key=model_key,
                    model_label=label,
                    seed=seed,
                )
                if not pred_df.empty:
                    per_sample_rows.extend(pred_df.to_dict("records"))

                if seed == seeds[-1]:
                    all_ct2, all_cp2 = [], []
                    model.eval()
                    with torch.no_grad():
                        for seqs, _, cids in te_ld:
                            try:
                                cl, _ = model(seqs)
                                all_ct2.extend(cids.tolist())
                                all_cp2.extend(cl.argmax(-1).cpu().tolist())
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    free_gpu()
                                    continue
                                raise
                    if all_ct2:
                        conf_data["proposed"] = confusion_matrix(all_ct2, all_cp2, labels=list(range(n_cls_total))).tolist()

                    if args.eval_uncertainty:
                        try:
                            mc_res = mc_dropout_eval(model, te_ld, device, n_passes=20)
                            pd.DataFrame([{
                                "model_key": model_key,
                                "seed": seed,
                                "ece": mc_res["ece"],
                                "uncertainty_mean": float(np.mean(mc_res["uncertainty"])),
                                "macro_f1_mc": mc_res["macro_f1_mc"],
                                "mcc_mc": mc_res["mcc_mc"],
                            }]).to_csv(out_dir / f"mc_dropout_proposed_s{seed}.csv", index=False)
                            plot_calibration_curve(mc_res, out_dir, tag=f"proposed_s{seed}")
                        except Exception as e:
                            print(f"  [WARN] uncertainty skipped: {e}")

                    if args.eval_embedding:
                        try:
                            emb_data = collect_embeddings_with_meta(model, te_ld, device, mode="proposed")
                            id2cls = {i: c for i, c in enumerate(cls_le.classes_)}
                            plot_embedding_space(emb_data, id2cls, out_dir, tag=f"proposed_s{seed}")
                        except Exception as e:
                            print(f"  [WARN] embedding skipped: {e}")

                free_gpu(model)

            else:
                print(f"  Bilinmeyen model: {model_key}")
                continue

            if fam_m:
                all_rows.append({
                    "model_key": model_key,
                    "model_label": label,
                    "seed": seed,
                    **{f"fam_{k}": v for k, v in fam_m.items()},
                    **{f"cls_{k}": v for k, v in cls_m.items()},
                })

            for fi, f1v in enumerate(per_fam):
                if fi < n_fam_total:
                    per_fam_rows.append({
                        "model_key": model_key,
                        "model_label": label,
                        "seed": seed,
                        "family": fam_le.classes_[fi],
                        "f1": f1v,
                    })

    fs_rows = []
    if args.eval_few_shot:
        print("\n[3] Gerçek few-shot...")
        for model_key in models:
            label = MODEL_LABEL.get(model_key, model_key)
            for k in k_shots:
                f1s = []
                for fsi in range(args.fewshot_seeds):
                    cseed = seeds[fsi % len(seeds)] + fsi
                    set_seed(cseed)
                    free_gpu()
                    print(f"  {label:25s} k={k:2d} fsi={fsi}", end="  ")
                    f1 = real_fewshot_run(model_key, tr_s, tr_f, te_s, te_f, fam_le, cls_le, device, k, cseed, args)
                    print(f"F1={f1:.4f}")
                    f1s.append(f1)
                fs_rows.append({
                    "model_key": model_key,
                    "model_label": label,
                    "k": k,
                    "f1_mean": np.mean(f1s),
                    "f1_std": np.std(f1s),
                    "n_seeds": len(f1s),
                })

    print("\n[4] Sonuçlar derleniyor...")
    per_seed_df = pd.DataFrame(all_rows)
    per_seed_df.to_csv(out_dir / "comparison_per_seed.csv", index=False)

    if per_sample_rows:
        per_sample_df = pd.DataFrame(per_sample_rows)
        per_sample_df.to_csv(out_dir / "per_sample_predictions.csv", index=False)
        print(f"  Yazıldı: {out_dir / 'per_sample_predictions.csv'}")

    summary_df = pd.DataFrame()
    if not per_seed_df.empty:
        metric_cols = [c for c in per_seed_df.columns if c not in ("model_key", "model_label", "seed")]
        rows = []
        for mkey in per_seed_df["model_key"].unique():
            sub = per_seed_df[per_seed_df["model_key"] == mkey]
            row = {"model_key": mkey, "model_label": MODEL_LABEL.get(mkey, mkey)}
            for col in metric_cols:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    row[col + "_mean"] = sub[col].mean()
                    row[col + "_std"] = sub[col].std()
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)

    fs_df = pd.DataFrame(fs_rows) if fs_rows else pd.DataFrame()
    if not fs_df.empty:
        fs_df.to_csv(out_dir / "fewshot_results.csv", index=False)

    pf_agg = pd.DataFrame()
    grp_df = pd.DataFrame()
    if per_fam_rows:
        pf_df = pd.DataFrame(per_fam_rows)
        pf_agg = pf_df.groupby(["model_key", "model_label", "family"])["f1"].agg(f1_mean="mean", f1_std="std").reset_index()
        pf_agg.to_csv(out_dir / "per_family_f1.csv", index=False)

        grp_df = compute_group_f1(per_fam_rows, fam_le, fam_counts)
        if not grp_df.empty:
            grp_df.to_csv(out_dir / "group_f1.csv", index=False)

        if "hbi" in pf_agg["model_key"].unique() and "proposed" in pf_agg["model_key"].unique():
            hbi_pf = pf_agg[pf_agg["model_key"] == "hbi"].copy()
            prop_pf = pf_agg[pf_agg["model_key"] == "proposed"].copy()
            save_hbi_vs_proposed_delta(hbi_pf, prop_pf, out_dir / "hbi_vs_proposed_delta.csv")

    print("\n[4b] İstatistiksel testler...")
    stat_results = run_statistical_tests(per_seed_df) if not per_seed_df.empty else {}
    if stat_results:
        pd.DataFrame([{"baseline": k, **v} for k, v in stat_results.items()]).to_csv(out_dir / "statistical_tests.csv", index=False)

    leakage_df = pd.DataFrame()
    homo_df = pd.DataFrame()
    rand_df = pd.DataFrame()
    if args.eval_leakage:
        print("\n[4c] Leakage analizi...")
        leakage_df, homo_df, rand_df = run_leakage_analysis(tr_s, tr_f, tr_c, te_s, te_f, te_c, fam_le, cls_le, device, args, seeds)
        if not leakage_df.empty:
            leakage_df.to_csv(out_dir / "leakage_analysis.csv", index=False)

    print("\n[5] Grafikler...")
    if not summary_df.empty:
        plot_main_bar(summary_df, out_dir)
        plot_radar(summary_df, out_dir)
    if not fs_df.empty:
        plot_fewshot_curve(fs_df, out_dir)
    if not pf_agg.empty:
        plot_family_heatmap(pf_agg, out_dir)
    if not grp_df.empty:
        plot_group_f1(grp_df, out_dir)
    if conf_data:
        plot_confusion(conf_data, cls_le, out_dir)
    if not homo_df.empty and not rand_df.empty:
        plot_homology_leakage(homo_df, rand_df, "f1_mean", out_dir)

    print("\n[6] LaTeX tabloları...")
    if not summary_df.empty:
        (out_dir / "table1_main.tex").write_text(make_table1(summary_df, stat_results), encoding="utf-8")
    if not fs_df.empty:
        (out_dir / "table2_fewshot.tex").write_text(make_table2(fs_df), encoding="utf-8")
    if not leakage_df.empty:
        (out_dir / "table3_leakage.tex").write_text(make_table3(leakage_df), encoding="utf-8")
    if not grp_df.empty:
        (out_dir / "table4_group_f1.tex").write_text(make_table4(grp_df), encoding="utf-8")

    print(f"\n{'═' * 78}")
    print("  SONUÇ ÖZETİ")
    print(f"{'═' * 78}")
    if not summary_df.empty:
        show_cols = ["model_label", "fam_macro_f1_mean", "fam_macro_f1_std", "cls_macro_f1_mean", "fam_mcc_mean", "fam_ece_mean"]
        show = summary_df[[c for c in show_cols if c in summary_df.columns]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if stat_results:
        print("\n  Wilcoxon (proposed vs baseline):")
        for mkey, res in stat_results.items():
            print(
                f"    vs {MODEL_LABEL.get(mkey,mkey):<22}: "
                f"ΔF1={res.get('delta_f1',0):+.4f}  "
                f"p={res.get('p_value',1):.4f}  {res.get('symbol','?')}"
            )
    print(f"\n  Çıktı: {out_dir}")
    print("  CSVs : comparison_summary | per_seed | per_sample_predictions | fewshot | per_family | group_f1 | stats | leakage")
    print("  Extras: hbi_vs_proposed_delta | mc_dropout_proposed | embedding_*")
    print("  Figs : fig1_main_bar | fig2_fewshot_curve | fig3_radar | fig4_family_heatmap | fig5_group_f1 | fig6_confusion_cls | fig7_homology_leakage")
    print("  LaTeX: table1_main | table2_fewshot | table3_leakage | table4_group_f1")
    print(f"{'═' * 78}")


if __name__ == "__main__":
    main()
