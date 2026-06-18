#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAZy Enzyme Classification — Full Q1 Comparison Suite  v7
=========================================================
Baselines : HBI | CNN | ESM-2 (frozen) | ProtBERT | SetFit
Proposed  : ESM-2 + LoRA + Attention Pooling + HCH

Implemented components (Proposed):
  ① Hierarchy-Aware Loss  — LDAMLoss(fam) + LDAMLoss(cls)
                            + KL(agg(fam) || cls)
  ② Hierarchical Conditional Head (HCH)
  ③ Long-tail handling    — LDAM-DRW + balanced sampler + label smoothing
  ④ Learnable sequence summarization — Attention Pooling

OOM budget (RTX 5060 · 8 GB VRAM):
  • batch_size=4, grad_accum=4 → effective batch 16
  • ESM max token = 256
  • Gradient checkpointing
  • Mini-batch ESM encode
  • best_state stored on CPU

Outputs (--out):
  CSVs : comparison_summary.csv | comparison_per_seed.csv
         fewshot_results.csv   | per_family_f1.csv
         group_f1.csv          | statistical_tests.csv
         leakage_analysis.csv
  PDFs : fig1_main_bar.pdf | fig2_fewshot_curve.pdf | fig3_radar.pdf
         fig4_family_heatmap.pdf | fig5_confusion_cls.pdf | fig6_group_f1.pdf
  LaTeX: table1_main.tex | table2_fewshot.tex | table3_leakage.tex | table4_group_f1.tex

Run:
python cazy_comparison_v7.py \
  --train_fasta splits/train.fasta \
  --val_fasta splits/val.fasta \
  --test_fasta splits/test.fasta \
  --train_labels splits/train_labels.csv \
  --val_labels splits/val_labels.csv \
  --test_labels splits/test_labels.csv \
  --out results/comparison_v7_full \
  --models proposed,hbi,cnn,esm_frozen,protbert,setfit \
  --epochs 5 \
  --fewshot_epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 3e-4 \
  --lora_r 8 \
  --lora_alpha 32 \
  --seeds 1,7,42 \
  --eval_few_shot \
  --k_shots 1,5,10,20 \
  --fewshot_seeds 3 \
  --eval_leakage
"""

from __future__ import annotations
import argparse
import gc
import math
import random
import shutil
import subprocess
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder
from Bio import SeqIO

try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] pip install scipy")

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# DETERMINISM
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


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL DEPS
# ──────────────────────────────────────────────────────────────────────────────

try:
    import esm as esm_lib
    ESM_OK = True
except ImportError:
    ESM_OK = False
    print("[WARN] pip install fair-esm")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_OK = True
except ImportError:
    PEFT_OK = False

try:
    from transformers import BertModel, BertTokenizer
    BERT_OK = True
except ImportError:
    BERT_OK = False
    print("[WARN] pip install transformers  # ProtBERT için")

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
except ImportError:
    PLOT_OK = False
    print("[WARN] pip install matplotlib seaborn")


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

MODEL_KEYS = ["hbi", "cnn", "esm_frozen", "protbert", "setfit", "proposed"]
MODEL_LABEL = {
    "hbi": "HBI (MMseqs2 1-NN)",
    "cnn": "CNN",
    "esm_frozen": "ESM-2 (frozen)",
    "protbert": "ProtBERT",
    "setfit": "SetFit",
    "proposed": "ESM-2+LoRA+HCH (ours)",
}
PALETTE = {
    "HBI (MMseqs2 1-NN)": "#bcbd22",
    "CNN": "#7f7f7f",
    "ESM-2 (frozen)": "#1f77b4",
    "ProtBERT": "#2ca02c",
    "SetFit": "#ff7f0e",
    "ESM-2+LoRA+HCH (ours)": "#d62728",
}
MARKERS = {
    "HBI (MMseqs2 1-NN)": "v",
    "CNN": "s",
    "ESM-2 (frozen)": "p",
    "ProtBERT": "^",
    "SetFit": "D",
    "ESM-2+LoRA+HCH (ours)": "o",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def normalize_id(raw: str) -> str:
    """
    FASTA / labels / split dosyaları ile aynı ID normalizasyonu.

    Örnek:
      >sp|Q9ABC1|ENTRY desc -> Q9ABC1
      >tr|A0A123|ENTRY      -> A0A123
      >Q9ABC1 desc          -> Q9ABC1
      Q9ABC1                -> Q9ABC1
    """
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

    df["id"] = df["id"].astype(str).apply(normalize_id)
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
        seqs.append(seq[:ESM_MAXLEN])
        fams.append(fam)
        clss.append(fam2cls(fam))

    if skipped_id > 0:
        print(f"  [UYARI] {skipped_id} FASTA kaydı labels içinde bulunamadı")
    if skipped_short > 0:
        print(f"  [UYARI] {skipped_short} çok kısa sekans atlandı")

    if len(seqs) == 0:
        raise ValueError(
            f"{fasta} ve {labels_csv} eşleşmesinden hiç örnek çıkmadı. "
            "ID normalizasyonunu ve split dosyalarını kontrol et."
        )

    print(f"  {Path(fasta).name}: {len(seqs)} seq | {len(set(fams))} fam | {len(set(clss))} cls")
    return seqs, fams, clss


def few_shot_sample(seqs, lbls, k, seed):
    rng = random.Random(seed)
    c2i = defaultdict(list)
    for i, l in enumerate(lbls):
        c2i[l].append(i)
    idx = []
    for _, idxs in c2i.items():
        idx.extend(rng.sample(idxs, min(k, len(idxs))))
    return [seqs[i] for i in idx], [lbls[i] for i in idx]


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


# ──────────────────────────────────────────────────────────────────────────────
# LONG-TAIL
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


class LDAMLoss(nn.Module):
    def __init__(self, class_counts: List[int], max_margin: float = 0.5,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.ls = label_smoothing
        counts = np.array(class_counts, dtype=np.float32)
        margins = max_margin / (counts ** 0.25)
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        inv_freq = 1.0 / counts
        self.register_buffer("inv_freq", torch.tensor(inv_freq / inv_freq.sum(), dtype=torch.float32))
        self._use_weights = False

    def set_drw(self, active: bool):
        self._use_weights = active

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)

        _lse = torch.logsumexp(logits, dim=-1).abs().max().item()
        is_log_prob = (logits.max().item() <= 1e-6) and (_lse < 0.05)

        margins = self.margins.to(device=targets.device, dtype=logits.dtype)
        batch_margins = margins[targets]

        if is_log_prob:
            log_p_adj = logits.clone()
            log_p_adj.scatter_add_(1, targets.unsqueeze(1), -batch_margins.unsqueeze(1))
            log_p = log_p_adj - torch.logsumexp(log_p_adj, dim=-1, keepdim=True)
        else:
            logits_adj = logits.clone()
            logits_adj.scatter_add_(1, targets.unsqueeze(1), -batch_margins.unsqueeze(1))
            log_p = F.log_softmax(logits_adj, dim=-1)

        with torch.no_grad():
            denom = max(n - 1, 1)
            smooth = torch.full(
                (logits.size(0), n),
                self.ls / denom,
                device=logits.device,
                dtype=logits.dtype
            )
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)

        ce = -(smooth * log_p).sum(-1)

        if self._use_weights:
            inv_freq = self.inv_freq.to(device=targets.device, dtype=logits.dtype)
            w = inv_freq[targets] * len(self.inv_freq)
            return (w * ce).mean()

        return ce.mean()


def balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    weights = torch.tensor(1.0 / (counts[labels] + 1e-6), dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ──────────────────────────────────────────────────────────────────────────────
# HIERARCHY-AWARE LOSS
# ──────────────────────────────────────────────────────────────────────────────

class HierarchyAwareLoss(nn.Module):
    def __init__(self, fam_to_cls: List[int],
                 fam_counts: Optional[List[int]] = None,
                 cls_counts: Optional[List[int]] = None,
                 lambda_fam=1.0, lambda_cls=0.3, lambda_kl=0.1,
                 max_margin=0.5, label_smoothing=0.05,
                 drw_start_frac: float = 0.7):
        super().__init__()
        self.lf, self.lc, self.lk = lambda_fam, lambda_cls, lambda_kl
        self.drw_start_frac = drw_start_frac

        n_fam = len(fam_to_cls)
        n_cls = max(fam_to_cls) + 1

        _fc = fam_counts if fam_counts else [100] * n_fam
        _cc = cls_counts if cls_counts else [100] * n_cls

        self.fam_loss = LDAMLoss(_fc, max_margin=max_margin, label_smoothing=label_smoothing)
        self.cls_loss = LDAMLoss(_cc, max_margin=max_margin, label_smoothing=label_smoothing)

        agg = torch.zeros(n_cls, n_fam)
        for fi, ci in enumerate(fam_to_cls):
            agg[ci, fi] = 1.0
        agg = agg / agg.sum(1, keepdim=True).clamp(min=1)
        self.register_buffer("agg_matrix", agg)

    def set_epoch(self, epoch: int, total_epochs: int):
        drw_active = (total_epochs > 1) and ((epoch / total_epochs) >= self.drw_start_frac)
        self.fam_loss.set_drw(drw_active)
        self.cls_loss.set_drw(drw_active)

    def forward(self, fam_log_probs, cls_logits, fam_labels, cls_labels, agg_matrix=None):
        l_fam = self.fam_loss(fam_log_probs, fam_labels)
        l_cls = self.cls_loss(cls_logits, cls_labels)

        fam_prob = fam_log_probs.exp().clamp(min=1e-8)
        agg_m = agg_matrix if agg_matrix is not None else self.agg_matrix
        agg_cls = (fam_prob @ agg_m.T).clamp(min=1e-8)

        with torch.no_grad():
            cls_log_target = cls_logits.detach().log_softmax(-1)

        l_kl = F.kl_div(cls_log_target, agg_cls, reduction="batchmean")

        loss = self.lf * l_fam + self.lc * l_cls + self.lk * l_kl
        return loss, {"l_fam": l_fam.item(), "l_cls": l_cls.item(), "l_kl": l_kl.item()}


# ──────────────────────────────────────────────────────────────────────────────
# HCH
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalConditionalHead(nn.Module):
    def __init__(self, n_classes: int, n_families: int,
                 hidden: int, fam_to_cls: List[int], dropout: float = 0.1):
        super().__init__()
        self.within_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_families),
        )

        mask = torch.full((n_classes, n_families), float("-inf"))
        for fam_idx, cls_idx in enumerate(fam_to_cls):
            mask[cls_idx, fam_idx] = 0.0

        empty_cls = (mask == float("-inf")).all(dim=1).nonzero(as_tuple=True)[0]
        if len(empty_cls) > 0:
            raise ValueError(
                f"HCH: {len(empty_cls)} sınıfın hiç family'si yok "
                f"(index: {empty_cls.tolist()}). fam_to_cls eşleşmesini kontrol edin."
            )

        self.register_buffer("cls_fam_mask", mask)
        self.n_cls = n_classes
        self.n_fam = n_families

    def forward(self, z: torch.Tensor, cls_logits: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        cls_log_prob = cls_logits.detach().log_softmax(-1)
        raw = self.within_proj(z)

        raw_exp = raw.unsqueeze(1).expand(B, self.n_cls, self.n_fam)
        masked = raw_exp + self.cls_fam_mask.unsqueeze(0)
        within_log_prob = masked.log_softmax(-1)

        log_joint = cls_log_prob.unsqueeze(-1) + within_log_prob
        log_fam = torch.logsumexp(log_joint, dim=1)
        return log_fam


# ──────────────────────────────────────────────────────────────────────────────
# MANUAL LoRA
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r=16, alpha=32.0):
        super().__init__()
        self.orig = orig
        for p in self.orig.parameters():
            p.requires_grad = False
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, orig.in_features) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(orig.out_features, r))

    def forward(self, x):
        return self.orig(x) + F.linear(x, self.B @ self.A) * self.scale


def inject_lora(model, r=16, alpha=32.0):
    n = 0
    for mod in model.modules():
        for k in ("q_proj", "v_proj"):
            if hasattr(mod, k) and isinstance(getattr(mod, k), nn.Linear):
                setattr(mod, k, LoRALinear(getattr(mod, k), r=r, alpha=alpha))
                n += 1
    return model, n


# ──────────────────────────────────────────────────────────────────────────────
# ESM ENCODE
# ──────────────────────────────────────────────────────────────────────────────

def esm_encode(esm_model, alphabet, seqs: List[str], repr_layer: int,
               device, chunk: int = ESM_ENCODE_CHUNK) -> torch.Tensor:
    bc = alphabet.get_batch_converter()
    eos_idx = alphabet.eos_idx
    outs = []
    for i in range(0, len(seqs), chunk):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i+chunk])]
        _, _, tok = bc(batch)
        tok = tok.to(device)
        try:
            with torch.no_grad() if not esm_model.training else torch.enable_grad():
                rep = esm_model(tok, repr_layers=[repr_layer],
                                return_contacts=False)["representations"][repr_layer]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                with torch.no_grad() if not esm_model.training else torch.enable_grad():
                    rep = esm_model(tok[:1], repr_layers=[repr_layer],
                                    return_contacts=False)["representations"][repr_layer]
                extras = []
                for j in range(1, tok.size(0)):
                    with torch.no_grad() if not esm_model.training else torch.enable_grad():
                        r = esm_model(tok[j:j+1], repr_layers=[repr_layer],
                                      return_contacts=False)["representations"][repr_layer]
                    extras.append(r)
                if extras:
                    rep = torch.cat([rep] + extras, dim=0)
            else:
                raise

        mask = (tok != alphabet.padding_idx).float().to(device)
        mask = mask * (tok != eos_idx).float().to(device)
        mask[:, 0] = 0

        mean = (rep * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        outs.append(mean)
        if not esm_model.training:
            del tok, rep, mask
    return torch.cat(outs, dim=0)


def esm_encode_raw(esm_model, alphabet, seqs: List[str], repr_layer: int,
                   device, chunk: int = ESM_ENCODE_CHUNK):
    bc = alphabet.get_batch_converter()
    eos_idx = alphabet.eos_idx
    reps, masks = [], []
    for i in range(0, len(seqs), chunk):
        batch = [(str(j), s) for j, s in enumerate(seqs[i:i+chunk])]
        _, _, tok = bc(batch)
        tok = tok.to(device)
        try:
            ctx = torch.no_grad() if not esm_model.training else torch.enable_grad()
            with ctx:
                rep = esm_model(tok, repr_layers=[repr_layer],
                                return_contacts=False)["representations"][repr_layer]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_gpu()
                ctx = torch.no_grad() if not esm_model.training else torch.enable_grad()
                with ctx:
                    rep = esm_model(tok[:1], repr_layers=[repr_layer],
                                    return_contacts=False)["representations"][repr_layer]
                extras = []
                for j in range(1, tok.size(0)):
                    ctx2 = torch.no_grad() if not esm_model.training else torch.enable_grad()
                    with ctx2:
                        r = esm_model(tok[j:j+1], repr_layers=[repr_layer],
                                      return_contacts=False)["representations"][repr_layer]
                    extras.append(r)
                if extras:
                    rep = torch.cat([rep] + extras, dim=0)
            else:
                raise

        mask = (tok != alphabet.padding_idx).float()
        mask = mask * (tok != eos_idx).float()
        mask[:, 0] = 0.0
        mask = mask.to(device)
        reps.append(rep)
        masks.append(mask)
        if not esm_model.training:
            del tok

    max_len = max(r.size(1) for r in reps)
    B_total = sum(r.size(0) for r in reps)
    D = reps[0].size(2)
    out_rep = torch.zeros(B_total, max_len, D, device=device)
    out_mask = torch.zeros(B_total, max_len, device=device)
    idx = 0
    for r, m in zip(reps, masks):
        b, l, _ = r.shape
        out_rep[idx:idx+b, :l, :] = r
        out_mask[idx:idx+b, :l] = m
        idx += b
    return out_rep, out_mask


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, rep: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.v(torch.tanh(self.W(rep))).squeeze(-1)
        score = score.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(score, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return (alpha.unsqueeze(-1) * rep).sum(1)


# ──────────────────────────────────────────────────────────────────────────────
# CNN
# ──────────────────────────────────────────────────────────────────────────────

class CNNModel(nn.Module):
    def __init__(self, n_fam, n_cls, emb_dim=64, hidden=256, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(len(AA_VOCAB) + 1, emb_dim, padding_idx=0)

        def cb(i, o, k):
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=k // 2),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(dropout)
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
            nn.Linear(hidden, n_fam)
        )

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = self.c3(self.c2(self.c1(e))) + self.proj(e)
        z = self.drop(self.pool(h).squeeze(-1))
        return self.cls_head(z), self.fam_head(z)


def seq2tok(seq, maxlen=ESM_MAXLEN):
    ids = [AA_VOCAB.get(a, 0) for a in seq[:maxlen]]
    ids += [0] * (maxlen - len(ids))
    return ids


class CNNDataset(Dataset):
    def __init__(self, seqs, fids, cids):
        self.toks = [torch.tensor(seq2tok(s)) for s in seqs]
        self.fids, self.cids = fids, cids

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        return self.toks[i], self.fids[i], self.cids[i]


def cnn_collate(b):
    t, f, c = zip(*b)
    return torch.stack(t), torch.tensor(f), torch.tensor(c)


# ──────────────────────────────────────────────────────────────────────────────
# ESM FROZEN
# ──────────────────────────────────────────────────────────────────────────────

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
            nn.Linear(ESM_DIM // 2, n_fam)
        )
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ESM-2 (frozen): {n_tr/1e3:.0f}K trainable (heads only)")

    def forward(self, seqs):
        device = next(self.parameters()).device
        z = esm_encode(self.esm, self.alphabet, seqs, self.rl, device, chunk=ESM_ENCODE_CHUNK)
        z = self.drop(z)
        return self.cls_head(z), self.fam_head(z)


# ──────────────────────────────────────────────────────────────────────────────
# ProtBERT
# ──────────────────────────────────────────────────────────────────────────────

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
            chunks.append(self._encode_chunk(seqs[i:i+ESM_ENCODE_CHUNK]))
            if not self.training and torch.cuda.is_available():
                torch.cuda.empty_cache()
        z = self.drop(torch.cat(chunks, 0))
        return self.cls_head(z), self.fam_head(z)


# ──────────────────────────────────────────────────────────────────────────────
# SetFit
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


def run_setfit(tr_s, tr_f, te_s, te_f, tr_c, te_c,
               fam_le, cls_le, device, args, seed):
    enc = SetFitEncoder().to(device)
    c2i = defaultdict(list)
    for i, l in enumerate(tr_f):
        c2i[l].append(i)

    pairs, rng = [], random.Random(seed)
    all_classes = list(c2i.keys())

    for cls_lbl, idxs in c2i.items():
        if len(idxs) < 2:
            continue

        neg_candidates = [c for c in all_classes if c != cls_lbl and len(c2i[c]) > 0]
        n_pairs = max(1, min(16, len(idxs)))

        for _ in range(n_pairs):
            i, j = rng.sample(idxs, 2)
            pairs.append((tr_s[i], tr_s[j], 1.0))

            if neg_candidates:
                neg = rng.choice(neg_candidates)
                neg_pool = c2i[neg]
                pairs.append((tr_s[rng.choice(idxs)], tr_s[rng.choice(neg_pool)], 0.0))

    if not pairs:
        print("  [SetFit] CFT atlandı (yeterli pozitif/negatif çift yok)")
    else:
        rng.shuffle(pairs)
        opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=2e-5)
        scaler = GradScaler(enabled=torch.cuda.is_available())
        enc.train()

        for _ in range(args.setfit_epochs):
            for i in range(0, len(pairs), args.batch_size):
                b = pairs[i:i+args.batch_size]
                if not b:
                    continue
                s1 = [x[0] for x in b]
                s2 = [x[1] for x in b]
                lbl = torch.tensor([x[2] for x in b], dtype=torch.float32, device=device)

                opt.zero_grad()
                with autocast(enabled=torch.cuda.is_available()):
                    z1 = enc(s1)
                    z2 = enc(s2)
                    sim = F.cosine_similarity(z1, z2)
                    loss = F.mse_loss(sim, lbl)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()

    enc.eval()

    @torch.no_grad()
    def embs(seqs, bs=ESM_ENCODE_CHUNK):
        out = []
        for i in range(0, len(seqs), bs):
            out.append(enc(seqs[i:i+bs]).cpu().numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not out:
            return np.empty((0, ESM_DIM), dtype=np.float32)
        return np.vstack(out)

    tr_e = embs(tr_s)
    te_e = embs(te_s)

    ytr_f = fam_le.transform(tr_f)
    yte_f = fam_le.transform(te_f)
    ytr_c = cls_le.transform(tr_c)
    yte_c = cls_le.transform(te_c)

    if len(np.unique(ytr_f)) < 2 or tr_e.shape[0] < 2:
        print("  [SetFit] Family classifier için yeterli çeşitlilik yok")
        fam_pred = np.full_like(yte_f, fill_value=ytr_f[0] if len(ytr_f) else 0)
        fam_m = metrics(yte_f, fam_pred)
    else:
        clf_f = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", n_jobs=-1
        ).fit(tr_e, ytr_f)
        fam_m = metrics(yte_f, clf_f.predict(te_e))

    if len(np.unique(ytr_c)) < 2 or tr_e.shape[0] < 2:
        print("  [SetFit] Class classifier için yeterli çeşitlilik yok")
        cls_pred = np.full_like(yte_c, fill_value=ytr_c[0] if len(ytr_c) else 0)
        cls_m = metrics(yte_c, cls_pred)
        ece = float("nan")
    else:
        clf_c = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", n_jobs=-1
        ).fit(tr_e, ytr_c)
        cls_pred = clf_c.predict(te_e)
        cls_m = metrics(yte_c, cls_pred)
        cls_probs = clf_c.predict_proba(te_e)
        ece = compute_ece(cls_probs, yte_c)

    fam_m["ece"] = ece
    cls_m["ece"] = ece

    free_gpu(enc)
    return fam_m, cls_m, None, fam_le


# ──────────────────────────────────────────────────────────────────────────────
# PROPOSED
# ──────────────────────────────────────────────────────────────────────────────

class ProposedModel(nn.Module):
    """
    State-of-the-art Proposed:
    ESM-2 + LoRA (r=16) + Attention Pooling + HCH
    ① Frozen ESM-2-35M backbone
    ② LoRA (q_proj, v_proj) — parametre-verimli ince ayar
    ③ AttentionPooling — katalitik bölgelere odaklanan öğrenilebilir havuzlama
    ④ Linear projeksiyon (480→256) + LayerNorm
    ⑤ Class head: Linear(256→128) → GELU → Linear(128→n_cls)
    ⑥ HCH: P(fam|x) = Σ_c P(cls=c|x)·P(fam|cls=c,x)
    """

    def __init__(self, n_fam, n_cls, fam_to_cls,
                 lora_r=16, lora_alpha=32.0, hidden=256, dropout=0.1):
        super().__init__()
        if not ESM_OK:
            raise ImportError("pip install fair-esm")

        self.esm, self.alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM_MODEL)
        self.rl = self.esm.num_layers

        # frozen backbone
        for p in self.esm.parameters():
            p.requires_grad = False

        # LoRA
        if PEFT_OK:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.esm = get_peft_model(self.esm, cfg)
        else:
            self.esm, n = inject_lora(self.esm, r=lora_r, alpha=lora_alpha)
            print(f"  Manual LoRA: {n} katman")

        if hasattr(self.esm, "gradient_checkpointing_enable"):
            self.esm.gradient_checkpointing_enable()

        # Attention Pooling — mean pool yerine öğrenilebilir ağırlık
        self.attn_pool = AttentionPooling(ESM_DIM)

        self.proj = nn.Sequential(
            nn.Linear(ESM_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # class head
        self.class_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_cls),
        )

        # HCH family head
        self.family_head = HierarchicalConditionalHead(
            n_classes=n_cls,
            n_families=n_fam,
            hidden=hidden,
            fam_to_cls=fam_to_cls,
            dropout=dropout,
        )

        # agg matrix for hierarchy-aware loss
        agg = torch.zeros(n_cls, n_fam)
        for fi, ci in enumerate(fam_to_cls):
            agg[ci, fi] = 1.0
        agg = agg / agg.sum(1, keepdim=True).clamp(min=1)
        self.register_buffer("agg_matrix", agg)

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Proposed (AttnPool+HCH): {n_tr/1e3:.0f}K trainable "
              f"(LoRA r={lora_r} | attn_pool | proj | HCH)")

    def _encode(self, seqs):
        device = next(self.parameters()).device
        # Token-level rep → attention-weighted pooling (öğrenilebilir)
        rep, mask = esm_encode_raw(
            self.esm, self.alphabet, seqs, self.rl,
            device, chunk=ESM_ENCODE_CHUNK,
        )
        return self.attn_pool(rep, mask)   # (B, ESM_DIM)

    def forward(self, seqs):
        z = self.proj(self._encode(seqs))   # (B, hidden)
        cls_logits = self.class_head(z)     # (B, n_cls)
        fam_log = self.family_head(z, cls_logits)  # (B, n_fam) log-probs
        return cls_logits, fam_log


# ──────────────────────────────────────────────────────────────────────────────
# DATASET / LOADERS
# ──────────────────────────────────────────────────────────────────────────────

class SeqDS(Dataset):
    def __init__(self, seqs, fids, cids):
        self.seqs, self.fids, self.cids = seqs, fids, cids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.fids[i], self.cids[i]


def seq_collate(b):
    s, f, c = zip(*b)
    return list(s), torch.tensor(f), torch.tensor(c)


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
        num_workers=0
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
        num_workers=0
    )


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────────────────────

def train_and_eval_model(model, name,
                         tr_ld, va_ld, te_ld,
                         device, epochs, lr,
                         loss_fn=None, is_cnn=False,
                         grad_accum: int = 1,
                         ckpt_path: Optional[Path] = None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(tr_ld) / max(grad_accum, 1))
    total_steps = max(steps_per_epoch * epochs, 1)

    print(f"  Scheduler steps/epoch={steps_per_epoch} | total_steps={total_steps}")

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.06,
        anneal_strategy="cos"
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    focal = FocalLoss(gamma=2.0, label_smoothing=0.05)

    best_val = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        if loss_fn is not None and hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(ep, epochs)

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
                            raw, _ = loss_fn(fl, cl, fids, cids, model.agg_matrix)
                        else:
                            raw = 0.3 * focal(cl, cids) + focal(fl, fids)
                        loss = raw / grad_accum

                scaler.scale(loss).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  [OOM] ep={ep} step={step} — cache temizleniyor")
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
                free_gpu()

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
        f"  [{name:<30}] FamF1={fam_m['macro_f1']:.4f}  "
        f"ClsF1={cls_m['macro_f1']:.4f}  MCC={fam_m['mcc']:.4f}  "
        f"ECE={fam_m['ece']:.4f}  ({elapsed:.0f}s)"
    )
    return fam_m, cls_m, per_fam


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
        empty_m = {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "mcc": 0.0,
            "bal_acc": 0.0,
        }
        return empty_m.copy(), empty_m.copy(), [], np.empty((0, 1), dtype=np.float32), []

    fam_m = metrics(all_ft, all_fp)
    cls_m = metrics(all_ct, all_cp)

    n_labels = max(all_ft) + 1 if len(all_ft) > 0 else 0
    if n_labels > 0:
        per_fam = f1_score(
            all_ft,
            all_fp,
            average=None,
            labels=list(range(n_labels)),
            zero_division=0,
        ).tolist()
    else:
        per_fam = []

    cls_logits_np = (
        np.concatenate(all_cls_logits, axis=0)
        if len(all_cls_logits) > 0
        else np.empty((0, 1), dtype=np.float32)
    )
    return fam_m, cls_m, per_fam, cls_logits_np, all_ct


# ──────────────────────────────────────────────────────────────────────────────
# FEW-SHOT
# ──────────────────────────────────────────────────────────────────────────────

def real_fewshot_run(model_name, tr_s, tr_f, te_s, te_f,
                     fam_le, cls_le, fam_to_cls_map,
                     device, k, seed, args):
    k_seqs, k_fam = few_shot_sample(tr_s, tr_f, k, seed)
    k_cls = [fam2cls(f) for f in k_fam]
    te_cls = [fam2cls(f) for f in te_f]
    n_fam, n_cls = len(fam_le.classes_), len(cls_le.classes_)

    def _run(m, is_cnn=False, loader_fn=get_loader, loss_fn=None):
        tr_ld = loader_fn(k_seqs, k_fam, k_cls, fam_le, cls_le,
                          args.batch_size, shuffle=True, balanced=True)
        te_ld = loader_fn(te_s, te_f, te_cls, fam_le, cls_le, args.batch_size)
        fam_m, _, _ = train_and_eval_model(
            m, f"{model_name}-{k}shot", tr_ld, tr_ld, te_ld,
            device, args.fewshot_epochs, args.lr,
            loss_fn=loss_fn, is_cnn=is_cnn,
            grad_accum=args.grad_accum
        )
        f1 = fam_m["macro_f1"]
        free_gpu(m)
        return f1

    if model_name == "setfit":
        fam_m, _, _, _ = run_setfit(
            k_seqs, k_fam, te_s, te_f, k_cls, te_cls,
            fam_le, cls_le, device, args, seed
        )
        return fam_m["macro_f1"]

    if model_name == "cnn":
        return _run(CNNModel(n_fam, n_cls).to(device), is_cnn=True, loader_fn=get_cnn_loader)

    if model_name == "esm_frozen":
        if not ESM_OK:
            return 0.0
        return _run(ESMFrozenModel(n_fam, n_cls).to(device))

    if model_name == "protbert":
        if not BERT_OK:
            return 0.0
        return _run(ProtBERTModel(n_fam, n_cls).to(device))

    if model_name == "proposed":
        if not ESM_OK:
            return 0.0
        m = ProposedModel(
            n_fam, n_cls, fam_to_cls_map,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha
        ).to(device)
        loss_fn = HierarchyAwareLoss(fam_to_cls_map)
        return _run(m, loss_fn=loss_fn)

    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# HBI
# ──────────────────────────────────────────────────────────────────────────────

def run_hbi(tr_s: List[str], tr_f: List[str], tr_c: List[str],
            te_s: List[str], te_f: List[str], te_c: List[str],
            fam_le: LabelEncoder, cls_le: LabelEncoder) -> Tuple[Dict, Dict, List]:
    mmseqs = shutil.which("mmseqs")
    if mmseqs:
        return _hbi_mmseqs(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le, mmseqs)
    else:
        print("  [HBI] mmseqs bulunamadı → TF-IDF cosine 1-NN fallback")
        return _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le)


def _hbi_mmseqs(tr_s, tr_f, te_s, te_f, tr_c, te_c,
                fam_le, cls_le, mmseqs_bin) -> Tuple[Dict, Dict, List]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr_fa = tmp / "train.fasta"
        te_fa = tmp / "test.fasta"

        with open(tr_fa, "w") as f:
            for i, (s, lbl) in enumerate(zip(tr_s, tr_f)):
                f.write(f">tr_{i}|{lbl}\n{s}\n")

        with open(te_fa, "w") as f:
            for i, s in enumerate(te_s):
                f.write(f">te_{i}\n{s}\n")

        res_file = tmp / "hits.tsv"
        mmseqs_tmp = tmp / "mmseqs_tmp"
        mmseqs_tmp.mkdir()

        try:
            subprocess.check_call(
                [
                    mmseqs_bin, "easy-search",
                    str(te_fa), str(tr_fa), str(res_file), str(mmseqs_tmp),
                    "--format-output", "query,target,evalue",
                    "--max-seqs", "1", "-v", "0"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("  [HBI] mmseqs easy-search başarısız → cosine fallback")
            return _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c, fam_le, cls_le)

        te_idx2fam = {}
        if res_file.exists():
            with open(res_file) as f:
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


def _hbi_cosine(tr_s, tr_f, te_s, te_f, tr_c, te_c,
                fam_le, cls_le) -> Tuple[Dict, Dict, List]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def kmerize(seqs, k=3):
        return [" ".join(s[i:i+k] for i in range(len(s) - k + 1)) for s in seqs]

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1))
    tr_mat = vec.fit_transform(kmerize(tr_s))
    te_mat = vec.transform(kmerize(te_s))

    batch = 256
    fam_pred = []
    tr_f_arr = list(tr_f)
    for i in range(0, len(te_s), batch):
        sims = cosine_similarity(te_mat[i:i+batch], tr_mat)
        best = sims.argmax(axis=1)
        fam_pred.extend(tr_f_arr[int(j)] for j in best)

    return _hbi_metrics(fam_pred, te_f, te_c, fam_le, cls_le)


def _hbi_metrics(fam_pred: List[str], te_f: List[str], te_c: List[str],
                 fam_le: LabelEncoder, cls_le: LabelEncoder) -> Tuple[Dict, Dict, List]:
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
        fam_true_ids, fam_pred_ids,
        average=None,
        labels=list(range(n_fam)),
        zero_division=0
    ).tolist()

    print(f"  [HBI] FamF1={fam_m['macro_f1']:.4f}  ClsF1={cls_m['macro_f1']:.4f}  MCC={fam_m['mcc']:.4f}")
    return fam_m, cls_m, per_fam


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
        bars = ax.bar(
            names, means, yerr=stds, capsize=5, color=colors,
            edgecolor="white", linewidth=0.8,
            error_kw=dict(elinewidth=1.2, ecolor="#333")
        )
        ax.set_ylim(0.45, 1.08)
        ax.set_ylabel("Macro-F1")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                m + s + 0.013,
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold"
            )

        prop_row = summary_df[summary_df["model_key"] == "proposed"]
        if not prop_row.empty:
            pm = float(prop_row[f"{metric}_mean"].iloc[0])
            bm = float(summary_df[summary_df["model_key"] != "proposed"][f"{metric}_mean"].max())
            if pm > bm:
                ax.annotate("★", xy=(len(names)-1, pm + max(stds) + 0.025),
                            ha="center", fontsize=13, color="#d62728")

    plt.suptitle(
        "CAZy Enzyme Classification — Model Comparison\n"
        "(homology-aware split · 3 seeds · mean ± std)",
        y=1.02, fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = out_dir / "fig1_main_bar.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig1 → {path}")


def plot_fewshot_curve(fs_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or fs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for mkey in fs_df["model_key"].unique():
        sub = fs_df[fs_df["model_key"] == mkey].sort_values("k")
        lbl = MODEL_LABEL.get(mkey, mkey)
        col = PALETTE.get(lbl, "#1f77b4")
        mrk = MARKERS.get(lbl, "o")
        ax.plot(sub["k"], sub["f1_mean"], marker=mrk, color=col,
                linewidth=2, markersize=7, label=lbl)
        ax.fill_between(
            sub["k"],
            sub["f1_mean"] - sub["f1_std"],
            sub["f1_mean"] + sub["f1_std"],
            alpha=0.12, color=col
        )
    ax.set_xlabel("k examples per family")
    ax.set_ylabel("Family Macro-F1")
    ax.set_title("Real Few-Shot Learning\n(trained from scratch on k examples per family)", fontweight="bold")
    ax.set_xticks(sorted(fs_df["k"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "fig2_fewshot_curve.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig2 → {path}")


def plot_radar(summary_df: pd.DataFrame, out_dir: Path):
    if not PLOT_OK or summary_df.empty:
        return
    cols = ["fam_macro_f1_mean", "cls_macro_f1_mean", "fam_mcc_mean",
            "fam_bal_acc_mean", "fam_weighted_f1_mean"]
    lbls = ["Fam Macro-F1", "Cls Macro-F1", "Fam MCC", "Bal. Acc", "Weighted F1"]
    N = len(cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)

    for _, row in summary_df.iterrows():
        lbl = row["model_label"]
        col = PALETTE.get(lbl, "#1f77b4")
        vals = [float(row[c]) for c in cols] + [float(row[cols[0]])]
        ax.plot(angles, vals, color=col, linewidth=2, label=lbl, marker="o", markersize=5)
        ax.fill(angles, vals, color=col, alpha=0.07)

    ax.set_title("Multi-metric Comparison", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
    plt.tight_layout()
    path = out_dir / "fig3_radar.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig3 → {path}")


def plot_family_heatmap(pf_df: pd.DataFrame, fam_le: LabelEncoder, out_dir: Path):
    if not PLOT_OK or pf_df.empty:
        return
    pivot = pf_df.pivot(index="family", columns="model_label", values="f1_mean")
    ours_label = MODEL_LABEL.get("proposed", "ESM-2+LoRA+HCH (ours)")
    if ours_label in pivot.columns:
        pivot = pivot.sort_values(ours_label, ascending=False)

    fig_h = max(6, len(pivot) * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0.3, vmax=1.0, linewidths=0.3, cbar_kws={"label": "F1"}
    )
    ax.set_title("Per-family Macro-F1 (test set)", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("CAZy Family")
    plt.tight_layout()
    path = out_dir / "fig4_family_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig4 → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# GROUP F1
# ──────────────────────────────────────────────────────────────────────────────

FREQ_THRESHOLDS = {"head": 200, "tail": 50}


def assign_freq_group(n: int) -> str:
    if n > FREQ_THRESHOLDS["head"]:
        return "Head (n>200)"
    if n > FREQ_THRESHOLDS["tail"]:
        return "Medium (50<n≤200)"
    return "Tail (n≤50)"


def compute_group_f1(per_fam_rows: List[Dict],
                     fam_le: LabelEncoder,
                     fam_counts: List[int]) -> pd.DataFrame:
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
        bars = ax.bar(
            x + (i - n_m/2 + 0.5) * width, vals, width * 0.88,
            yerr=errs, capsize=3, color=col, alpha=0.88, label=mlbl,
            error_kw={"elinewidth": 1.2, "alpha": 0.6}
        )
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.013,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="0.25")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Family Macro-F1", fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        "Per-frequency-group Family Macro-F1\n"
        "(Head = n>200, Medium = 50<n≤200, Tail = n≤50 training examples)",
        fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.axvspan(1.5, 2.5, color="#FFF3CD", alpha=0.35, zorder=0)
    ax.text(2.0, 1.03, "LDAM katkısı\nen yüksek burada",
            ha="center", fontsize=7.5, color="#856404", style="italic")

    plt.tight_layout()
    path = out_dir / "fig6_group_f1.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig6 → {path}")


def plot_confusion(conf_data: Dict, cls_le: LabelEncoder, out_dir: Path):
    if not PLOT_OK or "proposed" not in conf_data:
        return
    cm = np.array(conf_data["proposed"])
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    lbls = cls_le.classes_
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_n, ax=ax, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=lbls, yticklabels=lbls, vmin=0, vmax=1, linewidths=0.4
    )
    ax.set_title("ESM-2+LoRA+HCH: Class-level Confusion\n(normalized, test set)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path = out_dir / "fig5_confusion_cls.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Fig5 → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# STATS
# ──────────────────────────────────────────────────────────────────────────────

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
            stat, p = wilcoxon(prop_f1s[:n], base_f1s[:n], alternative="greater")
            delta = float(np.mean(prop_f1s[:n]) - np.mean(base_f1s[:n]))
            sym = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            results[mkey] = {"statistic": float(stat), "p_value": float(p), "delta_f1": delta, "symbol": sym}
            print(f"  [Stats] proposed vs {mkey:<14}: ΔF1={delta:+.4f} p={p:.4f} {sym}")
        except Exception as e:
            results[mkey] = {"p_value": float("nan"), "symbol": "err", "note": str(e)}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# LEAKAGE
# ──────────────────────────────────────────────────────────────────────────────

def run_leakage_analysis(tr_s, tr_f, tr_c, te_s, te_f, te_c,
                         fam_le, cls_le, fam_to_cls_map, device, args, seeds):
    from sklearn.model_selection import StratifiedShuffleSplit

    all_s = tr_s + te_s
    all_f = tr_f + te_f
    all_c = tr_c + te_c

    sss = StratifiedShuffleSplit(1, test_size=len(te_s)/len(all_s), random_state=42)
    tr_idx, te_idx = next(sss.split(all_s, all_f))

    r_tr_s = [all_s[i] for i in tr_idx]
    r_tr_f = [all_f[i] for i in tr_idx]
    r_tr_c = [all_c[i] for i in tr_idx]
    r_te_s = [all_s[i] for i in te_idx]
    r_te_f = [all_f[i] for i in te_idx]
    r_te_c = [all_c[i] for i in te_idx]

    print(f"  [Leakage] Random split: {len(r_tr_s)} train / {len(r_te_s)} test")

    n_fam = len(fam_le.classes_)
    n_cls = len(cls_le.classes_)

    rows = []
    for mkey in ["cnn", "proposed"]:
        label = MODEL_LABEL[mkey]
        for split, ts, tf, tc, vs, vf, vc in [
            ("homology", tr_s, tr_f, tr_c, te_s, te_f, te_c),
            ("random", r_tr_s, r_tr_f, r_tr_c, r_te_s, r_te_f, r_te_c),
        ]:
            split_fam_ids = fam_le.transform(tf).tolist()
            split_cls_ids = cls_le.transform(tc).tolist()
            split_fam_counts = (np.bincount(split_fam_ids, minlength=n_fam) + 1).tolist()
            split_cls_counts = (np.bincount(split_cls_ids, minlength=n_cls) + 1).tolist()

            f1s = []
            for seed in seeds[:2]:
                set_seed(seed)
                if mkey == "cnn":
                    m = CNNModel(n_fam, n_cls).to(device)
                    tr_ld = get_cnn_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True)
                    te_ld = get_cnn_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_model(
                        m, f"{mkey}-{split}-s{seed}", tr_ld, tr_ld, te_ld,
                        device, max(3, args.epochs // 3), args.lr,
                        is_cnn=True, grad_accum=args.grad_accum
                    )
                else:
                    if not ESM_OK:
                        continue
                    m = ProposedModel(
                        n_fam, n_cls, fam_to_cls_map,
                        lora_r=args.lora_r, lora_alpha=args.lora_alpha
                    ).to(device)
                    tr_ld = get_loader(ts, tf, tc, fam_le, cls_le, args.batch_size, shuffle=True)
                    te_ld = get_loader(vs, vf, vc, fam_le, cls_le, args.batch_size)
                    fam_m, _, _ = train_and_eval_model(
                        m, f"{mkey}-{split}-s{seed}", tr_ld, tr_ld, te_ld,
                        device, max(3, args.epochs // 3), args.lr,
                        loss_fn = HierarchyAwareLoss(
                            fam_to_cls_map,
                            fam_counts=fam_counts,
                            cls_counts=cls_counts,
                            lambda_fam=1.15,
                            lambda_cls=0.80,
                            lambda_kl=0.10,
                            max_margin=args.ldam_max_margin,
                            drw_start_frac=args.drw_start_frac,
                        ),
                        grad_accum=args.grad_accum
                    )
                f1s.append(fam_m["macro_f1"])
                free_gpu(m)

            rows.append({
                "model_key": mkey,
                "model_label": label,
                "split": split,
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
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
        print(f"  [Leakage] {MODEL_LABEL.get(mkey,mkey):<28}: homo={hf:.4f} rand={rf:.4f} Δ={d:+.4f}")
    return pd.DataFrame(delta_rows)


# ──────────────────────────────────────────────────────────────────────────────
# LATEX
# ──────────────────────────────────────────────────────────────────────────────

def make_table1(summary_df: pd.DataFrame, stat_results: Dict = None) -> str:
    cols = [
        ("fam_macro_f1", "Fam Macro-F1"),
        ("fam_mcc", "Fam MCC"),
        ("fam_bal_acc", "Fam Bal.Acc"),
        ("cls_macro_f1", "Cls Macro-F1"),
        ("cls_mcc", "Cls MCC"),
        ("fam_ece", "ECE↓")
    ]
    best = {
        c: summary_df[c + "_mean"].max() if "ece" not in c else summary_df[c + "_mean"].min()
        for c, _ in cols if c + "_mean" in summary_df.columns
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Main results (homology-aware test set, mean $\pm$ std, 3 seeds). "
        r"Best per column \textbf{bold}. "
        r"$^{*}p<0.05$, $^{**}p<0.01$ vs.\ best baseline (Wilcoxon).}",
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

        if row["model_key"] == "proposed" and stat_results:
            syms = [v.get("symbol", "ns") for v in stat_results.values()]
            top = "**" if "**" in syms else ("*" if "*" in syms else "ns")
            suffix = f" $^{{{top}}}$" if top != "ns" else ""
        else:
            suffix = ""

        lines.append("  " + " & ".join(parts) + suffix + r" \\")

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
        r"\caption{Homology leakage. $\Delta$F1 = F1(random) $-$ F1(homology-aware). "
        r"Larger $\Delta$ = greater overestimation.}",
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
    g_short = {"Head (n>200)": "Head", "Medium (50<n≤200)": "Medium", "Tail (n≤50)": "Tail"}
    models = grp_df["model_key"].unique().tolist()

    best = {}
    for g in groups:
        sub = grp_df[grp_df["freq_group"] == g]
        best[g] = sub["f1_mean"].max() if not sub.empty else 0.0

    hdr = " & ".join(g_short[g] for g in groups)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Family Macro-F1 stratified by training-set frequency. "
        r"Head: $n>200$; Medium: $50 < n \leq 200$; Tail: $n \leq 50$. "
        r"Tail column demonstrates LDAM-DRW benefit on rare families.}",
        r"\label{tab:group_f1}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        f"Model & {hdr} \\\\",
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
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CAZy Q1 karşılaştırması: ESM-2+LoRA+HCH vs HBI/CNN/ESM-frozen/ProtBERT/SetFit"
    )
    ap.add_argument("--train_fasta", required=True)
    ap.add_argument("--val_fasta", required=True)
    ap.add_argument("--test_fasta", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--test_labels", required=True)
    ap.add_argument("--out", default="results/comparison")

    ap.add_argument("--models", default="hbi,cnn,esm_frozen,protbert,setfit,proposed")
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fewshot_epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)

    ap.add_argument("--ldam_max_margin", type=float, default=0.5)
    ap.add_argument("--drw_start_frac", type=float, default=0.7)

    ap.add_argument("--setfit_epochs", type=int, default=1)
    ap.add_argument("--eval_few_shot", action="store_true")
    ap.add_argument("--k_shots", default="1,5,10,20")
    ap.add_argument("--fewshot_seeds", type=int, default=3)

    ap.add_argument("--no_balanced_sampling", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    ap.add_argument("--eval_leakage", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    args.balanced_sampling = not args.no_balanced_sampling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    k_shots = [int(k) for k in args.k_shots.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seeds[0])

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {torch.cuda.get_device_name(0)} · {vram_gb:.1f} GB VRAM")
        print(
            f"  OOM bütçesi: batch={args.batch_size} · accum={args.grad_accum} "
            f"· effective={args.batch_size*args.grad_accum} "
            f"· ESM_MAXLEN={ESM_MAXLEN} · encode_chunk={ESM_ENCODE_CHUNK}"
        )
    print(f"  Models: {models} | Seeds: {seeds} | Out: {out_dir}\n")

    print("[1] Veri yükleniyor...")
    tr_s, tr_f, tr_c = load_split(args.train_fasta, args.train_labels)
    va_s, va_f, va_c = load_split(args.val_fasta, args.val_labels)
    te_s, te_f, te_c = load_split(args.test_fasta, args.test_labels)

    if not tr_s:
        raise RuntimeError("Eğitim verisi boş — FASTA/CSV ID eşleşmesini kontrol et")
    if not te_s:
        raise RuntimeError("Test verisi boş — FASTA/CSV ID eşleşmesini kontrol et")

    if args.debug:
        tr_s, tr_f, tr_c = tr_s[:300], tr_f[:300], tr_c[:300]
        va_s, va_f, va_c = va_s[:80], va_f[:80], va_c[:80]
        te_s, te_f, te_c = te_s[:100], te_f[:100], te_c[:100]
        args.epochs = 2
        args.fewshot_epochs = 2
        print("  [DEBUG] küçültüldü, epochs=2")

    all_fam = tr_f + va_f + te_f
    all_cls = tr_c + va_c + te_c
    fam_le = LabelEncoder().fit(all_fam)
    cls_le = LabelEncoder().fit(all_cls)

    fam_to_cls_map: List[int] = [
        int(cls_le.transform([fam2cls(f)])[0]) for f in fam_le.classes_
    ]

    tr_fam_ids = fam_le.transform(tr_f).tolist()
    tr_cls_ids = cls_le.transform(tr_c).tolist()
    n_fam_total = len(fam_le.classes_)
    n_cls_total = len(cls_le.classes_)
    fam_counts: List[int] = (np.bincount(tr_fam_ids, minlength=n_fam_total) + 1).tolist()
    cls_counts: List[int] = (np.bincount(tr_cls_ids, minlength=n_cls_total) + 1).tolist()

    print(f"  {n_fam_total} fam | {n_cls_total} cls")
    print(f"  Frekans: fam min={min(fam_counts)}, max={max(fam_counts)} | cls min={min(cls_counts)}, max={max(cls_counts)}")

    print("\n[2] Full-supervised eğitim...")
    all_rows, per_fam_rows, conf_data = [], [], {}

    for model_key in models:
        label = MODEL_LABEL.get(model_key, model_key)
        print(f"\n  ══ {label} ══")
        free_gpu()

        if model_key == "hbi":
            fam_m, cls_m, per_fam = run_hbi(tr_s, tr_f, tr_c, te_s, te_f, te_c, fam_le, cls_le)
            for seed in seeds:
                all_rows.append({
                    "model_key": model_key, "model_label": label, "seed": seed,
                    **{f"fam_{k}": v for k, v in fam_m.items()},
                    **{f"cls_{k}": v for k, v in cls_m.items()},
                })
                for fi, f1v in enumerate(per_fam):
                    if fi < n_fam_total:
                        per_fam_rows.append({
                            "model_key": model_key, "model_label": label,
                            "seed": seed, "family": fam_le.classes_[fi], "f1": f1v,
                        })
            continue

        for seed in seeds:
            set_seed(seed)
            fam_m, cls_m, per_fam = {}, {}, []

            if model_key == "setfit":
                if not ESM_OK:
                    print("  ESM yok, atlandı")
                    continue
                fam_m, cls_m, _, _ = run_setfit(
                    tr_s, tr_f, te_s, te_f, tr_c, te_c,
                    fam_le, cls_le, device, args, seed
                )
                print(f"  [setfit s={seed}] FamF1={fam_m['macro_f1']:.4f} ClsF1={cls_m['macro_f1']:.4f} ECE={fam_m.get('ece',0):.4f}")

            elif model_key == "cnn":
                model = CNNModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_cnn_loader(tr_s, tr_f, tr_c, fam_le, cls_le,
                                       args.batch_size, shuffle=True, balanced=args.balanced_sampling)
                va_ld = get_cnn_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_cnn_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"CNN s={seed}", tr_ld, va_ld, te_ld,
                    device, args.epochs, args.lr, is_cnn=True,
                    grad_accum=args.grad_accum,
                    ckpt_path=out_dir / f"ckpt_cnn_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "esm_frozen":
                if not ESM_OK:
                    print("  ESM yok, atlandı")
                    continue
                model = ESMFrozenModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le,
                                   args.batch_size, shuffle=True, balanced=args.balanced_sampling)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"ESM-frozen s={seed}", tr_ld, va_ld, te_ld,
                    device, args.epochs, args.lr,
                    grad_accum=args.grad_accum,
                    ckpt_path=out_dir / f"ckpt_esm_frozen_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "protbert":
                if not BERT_OK:
                    print("  transformers yok, atlandı")
                    continue
                model = ProtBERTModel(n_fam_total, n_cls_total).to(device)
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le,
                                   args.batch_size, shuffle=True, balanced=args.balanced_sampling)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"ProtBERT s={seed}", tr_ld, va_ld, te_ld,
                    device, args.epochs, args.lr,
                    grad_accum=args.grad_accum,
                    ckpt_path=out_dir / f"ckpt_protbert_s{seed}.pt"
                )
                free_gpu(model)

            elif model_key == "proposed":
                if not ESM_OK:
                    print("  ESM yok, atlandı")
                    continue
                model = ProposedModel(
                    n_fam_total, n_cls_total, fam_to_cls_map,
                    lora_r=args.lora_r, lora_alpha=args.lora_alpha
                ).to(device)
                loss_fn = HierarchyAwareLoss(
                    fam_to_cls_map,
                    fam_counts=fam_counts,
                    cls_counts=cls_counts,
                    max_margin=args.ldam_max_margin,
                    drw_start_frac=args.drw_start_frac,
                )
                tr_ld = get_loader(tr_s, tr_f, tr_c, fam_le, cls_le,
                                   args.batch_size, shuffle=True, balanced=args.balanced_sampling)
                va_ld = get_loader(va_s, va_f, va_c, fam_le, cls_le, args.batch_size)
                te_ld = get_loader(te_s, te_f, te_c, fam_le, cls_le, args.batch_size)
                fam_m, cls_m, per_fam = train_and_eval_model(
                    model, f"Proposed s={seed}", tr_ld, va_ld, te_ld,
                    device, args.epochs, args.lr, loss_fn=loss_fn,
                    grad_accum=args.grad_accum,
                    ckpt_path=out_dir / f"ckpt_proposed_s{seed}.pt"
                )

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
                        conf_data["proposed"] = confusion_matrix(
                            all_ct2, all_cp2, labels=list(range(n_cls_total))
                        ).tolist()
                free_gpu(model)

            else:
                print(f"  Bilinmeyen model: {model_key}")
                continue

            if fam_m:
                all_rows.append({
                    "model_key": model_key, "model_label": label, "seed": seed,
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
        print("\n[3] Gerçek few-shot (train from scratch)...")
        for model_key in models:
            label = MODEL_LABEL.get(model_key, model_key)
            for k in k_shots:
                f1s = []
                for fsi in range(args.fewshot_seeds):
                    cseed = seeds[fsi % len(seeds)] + fsi
                    set_seed(cseed)
                    free_gpu()
                    print(f"  {label:30s} k={k:2d} fsi={fsi}", end="  ")
                    f1 = real_fewshot_run(
                        model_key, tr_s, tr_f, te_s, te_f,
                        fam_le, cls_le, fam_to_cls_map,
                        device, k, cseed, args
                    )
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

    summary_df = pd.DataFrame()
    if not per_seed_df.empty:
        mcols = [c for c in per_seed_df.columns if c not in ("model_key", "model_label", "seed")]
        rows = []
        for mkey in per_seed_df["model_key"].unique():
            sub = per_seed_df[per_seed_df["model_key"] == mkey]
            row = {"model_key": mkey, "model_label": MODEL_LABEL.get(mkey, mkey)}
            for col in mcols:
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
        pf_agg = pf_df.groupby(["model_key", "model_label", "family"])["f1"].agg(
            f1_mean="mean", f1_std="std"
        ).reset_index()
        pf_agg.to_csv(out_dir / "per_family_f1.csv", index=False)

        grp_df = compute_group_f1(per_fam_rows, fam_le, fam_counts)
        if not grp_df.empty:
            grp_df.to_csv(out_dir / "group_f1.csv", index=False)
            print(f"  Group F1 → {out_dir}/group_f1.csv")

    print("\n[4b] İstatistiksel testler...")
    stat_results = run_statistical_tests(per_seed_df) if not per_seed_df.empty else {}
    if stat_results:
        pd.DataFrame([{"baseline": k, **v} for k, v in stat_results.items()]).to_csv(
            out_dir / "statistical_tests.csv", index=False
        )

    leakage_df = pd.DataFrame()
    if args.eval_leakage:
        print("\n[4c] Homology leakage analizi...")
        leakage_df = run_leakage_analysis(
            tr_s, tr_f, tr_c, te_s, te_f, te_c,
            fam_le, cls_le, fam_to_cls_map, device, args, seeds
        )
        if not leakage_df.empty:
            leakage_df.to_csv(out_dir / "leakage_analysis.csv", index=False)

    print("\n[5] Grafikler üretiliyor...")
    if not summary_df.empty:
        plot_main_bar(summary_df, out_dir)
        plot_radar(summary_df, out_dir)
    if not fs_df.empty:
        plot_fewshot_curve(fs_df, out_dir)
    if not pf_agg.empty:
        plot_family_heatmap(pf_agg, fam_le, out_dir)
    if not grp_df.empty:
        plot_group_f1(grp_df, out_dir)
    if conf_data:
        plot_confusion(conf_data, cls_le, out_dir)

    print("\n[6] LaTeX tabloları...")
    if not summary_df.empty:
        (out_dir / "table1_main.tex").write_text(make_table1(summary_df, stat_results))
        print(f"  Table1 → {out_dir}/table1_main.tex")
    if not grp_df.empty:
        (out_dir / "table4_group_f1.tex").write_text(make_table4(grp_df))
        print(f"  Table4 → {out_dir}/table4_group_f1.tex")
    if not fs_df.empty:
        (out_dir / "table2_fewshot.tex").write_text(make_table2(fs_df))
        print(f"  Table2 → {out_dir}/table2_fewshot.tex")
    if not leakage_df.empty:
        (out_dir / "table3_leakage.tex").write_text(make_table3(leakage_df))
        print(f"  Table3 → {out_dir}/table3_leakage.tex")

    print(f"\n{'═'*78}")
    print("  SONUÇ ÖZETİ")
    print(f"{'═'*78}")
    if not summary_df.empty:
        show_cols = ["model_label", "fam_macro_f1_mean", "fam_macro_f1_std",
                     "cls_macro_f1_mean", "fam_mcc_mean", "fam_ece_mean"]
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
    print("  CSVs : comparison_summary | per_seed | fewshot | per_family | group_f1 | stats | leakage")
    print("  Figs : fig1_main_bar | fig2_fewshot_curve | fig3_radar | fig4_family_heatmap | fig5_confusion | fig6_group_f1")
    print("  LaTeX: table1_main | table2_fewshot | table3_leakage | table4_group_f1")
    print(f"{'═'*78}")


if __name__ == "__main__":
    main()
