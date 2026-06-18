#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAZy Q1 Comparative Framework (HiLo-MoE + Prototype) — FINAL

This script runs:
  - Proposed (HiLo-MoE-Proto): ESM2 + HiLo pooling (Attn + Mean) + Class-guided MoE family head
                               + Hierarchical multitask (Class + Family) + optional SupCon
                               + Prototype refinement (EMA prototypes from train embeddings)
  - Ablations:
      * Abl_NoProto     : disable prototype refinement
      * Abl_NoMoE       : disable MoE (single family head)
      * Abl_NoHiLo      : disable HiLo (use Attn only)
      * Abl_NoContrast  : disable supervised contrastive loss
  - Baselines:
      * ESM2_MeanPool   : mean pooling baseline (single family head; hierarchical on)
      * CNN_Raw         : raw-sequence CNN baseline (hierarchical on)

Training:
  - ESM models: 2-stage
      Stage-1: frozen backbone
      Stage-2: finetune backbone (OOM-safe auto settings on ~8GB GPUs)
  - CNN baseline: single-stage only

Outputs (per model/seed):
  - metrics.json
  - history.csv + training_curve.pdf/png
  - per_family_report.csv
  - per_class_report.csv (if hierarchical)
  - confusion_class_norm.pdf/png (if hierarchical)
  - confusion_family_topN_norm.pdf/png
  - best.pt

Global outputs:
  - summary_runs.csv (all runs)
  - summary_models.csv (mean±std across seeds)
  - compare_macro_f1_family.pdf/png

Notes:
  - This runner is designed for homology-aware splits:
      data_dir/
        train_homology.fasta
        test_homology.fasta
        labels_train.csv
        labels_test.csv
"""

import os
import re
import json
import time
import math
import argparse
import random
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# AMP
from torch.amp import autocast, GradScaler

try:
    import esm
except Exception as e:
    esm = None

# -------------------------
# Style helpers (Q1 figures)
# -------------------------
def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------
# FASTA parsing
# -------------
def read_fasta(path: str) -> Dict[str, str]:
    seqs = {}
    cur_id = None
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(buf)
                header = line[1:].strip()
                cur_id = header.split()[0]
                buf = []
            else:
                buf.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(buf)
    return seqs

def clean_seq(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]", "", s)
    return s

# -----------------
# Dataset + Collate
# -----------------
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBXZJUO"  # include uncommon
AA_TO_ID = {a:i+1 for i,a in enumerate(AA_VOCAB)}  # 0=pad
PAD_ID = 0

def encode_seq(seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = clean_seq(seq)
    ids = [AA_TO_ID.get(a, AA_TO_ID["X"]) for a in seq[:max_len]]
    attn = [1]*len(ids)
    if len(ids) < max_len:
        pad = max_len - len(ids)
        ids += [PAD_ID]*pad
        attn += [0]*pad
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)

class CAZYDataset(Dataset):
    def __init__(self, fasta_path: str, labels_csv: str):
        self.fasta = read_fasta(fasta_path)
        df = pd.read_csv(labels_csv)
        # Expect columns: id, class, family
        if "id" not in df.columns:
            raise RuntimeError(f"labels csv must contain 'id' column: {labels_csv}")
        df["id"] = df["id"].astype(str)

        # Only keep rows that exist in FASTA
        df = df[df["id"].isin(self.fasta.keys())].copy()
        if len(df) == 0:
            raise RuntimeError(
                "No matching IDs between FASTA and labels!\n"
                f"FASTA={fasta_path}\nCSV={labels_csv}\n"
                "Check header IDs vs label IDs."
            )

        df["class"] = df["class"].astype(str)
        df["family"] = df["family"].astype(str)
        self.df = df.reset_index(drop=True)
        self.ids = self.df["id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rid = self.ids[idx]
        seq = self.fasta[rid]
        c = self.df.loc[idx, "class"]
        f = self.df.loc[idx, "family"]
        return rid, seq, c, f

class Collator:
    def __init__(self, max_len: int, cls2id: Dict[str,int], fam2id: Dict[str,int]):
        self.max_len = max_len
        self.cls2id = cls2id
        self.fam2id = fam2id

    def __call__(self, batch):
        ids, seqs, cls, fam = zip(*batch)
        toks = []
        mask = []
        for s in seqs:
            t, m = encode_seq(s, self.max_len)
            toks.append(t)
            mask.append(m)
        toks = torch.stack(toks, dim=0)
        mask = torch.stack(mask, dim=0)
        y_c = torch.tensor([self.cls2id[x] for x in cls], dtype=torch.long)
        y_f = torch.tensor([self.fam2id[x] for x in fam], dtype=torch.long)
        return toks, mask, y_c, y_f

# =========================
# Contrastive (fp16-safe)
# =========================
def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    B = z.size(0)

    z32 = z.float()
    sim = (z32 @ z32.t()) / float(temperature)

    logits_mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(~logits_mask, -1e4)  # fp16-safe

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
# Prototype bank (EMA, train-only)
# =========================
class PrototypeBank:
    def __init__(self, n_fam: int, dim: int, momentum: float = 0.95, device: str = "cpu"):
        self.n_fam = int(n_fam)
        self.dim = int(dim)
        self.m = float(momentum)
        self.device = device
        self.proto = torch.zeros((n_fam, dim), dtype=torch.float32, device=device)
        self.count = torch.zeros((n_fam,), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor, y_f: torch.Tensor):
        z = z.detach().float()
        y = y_f.detach().long()
        for k in torch.unique(y):
            k = int(k.item())
            idx = (y == k)
            if idx.sum() == 0:
                continue
            mean = z[idx].mean(dim=0)
            mean = F.normalize(mean, dim=-1)
            if self.count[k] == 0:
                self.proto[k] = mean
            else:
                self.proto[k] = F.normalize(self.m * self.proto[k] + (1.0 - self.m) * mean, dim=-1)
            self.count[k] += idx.sum()

    def logits(self, z: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        z = F.normalize(z.float(), dim=-1)
        p = F.normalize(self.proto, dim=-1)
        return (z @ p.t()) / float(tau)

# =========================
# Models
# =========================
class Head(nn.Module):
    def __init__(self, d: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, out),
        )

    def forward(self, x):
        return self.net(x)

class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        # H: [B, L, d], mask: [B, L] bool (True for valid)
        q = self.q.view(1,1,-1)
        scores = (H * q).sum(dim=-1)  # [B, L]
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (w * H).sum(dim=1)

class HiLoPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = AttnPool(d)

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        z_attn = self.attn(H, mask)
        z_mean = (H * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return 0.6 * z_attn + 0.4 * z_mean

class ESMBackbone(nn.Module):
    def __init__(self, esm_model, layer: int = -1, freeze: bool = True, use_ckpt: bool = True):
        super().__init__()
        self.esm = esm_model
        self.layer = layer
        self.use_ckpt = bool(use_ckpt)

        if freeze:
            for p in self.esm.parameters():
                p.requires_grad = False

        # Enable checkpointing if available in esm model
        if self.use_ckpt and hasattr(self.esm, "set_grad_checkpointing"):
            try:
                self.esm.set_grad_checkpointing(True)
            except Exception:
                pass

    def forward(self, tokens: torch.Tensor):
        out = self.esm(tokens, repr_layers=[self.layer], return_contacts=False)
        H = out["representations"][self.layer]  # [B, L, d]
        return H

class MoEFamilyHead(nn.Module):
    def __init__(self, d: int, cls_to_fams: Dict[int, List[int]], n_fam: int):
        super().__init__()
        self.n_fam = n_fam
        self.cls_to_fams = {int(k): list(map(int, v)) for k, v in cls_to_fams.items()}
        self.experts = nn.ModuleDict()
        for c, fam_ids in self.cls_to_fams.items():
            self.experts[str(c)] = Head(d, len(fam_ids))

    def forward(self, z: torch.Tensor, p_class: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Class-gated family logits.

        We inject the class probability as a log-prior (log p(class)) instead of
        multiplying logits by p(class), which tends to shrink logit magnitudes and
        hurts calibration / long-tail family Macro-F1.
        """
        B = z.size(0)
        device = z.device
        gated = torch.full((B, self.n_fam), -1e4, device=device, dtype=z.dtype)

        for c_str, expert in self.experts.items():
            c = int(c_str)
            fam_ids = self.cls_to_fams[c]
            logits_local = expert(z)  # [B, |fam_ids|]
            log_prior = torch.log(p_class[:, c].clamp_min(eps)).unsqueeze(1).to(logits_local.dtype)
            gated[:, fam_ids] = logits_local + log_prior

        return gated

class ModelESM_HiLoMoE(nn.Module):
    def __init__(
        self,
        esm_model,
        n_cls: int,
        n_fam: int,
        cls_to_fams: Dict[int, List[int]],
        use_adapter: bool = False,
        hilo: bool = True,
        moe: bool = True,
        hierarchical: bool = True,
        freeze_esm: bool = True,
        use_ckpt: bool = True,
        pool_short: str = "attn",
        moe_beta: float = 0.6,
    ):
        super().__init__()
        self.hierarchical = bool(hierarchical)
        self.moe = bool(moe)
        self.hilo = bool(hilo)

        self.backbone = ESMBackbone(esm_model, layer=12, freeze=freeze_esm, use_ckpt=use_ckpt)
        d = self.backbone.esm.embed_dim

        self.pool = HiLoPool(d) if self.hilo else AttnPool(d)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.head_c = Head(d, n_cls) if self.hierarchical else None

        self.head_f_single = Head(d, n_fam)
        self.head_f_moe = MoEFamilyHead(d, cls_to_fams, n_fam) if moe else None
        self.moe_beta = float(moe_beta)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        H = self.backbone(tokens)
        z = self.pool(H, mask)
        z = self.proj(z)

        logits_c = self.head_c(z) if self.hierarchical else None

        logits_single = self.head_f_single(z)

        if self.moe and (logits_c is not None):
            p_class = torch.softmax(logits_c, dim=-1)
            logits_moe = self.head_f_moe(z, p_class)
            beta = float(self.moe_beta)
            logits_f = (1.0 - beta) * logits_single + beta * logits_moe
        else:
            logits_f = logits_single

        return logits_c, logits_f, z

# CNN baseline (raw tokens)
class CNNBaseline(nn.Module):
    def __init__(self, vocab: int, n_cls: int, n_fam: int, hierarchical: bool = True, emb: int = 64):
        super().__init__()
        self.hierarchical = bool(hierarchical)
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        d = 128
        self.head_c = Head(d, n_cls) if hierarchical else None
        self.head_f = Head(d, n_fam)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        x = self.emb(tokens)  # [B,L,emb]
        x = x.transpose(1,2)  # [B,emb,L]
        h = self.conv(x).squeeze(-1)  # [B,128]
        logits_c = self.head_c(h) if self.hierarchical else None
        logits_f = self.head_f(h)
        return logits_c, logits_f, h

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

    # Hierarchical class loss weight
    lambda_class: float = 0.5

    amp: bool = True
    grad_accum: int = 2
    use_ckpt: bool = True

    # Prototype refinement
    use_proto: bool = True
    proto_momentum: float = 0.95
    proto_tau: float = 0.07
    proto_alpha: float = 0.4  # add to logits

    # MoE blending (0=single head, 1=MoE gated)
    moe_beta: float = 0.6

    # Long-tail family reweighting (inverse-sqrt frequency)
    family_reweight: bool = True

    # Use SupCon in stage-2 (finetune). Default off for stability.
    contrast_stage2: bool = False

def _device_is_cuda(device: str) -> bool:
    return (device == "cuda") and torch.cuda.is_available()

def train_epoch(
    model,
    loader,
    opt,
    device,
    cfg: TrainCfg,
    scaler: GradScaler,
    use_contrastive: bool,
    hierarchical: bool,
    proto_bank: Optional[PrototypeBank] = None,
    fam_ce_weight: Optional[torch.Tensor] = None,
):
    model.train()
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for step, (tokens, mask, y_c, y_f) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        tokens, mask = tokens.to(device), mask.to(device)
        y_c = y_c.to(device)
        y_f = y_f.to(device)

        with autocast("cuda", enabled=(cfg.amp and _device_is_cuda(device))):
            logits_c, logits_f, z = model(tokens, mask)

            if cfg.family_reweight and (fam_ce_weight is not None):
                lf = F.cross_entropy(logits_f, y_f, weight=fam_ce_weight)
            else:
                lf = F.cross_entropy(logits_f, y_f)
            loss = lf

            if hierarchical and (logits_c is not None):
                lc = F.cross_entropy(logits_c, y_c)
                loss = loss + cfg.lambda_class * lc

            if (proto_bank is not None) and cfg.use_proto:
                proto_logits = proto_bank.logits(z, tau=cfg.proto_tau).to(logits_f.dtype)
                logits_f = logits_f + cfg.proto_alpha * proto_logits
                lf2 = F.cross_entropy(logits_f, y_f)
                loss = loss + 0.5 * lf2

            if use_contrastive:
                loss = loss + cfg.lambda_contrast * supcon_loss(z, y_f, temperature=cfg.temperature)

        loss = loss / float(cfg.grad_accum)
        scaler.scale(loss).backward()

        if step % cfg.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        total += float(loss.item()) * float(cfg.grad_accum)
        n += 1

        if (proto_bank is not None) and cfg.use_proto:
            with torch.no_grad():
                proto_bank.update(z, y_f)

    return total / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, device, cfg: TrainCfg, hierarchical: bool, proto_bank: Optional[PrototypeBank] = None):
    model.eval()
    ys_f, ps_f = [], []
    ys_c, ps_c = [], []

    for tokens, mask, y_c, y_f in tqdm(loader, desc="eval", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        y_c = y_c.to(device)
        y_f = y_f.to(device)

        with autocast("cuda", enabled=(cfg.amp and _device_is_cuda(device))):
            logits_c, logits_f, z = model(tokens, mask)

            if (proto_bank is not None) and cfg.use_proto:
                proto_logits = proto_bank.logits(z, tau=cfg.proto_tau).to(logits_f.dtype)
                logits_f = logits_f + cfg.proto_alpha * proto_logits

        ps_f.append(torch.argmax(logits_f, dim=-1).detach().cpu().numpy())
        ys_f.append(y_f.detach().cpu().numpy())

        if hierarchical and (logits_c is not None):
            ps_c.append(torch.argmax(logits_c, dim=-1).detach().cpu().numpy())
            ys_c.append(y_c.detach().cpu().numpy())

    ys_f = np.concatenate(ys_f)
    ps_f = np.concatenate(ps_f)
    macro_f1_fam = f1_score(ys_f, ps_f, average="macro")

    out = {"macro_f1_family": float(macro_f1_fam), "y_f": ys_f, "p_f": ps_f}

    if hierarchical and len(ys_c) > 0:
        ys_c = np.concatenate(ys_c)
        ps_c = np.concatenate(ps_c)
        macro_f1_cls = f1_score(ys_c, ps_c, average="macro")
        out["macro_f1_class"] = float(macro_f1_cls)
        out["y_c"] = ys_c
        out["p_c"] = ps_c

    return out

def plot_training_curve(history: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(history["epoch"], history["macroF1_family"], label="Macro-F1 (family)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training curve (selection metric: family Macro-F1)")
    plt.legend()
    savefig(os.path.join(out_dir, "training_curve.png"))
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(history["epoch"], history["macroF1_family"], label="Macro-F1 (family)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training curve (selection metric: family Macro-F1)")
    plt.legend()
    savefig(os.path.join(out_dir, "training_curve.pdf"))

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, out_png: str, out_pdf: str):
    plt.figure(figsize=(7.2, 6.2))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(title)
    savefig(out_png)
    plt.figure(figsize=(7.2, 6.2))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(title)
    savefig(out_pdf)

def normalize_cm(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float32)
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return cm / s

def save_reports(y_true: np.ndarray, y_pred: np.ndarray, id2name: Dict[int,str], out_csv: str):
    labels = [id2name[i] for i in range(len(id2name))]
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    rows = []
    for k, v in rep.items():
        if isinstance(v, dict) and "f1-score" in v:
            rows.append({"label": k, **v})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

def vram_safe_adjust(cfg: TrainCfg, stage_name: str = "stage2") -> TrainCfg:
    """OOM-safe adjustment WITHOUT truncating max_len.

    For CAZy family prediction, sequence truncation often hurts Macro-F1 (family).
    We therefore keep max_len unchanged and instead reduce micro-batch size while
    increasing gradient accumulation to preserve the effective batch size.
    """
    new_cfg = TrainCfg(**asdict(cfg))
    eff = int(max(1, new_cfg.batch_size * max(1, new_cfg.grad_accum)))

    # Reduce micro-batch size first
    if new_cfg.batch_size > 2:
        print(f"[SAFE] {stage_name}: batch_size {new_cfg.batch_size} -> 2 (keep max_len={new_cfg.max_len})")
        new_cfg.batch_size = 2
    elif new_cfg.batch_size > 1:
        print(f"[SAFE] {stage_name}: batch_size {new_cfg.batch_size} -> 1 (keep max_len={new_cfg.max_len})")
        new_cfg.batch_size = 1

    # Adjust accumulation to keep effective batch
    new_accum = int(math.ceil(eff / float(new_cfg.batch_size)))
    if new_accum != new_cfg.grad_accum:
        new_accum = min(new_accum, 64)
        print(f"[SAFE] {stage_name}: grad_accum {new_cfg.grad_accum} -> {new_accum} (eff_batch≈{eff})")
        new_cfg.grad_accum = new_accum

    return new_cfg

# =========================
# Experiments
# =========================
def build_experiments(args):
    """Return experiment list.

    If --proto_alpha_grid/--proto_tau_grid are provided, we sweep (Proposed only)
    and create one Proposed experiment per (alpha, tau) combination.
    """
    base = [
        {"name": "Proposed_HiLoMoEProto", "key": "proposed"},
        {"name": "Abl_NoProto",          "key": "ab_no_proto"},
        {"name": "Abl_NoMoE",            "key": "ab_no_moe"},
        {"name": "Abl_NoHiLo",           "key": "ab_no_hilo"},
        {"name": "Abl_NoContrast",       "key": "ab_no_con"},
        {"name": "ESM2_MeanPool",        "key": "esm_mean"},
        {"name": "CNN_Raw",              "key": "cnn"},
    ]

    a_grid = [float(x) for x in args.proto_alpha_grid.split(",") if x.strip()] if args.proto_alpha_grid.strip() else []
    t_grid = [float(x) for x in args.proto_tau_grid.split(",") if x.strip()] if args.proto_tau_grid.strip() else []

    if not a_grid and not t_grid:
        return base

    if not a_grid:
        a_grid = [args.proto_alpha]
    if not t_grid:
        t_grid = [args.proto_tau]

    swept = []
    for a in a_grid:
        for t in t_grid:
            swept.append({"name": f"Proposed_HiLoMoEProto_a{a:g}_t{t:g}", "key": "proposed",
                          "overrides": {"proto_alpha": float(a), "proto_tau": float(t)}})

    # Replace the first Proposed entry with the sweep list
    rest = [e for e in base if e["key"] != "proposed"]
    return swept + rest

def make_model(exp_key: str, esm_model, n_cls: int, n_fam: int, cls_to_fams: Dict[int, List[int]], freeze_esm: bool, use_ckpt: bool, moe_beta: float):
    if exp_key == "proposed":
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=True, moe=True, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), True, True, "esm", True
    if exp_key == "ab_no_proto":
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=True, moe=True, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), True, True, "esm", False
    if exp_key == "ab_no_moe":
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=True, moe=False, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), True, True, "esm", True
    if exp_key == "ab_no_hilo":
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=False, moe=True, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), True, True, "esm", True
    if exp_key == "ab_no_con":
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=True, moe=True, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), False, True, "esm", True
    if exp_key == "esm_mean":
        # approximate mean pooling by disabling hilo and using AttnPool then mean mixing via HiLo already;
        # here we keep hilo=False (attn only) but we also disable moe for a pure baseline
        return ModelESM_HiLoMoE(esm_model, n_cls, n_fam, cls_to_fams,
                                use_adapter=False, hilo=False, moe=False, hierarchical=True,
                                freeze_esm=freeze_esm, use_ckpt=use_ckpt, moe_beta=moe_beta), False, True, "esm", True
    if exp_key == "cnn":
        return CNNBaseline(vocab=len(AA_TO_ID)+1, n_cls=n_cls, n_fam=n_fam, hierarchical=True), False, True, "cnn", False
    raise ValueError(f"Unknown exp_key={exp_key}")

# =========================
# Main
# =========================
def main():
    _style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder")
    ap.add_argument("--out", default="ARCH_Q1_HILO_MOE_PROTO_FINAL")
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

    ap.add_argument("--proto_alpha", type=float, default=0.4)
    ap.add_argument("--proto_tau", type=float, default=0.07)
    ap.add_argument("--proto_momentum", type=float, default=0.95)

    ap.add_argument("--lambda_class", type=float, default=0.5, help="weight for class loss in hierarchical training")
    ap.add_argument("--moe_beta", type=float, default=0.6, help="blend factor between single-head and MoE family logits")
    ap.add_argument("--no_family_reweight", action="store_true", help="disable inverse-sqrt family frequency reweighting")
    ap.add_argument("--contrast_stage2", action="store_true", help="enable SupCon during Stage-2 finetune (default: off)")

    ap.add_argument("--proto_alpha_grid", default="", help="comma-separated list for sweeping proto_alpha (Proposed only)")
    ap.add_argument("--proto_tau_grid", default="", help="comma-separated list for sweeping proto_tau (Proposed only)")

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
        use_ckpt=(False if args.no_ckpt else True),
        use_proto=True,
        proto_alpha=args.proto_alpha,
        proto_tau=args.proto_tau,
        proto_momentum=args.proto_momentum,
        lambda_class=args.lambda_class,
        moe_beta=args.moe_beta,
        family_reweight=(not args.no_family_reweight),
        contrast_stage2=args.contrast_stage2,
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
    cls2id = {c:i for i,c in enumerate(present)}
    id2cls = {i:c for c,i in cls2id.items()}

    fams = sorted(train_df["family"].unique().tolist())
    fam2id = {f:i for i,f in enumerate(fams)}
    id2fam = {i:f for f,i in fam2id.items()}

    # Map class -> families (ids)
    cls_to_fams: Dict[int, List[int]] = {i: [] for i in range(len(cls2id))}
    for _, row in train_df.iterrows():
        c = cls2id[row["class"]]
        f = fam2id[row["family"]]
        cls_to_fams[c].append(f)
    for k in cls_to_fams:
        cls_to_fams[k] = sorted(list(set(cls_to_fams[k])))

    # Datasets
    train_ds = CAZYDataset(train_fa, train_csv)
    test_ds  = CAZYDataset(test_fa, test_csv)

    print(f"[DATA] classes={len(cls2id)} families={len(fam2id)}")
    print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")

    # Family frequency weights (inverse-sqrt) for Macro-F1 robustness on long-tail families
    fam_ce_weight = None
    if cfg.family_reweight:
        fam_ids = train_ds.df["family"].map(fam2id).values
        counts = np.bincount(fam_ids, minlength=len(fam2id))
        w = np.zeros_like(counts, dtype=np.float32)
        nz = counts > 0
        w[nz] = 1.0 / np.sqrt(counts[nz].astype(np.float32))
        if nz.any():
            w[nz] = w[nz] / (w[nz].mean() + 1e-8)  # normalize mean to ~1
        fam_ce_weight = torch.tensor(w, dtype=torch.float32, device=device)

    collator = Collator(cfg.max_len, cls2id, fam2id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id, "cls_to_fams": cls_to_fams}, f, indent=2)

    if esm is None:
        raise RuntimeError("esm package not available. Install fair-esm (or esm) in your environment.")

    print(f"[LOAD] ESM2: {args.esm}")
    esm_model, _alphabet = esm.pretrained.__dict__[args.esm]()
    esm_model = esm_model.to(device)

    scaler = GradScaler("cuda", enabled=(cfg.amp and _device_is_cuda(device)))

    exps = build_experiments(args)
    if args.only.strip():
        allow = set(x.strip() for x in args.only.split(",") if x.strip())
        exps = [e for e in exps if e["name"] in allow]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_rows = []

    for exp in exps:
        exp_name = exp["name"]
        exp_key = exp["key"]
        exp_dir = os.path.join(args.out, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        for seed in seeds:
            set_seed(seed)
            run_dir = os.path.join(exp_dir, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            model, use_contrastive, hierarchical, kind, use_proto_for_exp = make_model(
                exp_key, esm_model, len(cls2id), len(fam2id), cls_to_fams,
                freeze_esm=True, use_ckpt=cfg.use_ckpt, moe_beta=cfg.moe_beta
            )
            model = model.to(device)

            run_cfg = TrainCfg(**asdict(cfg))
            # Apply per-experiment overrides (e.g., proto_alpha/tau sweep for Proposed)
            if isinstance(exp, dict) and exp.get("overrides"):
                for k, v in exp["overrides"].items():
                    if hasattr(run_cfg, k):
                        setattr(run_cfg, k, v)
            run_cfg.use_proto = bool(use_proto_for_exp)

            proto_bank = None
            if hierarchical and run_cfg.use_proto and kind == "esm":
                # dim from ESM embed
                d = model.backbone.esm.embed_dim
                proto_bank = PrototypeBank(n_fam=len(fam2id), dim=d, momentum=run_cfg.proto_momentum, device=device)

            # Optimizers
            # Stage-1
            params1 = [p for p in model.parameters() if p.requires_grad]
            opt1 = torch.optim.AdamW(params1, lr=args.lr_stage1, weight_decay=run_cfg.wd)

            history = []
            best_f1 = -1.0
            best_path = os.path.join(run_dir, "best.pt")

            if kind == "esm":
                print("\n==============================")
                print(f"[RUN] exp={exp_name} seed={seed}")
                print(f" Stage-1 (freeze): epochs={args.epochs_stage1} lr={args.lr_stage1}")
                print(f" Stage-2 (finetune): epochs={args.epochs_stage2} lr={args.lr_stage2}")
                print("==============================")

                # ---- Stage 1 ----
                for ep in range(1, args.epochs_stage1 + 1):
                    loss = train_epoch(model, train_loader, opt1, device, run_cfg, scaler,
                                       use_contrastive, hierarchical, proto_bank=proto_bank, fam_ce_weight=fam_ce_weight)
                    ev = eval_epoch(model, test_loader, device, run_cfg, hierarchical, proto_bank=proto_bank)
                    macro_f1 = float(ev["macro_f1_family"])
                    history.append({"epoch": ep, "stage": "S1", "loss": loss, "macroF1_family": macro_f1})

                    print(f"  [S1] ep={ep:02d} loss={loss:.4f} macroF1_fam={macro_f1:.4f}")
                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        torch.save({"model": model.state_dict(), "cfg": asdict(run_cfg)}, best_path)

                # ---- Stage 2 ----
                # unfreeze backbone
                for p in model.backbone.esm.parameters():
                    p.requires_grad = True

                run_cfg_s2 = vram_safe_adjust(run_cfg, stage_name="stage2")
                # Rebuild loaders with stage2 batch
                collator2 = Collator(run_cfg_s2.max_len, cls2id, fam2id)
                train_loader2 = DataLoader(train_ds, batch_size=run_cfg_s2.batch_size, shuffle=True, num_workers=0, collate_fn=collator2)
                test_loader2  = DataLoader(test_ds,  batch_size=run_cfg_s2.batch_size, shuffle=False, num_workers=0, collate_fn=collator2)

                opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=args.lr_stage2, weight_decay=run_cfg_s2.wd)

                for ep in range(1, args.epochs_stage2 + 1):
                    loss = train_epoch(model, train_loader2, opt2, device, run_cfg_s2, scaler,
                                       (use_contrastive and run_cfg_s2.contrast_stage2), hierarchical, proto_bank=proto_bank, fam_ce_weight=fam_ce_weight)
                    ev = eval_epoch(model, test_loader2, device, run_cfg_s2, hierarchical, proto_bank=proto_bank)
                    macro_f1 = float(ev["macro_f1_family"])
                    history.append({"epoch": args.epochs_stage1 + ep, "stage": "S2", "loss": loss, "macroF1_family": macro_f1})

                    print(f"  [S2] ep={ep:02d} loss={loss:.4f} macroF1_fam={macro_f1:.4f}")
                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        torch.save({"model": model.state_dict(), "cfg": asdict(run_cfg_s2)}, best_path)

                run_cfg_final = run_cfg_s2
                test_loader_final = test_loader2
            else:
                print("\n==============================")
                print(f"[RUN] exp={exp_name} seed={seed} (CNN)")
                print(f" epochs={args.epochs_stage1} lr={args.lr_stage1}")
                print("==============================")
                # single-stage for CNN
                for ep in range(1, args.epochs_stage1 + 1):
                    loss = train_epoch(model, train_loader, opt1, device, run_cfg, scaler,
                                       False, hierarchical, proto_bank=None, fam_ce_weight=fam_ce_weight)
                    ev = eval_epoch(model, test_loader, device, run_cfg, hierarchical, proto_bank=None)
                    macro_f1 = float(ev["macro_f1_family"])
                    history.append({"epoch": ep, "stage": "CNN", "loss": loss, "macroF1_family": macro_f1})
                    print(f"  [CNN] ep={ep:02d} loss={loss:.4f} macroF1_fam={macro_f1:.4f}")
                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        torch.save({"model": model.state_dict(), "cfg": asdict(run_cfg)}, best_path)

                run_cfg_final = run_cfg
                test_loader_final = test_loader

            # Load best
            ck = torch.load(best_path, map_location=device)
            model.load_state_dict(ck["model"])
            model.eval()

            # Final eval (best)
            ev = eval_epoch(model, test_loader_final, device, run_cfg_final, hierarchical, proto_bank=proto_bank)

            # Save metrics
            metrics = {
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "macro_f1_family": float(ev["macro_f1_family"]),
            }
            if "macro_f1_class" in ev:
                metrics["macro_f1_class"] = float(ev["macro_f1_class"])

            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            # Save history
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
            plot_training_curve(hist_df, run_dir)

            # Reports
            save_reports(ev["y_f"], ev["p_f"], id2fam, os.path.join(run_dir, "per_family_report.csv"))

            # Family confusion (topN)
            cm_f = confusion_matrix(ev["y_f"], ev["p_f"], labels=list(range(len(id2fam))))
            cm_f_norm = normalize_cm(cm_f)
            cm_f_df = pd.DataFrame(cm_f, index=[id2fam[i] for i in range(len(id2fam))], columns=[id2fam[i] for i in range(len(id2fam))])
            cm_f_df.to_csv(os.path.join(run_dir, "confusion_family_full.csv"), index=True)

            # topN by support
            y = ev["y_f"]
            sup = np.bincount(y, minlength=len(id2fam))
            top = np.argsort(-sup)[:args.cm_topn]
            labels_top = [id2fam[i] for i in top]
            cm_top = cm_f_norm[np.ix_(top, top)]
            pd.DataFrame(cm_top, index=labels_top, columns=labels_top).to_csv(os.path.join(run_dir, "confusion_family_top30.csv"), index=True)
            plot_confusion(
                cm_top,
                labels_top,
                f"Family confusion top{len(labels_top)} (norm)",
                os.path.join(run_dir, "confusion_family_top30_norm.png"),
                os.path.join(run_dir, "confusion_family_top30_norm.pdf")
            )

            if "y_c" in ev and "p_c" in ev:
                save_reports(ev["y_c"], ev["p_c"], id2cls, os.path.join(run_dir, "per_class_report.csv"))
                cm_c = confusion_matrix(ev["y_c"], ev["p_c"], labels=list(range(len(id2cls))))
                cm_c_norm = normalize_cm(cm_c)
                pd.DataFrame(cm_c, index=[id2cls[i] for i in range(len(id2cls))], columns=[id2cls[i] for i in range(len(id2cls))]).to_csv(
                    os.path.join(run_dir, "confusion_class.csv"), index=True
                )
                plot_confusion(
                    cm_c_norm,
                    [id2cls[i] for i in range(len(id2cls))],
                    "Class confusion (norm)",
                    os.path.join(run_dir, "confusion_class_norm.png"),
                    os.path.join(run_dir, "confusion_class_norm.pdf")
                )

            run_rows.append({
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "macro_f1_family": float(metrics["macro_f1_family"]),
                "macro_f1_class": float(metrics.get("macro_f1_class", np.nan)),
                "best_f1_during_train": float(best_f1),
                "proto_alpha": float(run_cfg_final.proto_alpha),
                "proto_tau": float(run_cfg_final.proto_tau),
                "lambda_class": float(run_cfg_final.lambda_class),
                "moe_beta": float(run_cfg_final.moe_beta),
                "family_reweight": bool(run_cfg_final.family_reweight),
                "contrast_stage2": bool(run_cfg_final.contrast_stage2),
            })

            # Clear CUDA cache between runs
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- Global summaries ----
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)

    # mean±std per exp
    g = runs_df.groupby("exp")["macro_f1_family"]
    summ = g.agg(["mean", "std", "count"]).reset_index()
    summ.columns = ["exp", "macro_f1_family_mean", "macro_f1_family_std", "n"]
    summ.to_csv(os.path.join(args.out, "summary_models.csv"), index=False)

    # Plot comparison
    plt.figure(figsize=(7.4, 4.4))
    order = summ.sort_values("macro_f1_family_mean", ascending=False)["exp"].tolist()
    xs = np.arange(len(order))
    vals = [summ[summ["exp"] == e]["macro_f1_family_mean"].values[0] for e in order]
    err = [summ[summ["exp"] == e]["macro_f1_family_std"].values[0] for e in order]
    plt.bar(xs, vals, yerr=err, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (family)")
    plt.title("Comparative Macro-F1 (family) across models (mean±std)")
    savefig(os.path.join(args.out, "compare_macro_f1_family.png"))
    plt.figure(figsize=(7.4, 4.4))
    plt.bar(xs, vals, yerr=err, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (family)")
    plt.title("Comparative Macro-F1 (family) across models (mean±std)")
    savefig(os.path.join(args.out, "compare_macro_f1_family.pdf"))

    print("\n✅ DONE")
    print(f"Output folder: {args.out}")
    print(f"- summary_runs.csv / summary_models.csv / compare_macro_f1_family.(png|pdf)")
    print("Per run outputs under: out/EXP_NAME/seed_k/ ...")

if __name__ == "__main__":
    main()
