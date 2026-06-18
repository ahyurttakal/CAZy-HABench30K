#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAZy ESM-2 — Q1 Runner  (v3 — full Q1 build)
==============================================
Proposed : ESM-2 + Mean Pooling + Hierarchical Multi-Task
Baselines: CNN, ProtBERT (optional)

YENİ (v3):
  [FIX-1]  Val split — best model VAL F1'e göre seçilir, TEST sadece 1 kez
  [FIX-2]  copy.deepcopy(esm_model) her seed'de — bağımsız ağırlıklar
  [FIX-3]  Yeni GradScaler her seed'de — state leakage yok
  [FIX-4]  Warmup + cosine LR scheduler
  [FIX-5]  MCC (Matthews Correlation Coefficient) tüm evaluate() çıktılarında
  [FIX-6]  MC-Dropout uncertainty quantification (N=20 pass)
  [NEW-1]  Random split vs Homology split karşılaştırması  ← ANA KATKI
  [NEW-2]  Paired permutation test (n=3 seed için geçerli, Bonferroni düzeltmeli)
  [NEW-3]  Calibration curve (reliability diagram)
  [NEW-4]  UMAP / t-SNE embedding görselleştirme
  [NEW-5]  Per-family F1 vs family size scatter (log-log)
  [NEW-6]  Few-shot error bars (std across seeds)
  [NEW-7]  Permutation test p-value (Cohen's d) heatmap
  [NEW-8]  Publication-ready summary table (LaTeX + CSV)

Veri klasörü yapısı (homology):
  train_homology.fasta / labels_train.csv
  val_homology.fasta   / labels_val.csv    ← YENİ (yoksa auto-split)
  test_homology.fasta  / labels_test.csv
  sütunlar: id, class, family

Kullanım:
  python runner_q1_final.py \\
      --data_dir data/splits/ \\
      --out      results/Q1/ \\
      --seeds    1,7,42 \\
      --add_random_split \\
      --fewshot_episodes 200
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import copy
import gc
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
# scipy.stats.wilcoxon kaldırıldı — permutation test kullanılıyor (n=3 için geçerli)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import esm as esm_lib

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import umap as umap_lib
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# =============================================================================
# Style
# =============================================================================

CLASS_COLORS = {
    "GH":  "#E74C3C", "GT": "#3498DB", "PL": "#2ECC71",
    "CE":  "#F39C12", "AA": "#9B59B6", "CBM": "#1ABC9C",
}

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

def save_fig(png: str, pdf: str = None):
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight", dpi=300)
    if pdf:
        plt.savefig(pdf, bbox_inches="tight")
    plt.close()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_cuda(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_available()


# =============================================================================
# Data helpers
# =============================================================================

def norm_id(x: str) -> str:
    s = str(x).strip()
    if "|" in s:
        parts = s.split("|")
        s = parts[1] if len(parts) >= 2 and parts[1] else parts[-1]
    return s.split(".")[0]

def normalize_seq(seq: str) -> str:
    seq = seq.strip().upper().replace("*", "")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    return "".join(c if c in allowed else "X" for c in seq)

def read_fasta_dict(path: str) -> Dict[str, str]:
    d = {}
    for rec in SeqIO.parse(path, "fasta"):
        rid = norm_id(rec.id)
        if rid not in d:
            d[rid] = normalize_seq(str(rec.seq))
    if not d:
        raise RuntimeError(f"FASTA boş/okunamıyor: {path}")
    return d


class CAZyDataset(Dataset):
    def __init__(self, fasta_path: str, labels_csv: str,
                 cls2id: Dict[str, int], fam2id: Dict[str, int]):
        self.seqs = read_fasta_dict(fasta_path)
        df = pd.read_csv(labels_csv).copy()
        req = {"id", "class", "family"}
        if not req.issubset(df.columns):
            raise RuntimeError(f"{labels_csv} sütunları eksik: {req}")
        df["id"]     = df["id"].astype(str).map(norm_id)
        df["class"]  = df["class"].astype(str)
        df["family"] = df["family"].astype(str)
        df = df[df["id"].isin(self.seqs)].reset_index(drop=True)
        n_before = len(df)
        df = df[df["class"].isin(cls2id) & df["family"].isin(fam2id)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"[WARN] {labels_csv}: {n_dropped}/{n_before} satır düşürüldü "
                  f"(eğitim setinde görülmemiş class/family). "
                  f"Kalan: {len(df)} sekans.")
        if len(df) == 0:
            raise RuntimeError(f"Etiket/FASTA örtüşmesi yok: {labels_csv}")
        self.df     = df
        self.cls2id = cls2id
        self.fam2id = fam2id

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]; rid = r["id"]
        return {"id": rid, "seq": self.seqs[rid],
                "y_class": self.cls2id[r["class"]],
                "y_family": self.fam2id[r["family"]]}


def auto_val_split(train_ds: CAZyDataset, val_frac: float = 0.15,
                   seed: int = 42) -> Tuple["CAZyDataset", "CAZyDataset"]:
    """
    Val dosyası yoksa train'den stratified split yap.
    Family bazlı stratification — küçük family'ler test'e sızmasın.
    """
    labels = train_ds.df["family"].tolist()
    idx = np.arange(len(train_ds))
    # Eğer bazı family'ler çok küçükse stratify bozulabilir
    counts = pd.Series(labels).value_counts()
    rare   = set(counts[counts < 2].index)
    safe   = [i for i, l in enumerate(labels) if l not in rare]
    rare_i = [i for i, l in enumerate(labels) if l in rare]

    if len(safe) < 2:
        tr_i, va_i = train_test_split(idx, test_size=val_frac, random_state=seed)
    else:
        safe_labels = [labels[i] for i in safe]
        safe_tr, safe_va = train_test_split(
            safe, test_size=val_frac, random_state=seed, stratify=safe_labels
        )
        tr_i = np.array(safe_tr + rare_i)
        va_i = np.array(safe_va)

    # Subset dataset'leri oluştur
    tr_ds      = copy.copy(train_ds)
    tr_ds.df   = train_ds.df.iloc[tr_i].reset_index(drop=True)
    va_ds      = copy.copy(train_ds)
    va_ds.df   = train_ds.df.iloc[va_i].reset_index(drop=True)
    return tr_ds, va_ds


def make_random_split_datasets(
    train_ds: CAZyDataset, test_ds: CAZyDataset,
    test_frac: float = 0.20, seed: int = 0
) -> Tuple["CAZyDataset", "CAZyDataset"]:
    """
    [NEW-1] Random split — tüm veriyi birleştir, family-stratified böl.
    Nadir family'lerin test setine tamamen düşmesini önlemek için
    stratified split yapılır. Bu sayede ΔF1 farkı gerçekten homology
    leakage'dan kaynaklanır, nadir family datasızlığından değil.
    """
    combined_df   = pd.concat([train_ds.df, test_ds.df], ignore_index=True)
    combined_seqs = {**train_ds.seqs, **test_ds.seqs}
    idx           = np.arange(len(combined_df))

    # Stratify için family label'ları — çok nadir family'ler tek grup
    fam_labels  = combined_df["family"].values
    fam_counts  = pd.Series(fam_labels).value_counts()
    # ≥2 örnekli family'lerde stratify, tek örneklileri train'e koy
    singleton_mask = np.array([fam_counts.get(f, 0) < 2 for f in fam_labels])
    safe_idx   = idx[~singleton_mask]
    single_idx = idx[singleton_mask]

    if len(safe_idx) < 4:
        # Yeterli veri yoksa unstratified
        tr_i, te_i = train_test_split(idx, test_size=test_frac, random_state=seed)
    else:
        safe_labels = fam_labels[safe_idx]
        safe_tr, safe_te = train_test_split(
            safe_idx, test_size=test_frac, random_state=seed, stratify=safe_labels
        )
        # Singleton'lar her zaman train'e
        tr_i = np.concatenate([safe_tr, single_idx])
        te_i = safe_te

    def _make(indices):
        ds      = copy.copy(train_ds)
        ds.df   = combined_df.iloc[indices].reset_index(drop=True)
        ds.seqs = combined_seqs
        return ds

    return _make(tr_i), _make(te_i)


# =============================================================================
# Collators
# =============================================================================

class ESMCollator:
    def __init__(self, alphabet, max_len: int = 768):
        self.bc = alphabet.get_batch_converter(); self.max_len = max_len

    def __call__(self, batch):
        items, yc, yf, ids = [], [], [], []
        for b in batch:
            s = b["seq"][:self.max_len] if self.max_len else b["seq"]
            items.append((b["id"], s)); ids.append(b["id"])
            yc.append(b["y_class"]); yf.append(b["y_family"])
        _, _, tokens = self.bc(items)
        mask = (tokens != 1).long()
        return ids, tokens, mask, torch.tensor(yc, dtype=torch.long), torch.tensor(yf, dtype=torch.long)


class CNNRawCollator:
    AA   = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    AA2I = {a: i+1 for i, a in enumerate(AA)}
    def __init__(self, max_len: int = 768): self.max_len = max_len

    def __call__(self, batch):
        yc, yf, ids, xs = [], [], [], []
        for b in batch:
            s = b["seq"][:self.max_len] if self.max_len else b["seq"]
            xs.append([self.AA2I.get(c, self.AA2I["X"]) for c in s])
            ids.append(b["id"]); yc.append(b["y_class"]); yf.append(b["y_family"])
        L = max(len(x) for x in xs)
        X = torch.zeros(len(xs), L, dtype=torch.long)
        M = torch.zeros(len(xs), L, dtype=torch.long)
        for i, x in enumerate(xs):
            X[i, :len(x)] = torch.tensor(x, dtype=torch.long); M[i, :len(x)] = 1
        return ids, X, M, torch.tensor(yc, dtype=torch.long), torch.tensor(yf, dtype=torch.long)


class ProtBERTCollator:
    def __init__(self, tokenizer, max_len: int = 768):
        self.tokenizer = tokenizer; self.max_len = max_len

    def __call__(self, batch):
        ids, seqs, yc, yf = [], [], [], []
        for b in batch:
            s = b["seq"][:self.max_len] if self.max_len else b["seq"]
            ids.append(b["id"]); seqs.append(" ".join(list(s)))
            yc.append(b["y_class"]); yf.append(b["y_family"])
        tok = self.tokenizer(seqs, return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_len)
        return ids, tok["input_ids"], tok["attention_mask"], \
               torch.tensor(yc, dtype=torch.long), torch.tensor(yf, dtype=torch.long)


# =============================================================================
# Models
# =============================================================================

class MeanPool(nn.Module):
    def forward(self, H, mask):
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp_min(1.0)

class Head(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, out_dim)
        )
    def forward(self, z): return self.net(z)


class ESMBackbone(nn.Module):
    def __init__(self, esm_model, freeze=True):
        super().__init__()
        self.esm = esm_model; self.d = esm_model.embed_dim
        self.last_layer = int(getattr(esm_model, "num_layers"))
        self.set_freeze(freeze)

    def set_freeze(self, freeze):
        for p in self.esm.parameters(): p.requires_grad = not freeze

    def unfreeze_last_n_layers(self, n):
        for p in self.esm.parameters(): p.requires_grad = False
        if n <= 0: return
        for attr in ["emb_layer_norm_after", "lm_head", "contact_head"]:
            mod = getattr(self.esm, attr, None)
            if mod is not None:
                for p in mod.parameters(): p.requires_grad = True
        layers = getattr(self.esm, "layers", None)
        if layers is None:
            for p in self.esm.parameters(): p.requires_grad = True; return
        for i in range(max(0, len(layers) - n), len(layers)):
            for p in layers[i].parameters(): p.requires_grad = True

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.last_layer], return_contacts=False)
        return out["representations"][self.last_layer]


class ESMHierModel(nn.Module):
    def __init__(self, esm_model, n_cls, n_fam, freeze_backbone=True):
        super().__init__()
        self.backbone = ESMBackbone(esm_model, freeze=freeze_backbone)
        d = self.backbone.d
        self.pool = MeanPool()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1))
        self.head_c = Head(d, n_cls); self.head_f = Head(d, n_fam)

    def unfreeze_last_n_layers(self, n): self.backbone.unfreeze_last_n_layers(n)

    def forward(self, tokens, mask):
        H = self.backbone(tokens); z = self.pool(H, mask); z = self.proj(z)
        return self.head_c(z), self.head_f(z), z


class ProtBERTBackbone(nn.Module):
    def __init__(self, model, freeze=True):
        super().__init__()
        self.model = model; self.d = model.config.hidden_size; self.set_freeze(freeze)

    def set_freeze(self, freeze):
        for p in self.model.parameters(): p.requires_grad = not freeze

    def unfreeze_last_n_layers(self, n):
        for p in self.model.parameters(): p.requires_grad = False
        encoder = getattr(self.model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            for p in self.model.parameters(): p.requires_grad = True; return
        for i in range(max(0, len(encoder.layer) - n), len(encoder.layer)):
            for p in encoder.layer[i].parameters(): p.requires_grad = True

    def forward(self, tokens, mask):
        return self.model(input_ids=tokens, attention_mask=mask).last_hidden_state


class ProtBERTHierModel(nn.Module):
    def __init__(self, backbone_model, n_cls, n_fam, freeze_backbone=True):
        super().__init__()
        self.backbone = ProtBERTBackbone(backbone_model, freeze=freeze_backbone)
        d = self.backbone.d
        self.pool = MeanPool()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1))
        self.head_c = Head(d, n_cls); self.head_f = Head(d, n_fam)

    def unfreeze_last_n_layers(self, n): self.backbone.unfreeze_last_n_layers(n)

    def forward(self, tokens, mask):
        H = self.backbone(tokens, mask); z = self.pool(H, mask); z = self.proj(z)
        return self.head_c(z), self.head_f(z), z


class CNNBaseline(nn.Module):
    def __init__(self, n_cls, n_fam, emb=64):
        super().__init__()
        vocab = len(CNNRawCollator.AA) + 1
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 128, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2), nn.GELU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head_f = Head(128, n_fam); self.head_c = Head(128, n_cls)

    def forward(self, tokens, mask):
        z = self.conv(self.emb(tokens).transpose(1,2)).squeeze(-1)
        return self.head_c(z), self.head_f(z), z


# =============================================================================
# [FIX-4] Warmup + Cosine LR Scheduler
# =============================================================================

def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup sonrası cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Few-shot (prototype-based)
# =============================================================================

@torch.no_grad()
def collect_embeddings(model, loader, device, fam_ids) -> Tuple[np.ndarray, np.ndarray]:
    model.eval(); Z, Y = [], []; fam_set = set(fam_ids)
    for _, tokens, mask, _, yf in tqdm(loader, desc="embed", leave=False):
        keep = [i for i, f in enumerate(yf.tolist()) if f in fam_set]
        if not keep: continue
        _, _, z = model(tokens[keep].to(device), mask[keep].to(device))
        Z.append(z.detach().cpu().float().numpy())
        Y.append(np.array([yf.tolist()[i] for i in keep], dtype=np.int64))
    if not Z:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(Z), np.concatenate(Y)


def prototype_predict(z_query, z_support, y_support, fam_ids):
    fam_ids = sorted(set(fam_ids))
    P = np.stack([
        (lambda v: v / (np.linalg.norm(v) + 1e-8))(z_support[y_support == f].mean(0))
        for f in fam_ids
    ])
    Z = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + 1e-8)
    return np.array([fam_ids[i] for i in (Z @ P.T).argmax(axis=1)], dtype=np.int64)


def fewshot_episode_eval(train_Z, train_Y, test_Z, test_Y,
                          fam_ids, shots, episodes, seed) -> Dict:
    rng = np.random.default_rng(seed)
    fam_ids = [f for f in fam_ids
               if (train_Y == f).sum() >= shots and (test_Y == f).sum() >= 1]
    if len(fam_ids) < 2:
        return {"macro_f1": np.nan, "macro_f1_std": np.nan,
                "n_families": len(fam_ids), "shots": shots, "episodes": episodes}
    scores = []
    for _ in range(episodes):
        sup_i, q_i, used = [], [], []
        for f in fam_ids:
            tr_i = np.where(train_Y == f)[0]
            te_i = np.where(test_Y == f)[0]
            if len(tr_i) < shots or len(te_i) < 1: continue
            sup_i.extend(rng.choice(tr_i, shots, replace=False).tolist())
            q_i.extend(te_i.tolist()); used.append(f)
        if len(set(used)) < 2: continue
        scores.append(f1_score(test_Y[q_i],
                               prototype_predict(test_Z[q_i], train_Z[sup_i],
                                                 train_Y[sup_i], used),
                               average="macro"))
    return {
        "macro_f1":     float(np.mean(scores)) if scores else np.nan,
        "macro_f1_std": float(np.std(scores))  if scores else np.nan,
        "n_families":   len(fam_ids), "shots": shots, "episodes": episodes,
    }


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainCfg:
    batch_size: int = 16; max_len: int = 768; wd: float = 0.01
    grad_clip: float = 1.0; amp: bool = True; grad_accum: int = 2
    lambda_class: float = 0.5; family_reweight: bool = True
    unfreeze_last_n: int = 2; warmup_frac: float = 0.1


def train_epoch(model, loader, opt, sched, device, cfg: TrainCfg,
                scaler: GradScaler, fam_weight):
    model.train(); total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (_, tokens, mask, yc, yf) in enumerate(
            tqdm(loader, desc="train", leave=False), start=1):
        tokens, mask = tokens.to(device), mask.to(device)
        yc, yf = yc.to(device), yf.to(device)
        with autocast("cuda", enabled=(cfg.amp and is_cuda(device))):
            lc, lf, _ = model(tokens, mask)
            loss_f = F.cross_entropy(lf, yf,
                                     weight=fam_weight if cfg.family_reweight else None)
            loss_c = F.cross_entropy(lc, yc)
            loss   = (loss_f + cfg.lambda_class * loss_c) / cfg.grad_accum
        if cfg.amp and is_cuda(device):
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % cfg.grad_accum == 0:
            if cfg.amp and is_cuda(device):
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            opt.zero_grad(set_to_none=True)
            if sched: sched.step()
        total += float(loss.detach().cpu()) * cfg.grad_accum; n += 1
    # leftover
    if len(loader) % cfg.grad_accum != 0:
        if cfg.amp and is_cuda(device):
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        opt.zero_grad(set_to_none=True)
        if sched: sched.step()
    return total / max(n, 1)


# =============================================================================
# [FIX-5] Evaluate — MCC eklendi
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    ytf, ypf, ytc, ypc = [], [], [], []
    for _, tokens, mask, yc, yf in tqdm(loader, desc="eval", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        lc, lf, _ = model(tokens, mask)
        ypf += lf.argmax(1).cpu().tolist(); ytf += yf.tolist()
        ypc += lc.argmax(1).cpu().tolist(); ytc += yc.tolist()
    return {
        "macro_f1_family":    float(f1_score(ytf, ypf, average="macro", zero_division=0)),
        "balanced_acc_family":float(balanced_accuracy_score(ytf, ypf)),
        "mcc_family":         float(matthews_corrcoef(ytf, ypf)),
        "macro_f1_class":     float(f1_score(ytc, ypc, average="macro", zero_division=0)),
        "mcc_class":          float(matthews_corrcoef(ytc, ypc)),
        "y_f_true": ytf, "y_f_pred": ypf,
        "y_c_true": ytc, "y_c_pred": ypc,
    }


# =============================================================================
# [FIX-6] MC-Dropout Uncertainty + Calibration
# =============================================================================

def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()


@torch.no_grad()
def mc_dropout_eval(model, loader, device, n_passes: int = 20) -> Dict:
    """
    MC-Dropout: N stochastic forward pass.
    Returns per-sequence confidence, epistemic uncertainty, ECE, F1, MCC.
    ECE doğrudan burada hesaplanır — dışarıya NaN dönmez.
    """
    model.eval(); _enable_dropout(model)
    all_conf, all_uncert, all_true, all_pred = [], [], [], []
    for _, tokens, mask, _, yf in tqdm(loader, desc="MC-Dropout", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        probs_list = []
        for _ in range(n_passes):
            _, lf, _ = model(tokens, mask)
            probs_list.append(F.softmax(lf, -1).cpu().float().numpy())
        probs_arr = np.stack(probs_list)           # (n_passes, B, n_fam)
        probs     = probs_arr.mean(0)               # (B, n_fam) mean
        std_arr   = probs_arr.std(0)                # (B, n_fam) epistemic uncertainty
        pred      = probs.argmax(1)                 # (B,)
        conf      = probs.max(1)                    # (B,) mean confidence
        uncert    = std_arr[np.arange(len(pred)), pred]  # uncertainty of predicted class
        all_conf.extend(conf.tolist())
        all_uncert.extend(uncert.tolist())
        all_true.extend(yf.tolist())
        all_pred.extend(pred.tolist())
    model.eval()  # dropout off

    confs   = np.array(all_conf)
    correct = np.array([p == t for p, t in zip(all_pred, all_true)], dtype=bool)

    # ECE (Expected Calibration Error) — 15 eşit genişlikte bin
    n_bins = 15
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf_m, bin_count = [], [], []
    for i in range(n_bins):
        lo, hi  = bins[i], bins[i + 1]
        mask_b  = (confs >= lo) & (confs < hi)
        if mask_b.sum() == 0:
            continue
        bin_acc.append(float(correct[mask_b].mean()))
        bin_conf_m.append(float(confs[mask_b].mean()))
        bin_count.append(int(mask_b.sum()))
    ece = float(
        sum(abs(a - c) * n for a, c, n in zip(bin_acc, bin_conf_m, bin_count))
        / max(sum(bin_count), 1)
    )

    return {
        "confidences": confs,
        "uncertainty": np.array(all_uncert),
        "correct":     correct,
        "predictions": np.array(all_pred),
        "true":        np.array(all_true),
        "ece":         ece,          # float, artık NaN değil
        "macro_f1_mc": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "mcc_mc":      float(matthews_corrcoef(all_true, all_pred)),
        "bin_acc":     bin_acc,
        "bin_conf":    bin_conf_m,
        "bin_count":   bin_count,
    }


def plot_calibration_curve(mc_result: Dict, out_dir: str, tag: str = ""):
    """
    [NEW-3] Reliability diagram (calibration curve).
    ECE mc_dropout_eval içinde hesaplanmış olarak gelir — burada yeniden hesaplanmaz.
    """
    ece      = mc_result["ece"]
    bin_conf = mc_result["bin_conf"]
    bin_acc  = mc_result["bin_acc"]
    bin_count= mc_result["bin_count"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    scatter = ax.scatter(bin_conf, bin_acc, c=bin_count, s=80,
                         cmap="Blues", edgecolors="k", linewidths=0.5, zorder=3)
    plt.colorbar(scatter, ax=ax, label="Sample count per bin")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Calibration Curve{(' — ' + tag) if tag else ''}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    name = f"calibration_curve{'_' + tag if tag else ''}"
    save_fig(os.path.join(out_dir, name + ".png"), os.path.join(out_dir, name + ".pdf"))
    return ece


# =============================================================================
# [NEW-4] UMAP / t-SNE Embedding Görselleştirme
# =============================================================================

@torch.no_grad()
def collect_embeddings_with_meta(model, loader, device, id2fam, id2cls) -> Dict:
    model.eval(); Z, YF, YC = [], [], []
    for _, tokens, mask, yc, yf in tqdm(loader, desc="embed-vis", leave=False):
        _, _, z = model(tokens.to(device), mask.to(device))
        Z.append(z.detach().cpu().float().numpy())
        YF.append(yf.numpy()); YC.append(yc.numpy())
    Z  = np.concatenate(Z)
    YF = np.concatenate(YF); YC = np.concatenate(YC)
    return {"Z": Z, "YF": YF, "YC": YC}


def plot_embedding_space(emb_data: Dict, id2cls: Dict, out_dir: str, tag: str = ""):
    """
    [NEW-4] UMAP (veya t-SNE fallback) ile embedding uzayı.
    CAZy class bazlı renklendirilmiş scatter.
    """
    Z  = emb_data["Z"]; YC = emb_data["YC"]
    n  = min(len(Z), 5000)  # Bellek için örnekle
    if n < len(Z):
        rng_sub = np.random.default_rng(42)           # reproducible subsample
        idx = rng_sub.choice(len(Z), n, replace=False)
        Z = Z[idx]; YC = YC[idx]

    print(f"[UMAP] {len(Z)} nokta için boyut indirgeme...")
    if HAS_UMAP:
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        coords  = reducer.fit_transform(Z)
        method  = "UMAP"
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1)).fit_transform(Z)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_id, cls_name in id2cls.items():
        mask = YC == cls_id
        if mask.sum() == 0: continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CLASS_COLORS.get(cls_name, "#95A5A6"),
                   s=5, alpha=0.5, label=cls_name, linewidths=0)
    ax.set_xlabel(f"{method} dim-1"); ax.set_ylabel(f"{method} dim-2")
    ax.set_title(f"ESM-2 Embedding Space — {method}{(' ('+tag+')') if tag else ''}")
    ax.legend(markerscale=4, frameon=False, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    name = f"embedding_{method.lower()}{'_'+tag if tag else ''}"
    save_fig(os.path.join(out_dir, name+".png"), os.path.join(out_dir, name+".pdf"))


# =============================================================================
# [NEW-5] Per-family F1 vs Family Size
# =============================================================================

def plot_family_f1_vs_size(per_fam_csv: str, fam_dist_csv: str, out_dir: str, tag: str = ""):
    """
    [NEW-5] Log-log scatter: family size (train) vs test F1.
    Rare family performance = en önemli gösterge.
    """
    try:
        rep_df  = pd.read_csv(per_fam_csv)
        dist_df = pd.read_csv(fam_dist_csv)
    except Exception as e:
        print(f"[WARN] per-family scatter: {e}"); return

    rep_df  = rep_df[rep_df["label"].str.match(r'^[A-Z]{2,3}\d+')]  # GH/GT/PL/CE/AA + CBM
    merged  = rep_df.merge(dist_df[["family","train_count"]], left_on="label", right_on="family", how="inner")
    if merged.empty: return

    cls_col = merged["label"].str.extract(r'^(GH|GT|PL|CE|AA|CBM)')[0].fillna("OTHER")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls_name, grp in merged.groupby(cls_col):
        ax.scatter(grp["train_count"], grp["f1-score"],
                   c=CLASS_COLORS.get(cls_name, "#95A5A6"),
                   s=25, alpha=0.7, label=cls_name, linewidths=0)
    # Trend line
    x = np.log10(merged["train_count"].values.clip(1))
    y = merged["f1-score"].values
    if len(x) > 5:
        m, b = np.polyfit(x, y, 1)
        xs   = np.linspace(x.min(), x.max(), 100)
        ax.plot(10**xs, m*xs + b, "k--", alpha=0.4, lw=1.2, label=f"Trend (slope={m:.3f})")
    ax.set_xscale("log"); ax.set_xlim(left=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("F1-score (test set)")
    ax.set_title(f"Per-family F1 vs Training Size{(' — '+tag) if tag else ''}")
    ax.legend(markerscale=2, frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    name = f"family_f1_vs_size{'_'+tag if tag else ''}"
    save_fig(os.path.join(out_dir, name+".png"), os.path.join(out_dir, name+".pdf"))


# =============================================================================
# [NEW-2] Permutation Test — n'den bağımsız, n=3 için de geçerli
# =============================================================================

def run_permutation_tests(runs_df: pd.DataFrame, metric: str, out_dir: str,
                           n_perms: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """
    Paired permutation test (aka randomization test).
    Wilcoxon signed-rank testi n<10 için minimum ulaşılabilir p ≈ 0.0625 sınırına
    takılır. Permutation test bu sınırı aşar: n=3'te bile p≈0.125 (tek taraflı)
    elde edilebilir, iki taraflı p≈0.25 — bu zaten elimizdeki verinin bilgi sınırı.

    Test istatistiği: observed mean difference
    Null: iki modelin skorları değiştirilebilir (paired by seed)
    Bonferroni düzeltmesi model çifti sayısına göre uygulanır.
    Effect size: Cohen's d (paired)
    """
    from itertools import combinations
    rng    = np.random.default_rng(seed)
    exps   = runs_df["exp"].unique().tolist()
    n_comp = len(list(combinations(exps, 2)))
    rows   = []

    for a, b in combinations(exps, 2):
        va = runs_df[runs_df["exp"] == a][metric].dropna().values
        vb = runs_df[runs_df["exp"] == b][metric].dropna().values
        n  = min(len(va), len(vb))

        if n < 2:
            rows.append({"model_a": a, "model_b": b,
                         "observed_delta": np.nan, "p_value": np.nan,
                         "p_bonferroni": np.nan, "cohens_d": np.nan,
                         "mean_a": float(np.mean(va)) if len(va) else np.nan,
                         "mean_b": float(np.mean(vb)) if len(vb) else np.nan, "n": n})
            continue

        va, vb      = va[:n], vb[:n]
        diffs       = va - vb
        obs_delta   = float(diffs.mean())

        # Permutation: her çift için ±1 işareti rastgele ata
        perm_deltas = np.array([
            (rng.choice([-1.0, 1.0], size=n) * diffs).mean()
            for _ in range(n_perms)
        ])
        # İki taraflı p
        p = float(np.mean(np.abs(perm_deltas) >= abs(obs_delta)))
        p = max(p, 1.0 / n_perms)   # 0 p değerinden kaçın

        # Cohen's d (paired)
        std_d = float(diffs.std(ddof=1))
        d     = (obs_delta / std_d) if std_d > 0 else 0.0

        rows.append({
            "model_a":       a,
            "model_b":       b,
            "observed_delta":round(obs_delta, 5),
            "p_value":       round(p, 6),
            "p_bonferroni":  round(min(p * n_comp, 1.0), 6),
            "cohens_d":      round(d, 4),
            "mean_a":        round(float(np.mean(va)), 5),
            "mean_b":        round(float(np.mean(vb)), 5),
            "n":             n,
            "n_perms":       n_perms,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"permtest_{metric}.csv"), index=False)

    if len(exps) >= 2:
        _plot_pvalue_heatmap(df, exps, out_dir, metric)

    return df


def _plot_pvalue_heatmap(perm_df: pd.DataFrame, exps: List[str],
                          out_dir: str, metric: str):
    """
    n=3'te minimum p=0.25 (Bonferroni ile 0.75), dolayısıyla p heatmap'i
    düz kırmızı çıkar ve yanıltıcı olur. Bunun yerine Cohen's d gösterilir
    — tablo ile tutarlı, etki büyüklüğü reviewerlar tarafından kabul görür.
    """
    n = len(exps); idx = {e: i for i, e in enumerate(exps)}
    mat = np.zeros((n, n))   # 0 = no effect on diagonal
    for _, row in perm_df.iterrows():
        i = idx.get(row["model_a"]); j = idx.get(row["model_b"])
        if i is None or j is None: continue
        d = row.get("cohens_d", np.nan)
        if not np.isnan(d):
            mat[i, j] = abs(d)   # symmetric, magnitude only
            mat[j, i] = abs(d)

    labels = [e.replace("Proposed_", "P_").replace("Baseline_", "B_") for e in exps]
    fig, ax = plt.subplots(figsize=(max(5, 0.7*n), max(4, 0.6*n)))
    im = ax.imshow(mat, vmin=0, vmax=2.0, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="|Cohen's d|")
    # Reference lines on colorbar
    for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        cbar.ax.axhline(thresh / 2.0, color="k", lw=0.6, alpha=0.4)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                v = mat[i, j]
                star = ("***" if v > 0.8 else "**" if v > 0.5 else "*" if v > 0.2 else "")
                txt  = f"{v:.2f}{star}" if not np.isnan(v) else "?"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="black" if v < 1.4 else "white")
    ax.set_title(f"|Cohen's d| — {metric}\n"
                 f"(n=3 seeds; *|d|>0.2, **|d|>0.5, ***|d|>0.8)")
    save_fig(os.path.join(out_dir, f"permtest_heatmap_{metric}.png"),
             os.path.join(out_dir, f"permtest_heatmap_{metric}.pdf"))


# =============================================================================
# [NEW-1] Random vs Homology Split Karşılaştırma Görseli (ANA KATKI)
# =============================================================================

def plot_homology_leakage(homo_df: pd.DataFrame, rand_df: pd.DataFrame,
                           metric: str, out_dir: str):
    """
    [NEW-1] Homology-aware vs Random split performans karşılaştırması.
    ΔF1 = random_F1 − homology_F1 = homology leakage miktarı
    Bu grafik makalenin ana katkısını gösterir.
    """
    homo_g = homo_df.groupby("exp")[metric].agg(["mean","std"]).reset_index()
    rand_g = rand_df.groupby("exp")[metric].agg(["mean","std"]).reset_index()
    merged = homo_g.merge(rand_g, on="exp", suffixes=("_homo","_rand"))
    merged["delta"] = merged["mean_rand"] - merged["mean_homo"]
    merged = merged.sort_values("delta", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: grouped bar
    ax = axes[0]; n = len(merged); xs = np.arange(n); w = 0.35
    bars1 = ax.bar(xs - w/2, merged["mean_homo"], w, yerr=merged["std_homo"],
                   capsize=3, label="Homology-aware", color="#3498DB", alpha=0.85)
    bars2 = ax.bar(xs + w/2, merged["mean_rand"],  w, yerr=merged["std_rand"],
                   capsize=3, label="Random split",   color="#E74C3C", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([e.replace("Proposed_","P_").replace("Baseline_","B_")
                        for e in merged["exp"]], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.set_title("(A) Homology-Aware vs Random Split")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.05, merged[["mean_homo","mean_rand"]].max().max() + 0.1))

    # Panel B: delta (leakage) bar chart
    ax2 = axes[1]
    colors = ["#E74C3C" if d > 0 else "#2ECC71" for d in merged["delta"]]
    ax2.barh(range(n), merged["delta"], color=colors, alpha=0.85)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([e.replace("Proposed_","P_").replace("Baseline_","B_")
                          for e in merged["exp"]], fontsize=9)
    ax2.set_xlabel("ΔF1 (random − homology) = Leakage")
    ax2.set_title("(B) Homology Leakage per Model\n"
                  "(positive = inflated by sequence similarity)")
    ax2.grid(axis="x", alpha=0.3)
    for i, (d, row) in enumerate(zip(merged["delta"], merged.itertuples())):
        ax2.text(d + 0.002 if d >= 0 else d - 0.002,
                 i, f"{d:+.3f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9)

    save_fig(os.path.join(out_dir, "homology_leakage_comparison.png"),
             os.path.join(out_dir, "homology_leakage_comparison.pdf"))

    merged.to_csv(os.path.join(out_dir, "homology_leakage_table.csv"), index=False)
    print(f"\n[LEAKAGE] Ortalama leakage ({metric}): {merged['delta'].mean():+.4f}")


# =============================================================================
# Raporlama yardımcıları
# =============================================================================

def normalize_cm(cm): s = cm.sum(1, keepdims=True); s[s==0]=1; return cm.astype(np.float32)/s

def plot_confusion(cm, labels, title, out_png, out_pdf=None):
    n   = len(labels)
    fsz = 10 if n<=20 else 9 if n<=35 else 8
    fig, ax = plt.subplots(figsize=(min(24, max(7, 0.38*n)), min(24, max(6, 0.38*n))))
    im = ax.imshow(cm, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=fsz)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fsz)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    save_fig(out_png, out_pdf)

def plot_training_curve(hist_df, out_dir, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(hist_df["epoch"], hist_df["macro_f1_family"], label="Family F1")
    ax1.plot(hist_df["epoch"], hist_df["macro_f1_class"],  label="Class F1")
    s1 = hist_df[hist_df["stage"]=="S1"]["epoch"].max()
    if not np.isnan(s1): ax1.axvline(s1, color="gray", ls="--", alpha=0.5, label="S1→S2")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Macro-F1"); ax1.set_ylim(0,1)
    ax1.legend(frameon=False); ax1.grid(alpha=0.3); ax1.set_title("(A) F1 Curves")
    ax2.plot(hist_df["epoch"], hist_df["loss"], color="coral")
    if not np.isnan(s1): ax2.axvline(s1, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.grid(alpha=0.3)
    ax2.set_title("(B) Training Loss"); plt.suptitle(title, fontsize=12)
    save_fig(os.path.join(out_dir,"training_curve.png"), os.path.join(out_dir,"training_curve.pdf"))

def save_per_family_report(yt, yp, id2name, out_csv):
    labels = [id2name[i] for i in range(len(id2name))]
    rep    = classification_report(yt, yp, target_names=labels, output_dict=True, zero_division=0)
    rows   = [{"label":k,**v} for k,v in rep.items() if isinstance(v,dict) and "f1-score" in v]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def save_dataset_summary(train_ds, test_ds, val_ds, out_dir):
    rows = []
    for name, ds in [("train",train_ds),("val",val_ds),("test",test_ds)]:
        if ds is None: continue
        lens = [len(ds.seqs[r]) for r in ds.df["id"]]
        rows.append({"split":name,"n_sequences":len(ds),
                     "n_classes":ds.df["class"].nunique(),"n_families":ds.df["family"].nunique(),
                     "len_mean":round(np.mean(lens),1),"len_std":round(np.std(lens),1),
                     "len_min":int(np.min(lens)),"len_median":round(np.median(lens),1),
                     "len_max":int(np.max(lens))})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"dataset_summary.csv"), index=False)

    fam_tr = train_ds.df["family"].value_counts().rename_axis("family").reset_index(name="train_count")
    fam_te = test_ds.df["family"].value_counts().rename_axis("family").reset_index(name="test_count")
    fam_dist = fam_tr.merge(fam_te, on="family", how="outer").fillna(0)
    fam_dist["total"] = fam_dist["train_count"] + fam_dist["test_count"]
    fam_dist = fam_dist.sort_values("total", ascending=False)
    fam_dist.to_csv(os.path.join(out_dir,"family_distribution.csv"), index=False)

    topn = min(30, len(fam_dist))
    top  = fam_dist.head(topn)
    plt.figure(figsize=(10, max(4.5, 0.28*topn)))
    plt.barh(top["family"].tolist()[::-1], top["total"].tolist()[::-1], color="#3498DB", alpha=0.8)
    plt.xlabel("Count"); plt.title(f"Top-{topn} CAZy Family Frequencies (train+test)")
    save_fig(os.path.join(out_dir,"family_distribution_top30.png"),
             os.path.join(out_dir,"family_distribution_top30.pdf"))
    return fam_dist


def save_run_outputs(out_dir, metrics, id2cls, id2fam, cm_topn):
    os.makedirs(out_dir, exist_ok=True)
    # metrics.json (y_true/pred listelerini çıkar — çok büyük)
    clean = {k:v for k,v in metrics.items() if not k.startswith("y_")}
    with open(os.path.join(out_dir,"metrics.json"),"w") as f: json.dump(clean,f,indent=2)

    save_per_family_report(metrics["y_f_true"], metrics["y_f_pred"], id2fam,
                           os.path.join(out_dir,"per_family_report.csv"))
    save_per_family_report(metrics["y_c_true"], metrics["y_c_pred"], id2cls,
                           os.path.join(out_dir,"per_class_report.csv"))
    cm_f = confusion_matrix(metrics["y_f_true"], metrics["y_f_pred"],
                             labels=list(range(len(id2fam))))
    pd.DataFrame(cm_f, index=[id2fam[i] for i in range(len(id2fam))],
                 columns=[id2fam[i] for i in range(len(id2fam))]).to_csv(
        os.path.join(out_dir,"confusion_family_full.csv"))
    top_idx = np.argsort(-np.bincount(np.array(metrics["y_f_true"]), minlength=len(id2fam)))[:cm_topn]
    cm_top  = normalize_cm(cm_f)[np.ix_(top_idx,top_idx)]
    labs    = [id2fam[i] for i in top_idx]
    plot_confusion(cm_top, labs, f"Family top-{len(labs)} (row-norm)",
                   os.path.join(out_dir,f"confusion_family_top{cm_topn}_norm.png"),
                   os.path.join(out_dir,f"confusion_family_top{cm_topn}_norm.pdf"))
    cm_c = confusion_matrix(metrics["y_c_true"],metrics["y_c_pred"],labels=list(range(len(id2cls))))
    plot_confusion(normalize_cm(cm_c),[id2cls[i] for i in range(len(id2cls))],
                   "Class confusion (row-norm)",
                   os.path.join(out_dir,"confusion_class_norm.png"),
                   os.path.join(out_dir,"confusion_class_norm.pdf"))


# =============================================================================
# Publication summary table (LaTeX + CSV)
# =============================================================================

def save_summary_table(model_summ: pd.DataFrame, perm_f1: pd.DataFrame,
                        perm_mcc: pd.DataFrame, out_dir: str):
    """
    Q1 makalesi Table 1 formatında özet tablo.
    n=3 seed ile paired permutation testte minimum ulaşılabilir p=0.25
    (2^3=8 permütasyon, iki taraflı), dolayısıyla p<0.05 hiçbir zaman
    ulaşılamaz. Bunun yerine Cohen's d (etki büyüklüğü) eşikleri kullanılır.
    LaTeX ve CSV olarak kaydedilir.
    """
    # Cohen's d eşikleri (Cohen 1988): |d|>0.2 küçük, >0.5 orta, >0.8 büyük
    def asterisk_d(d):
        if pd.isna(d): return ""
        ad = abs(d)
        if ad > 0.8:  return "***"   # büyük etki
        if ad > 0.5:  return "**"    # orta etki
        if ad > 0.2:  return "*"     # küçük etki
        return ""

    # Proposed modelin her baseline ile karşılaştırması için asterisk map
    def get_asterisk_map(perm_df: pd.DataFrame, proposed: str) -> Dict[str, str]:
        if perm_df is None or len(perm_df) == 0:
            return {}
        amap = {}
        for _, row in perm_df.iterrows():
            if row["model_a"] == proposed:
                amap[row["model_b"]] = asterisk_d(row.get("cohens_d", np.nan))
            elif row["model_b"] == proposed:
                amap[row["model_a"]] = asterisk_d(row.get("cohens_d", np.nan))
        return amap

    proposed_name = "Proposed_ESM2_Mean_Hier"
    ast_f1  = get_asterisk_map(perm_f1,  proposed_name)
    ast_mcc = get_asterisk_map(perm_mcc, proposed_name)

    def fmt(row, mean_col, std_col, exp_name="", ast_map=None):
        if mean_col not in row.index: return "—"
        m = row[mean_col]
        if pd.isna(m): return "—"        # CNN için ECE NaN → "—"
        s = row.get(std_col, np.nan)
        star = (ast_map or {}).get(exp_name, "")
        val  = f"{m:.3f}$\\pm${s:.3f}" if not pd.isna(s) else f"{m:.3f}"
        return val + (f"$^{{{star}}}$" if star else "")

    # ECE sütunu ekle
    has_ece = "ece_mean" in model_summ.columns

    cols = ["exp", "macro_f1_family_mean", "macro_f1_family_std",
            "mcc_family_mean", "mcc_family_std",
            "balanced_acc_family_mean", "balanced_acc_family_std",
            "macro_f1_class_mean", "macro_f1_class_std"]
    if has_ece:
        cols += ["ece_mean", "ece_std"]
    avail = [c for c in cols if c in model_summ.columns]
    model_summ[avail].to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    # LaTeX
    n_cols = 5 + (1 if has_ece else 0)
    col_fmt = "l" + "c" * (n_cols - 1)
    header_row = r"Model & Macro-F1 (Fam) & MCC (Fam) & Bal.Acc (Fam) & Macro-F1 (Cls)"
    if has_ece:
        header_row += r" & ECE$\downarrow$"
    header_row += r" \\"

    latex_rows = []
    for _, row in model_summ.iterrows():
        exp  = row["exp"]
        name = exp.replace("Proposed_", "\\textbf{").replace("_", " ")
        if "textbf" in name: name += "}"
        cells = [
            fmt(row, "macro_f1_family_mean",     "macro_f1_family_std",     exp, ast_f1),
            fmt(row, "mcc_family_mean",           "mcc_family_std",           exp, ast_mcc),
            fmt(row, "balanced_acc_family_mean",  "balanced_acc_family_std",  exp, {}),
            fmt(row, "macro_f1_class_mean",       "macro_f1_class_std",       exp, {}),
        ]
        if has_ece:
            cells.append(fmt(row, "ece_mean", "ece_std", exp, {}))
        latex_rows.append(f"  {name} & " + " & ".join(cells) + r" \\")

    header = (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\caption{Model comparison (3-seed mean $\pm$ std, homology-aware split). "
        r"Symbols denote effect size of proposed model vs.\ each baseline "
        r"(paired permutation test, Bonferroni corrected, $n_{\mathrm{perm}}=10\,000$; "
        r"at $k=3$ seeds minimum achievable two-tailed $p=0.25$, so we report "
        r"Cohen's $d$): $^{*}|d|{>}0.2$, $^{**}|d|{>}0.5$, $^{***}|d|{>}0.8$.}" + "\n"
        r"\label{tab:results}" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
        r"\hline" + "\n" +
        header_row + "\n"
        r"\hline"
    )
    footer = r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    note   = ("% ECE = Expected Calibration Error (MC-Dropout, 15 bins, lower is better).\n"
              "% '—' in ECE column = model has no dropout (CNN baseline).\n"
              if has_ece else "")
    with open(os.path.join(out_dir, "table1_latex.tex"), "w") as f:
        f.write(header + "\n" + "\n".join(latex_rows) + "\n" + footer + "\n" + note)


# =============================================================================
# Experiment setup helpers
# =============================================================================

def build_experiments(enable_protbert):
    exps = [{"name":"Proposed_ESM2_Mean_Hier","key":"proposed"},
             {"name":"Baseline_CNN","key":"cnn"}]
    if enable_protbert:
        exps.insert(1, {"name":"Baseline_ProtBERT_Mean_Hier","key":"protbert"})
    return exps

def make_model(exp_key, esm_model, n_cls, n_fam, freeze_backbone, protbert_model=None):
    if exp_key == "proposed":
        # [FIX-2] deepcopy → her seed'de bağımsız ağırlıklar
        return ESMHierModel(copy.deepcopy(esm_model), n_cls, n_fam, freeze_backbone), "esm"
    if exp_key == "protbert":
        if protbert_model is None: raise RuntimeError("ProtBERT modeli yok")
        return ProtBERTHierModel(copy.deepcopy(protbert_model), n_cls, n_fam, freeze_backbone), "protbert"
    if exp_key == "cnn":
        return CNNBaseline(n_cls, n_fam), "cnn"
    raise ValueError(exp_key)

def vram_safe_stage2(cfg):
    c = TrainCfg(**asdict(cfg))
    eff = max(1, c.batch_size * max(1, c.grad_accum))
    c.batch_size = max(1, min(c.batch_size, 2))
    c.grad_accum = min(int(math.ceil(eff / c.batch_size)), 64)
    return c

def build_loader(kind, ds, batch_size, shuffle, max_len, alphabet=None, pb_tok=None):
    coll = (ESMCollator(alphabet, max_len) if kind=="esm" else
            ProtBERTCollator(pb_tok, max_len) if kind=="protbert" else
            CNNRawCollator(max_len))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=coll, pin_memory=is_cuda("cuda"))

def build_fam_weight(train_ds, fam2id, device):
    ids  = train_ds.df["family"].map(fam2id).values
    cnt  = np.bincount(ids, minlength=len(fam2id)).astype(np.float32)
    w    = np.where(cnt>0, 1.0/np.sqrt(cnt), 0.0)
    if w.sum() > 0: w = w / (w[w>0].mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32, device=device)


# =============================================================================
# Core training run
# =============================================================================

def run_one_experiment(
    exp_name, exp_key, seed, args, cfg,
    train_ds, val_ds, test_ds,
    esm_model, alphabet, protbert_model, pb_tok,
    fam2id, id2fam, cls2id, id2cls, fam_ids,
    device, fewshot_shots, tag=""
) -> Tuple[Dict, List[Dict]]:
    """
    Tek experiment + seed çalıştır. Val split ile model selection.
    Returns: (run_metrics_row, fewshot_rows)
    """
    set_seed(seed)
    run_dir = os.path.join(args.out, exp_name + tag, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    model, kind = make_model(exp_key, esm_model, len(cls2id), len(fam2id),
                              freeze_backbone=True, protbert_model=protbert_model)
    model = model.to(device)
    fam_weight = build_fam_weight(train_ds, fam2id, device)

    # [FIX-3] Yeni GradScaler her seed'de
    scaler = GradScaler("cuda", enabled=(cfg.amp and is_cuda(device)))

    run_cfg = TrainCfg(**asdict(cfg))

    # Loader'lar
    def _loader(ds, bs, shuffle):
        return build_loader(kind, ds, bs, shuffle, run_cfg.max_len,
                             alphabet=alphabet, pb_tok=pb_tok)

    tr_loader  = _loader(train_ds, run_cfg.batch_size, True)
    val_loader = _loader(val_ds,   run_cfg.batch_size, False)
    te_loader  = _loader(test_ds,  run_cfg.batch_size, False)

    history   = []
    best_val  = -1.0
    best_state= None

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    total_steps = args.epochs_stage1 * len(tr_loader) // run_cfg.grad_accum
    warmup_steps= max(1, int(total_steps * run_cfg.warmup_frac))
    opt1  = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr_stage1, weight_decay=run_cfg.wd)
    sched1 = get_warmup_cosine_scheduler(opt1, warmup_steps, total_steps)

    for ep in range(1, args.epochs_stage1+1):
        loss = train_epoch(model, tr_loader, opt1, sched1, device, run_cfg, scaler, fam_weight)
        ev   = evaluate(model, val_loader, device)   # [FIX-1] val, not test
        history.append({"epoch":ep,"stage":"S1","loss":loss,**{k:ev[k] for k in ev if not k.startswith("y_")}})
        print(f"  [S1/{exp_name}/seed={seed}] ep={ep:02d} "
              f"loss={loss:.4f} val_F1={ev['macro_f1_family']:.4f} val_MCC={ev['mcc_family']:.4f}")
        if ev["macro_f1_family"] > best_val:
            best_val  = ev["macro_f1_family"]
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if kind != "cnn" and args.epochs_stage2 > 0:
        if best_state: model.load_state_dict(best_state, strict=True)
        gc.collect()
        if is_cuda(device): torch.cuda.empty_cache()
        model.unfreeze_last_n_layers(run_cfg.unfreeze_last_n)
        s2cfg = vram_safe_stage2(run_cfg)
        tr2   = _loader(train_ds, s2cfg.batch_size, True)
        val2  = _loader(val_ds,   s2cfg.batch_size, False)
        te2   = _loader(test_ds,  s2cfg.batch_size, False)
        t2s   = args.epochs_stage2 * len(tr2) // s2cfg.grad_accum
        w2s   = max(1, int(t2s * s2cfg.warmup_frac))
        opt2  = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr_stage2, weight_decay=s2cfg.wd)
        sch2  = get_warmup_cosine_scheduler(opt2, w2s, t2s)
        for ep2 in range(1, args.epochs_stage2+1):
            loss2 = train_epoch(model, tr2, opt2, sch2, device, s2cfg, scaler, fam_weight)
            ev2   = evaluate(model, val2, device)   # [FIX-1] val
            history.append({"epoch":args.epochs_stage1+ep2,"stage":"S2","loss":loss2,
                             **{k:ev2[k] for k in ev2 if not k.startswith("y_")}})
            print(f"  [S2/{exp_name}/seed={seed}] ep={ep2:02d} "
                  f"loss={loss2:.4f} val_F1={ev2['macro_f1_family']:.4f}")
            if ev2["macro_f1_family"] > best_val:
                best_val  = ev2["macro_f1_family"]
                best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
        te_loader = te2; run_cfg = s2cfg

    # ── Final test (1 kez) ────────────────────────────────────────────────────
    if best_state: model.load_state_dict(best_state, strict=True)
    metrics = evaluate(model, te_loader, device)   # [FIX-1] TEST SADECE BURADA
    metrics.update({"exp":exp_name,"seed":seed,"tag":tag})

    save_run_outputs(run_dir, metrics, id2cls, id2fam, args.cm_topn)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(run_dir,"history.csv"), index=False)
    plot_training_curve(hist_df, run_dir, f"{exp_name} seed={seed}{tag}")

    # Checkpoint
    torch.save({"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},
                "metrics": {k:v for k,v in metrics.items() if not k.startswith("y_")},
                "cfg": asdict(run_cfg), "cls2id":cls2id, "fam2id":fam2id},
               os.path.join(run_dir,"best.pt"))

    # ── MC-Dropout calibration ─────────────────────────────────────────────
    if kind in ("esm", "protbert"):
        mc  = mc_dropout_eval(model, te_loader, device, n_passes=args.mc_passes)
        ece = plot_calibration_curve(mc, run_dir, tag=f"{exp_name}_seed{seed}")
        pd.DataFrame([{
            "exp":             exp_name,
            "seed":            seed,
            "tag":             tag,
            "ece":             round(mc["ece"], 6),          # artık NaN değil
            "uncertainty_mean":round(float(mc["uncertainty"].mean()), 6),
            "macro_f1_mc":     mc["macro_f1_mc"],
            "mcc_mc":          mc["mcc_mc"],
        }]).to_csv(os.path.join(run_dir, "mc_dropout.csv"), index=False)

    # ── UMAP (sadece proposed, ilk seed) ───────────────────────────────────
    if exp_key == "proposed" and seed == args.seeds_list[0]:
        emb = collect_embeddings_with_meta(model, te_loader, device, id2fam, id2cls)
        plot_embedding_space(emb, id2cls, run_dir, tag=f"{exp_name}_seed{seed}")

    # ── Few-shot evaluation ────────────────────────────────────────────────
    fewshot_rows = []
    embed_tr = _loader(train_ds, max(1,min(16,run_cfg.batch_size)), False)
    embed_te = _loader(test_ds,  max(1,min(16,run_cfg.batch_size)), False)
    tr_Z, tr_Y = collect_embeddings(model, embed_tr, device, fam_ids)
    te_Z, te_Y = collect_embeddings(model, embed_te, device, fam_ids)
    for shots in fewshot_shots:
        fs = fewshot_episode_eval(tr_Z, tr_Y, te_Z, te_Y, fam_ids,
                                   shots=shots, episodes=args.fewshot_episodes, seed=seed)
        fs.update({"exp":exp_name,"seed":seed,"tag":tag})
        fewshot_rows.append(fs)
    pd.DataFrame(fewshot_rows).to_csv(os.path.join(run_dir,"fewshot.csv"), index=False)

    # Per-family F1 vs size
    plot_family_f1_vs_size(
        os.path.join(run_dir,"per_family_report.csv"),
        os.path.join(args.out,"family_distribution.csv"),
        run_dir, tag=f"{exp_name}_seed{seed}"
    )

    del model; gc.collect()
    if is_cuda(device): torch.cuda.empty_cache()

    run_row = {
        "exp": exp_name, "tag": tag, "seed": seed,
        "macro_f1_family":     metrics["macro_f1_family"],
        "mcc_family":          metrics["mcc_family"],
        "balanced_acc_family": metrics["balanced_acc_family"],
        "macro_f1_class":      metrics["macro_f1_class"],
        "mcc_class":           metrics["mcc_class"],
    }
    return run_row, fewshot_rows


# =============================================================================
# Summary plots
# =============================================================================

def plot_multi_metric_comparison(model_summ: pd.DataFrame, out_dir: str):
    """4 metrik için tek figure — yan yana."""
    metrics = [
        ("macro_f1_family_mean",     "macro_f1_family_std",     "Macro-F1 (Family)"),
        ("mcc_family_mean",          "mcc_family_std",           "MCC (Family)"),
        ("balanced_acc_family_mean", "balanced_acc_family_std",  "Balanced Acc (Family)"),
        ("macro_f1_class_mean",      "macro_f1_class_std",       "Macro-F1 (Class)"),
    ]
    avail = [(a,b,t) for a,b,t in metrics if a in model_summ.columns]
    fig, axes = plt.subplots(1, len(avail), figsize=(4.5*len(avail), 5))
    if len(avail)==1: axes = [axes]
    order = model_summ.sort_values(avail[0][0], ascending=False)["exp"].tolist()
    xs    = np.arange(len(order))
    short = [e.replace("Proposed_","P_").replace("Baseline_","B_") for e in order]
    for ax, (mc, sc, title) in zip(axes, avail):
        vals = [model_summ[model_summ["exp"]==e][mc].values[0] for e in order]
        errs = [model_summ[model_summ["exp"]==e][sc].values[0] for e in order]
        colors = ["#E74C3C" if "Proposed" in e else "#3498DB" for e in order]
        ax.bar(xs, vals, yerr=errs, capsize=4, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(xs); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(title); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, min(1.05, max(v+e for v,e in zip(vals,errs))+0.05))
        for i,(v,e) in enumerate(zip(vals,errs)):
            ax.text(i, v+e+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Model Comparison — Homology-Aware Split", fontsize=13, y=1.02)
    save_fig(os.path.join(out_dir,"compare_all_metrics.png"),
             os.path.join(out_dir,"compare_all_metrics.pdf"))


def plot_fewshot_summary(fewshot_df: pd.DataFrame, out_dir: str):
    """Few-shot comparison: mean ± std across seeds per model."""
    if len(fewshot_df)==0: return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    grouped = fewshot_df.groupby(["exp","shots"])["macro_f1"].agg(["mean","std"]).reset_index()
    for exp_name, sub in grouped.groupby("exp"):
        sub = sub.sort_values("shots")
        color = "#E74C3C" if "Proposed" in exp_name else "#3498DB" if "ESM" in exp_name else "#95A5A6"
        ax.plot(sub["shots"], sub["mean"], marker="o", label=exp_name.replace("_"," "),
                color=color, linewidth=2)
        ax.fill_between(sub["shots"],
                         sub["mean"] - sub["std"].fillna(0),
                         sub["mean"] + sub["std"].fillna(0),
                         alpha=0.15, color=color)
    ax.set_xlabel("Shots per family"); ax.set_ylabel("Prototype Macro-F1")
    ax.set_title("Few-Shot Learning Comparison (mean ± std across seeds)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    save_fig(os.path.join(out_dir,"compare_fewshot.png"),
             os.path.join(out_dir,"compare_fewshot.pdf"))


# =============================================================================
# Main
# =============================================================================

def main():
    set_pub_style()
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    ap.add_argument("--data_dir",  required=True, help="homology split klasörü")
    ap.add_argument("--out",       default="results/Q1_Final")
    ap.add_argument("--only",      default="", help="çalıştırılacak experiment isimleri (virgüllü)")

    # Model
    ap.add_argument("--esm",               default="esm2_t12_35M_UR50D")
    ap.add_argument("--protbert_model",    default="Rostlab/prot_bert")
    ap.add_argument("--disable_protbert",  action="store_true")

    # Eğitim
    ap.add_argument("--epochs_stage1",  type=int,   default=10)
    ap.add_argument("--lr_stage1",      type=float, default=2e-4)
    ap.add_argument("--epochs_stage2",  type=int,   default=5)
    ap.add_argument("--lr_stage2",      type=float, default=5e-5)
    ap.add_argument("--batch_size",     type=int,   default=16)
    ap.add_argument("--grad_accum",     type=int,   default=2)
    ap.add_argument("--max_len",        type=int,   default=768)
    ap.add_argument("--lambda_class",   type=float, default=0.5)
    ap.add_argument("--unfreeze_last_n",type=int,   default=2)
    ap.add_argument("--warmup_frac",    type=float, default=0.1)
    ap.add_argument("--amp",            action="store_true", default=True)
    ap.add_argument("--no_amp",         action="store_true")

    # Değerlendirme
    ap.add_argument("--seeds",             default="1,7,42")
    ap.add_argument("--cm_topn",           type=int, default=30)
    ap.add_argument("--auto_val_frac",     type=float, default=0.15,
                    help="Val dosyası yoksa train'den kaçı val yapılacak")
    ap.add_argument("--mc_passes",         type=int, default=20,
                    help="MC-Dropout forward pass sayısı")
    ap.add_argument("--n_perms",           type=int, default=10_000,
                    help="Permutation test tekrar sayısı")
    ap.add_argument("--fewshot_shots",     default="1,5,10,20")
    ap.add_argument("--fewshot_episodes",  type=int, default=200)

    # [NEW-1] Random split karşılaştırması
    ap.add_argument("--add_random_split",  action="store_true",
                    help="Homology leakage demostrasyon için random split ekle")
    ap.add_argument("--random_split_frac", type=float, default=0.20)

    args = ap.parse_args()

    # Seeds
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    args.seeds_list = seeds
    fewshot_shots   = [int(x.strip()) for x in args.fewshot_shots.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device}  |  seeds={seeds}  |  ESM={args.esm}")

    # ── Veri yükle ────────────────────────────────────────────────────────────
    cfg = TrainCfg(
        batch_size=args.batch_size, max_len=args.max_len,
        amp=(False if args.no_amp else True),
        grad_accum=max(1, args.grad_accum),
        lambda_class=args.lambda_class,
        family_reweight=True,
        unfreeze_last_n=max(0, args.unfreeze_last_n),
        warmup_frac=args.warmup_frac,
    )

    tr_fa   = os.path.join(args.data_dir, "train_homology.fasta")
    te_fa   = os.path.join(args.data_dir, "test_homology.fasta")
    tr_csv  = os.path.join(args.data_dir, "labels_train.csv")
    te_csv  = os.path.join(args.data_dir, "labels_test.csv")
    val_fa  = os.path.join(args.data_dir, "val_homology.fasta")
    val_csv = os.path.join(args.data_dir, "labels_val.csv")

    # Label map
    tr_df = pd.read_csv(tr_csv).copy()
    tr_df["class"] = tr_df["class"].astype(str); tr_df["family"] = tr_df["family"].astype(str)
    cls_order = ["GH","GT","PL","CE","AA","CBM"]
    present   = [c for c in cls_order if c in set(tr_df["class"])] or sorted(tr_df["class"].unique())
    cls2id    = {c:i for i,c in enumerate(present)}; id2cls = {i:c for c,i in cls2id.items()}
    fams      = sorted(tr_df["family"].unique())
    fam2id    = {f:i for i,f in enumerate(fams)};   id2fam = {i:f for f,i in fam2id.items()}
    fam_ids   = sorted(id2fam.keys())

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,"mappings.json"),"w") as f:
        json.dump({"cls2id":cls2id,"fam2id":fam2id},f,indent=2)

    train_ds_full = CAZyDataset(tr_fa, tr_csv, cls2id, fam2id)
    test_ds       = CAZyDataset(te_fa, te_csv, cls2id, fam2id)

    # [FIX-1] Val split
    if os.path.exists(val_fa) and os.path.exists(val_csv):
        print("[DATA] Val dosyası bulundu ✓")
        train_ds = train_ds_full
        val_ds   = CAZyDataset(val_fa, val_csv, cls2id, fam2id)
    else:
        print(f"[DATA] Val dosyası yok — train'den %{int(args.auto_val_frac*100)} auto-split")
        train_ds, val_ds = auto_val_split(train_ds_full, args.auto_val_frac, seed=42)

    print(f"[DATA] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"[DATA] classes={len(cls2id)}  families={len(fam2id)}")

    fam_dist_df = save_dataset_summary(train_ds, test_ds, val_ds, args.out)

    # ── Model yükle ───────────────────────────────────────────────────────────
    print(f"[LOAD] ESM-2: {args.esm}")
    esm_model, alphabet = esm_lib.pretrained.__dict__[args.esm]()
    esm_model = esm_model.to(device)

    enable_protbert = HAS_TRANSFORMERS and (not args.disable_protbert)
    protbert_model = pb_tok = None
    if enable_protbert:
        try:
            print(f"[LOAD] ProtBERT: {args.protbert_model}")
            pb_tok         = AutoTokenizer.from_pretrained(args.protbert_model, do_lower_case=False)
            protbert_model = AutoModel.from_pretrained(args.protbert_model).to(device)
        except Exception as e:
            print(f"[WARN] ProtBERT yüklenemedi: {e}"); enable_protbert = False

    exps = build_experiments(enable_protbert)
    if args.only.strip():
        allow = set(x.strip() for x in args.only.split(","))
        exps  = [e for e in exps if e["name"] in allow]

    # ── Homology split deneyleri ──────────────────────────────────────────────
    runs_rows    = []
    fewshot_rows = []

    for exp in exps:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"[RUN] {exp['name']} | seed={seed} | HOMOLOGY SPLIT")
            print(f"{'='*60}")
            row, fs = run_one_experiment(
                exp["name"], exp["key"], seed, args, cfg,
                train_ds, val_ds, test_ds,
                esm_model, alphabet, protbert_model, pb_tok,
                fam2id, id2fam, cls2id, id2cls, fam_ids,
                device, fewshot_shots, tag=""
            )
            runs_rows.append(row); fewshot_rows.extend(fs)

    # ── [NEW-1] Random split deneyleri ───────────────────────────────────────
    # Adil karşılaştırma için: veri bölmesi SABİT (seed=0), sadece training random
    rand_runs_rows = []
    if args.add_random_split:
        print(f"\n{'='*60}")
        print("[RANDOM SPLIT] Homology leakage demostrasyon başlıyor...")
        print("[RANDOM SPLIT] Veri bölmesi sabit (seed=0) — sadece training seed değişiyor")
        print(f"{'='*60}")
        # Veri bölmesi 1 kez yapılır — seed=0 sabit
        rand_tr_fixed, rand_te_fixed = make_random_split_datasets(
            train_ds_full, test_ds, args.random_split_frac, seed=0
        )
        rand_tr_base, rand_val_base = auto_val_split(rand_tr_fixed, args.auto_val_frac, seed=0)
        for exp in exps:
            for seed in seeds:
                print(f"\n[RUN] {exp['name']} | seed={seed} | RANDOM SPLIT (fixed data)")
                row_r, _ = run_one_experiment(
                    exp["name"], exp["key"], seed, args, cfg,
                    rand_tr_base, rand_val_base, rand_te_fixed,
                    esm_model, alphabet, protbert_model, pb_tok,
                    fam2id, id2fam, cls2id, id2cls, fam_ids,
                    device, fewshot_shots, tag="_RandomSplit"
                )
                rand_runs_rows.append(row_r)

    # ── Summary CSVs ─────────────────────────────────────────────────────────
    runs_df = pd.DataFrame(runs_rows)
    runs_df.to_csv(os.path.join(args.out,"summary_runs.csv"), index=False)

    def _agg(df):
        g = df.groupby("exp")
        agg_cols = ["macro_f1_family","mcc_family","balanced_acc_family",
                     "macro_f1_class","mcc_class"]
        parts = []
        for c in agg_cols:
            if c not in df.columns: continue
            t = g[c].agg(["mean","std"]).reset_index()
            t.columns = ["exp", f"{c}_mean", f"{c}_std"]
            parts.append(t)
        if not parts: return pd.DataFrame()
        from functools import reduce
        return reduce(lambda a,b: a.merge(b,on="exp"), parts)

    model_summ = _agg(runs_df)

    # ECE seed'ler arası ortalama — per-run mc_dropout.csv'lerini topla
    mc_rows = []
    for exp in exps:
        for seed in seeds:
            mc_csv = os.path.join(args.out, exp["name"], f"seed_{seed}", "mc_dropout.csv")
            if os.path.exists(mc_csv):
                mc_rows.append(pd.read_csv(mc_csv))
    if mc_rows:
        mc_df  = pd.concat(mc_rows, ignore_index=True)
        ece_agg = mc_df.groupby("exp")[["ece","uncertainty_mean"]].agg(["mean","std"]).reset_index()
        ece_agg.columns = ["exp", "ece_mean", "ece_std", "uncertainty_mean", "uncertainty_std"]
        model_summ = model_summ.merge(ece_agg, on="exp", how="left")

    model_summ.to_csv(os.path.join(args.out, "summary_models.csv"), index=False)

    fewshot_df = pd.DataFrame(fewshot_rows)
    fewshot_df.to_csv(os.path.join(args.out,"fewshot_summary.csv"), index=False)

    # ── İstatistiksel testler (paired permutation, n=3 için geçerli) ────────────
    if len(runs_df["exp"].unique()) >= 2:
        wf1  = run_permutation_tests(runs_df, "macro_f1_family", args.out, n_perms=args.n_perms)
        wmcc = run_permutation_tests(runs_df, "mcc_family",      args.out, n_perms=args.n_perms)
    else:
        wf1 = wmcc = pd.DataFrame()

    # ── Publication summary table ─────────────────────────────────────────────
    save_summary_table(model_summ, wf1, wmcc, args.out)

    # ── Karşılaştırma grafikleri ───────────────────────────────────────────────
    plot_multi_metric_comparison(model_summ, args.out)
    plot_fewshot_summary(fewshot_df, args.out)

    # ── [NEW-1] Homology leakage görseli ──────────────────────────────────────
    if args.add_random_split and rand_runs_rows:
        rand_df = pd.DataFrame(rand_runs_rows)
        rand_df.to_csv(os.path.join(args.out,"summary_runs_random.csv"), index=False)
        plot_homology_leakage(runs_df, rand_df, "macro_f1_family", args.out)

    # ── Final özet ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  Q1 RUNNER — TAMAMLANDI")
    print("="*70)
    if len(model_summ) > 0:
        print("\n  Model Özeti (Homology Split):")
        cols = [c for c in ["exp","macro_f1_family_mean","mcc_family_mean",
                              "balanced_acc_family_mean"] if c in model_summ.columns]
        print(model_summ[cols].to_string(index=False))
    print(f"\n  Çıktı klasörü: {args.out}")
    print("\n  Temel çıktı dosyaları:")
    print("    summary_models.csv        ← Table 1")
    print("    summary_table.csv + .tex  ← LaTeX Table 1")
    print("    permtest_macro_f1_family.csv ← istatistiksel testler")
    print("    permtest_heatmap_macro_f1_family.(png|pdf)")
    print("    compare_all_metrics.png   ← Figure 2")
    print("    compare_fewshot.png       ← Figure 3")
    if args.add_random_split:
        print("    homology_leakage_comparison.png ← Figure 4 (ANA KATKI)")
        print("    homology_leakage_table.csv")
    print("    embedding_umap/*.png      ← Figure 5")
    print("    calibration_curve*.png    ← Supplementary")
    print("    family_f1_vs_size*.png    ← Supplementary")
    print("="*70)


if __name__ == "__main__":
    main()
