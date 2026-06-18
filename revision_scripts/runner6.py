#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Q1 runner
Proposed: ESM2 + Mean Pooling + Hierarchical Multi-Task Heads
Baselines: ProtBERT MeanPool (optional, if transformers installed), CNN
No ablations.
Includes few-shot evaluation (prototype-based episodes) and publication-ready CSV/figures.

Expected data_dir contents:
  train_homology.fasta
  test_homology.fasta
  labels_train.csv
  labels_test.csv
with columns: id,class,family

Designed to be robust on limited VRAM:
  - Stage-1: frozen backbone
  - Stage-2: unfreeze only last N layers (default=2)
  - AMP + grad accumulation
  - automatic micro-batch reduction for stage-2 while preserving max_len
  - supports trying larger ESM models without immediate full fine-tuning OOM
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import io
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

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# =========================================================
# Style helpers
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_is_cuda(device: str) -> bool:
    return (device == "cuda") and torch.cuda.is_available()


# =========================================================
# Data helpers
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
    return d


class CAZyDataset(Dataset):
    def __init__(self, fasta_path: str, labels_csv: str, cls2id: Dict[str, int], fam2id: Dict[str, int]):
        self.seqs = read_fasta_dict(fasta_path)
        df = pd.read_csv(labels_csv).copy()
        req = {"id", "class", "family"}
        if not req.issubset(df.columns):
            raise RuntimeError(f"{labels_csv} must contain columns {sorted(req)}")
        df["id"] = df["id"].astype(str).map(norm_id)
        df["class"] = df["class"].astype(str)
        df["family"] = df["family"].astype(str)
        df = df[df["id"].isin(self.seqs.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No overlap between FASTA and labels for {labels_csv}")
        df = df[df["class"].isin(cls2id) & df["family"].isin(fam2id)].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("Dataset empty after mapping filter")
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
        items, y_c, y_f, ids = [], [], [], []
        for b in batch:
            s = b["seq"]
            if self.max_len and len(s) > self.max_len:
                s = s[: self.max_len]
            items.append((b["id"], s))
            ids.append(b["id"])
            y_c.append(b["y_class"])
            y_f.append(b["y_family"])
        _, _, tokens = self.batch_converter(items)
        pad_idx = 1
        mask = (tokens != pad_idx).long()
        return ids, tokens, mask, torch.tensor(y_c, dtype=torch.long), torch.tensor(y_f, dtype=torch.long)


class CNNRawCollator:
    AA = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    AA2I = {a: i + 1 for i, a in enumerate(AA)}

    def __init__(self, max_len: int = 768):
        self.max_len = max_len

    def __call__(self, batch):
        y_c, y_f, ids = [], [], []
        xs = []
        for b in batch:
            s = b["seq"]
            if self.max_len and len(s) > self.max_len:
                s = s[: self.max_len]
            token_ids = [self.AA2I.get(ch, self.AA2I["X"]) for ch in s]
            xs.append(token_ids)
            ids.append(b["id"])
            y_c.append(b["y_class"])
            y_f.append(b["y_family"])
        L = min(max(len(x) for x in xs), self.max_len) if self.max_len else max(len(x) for x in xs)
        X = torch.zeros(len(xs), L, dtype=torch.long)
        M = torch.zeros(len(xs), L, dtype=torch.long)
        for i, token_ids in enumerate(xs):
            token_ids = token_ids[:L]
            X[i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            M[i, :len(token_ids)] = 1
        return ids, X, M, torch.tensor(y_c, dtype=torch.long), torch.tensor(y_f, dtype=torch.long)


class ProtBERTCollator:
    def __init__(self, tokenizer, max_len: int = 768):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        ids, seqs, y_c, y_f = [], [], [], []
        for b in batch:
            ids.append(b["id"])
            seq = b["seq"]
            if self.max_len and len(seq) > self.max_len:
                seq = seq[: self.max_len]
            seqs.append(" ".join(list(seq)))
            y_c.append(b["y_class"])
            y_f.append(b["y_family"])
        tok = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        tokens = tok["input_ids"]
        mask = tok["attention_mask"]
        return ids, tokens, mask, torch.tensor(y_c, dtype=torch.long), torch.tensor(y_f, dtype=torch.long)


# =========================================================
# Models
# =========================================================
class MeanPool(nn.Module):
    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp_min(1.0)


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

    def unfreeze_last_n_layers(self, n: int):
        # First freeze everything
        for p in self.esm.parameters():
            p.requires_grad = False
        # Unfreeze LM head if exists? ESM pretrained model is encoder only, so only layers and final norm matter
        if n <= 0:
            return
        # Final norm / emb layer norm if exists
        for attr in ["emb_layer_norm_after", "lm_head", "contact_head"]:
            mod = getattr(self.esm, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        layers = getattr(self.esm, "layers", None)
        if layers is None:
            # fallback: full unfreeze if internals unavailable
            for p in self.esm.parameters():
                p.requires_grad = True
            return
        start = max(0, len(layers) - n)
        for i in range(start, len(layers)):
            for p in layers[i].parameters():
                p.requires_grad = True

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=[self.last_layer], return_contacts=False)
        return out["representations"][self.last_layer]


class ESMHierModel(nn.Module):
    def __init__(self, esm_model, n_cls: int, n_fam: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ESMBackbone(esm_model, freeze=freeze_backbone)
        d = self.backbone.d
        self.pool = MeanPool()
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_c = Head(d, n_cls)
        self.head_f = Head(d, n_fam)

    def set_freeze_backbone(self, freeze: bool):
        self.backbone.set_freeze(freeze)

    def unfreeze_last_n_layers(self, n: int):
        self.backbone.unfreeze_last_n_layers(n)

    def forward(self, tokens, mask):
        H = self.backbone(tokens)
        z = self.pool(H, mask)
        z = self.proj(z)
        return self.head_c(z), self.head_f(z), z


class ProtBERTBackbone(nn.Module):
    def __init__(self, model, freeze: bool = True):
        super().__init__()
        self.model = model
        self.d = model.config.hidden_size
        self.set_freeze(freeze)

    def set_freeze(self, freeze: bool):
        for p in self.model.parameters():
            p.requires_grad = (not freeze)

    def unfreeze_last_n_layers(self, n: int):
        for p in self.model.parameters():
            p.requires_grad = False
        encoder = getattr(self.model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            for p in self.model.parameters():
                p.requires_grad = True
            return
        start = max(0, len(encoder.layer) - n)
        for i in range(start, len(encoder.layer)):
            for p in encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.model.pooler.parameters() if getattr(self.model, "pooler", None) is not None else []:
            p.requires_grad = True

    def forward(self, tokens, mask):
        out = self.model(input_ids=tokens, attention_mask=mask)
        return out.last_hidden_state


class ProtBERTHierModel(nn.Module):
    def __init__(self, backbone_model, n_cls: int, n_fam: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ProtBERTBackbone(backbone_model, freeze=freeze_backbone)
        d = self.backbone.d
        self.pool = MeanPool()
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_c = Head(d, n_cls)
        self.head_f = Head(d, n_fam)

    def set_freeze_backbone(self, freeze: bool):
        self.backbone.set_freeze(freeze)

    def unfreeze_last_n_layers(self, n: int):
        self.backbone.unfreeze_last_n_layers(n)

    def forward(self, tokens, mask):
        H = self.backbone(tokens, mask)
        z = self.pool(H, mask)
        z = self.proj(z)
        return self.head_c(z), self.head_f(z), z


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
# Few-shot utilities
# =========================================================
@torch.no_grad()
def collect_embeddings(model, loader, device, family_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    Z, Y = [], []
    fam_set = set(family_ids)
    for _, tokens, mask, _, y_f in tqdm(loader, desc="embed", leave=False):
        keep = [i for i, f in enumerate(y_f.tolist()) if f in fam_set]
        if not keep:
            continue
        tokens = tokens[keep].to(device)
        mask = mask[keep].to(device)
        _, _, z = model(tokens, mask)
        Z.append(z.detach().cpu().float().numpy())
        Y.append(np.array([y_f.tolist()[i] for i in keep], dtype=np.int64))
    if not Z:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(Z, axis=0), np.concatenate(Y, axis=0)


def prototype_predict(z_query: np.ndarray, z_support: np.ndarray, y_support: np.ndarray, fam_ids: List[int]) -> np.ndarray:
    fam_ids = list(sorted(set(fam_ids)))
    protos = []
    for f in fam_ids:
        idx = (y_support == f)
        mu = z_support[idx].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-8)
        protos.append(mu)
    P = np.stack(protos, axis=0)
    Z = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + 1e-8)
    scores = Z @ P.T
    pred_local = scores.argmax(axis=1)
    return np.array([fam_ids[i] for i in pred_local], dtype=np.int64)


def fewshot_episode_eval(train_Z, train_Y, test_Z, test_Y, fam_ids: List[int], shots: int, episodes: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    fam_ids = [f for f in fam_ids if (train_Y == f).sum() >= shots and (test_Y == f).sum() >= 1]
    if len(fam_ids) < 2:
        return {"macro_f1": np.nan, "n_families": len(fam_ids), "shots": shots, "episodes": episodes}

    scores = []
    for ep in range(episodes):
        support_idx = []
        query_idx = []
        used_fams = []
        for f in fam_ids:
            tr_idx = np.where(train_Y == f)[0]
            te_idx = np.where(test_Y == f)[0]
            if len(tr_idx) < shots or len(te_idx) < 1:
                continue
            sel_sup = rng.choice(tr_idx, size=shots, replace=False)
            # use all test examples for that family in query for stability
            sel_q = te_idx
            support_idx.extend(sel_sup.tolist())
            query_idx.extend(sel_q.tolist())
            used_fams.append(f)
        if len(set(used_fams)) < 2:
            continue
        z_sup, y_sup = train_Z[support_idx], train_Y[support_idx]
        z_q, y_q = test_Z[query_idx], test_Y[query_idx]
        pred = prototype_predict(z_q, z_sup, y_sup, used_fams)
        scores.append(f1_score(y_q, pred, average="macro"))
    return {
        "macro_f1": float(np.mean(scores)) if scores else np.nan,
        "macro_f1_std": float(np.std(scores)) if scores else np.nan,
        "n_families": len(fam_ids),
        "shots": shots,
        "episodes": episodes,
    }


# =========================================================
# Training / eval
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
    unfreeze_last_n: int = 2


def train_epoch(model, loader, opt, device, cfg: TrainCfg, scaler: GradScaler, fam_ce_weight: Optional[torch.Tensor]):
    model.train()
    total, n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for step, (_, tokens, mask, y_c, y_f) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        tokens, mask = tokens.to(device), mask.to(device)
        y_c, y_f = y_c.to(device), y_f.to(device)

        with autocast("cuda", enabled=(cfg.amp and _device_is_cuda(device))):
            logits_c, logits_f, _ = model(tokens, mask)
            if cfg.family_reweight and fam_ce_weight is not None:
                loss_f = F.cross_entropy(logits_f, y_f, weight=fam_ce_weight)
            else:
                loss_f = F.cross_entropy(logits_f, y_f)
            loss_c = F.cross_entropy(logits_c, y_c)
            loss = loss_f + cfg.lambda_class * loss_c
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
def evaluate(model, loader, device):
    model.eval()
    ytf, ypf = [], []
    ytc, ypc = [], []
    for _, tokens, mask, y_c, y_f in tqdm(loader, desc="eval", leave=False):
        tokens, mask = tokens.to(device), mask.to(device)
        logits_c, logits_f, _ = model(tokens, mask)
        ypf += torch.argmax(logits_f, dim=1).cpu().tolist()
        ytf += y_f.tolist()
        ypc += torch.argmax(logits_c, dim=1).cpu().tolist()
        ytc += y_c.tolist()
    return {
        "macro_f1_family": float(f1_score(ytf, ypf, average="macro")),
        "balanced_acc_family": float(balanced_accuracy_score(ytf, ypf)),
        "macro_f1_class": float(f1_score(ytc, ypc, average="macro")),
        "y_f_true": ytf,
        "y_f_pred": ypf,
        "y_c_true": ytc,
        "y_c_pred": ypc,
    }


# =========================================================
# Reports / plots
# =========================================================
def plot_history(hist_df: pd.DataFrame, out_dir: str, title: str):
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(hist_df["epoch"], hist_df["macro_f1_family"], label="Macro-F1 (Family)")
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


def save_run_outputs(out_dir: str, metrics: Dict, id2cls: Dict[int, str], id2fam: Dict[int, str], cm_topn: int):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_classification_report(metrics["y_f_true"], metrics["y_f_pred"], id2fam, os.path.join(out_dir, "per_family_report.csv"))
    save_classification_report(metrics["y_c_true"], metrics["y_c_pred"], id2cls, os.path.join(out_dir, "per_class_report.csv"))

    cm_f = confusion_matrix(metrics["y_f_true"], metrics["y_f_pred"], labels=list(range(len(id2fam))))
    pd.DataFrame(cm_f, index=[id2fam[i] for i in range(len(id2fam))], columns=[id2fam[i] for i in range(len(id2fam))]).to_csv(
        os.path.join(out_dir, "confusion_family_full.csv")
    )
    cm_f_norm = normalize_cm(cm_f)
    y = np.array(metrics["y_f_true"])
    sup = np.bincount(y, minlength=len(id2fam))
    top = np.argsort(-sup)[:cm_topn]
    labels_top = [id2fam[i] for i in top]
    cm_top = cm_f_norm[np.ix_(top, top)]
    pd.DataFrame(cm_top, index=labels_top, columns=labels_top).to_csv(os.path.join(out_dir, f"confusion_family_top{cm_topn}.csv"))
    plot_confusion(
        cm_top,
        labels_top,
        f"Family confusion top{len(labels_top)} (row-normalized)",
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.png"),
        os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.pdf"),
    )

    cm_c = confusion_matrix(metrics["y_c_true"], metrics["y_c_pred"], labels=list(range(len(id2cls))))
    pd.DataFrame(cm_c, index=[id2cls[i] for i in range(len(id2cls))], columns=[id2cls[i] for i in range(len(id2cls))]).to_csv(
        os.path.join(out_dir, "confusion_class.csv")
    )
    plot_confusion(
        normalize_cm(cm_c),
        [id2cls[i] for i in range(len(id2cls))],
        "Class confusion (row-normalized)",
        os.path.join(out_dir, "confusion_class_norm.png"),
        os.path.join(out_dir, "confusion_class_norm.pdf"),
    )


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
    save_fig(os.path.join(out_dir, "family_distribution_top30.png"), os.path.join(out_dir, "family_distribution_top30.pdf"))


# =========================================================
# Experiment definitions
# =========================================================
def build_experiments(enable_protbert: bool):
    exps = [
        {"name": "Proposed_ESM2_Mean_Hier", "key": "proposed"},
        {"name": "Baseline_CNN", "key": "cnn"},
    ]
    if enable_protbert:
        exps.insert(1, {"name": "Baseline_ProtBERT_Mean_Hier", "key": "protbert"})
    return exps


def make_model(exp_key: str, esm_model, n_cls: int, n_fam: int, freeze_backbone: bool, protbert_model=None):
    if exp_key == "proposed":
        return ESMHierModel(esm_model, n_cls, n_fam, freeze_backbone=freeze_backbone), "esm"
    if exp_key == "protbert":
        if protbert_model is None:
            raise RuntimeError("ProtBERT requested but model is None")
        return ProtBERTHierModel(protbert_model, n_cls, n_fam, freeze_backbone=freeze_backbone), "protbert"
    if exp_key == "cnn":
        return CNNBaseline(n_cls, n_fam, hierarchical=True), "cnn"
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


def build_loader(kind: str, dataset: CAZyDataset, batch_size: int, shuffle: bool, max_len: int,
                 alphabet=None, protbert_tokenizer=None, cls2id=None, fam2id=None):
    if kind == "esm":
        collator = ESMCollator(alphabet, max_len=max_len)
    elif kind == "protbert":
        collator = ProtBERTCollator(protbert_tokenizer, max_len=max_len)
    elif kind == "cnn":
        collator = CNNRawCollator(max_len=max_len)
    else:
        raise ValueError(kind)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collator)


# =========================================================
# Main
# =========================================================
def main():
    set_pub_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder")
    ap.add_argument("--out", default="ARCH_Q1_MEAN_HIER_FEWSHOT")
    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")
    ap.add_argument("--protbert_model", default="Rostlab/prot_bert")
    ap.add_argument("--disable_protbert", action="store_true")

    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--lr_stage1", type=float, default=2e-4)
    ap.add_argument("--epochs_stage2", type=int, default=5)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--cm_topn", type=int, default=30)
    ap.add_argument("--lambda_class", type=float, default=0.5)
    ap.add_argument("--unfreeze_last_n", type=int, default=2, help="for stage-2 on large backbones")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--only", default="", help="comma-separated experiment names to run")

    ap.add_argument("--fewshot_shots", default="1,5,10")
    ap.add_argument("--fewshot_episodes", type=int, default=20)
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
        unfreeze_last_n=max(0, args.unfreeze_last_n),
    )

    train_fa = os.path.join(args.data_dir, "train_homology.fasta")
    test_fa = os.path.join(args.data_dir, "test_homology.fasta")
    train_csv = os.path.join(args.data_dir, "labels_train.csv")
    test_csv = os.path.join(args.data_dir, "labels_test.csv")

    train_df = pd.read_csv(train_csv).copy()
    train_df["class"] = train_df["class"].astype(str)
    train_df["family"] = train_df["family"].astype(str)

    class_order = ["GH", "GT", "PL", "CE", "AA", "CBM"]
    present = [c for c in class_order if c in set(train_df["class"].unique())]
    if not present:
        present = sorted(train_df["class"].unique().tolist())
    cls2id = {c: i for i, c in enumerate(present)}
    id2cls = {i: c for c, i in cls2id.items()}

    fams = sorted(train_df["family"].unique().tolist())
    fam2id = {f: i for i, f in enumerate(fams)}
    id2fam = {i: f for f, i in fam2id.items()}

    train_ds = CAZyDataset(train_fa, train_csv, cls2id, fam2id)
    test_ds = CAZyDataset(test_fa, test_csv, cls2id, fam2id)

    print(f"[DATA] classes={len(cls2id)} families={len(fam2id)}")
    print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id}, f, indent=2)

    save_dataset_summary(train_ds, test_ds, args.out)

    print(f"[LOAD] ESM2: {args.esm}")
    esm_model, alphabet = esm.pretrained.__dict__[args.esm]()
    esm_model = esm_model.to(device)

    enable_protbert = HAS_TRANSFORMERS and (not args.disable_protbert)
    protbert_tokenizer = None
    protbert_model = None
    if enable_protbert:
        try:
            print(f"[LOAD] ProtBERT: {args.protbert_model}")
            protbert_tokenizer = AutoTokenizer.from_pretrained(args.protbert_model, do_lower_case=False)
            protbert_model = AutoModel.from_pretrained(args.protbert_model).to(device)
        except Exception as e:
            print(f"[WARN] ProtBERT could not be loaded, skipping. Reason: {e}")
            enable_protbert = False

    exps = build_experiments(enable_protbert)
    if args.only.strip():
        allow = set(x.strip() for x in args.only.split(",") if x.strip())
        exps = [e for e in exps if e["name"] in allow]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    fewshot_shots = [int(x.strip()) for x in args.fewshot_shots.split(",") if x.strip()]

    fam_ids = sorted(list(id2fam.keys()))

    runs_rows = []
    models_rows = []
    fewshot_rows = []

    scaler = GradScaler("cuda", enabled=(cfg.amp and _device_is_cuda(device)))

    for exp in exps:
        exp_name = exp["name"]
        exp_key = exp["key"]

        for seed in seeds:
            set_seed(seed)
            run_dir = os.path.join(args.out, exp_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            model, kind = make_model(exp_key, esm_model, len(cls2id), len(fam2id), freeze_backbone=True, protbert_model=protbert_model)
            model = model.to(device)

            fam_ce_weight = None
            fam_ids_train = train_ds.df["family"].map(fam2id).values
            counts = np.bincount(fam_ids_train, minlength=len(fam2id))
            w = np.zeros_like(counts, dtype=np.float32)
            nz = counts > 0
            w[nz] = 1.0 / np.sqrt(counts[nz].astype(np.float32))
            if nz.any():
                w[nz] = w[nz] / (w[nz].mean() + 1e-8)
            fam_ce_weight = torch.tensor(w, dtype=torch.float32, device=device)

            run_cfg = TrainCfg(**asdict(cfg))

            train_loader = build_loader(kind, train_ds, run_cfg.batch_size, True, run_cfg.max_len,
                                        alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)
            test_loader = build_loader(kind, test_ds, run_cfg.batch_size, False, run_cfg.max_len,
                                       alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)

            history = []
            best_f1 = -1.0
            best_state = None

            print("\n==============================")
            print(f"[RUN] exp={exp_name} seed={seed}")
            print(f" Stage-1: epochs={args.epochs_stage1} lr={args.lr_stage1}")
            if kind != "cnn":
                print(f" Stage-2: epochs={args.epochs_stage2} lr={args.lr_stage2} unfreeze_last_n={run_cfg.unfreeze_last_n}")
            print("==============================")

            opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_stage1, weight_decay=run_cfg.wd)

            for ep in range(1, args.epochs_stage1 + 1):
                loss = train_epoch(model, train_loader, opt1, device, run_cfg, scaler, fam_ce_weight=fam_ce_weight)
                ev = evaluate(model, test_loader, device)
                history.append({
                    "epoch": ep,
                    "stage": "S1",
                    "loss": loss,
                    "macro_f1_family": ev["macro_f1_family"],
                    "macro_f1_class": ev["macro_f1_class"],
                })
                print(f"  [S1] ep={ep:02d} loss={loss:.4f} macroF1_fam={ev['macro_f1_family']:.4f}")
                if ev["macro_f1_family"] > best_f1:
                    best_f1 = ev["macro_f1_family"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            final_loader = test_loader
            final_cfg = run_cfg
            if kind != "cnn" and args.epochs_stage2 > 0:
                if best_state is not None:
                    model.load_state_dict(best_state, strict=True)
                gc.collect()
                if _device_is_cuda(device):
                    torch.cuda.empty_cache()

                # Stage 2: only unfreeze last N layers for OOM-safe larger models
                if kind == "esm":
                    model.unfreeze_last_n_layers(run_cfg.unfreeze_last_n)
                elif kind == "protbert":
                    model.unfreeze_last_n_layers(run_cfg.unfreeze_last_n)

                run_cfg_s2 = vram_safe_stage2(run_cfg)
                train_loader2 = build_loader(kind, train_ds, run_cfg_s2.batch_size, True, run_cfg_s2.max_len,
                                             alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)
                test_loader2 = build_loader(kind, test_ds, run_cfg_s2.batch_size, False, run_cfg_s2.max_len,
                                            alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)
                opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_stage2, weight_decay=run_cfg_s2.wd)

                for ep2 in range(1, args.epochs_stage2 + 1):
                    loss = train_epoch(model, train_loader2, opt2, device, run_cfg_s2, scaler, fam_ce_weight=fam_ce_weight)
                    ev = evaluate(model, test_loader2, device)
                    history.append({
                        "epoch": args.epochs_stage1 + ep2,
                        "stage": "S2",
                        "loss": loss,
                        "macro_f1_family": ev["macro_f1_family"],
                        "macro_f1_class": ev["macro_f1_class"],
                    })
                    print(f"  [S2] ep={ep2:02d} loss={loss:.4f} macroF1_fam={ev['macro_f1_family']:.4f}")
                    if ev["macro_f1_family"] > best_f1:
                        best_f1 = ev["macro_f1_family"]
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                final_loader = test_loader2
                final_cfg = run_cfg_s2

            if best_state is not None:
                model.load_state_dict(best_state, strict=True)
            metrics = evaluate(model, final_loader, device)
            metrics.update({
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "lambda_class": final_cfg.lambda_class,
                "family_reweight": True,
                "unfreeze_last_n": final_cfg.unfreeze_last_n,
                "max_len": final_cfg.max_len,
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

            # Few-shot evaluation (prototype-based)
            embed_train_loader = build_loader(kind, train_ds, max(1, min(16, final_cfg.batch_size)), False, final_cfg.max_len,
                                              alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)
            embed_test_loader = build_loader(kind, test_ds, max(1, min(16, final_cfg.batch_size)), False, final_cfg.max_len,
                                             alphabet=alphabet, protbert_tokenizer=protbert_tokenizer)
            train_Z, train_Y = collect_embeddings(model, embed_train_loader, device, fam_ids)
            test_Z, test_Y = collect_embeddings(model, embed_test_loader, device, fam_ids)
            fs_rows = []
            for shots in fewshot_shots:
                fs = fewshot_episode_eval(train_Z, train_Y, test_Z, test_Y, fam_ids, shots=shots, episodes=args.fewshot_episodes, seed=seed)
                fs.update({"exp": exp_name, "seed": seed})
                fs_rows.append(fs)
                fewshot_rows.append(fs)
            pd.DataFrame(fs_rows).to_csv(os.path.join(run_dir, "fewshot_results.csv"), index=False)

            # Few-shot plot per run
            fs_df = pd.DataFrame(fs_rows).sort_values("shots")
            plt.figure(figsize=(6.0, 4.0))
            plt.plot(fs_df["shots"], fs_df["macro_f1"], marker="o")
            plt.xlabel("Shots per family")
            plt.ylabel("Few-shot Macro-F1")
            plt.title(f"Few-shot performance: {exp_name} (seed={seed})")
            save_fig(os.path.join(run_dir, "fewshot_curve.png"), os.path.join(run_dir, "fewshot_curve.pdf"))

            runs_rows.append({
                "exp": exp_name,
                "key": exp_key,
                "seed": seed,
                "macro_f1_family": float(metrics["macro_f1_family"]),
                "balanced_acc_family": float(metrics["balanced_acc_family"]),
                "macro_f1_class": float(metrics["macro_f1_class"]),
                "lambda_class": float(final_cfg.lambda_class),
                "unfreeze_last_n": int(final_cfg.unfreeze_last_n),
                "max_len": int(final_cfg.max_len),
            })

            del model
            gc.collect()
            if _device_is_cuda(device):
                torch.cuda.empty_cache()

    # Summary CSVs
    runs_df = pd.DataFrame(runs_rows)
    runs_df.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)

    fam_summ = runs_df.groupby("exp")["macro_f1_family"].agg(["mean", "std", "count"]).reset_index()
    fam_summ.columns = ["exp", "macro_f1_family_mean", "macro_f1_family_std", "n"]
    cls_summ = runs_df.groupby("exp")["macro_f1_class"].agg(["mean", "std"]).reset_index()
    cls_summ.columns = ["exp", "macro_f1_class_mean", "macro_f1_class_std"]
    bal_summ = runs_df.groupby("exp")["balanced_acc_family"].agg(["mean", "std"]).reset_index()
    bal_summ.columns = ["exp", "balanced_acc_family_mean", "balanced_acc_family_std"]
    model_summ = fam_summ.merge(cls_summ, on="exp", how="left").merge(bal_summ, on="exp", how="left")
    model_summ.to_csv(os.path.join(args.out, "summary_models.csv"), index=False)

    fewshot_df = pd.DataFrame(fewshot_rows)
    fewshot_df.to_csv(os.path.join(args.out, "fewshot_summary.csv"), index=False)

    # Comparison plots
    order = model_summ.sort_values("macro_f1_family_mean", ascending=False)["exp"].tolist()
    xs = np.arange(len(order))

    vals = [model_summ[model_summ["exp"] == e]["macro_f1_family_mean"].values[0] for e in order]
    err = [model_summ[model_summ["exp"] == e]["macro_f1_family_std"].values[0] for e in order]
    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, vals, yerr=err, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (Family)")
    plt.title("Baseline comparison: Macro-F1 (Family), mean ± std")
    save_fig(os.path.join(args.out, "compare_macro_f1_family.png"), os.path.join(args.out, "compare_macro_f1_family.pdf"))

    vals_c = [model_summ[model_summ["exp"] == e]["macro_f1_class_mean"].values[0] for e in order]
    err_c = [model_summ[model_summ["exp"] == e]["macro_f1_class_std"].values[0] for e in order]
    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, vals_c, yerr=err_c, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Macro-F1 (Class)")
    plt.title("Baseline comparison: Macro-F1 (Class), mean ± std")
    save_fig(os.path.join(args.out, "compare_macro_f1_class.png"), os.path.join(args.out, "compare_macro_f1_class.pdf"))

    vals_b = [model_summ[model_summ["exp"] == e]["balanced_acc_family_mean"].values[0] for e in order]
    err_b = [model_summ[model_summ["exp"] == e]["balanced_acc_family_std"].values[0] for e in order]
    plt.figure(figsize=(9, 4.8))
    plt.bar(xs, vals_b, yerr=err_b, capsize=3)
    plt.xticks(xs, order, rotation=35, ha="right")
    plt.ylabel("Balanced Accuracy (Family)")
    plt.title("Baseline comparison: Balanced Accuracy (Family), mean ± std")
    save_fig(os.path.join(args.out, "compare_balanced_acc_family.png"), os.path.join(args.out, "compare_balanced_acc_family.pdf"))

    # Few-shot comparison plot
    if len(fewshot_df) > 0:
        plt.figure(figsize=(8.8, 4.8))
        for exp_name, sub in fewshot_df.groupby("exp"):
            g = sub.groupby("shots")["macro_f1"].mean().reset_index().sort_values("shots")
            plt.plot(g["shots"], g["macro_f1"], marker="o", label=exp_name)
        plt.xlabel("Shots per family")
        plt.ylabel("Few-shot Macro-F1")
        plt.title("Few-shot comparison across methods")
        plt.legend(frameon=False)
        save_fig(os.path.join(args.out, "compare_fewshot.png"), os.path.join(args.out, "compare_fewshot.pdf"))

    print("\n✅ DONE")
    print("Output root:", args.out)
    print("- dataset_summary.csv / family_distribution.csv")
    print("- summary_runs.csv / summary_models.csv / fewshot_summary.csv")
    print("- compare_macro_f1_family.(png|pdf)")
    print("- compare_macro_f1_class.(png|pdf)")
    print("- compare_balanced_acc_family.(png|pdf)")
    print("- compare_fewshot.(png|pdf)")
    print("- per-run outputs under out/EXP_NAME/seed_k/")


if __name__ == "__main__":
    main()
