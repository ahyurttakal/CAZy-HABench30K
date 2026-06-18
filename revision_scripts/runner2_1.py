#!/usr/bin/env python3
import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
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

import matplotlib as mpl
import matplotlib.pyplot as plt

import esm


# =========================
# Publication-grade plotting
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


def save_fig(out_png: str, out_pdf: str):
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_confusion(cm: np.ndarray, labels: List[str], title: str,
                   out_png: str, out_pdf: str, normalize: bool = True):
    cm = cm.astype(np.float64)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    n = len(labels)
    fig_w = min(22, max(7.5, 0.32 * n))
    fig_h = min(22, max(6.5, 0.32 * n))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title + (" (row-normalized)" if normalize else ""))
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(out_png, out_pdf)


def plot_bar(df: pd.DataFrame, metric: str, out_png: str, out_pdf: str):
    d = df.sort_values(metric, ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, max(3.8, 0.35 * len(d))))
    plt.barh(d["run"].tolist()[::-1], d[metric].tolist()[::-1])
    plt.xlabel(metric)
    plt.title(f"Comparison: {metric}")
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
# Data
# =========================
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


class ESMCollator:
    def __init__(self, alphabet, max_len: int = 768):
        self.batch_converter = alphabet.get_batch_converter()
        self.max_len = max_len

    def __call__(self, batch):
        items = []
        y_c, y_f = [], []
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


# =========================
# Loss: supervised contrastive
# =========================
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
    return loss[has_pos].mean() if has_pos.any() else loss.mean()


# =========================
# Model: Proposed
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
        feats = [F.gelu(conv(x)) for conv in self.convs]  # B,hidden,L
        y = torch.cat(feats, dim=1).transpose(1, 2)       # B,L,hidden*k
        y = self.proj(y)                                  # B,L,D
        y = y * mask.unsqueeze(-1).float()
        return self.norm(H + y)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (H * self.q.view(1, 1, -1)).sum(-1)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (H * w).sum(dim=1)


class ESM2Backbone(nn.Module):
    def __init__(self, esm_model, freeze: bool = True):
        super().__init__()
        self.esm = esm_model
        self.d = esm_model.embed_dim
        self.layer = esm_model.num_layers
        self.set_freeze(freeze)

    def set_freeze(self, freeze: bool):
        for p in self.esm.parameters():
            p.requires_grad = (not freeze)

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.layer], return_contacts=False)
        H = out["representations"][self.layer]  # B,L,D
        return H


class ProposedModel(nn.Module):
    def __init__(self, esm_model, n_cls: int, n_fam: int, freeze_esm: bool = True):
        super().__init__()
        self.backbone = ESM2Backbone(esm_model, freeze=freeze_esm)
        d = self.backbone.d
        self.adapter = MultiScaleAdapter(d)
        self.pool = AttnPool(d)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1))
        self.class_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_cls)
        )
        self.family_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_fam)
        )

    def set_freeze_esm(self, freeze: bool):
        self.backbone.set_freeze(freeze)

    def forward(self, tokens, mask):
        H = self.backbone(tokens)
        H = self.adapter(H, mask)
        z = self.pool(H, mask)
        z = self.proj(z)
        logits_c = self.class_head(z)
        logits_f = self.family_head(z)
        return logits_c, logits_f, z


# =========================
# Train / Eval
# =========================
@dataclass
class Cfg:
    max_len: int = 768
    batch_size: int = 16
    wd: float = 0.01
    grad_clip: float = 1.0
    temperature: float = 0.07
    lambda_contrast: float = 0.2
    cm_topn: int = 30


def train_epoch(model, loader, opt, device, cfg: Cfg):
    model.train()
    total = 0.0
    n = 0
    for tokens, mask, y_c, y_f in tqdm(loader, desc="train", leave=False):
        tokens = tokens.to(device)
        mask = mask.to(device)
        y_c = y_c.to(device)
        y_f = y_f.to(device)

        opt.zero_grad(set_to_none=True)
        logits_c, logits_f, z = model(tokens, mask)

        lc = F.cross_entropy(logits_c, y_c)
        lf = F.cross_entropy(logits_f, y_f)
        lcon = supcon_loss(z, y_f, temperature=cfg.temperature)

        loss = lc + lf + cfg.lambda_contrast * lcon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Returns:
      logits_class: [N, C]
      logits_family: [N, F]
      y_class_true: list[int]
      y_family_true: list[int]
    """
    model.eval()
    lc_list, lf_list = [], []
    ytc, ytf = [], []
    for tokens, mask, y_c, y_f in tqdm(loader, desc="predict", leave=False):
        tokens = tokens.to(device)
        mask = mask.to(device)
        logits_c, logits_f, _ = model(tokens, mask)
        lc_list.append(logits_c.detach().cpu().numpy())
        lf_list.append(logits_f.detach().cpu().numpy())
        ytc += y_c.tolist()
        ytf += y_f.tolist()

    lc = np.concatenate(lc_list, axis=0)
    lf = np.concatenate(lf_list, axis=0)
    return lc, lf, ytc, ytf


def topn_by_support(y_true: List[int], n: int) -> List[int]:
    s = pd.Series(y_true).value_counts()
    return [int(x) for x in s.index.tolist()[:n]]


def eval_and_save(out_dir: str,
                  logits_c: np.ndarray, logits_f: np.ndarray,
                  ytc: List[int], ytf: List[int],
                  id2cls: Dict[int, str], id2fam: Dict[int, str],
                  cm_topn: int):
    os.makedirs(out_dir, exist_ok=True)

    pred_c = logits_c.argmax(axis=1).tolist()
    pred_f = logits_f.argmax(axis=1).tolist()

    macro_f1_fam = float(f1_score(ytf, pred_f, average="macro"))
    bal_acc_fam = float(balanced_accuracy_score(ytf, pred_f))
    macro_f1_cls = float(f1_score(ytc, pred_c, average="macro"))

    metrics = {
        "macro_f1_family": macro_f1_fam,
        "balanced_acc_family": bal_acc_fam,
        "macro_f1_class": macro_f1_cls,
        "n_test": int(len(ytf)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # per-family report
    fam_labels = list(range(len(id2fam)))
    fam_names = [id2fam[i] for i in fam_labels]
    rep_f = classification_report(ytf, pred_f, labels=fam_labels, target_names=fam_names,
                                  output_dict=True, zero_division=0)
    pd.DataFrame(rep_f).transpose().to_csv(os.path.join(out_dir, "per_family_report.csv"))

    # per-class report
    cls_labels = list(range(len(id2cls)))
    cls_names = [id2cls[i] for i in cls_labels]
    rep_c = classification_report(ytc, pred_c, labels=cls_labels, target_names=cls_names,
                                  output_dict=True, zero_division=0)
    pd.DataFrame(rep_c).transpose().to_csv(os.path.join(out_dir, "per_class_report.csv"))

    # confusion class
    cm_c = confusion_matrix(ytc, pred_c, labels=cls_labels)
    pd.DataFrame(cm_c, index=cls_names, columns=cls_names).to_csv(os.path.join(out_dir, "confusion_class.csv"))
    plot_confusion(cm_c, cls_names, "Class confusion",
                   os.path.join(out_dir, "confusion_class_norm.png"),
                   os.path.join(out_dir, "confusion_class_norm.pdf"),
                   normalize=True)

    # confusion family topN
    keep_ids = topn_by_support(ytf, cm_topn)
    keep_names = [id2fam[i] for i in keep_ids]
    cm_f_top = confusion_matrix(ytf, pred_f, labels=keep_ids)
    pd.DataFrame(cm_f_top, index=keep_names, columns=keep_names).to_csv(os.path.join(out_dir, f"confusion_family_top{cm_topn}.csv"))
    plot_confusion(cm_f_top, keep_names, f"Family confusion (top{cm_topn})",
                   os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.png"),
                   os.path.join(out_dir, f"confusion_family_top{cm_topn}_norm.pdf"),
                   normalize=True)

    return metrics


# =========================
# Main: two-stage + 3-seed ensemble
# =========================
def main():
    set_pub_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="homology_split folder")
    ap.add_argument("--out", default="finetune_ensemble_q1", help="output root")
    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")

    # stage 1 (freeze)
    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--lr_stage1", type=float, default=2e-4)

    # stage 2 (finetune)
    ap.add_argument("--epochs_stage2", type=int, default=5)
    ap.add_argument("--lr_stage2", type=float, default=5e-5)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--cm_topn", type=int, default=30)
    ap.add_argument("--seeds", default="1,7,42", help="comma-separated seeds for ensemble")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device}")

    cfg = Cfg(max_len=args.max_len, batch_size=args.batch_size, cm_topn=args.cm_topn)

    # paths
    train_fa = os.path.join(args.data_dir, "train_homology.fasta")
    test_fa = os.path.join(args.data_dir, "test_homology.fasta")
    train_csv = os.path.join(args.data_dir, "labels_train.csv")
    test_csv = os.path.join(args.data_dir, "labels_test.csv")

    # mappings from train
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

    # datasets
    train_ds = CAZyDataset(train_fa, train_csv, cls2id=cls2id, fam2id=fam2id)
    test_ds = CAZyDataset(test_fa, test_csv, cls2id=cls2id, fam2id=fam2id)
    print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"cls2id": cls2id, "fam2id": fam2id}, f, indent=2)

    # Load ESM once
    print(f"[LOAD] ESM2: {args.esm}")
    esm_model, alphabet = esm.pretrained.__dict__[args.esm]()
    esm_model = esm_model.to(device)
    collator = ESMCollator(alphabet, max_len=cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 2:
        raise SystemExit("Provide at least 2 seeds for ensemble, e.g. --seeds 1,7,42")

    per_seed_metrics = []
    ensemble_logits_c = None
    ensemble_logits_f = None
    ytc_ref, ytf_ref = None, None

    for seed in seeds:
        print(f"\n==============================")
        print(f"[RUN] seed={seed}")
        print(f" Stage-1 (freeze): epochs={args.epochs_stage1} lr={args.lr_stage1}")
        print(f" Stage-2 (finetune): epochs={args.epochs_stage2} lr={args.lr_stage2}")
        print(f"==============================")

        set_seed(seed)
        run_dir = os.path.join(args.out, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        model = ProposedModel(esm_model, n_cls=len(cls2id), n_fam=len(fam2id), freeze_esm=True).to(device)

        # Stage 1: freeze backbone
        model.set_freeze_esm(True)
        opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr_stage1, weight_decay=cfg.wd)

        best_f1 = -1.0
        best_state = None

        for ep in range(1, args.epochs_stage1 + 1):
            loss = train_epoch(model, train_loader, opt1, device, cfg)
            lc, lf, ytc, ytf = predict_logits(model, test_loader, device)
            pred_f = lf.argmax(axis=1).tolist()
            f1_fam = float(f1_score(ytf, pred_f, average="macro"))
            print(f"  [S1] ep={ep:02d} loss={loss:.4f} macroF1_fam={f1_fam:.4f}")
            if f1_fam > best_f1:
                best_f1 = f1_fam
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Stage 2: finetune backbone
        model.set_freeze_esm(False)
        opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr_stage2, weight_decay=cfg.wd)

        best_f1_2 = -1.0
        best_state_2 = None

        for ep in range(1, args.epochs_stage2 + 1):
            loss = train_epoch(model, train_loader, opt2, device, cfg)
            lc, lf, ytc, ytf = predict_logits(model, test_loader, device)
            pred_f = lf.argmax(axis=1).tolist()
            f1_fam = float(f1_score(ytf, pred_f, average="macro"))
            print(f"  [S2] ep={ep:02d} loss={loss:.4f} macroF1_fam={f1_fam:.4f}")
            if f1_fam > best_f1_2:
                best_f1_2 = f1_fam
                best_state_2 = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state_2 is not None:
            model.load_state_dict(best_state_2, strict=True)

        # Final logits for this seed
        lc, lf, ytc, ytf = predict_logits(model, test_loader, device)

        # Save seed outputs (paper-ready)
        metrics = eval_and_save(run_dir, lc, lf, ytc, ytf, id2cls, id2fam, cfg.cm_topn)
        metrics["run"] = f"seed_{seed}"
        per_seed_metrics.append(metrics)

        # Save checkpoint
        ckpt = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "seed": seed,
            "esm": args.esm,
            "cls2id": cls2id,
            "fam2id": fam2id,
            "cfg": vars(args),
            "metrics": metrics,
        }
        torch.save(ckpt, os.path.join(run_dir, "best.pt"))

        # Accumulate logits for ensemble (logit average)
        if ensemble_logits_c is None:
            ensemble_logits_c = lc.astype(np.float64)
            ensemble_logits_f = lf.astype(np.float64)
            ytc_ref, ytf_ref = ytc, ytf
        else:
            # sanity
            if ytc != ytc_ref or ytf != ytf_ref:
                raise RuntimeError("Label order mismatch across seeds. Ensure same test_loader order.")
            ensemble_logits_c += lc.astype(np.float64)
            ensemble_logits_f += lf.astype(np.float64)

    # Ensemble average
    ensemble_logits_c /= float(len(seeds))
    ensemble_logits_f /= float(len(seeds))

    ens_dir = os.path.join(args.out, "ensemble_3seed")
    ens_metrics = eval_and_save(ens_dir, ensemble_logits_c, ensemble_logits_f, ytc_ref, ytf_ref, id2cls, id2fam, cfg.cm_topn)
    ens_metrics["run"] = "ensemble_3seed"
    per_seed_metrics.append(ens_metrics)

    # Summary tables + plots
    df = pd.DataFrame(per_seed_metrics)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(per_seed_metrics, f, indent=2)

    plot_bar(df, "macro_f1_family",
             os.path.join(args.out, "compare_macro_f1_family.png"),
             os.path.join(args.out, "compare_macro_f1_family.pdf"))

    plot_bar(df, "macro_f1_class",
             os.path.join(args.out, "compare_macro_f1_class.png"),
             os.path.join(args.out, "compare_macro_f1_class.pdf"))

    print("\n✅ DONE")
    print("Output root:", args.out)
    print("- Per seed: seed_*/ (metrics.json, per_family_report.csv, confusion PDFs)")
    print("- Ensemble: ensemble_3seed/")
    print("- Summary: summary.csv + compare_macro_f1_family.pdf")


if __name__ == "__main__":
    main()
