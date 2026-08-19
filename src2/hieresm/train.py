from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import load_yaml
from .data import load_all_frames, CAZySequenceDataset
from .model import HierESM, hierarchy_aware_loss
from .metrics import summarize_family_and_class
from .utils import ensure_dir, set_seed, get_device, save_json


def count_labels(values, n):
    counts = [0] * n
    for v in values:
        counts[int(v)] += 1
    return counts


def predict(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            protein_ids = batch.pop("protein_id")
            family_true = batch.pop("family_label").cpu().numpy()
            class_true = batch.pop("class_label").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            family_prob = torch.softmax(out["family_logits"], dim=-1).cpu().numpy()
            class_prob = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()
            family_pred = family_prob.argmax(axis=1)
            class_pred = class_prob.argmax(axis=1)

            for i, pid in enumerate(protein_ids):
                rows.append({
                    "protein_id": pid,
                    "family_true": int(family_true[i]),
                    "family_pred": int(family_pred[i]),
                    "class_true": int(class_true[i]),
                    "class_pred": int(class_pred[i]),
                    "family_conf": float(family_prob[i].max()),
                    "class_conf": float(class_prob[i].max()),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg["training"].get("seed", 1)
    set_seed(seed)

    out_dir = ensure_dir(args.out_dir)
    device = get_device()

    train_df, val_df, test_df, label_maps = load_all_frames(args.data_dir, cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"].get("backbone", "facebook/esm2_t12_35M_UR50D"))

    ds_kwargs = dict(
        tokenizer=tokenizer,
        label_maps=label_maps,
        sequence_col=cfg["data"].get("sequence_column", "sequence"),
        family_col=cfg["data"].get("family_column", "family"),
        class_col=cfg["data"].get("class_column", "class"),
        protein_id_col=cfg["data"].get("protein_id_column", "protein_id"),
        max_length=cfg["model"].get("max_length", 512),
    )

    train_ds = CAZySequenceDataset(train_df, **ds_kwargs)
    val_ds = CAZySequenceDataset(val_df, **ds_kwargs)
    test_ds = CAZySequenceDataset(test_df, **ds_kwargs)

    batch_size = cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["training"].get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = HierESM(cfg, label_maps).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-2),
    )

    family_col = cfg["data"].get("family_column", "family")
    class_col = cfg["data"].get("class_column", "class")
    train_family_ids = [label_maps.family_to_id[x] for x in train_df[family_col]]
    train_class_ids = [label_maps.class_to_id[x] for x in train_df[class_col]]
    family_counts = count_labels(train_family_ids, len(label_maps.families))
    class_counts = count_labels(train_class_ids, len(label_maps.classes))

    best_val = -1.0
    best_path = out_dir / f"best_hieresm_seed{seed}.pt"
    grad_accum = cfg["training"].get("grad_accum", 4)

    for epoch in range(cfg["training"].get("epochs", 25)):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}")):
            batch.pop("protein_id")
            family_labels = batch.pop("family_label").to(device)
            class_labels = batch.pop("class_label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = hierarchy_aware_loss(out, family_labels, class_labels, family_counts, class_counts, cfg)
            (loss / grad_accum).backward()
            running += float(loss.detach().cpu())

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)

        val_pred = predict(model, val_loader, device)
        val_metrics = summarize_family_and_class(
            val_pred["family_true"], val_pred["family_pred"],
            val_pred["class_true"], val_pred["class_pred"],
        )
        print({"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), **val_metrics})

        if val_metrics["family_macro_f1"] > best_val:
            best_val = val_metrics["family_macro_f1"]
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)

    test_pred = predict(model, test_loader, device)
    test_pred.to_csv(out_dir / f"predictions_seed{seed}.csv", index=False)

    test_metrics = summarize_family_and_class(
        test_pred["family_true"], test_pred["family_pred"],
        test_pred["class_true"], test_pred["class_pred"],
    )
    save_json(test_metrics, out_dir / f"metrics_seed{seed}.json")
    print(test_metrics)


if __name__ == "__main__":
    main()
