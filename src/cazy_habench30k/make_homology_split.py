#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAZy Homology-Aware Split v7 (Ubuntu / Linux)
=============================================

Amaç:
  - Homology-aware split
  - Family coverage'i mümkün olduğunca korumak
  - Train/val/test oranlarını hedefe yaklaştırmak
  - ID normalizasyonu ve cluster temizleme hatalarını önlemek

Öne çıkanlar:
  1) UniProt pipe ID normalizasyonu
  2) MMseqs TSV uyum doğrulaması
  3) Cluster coverage debug
  4) Family-aware + ratio-aware split
  5) Küçük family'lerde sequence-level fallback
  6) Singleton cluster tabanlı final rebalance
  7) Detaylı split_report.json

Önerilen kullanım:
  python homology_split_v7.py \
      --dataset_dir data/UniProt-CAZy-benchmark-30k \
      --out         data/splits_v7 \
      --test_ratio  0.15 \
      --val_ratio   0.15 \
      --min_seq_id  0.20 \
      --cov         0.60 \
      --threads     4
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


CAZY_CLS = ["GH", "GT", "PL", "CE", "AA", "CBM"]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalize_id(raw: str) -> str:
    """
    Örnek:
      >sp|Q9ABC1|NAME desc -> Q9ABC1
      >tr|A0A123|NAME      -> A0A123
      >Q9ABC1 desc         -> Q9ABC1
      Q9ABC1               -> Q9ABC1
    """
    s = str(raw).strip().lstrip(">")
    s = s.split()[0]

    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1].strip()

    return s.strip()


def fam2cls(fam: str) -> str:
    fam = str(fam).upper()
    for c in sorted(CAZY_CLS, key=len, reverse=True):
        if fam.startswith(c):
            return c
    return fam[:2]


def validate_ratios(val_ratio: float, test_ratio: float) -> float:
    for name, value in [("val_ratio", val_ratio), ("test_ratio", test_ratio)]:
        if not (0.0 <= value < 1.0):
            sys.exit(f"[HATA] {name} 0 ile 1 arasında olmalı. Gelen: {value}")

    if val_ratio + test_ratio >= 1.0:
        sys.exit(
            f"[HATA] val_ratio + test_ratio < 1.0 olmalı. "
            f"Gelen toplam: {val_ratio + test_ratio:.4f}"
        )

    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0.0:
        sys.exit(f"[HATA] train oranı sıfır veya negatif oldu: {train_ratio:.4f}")

    return train_ratio


def compute_cluster_stats(rep2mem: Dict[str, List[str]]) -> Dict[str, float]:
    sizes = [len(v) for v in rep2mem.values()]
    n_clusters = len(sizes)
    total_memberships = sum(sizes)

    if n_clusters == 0:
        return {
            "n_clusters": 0,
            "total_memberships": 0,
            "singleton_clusters": 0,
            "singleton_ratio": 0.0,
            "multi_clusters": 0,
            "multi_ratio": 0.0,
            "max_cluster_size": 0,
            "mean_cluster_size": 0.0,
        }

    singleton_clusters = sum(1 for s in sizes if s == 1)
    multi_clusters = sum(1 for s in sizes if s > 1)

    return {
        "n_clusters": n_clusters,
        "total_memberships": total_memberships,
        "singleton_clusters": singleton_clusters,
        "singleton_ratio": singleton_clusters / n_clusters,
        "multi_clusters": multi_clusters,
        "multi_ratio": multi_clusters / n_clusters,
        "max_cluster_size": max(sizes),
        "mean_cluster_size": total_memberships / n_clusters,
    }


def validate_existing_clusters_tsv(
    tsv_path: Path,
    common_ids: Set[str],
    min_overlap_ratio: float = 0.50,
    sample_lines: int = 20000,
) -> None:
    if not tsv_path.exists():
        sys.exit(f"[HATA] --skip_mmseqs verildi ama TSV yok: {tsv_path}")

    seen_ids: Set[str] = set()
    n_lines = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            seen_ids.add(normalize_id(parts[0]))
            seen_ids.add(normalize_id(parts[1]))
            n_lines += 1
            if n_lines >= sample_lines:
                break

    if n_lines == 0:
        sys.exit(f"[HATA] clusters.tsv boş görünüyor: {tsv_path}")

    overlap = len(seen_ids & common_ids)
    denom = max(1, min(len(seen_ids), len(common_ids)))
    overlap_ratio = overlap / denom

    print(
        f"  TSV doğrulama: satır_ornek={n_lines} | "
        f"benzersiz_id={len(seen_ids)} | ortak_id={overlap} | "
        f"uyum={overlap_ratio:.1%}"
    )

    if overlap_ratio < min_overlap_ratio:
        sys.exit(
            "[HATA] Mevcut clusters.tsv bu veriyle uyumlu görünmüyor. "
            "Muhtemelen eski veya farklı FASTA/labels setinden kalma."
        )


# ──────────────────────────────────────────────────────────────────────────────
# MMSEQS2
# ──────────────────────────────────────────────────────────────────────────────

def run_mmseqs2(
    all_fasta: Path,
    out_dir: Path,
    min_seq_id: float,
    cov: float,
    threads: int = 4,
) -> Path:
    mmseqs = shutil.which("mmseqs")
    if not mmseqs:
        sys.exit(
            "\n[HATA] mmseqs bulunamadı.\n"
            "  conda install -c conda-forge -c bioconda mmseqs2\n"
        )

    tmp = out_dir / "mmseqs_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    db = tmp / "seqdb"
    clu = tmp / "cludb"
    tsv = out_dir / "clusters.tsv"

    def run(cmd: List[str], step: str) -> None:
        print(f"  [{step}] {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n  [HATA] {step} başarısız (kod {result.returncode}):")
            print(result.stderr[-1500:] if result.stderr else "(çıktı yok)")
            sys.exit(1)

    run([mmseqs, "createdb", str(all_fasta), str(db)], "createdb")

    if not db.exists():
        sys.exit(f"[HATA] createdb çıktısı yok: {db}")

    run([
        mmseqs, "linclust",
        str(db), str(clu), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(cov),
        "--cov-mode", "0",
        "--threads", str(threads),
        "-v", "1",
    ], "linclust")

    run([
        mmseqs, "createtsv",
        str(db), str(db), str(clu), str(tsv)
    ], "createtsv")

    with open(tsv, "r", encoding="utf-8") as fh:
        n_lines = sum(1 for _ in fh)

    print(f"  TSV: {tsv}  ({n_lines} satır)")
    if n_lines == 0:
        sys.exit("[HATA] MMseqs2 TSV boş döndü.")

    return tsv


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER TSV
# ──────────────────────────────────────────────────────────────────────────────

def read_cluster_tsv(tsv_path: Path) -> Dict[str, List[str]]:
    rep2mem: Dict[str, List[str]] = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue

            rep = normalize_id(parts[0])
            mem = normalize_id(parts[1])
            rep2mem.setdefault(rep, []).append(mem)

    stats = compute_cluster_stats(rep2mem)
    print(
        f"  Kümeler: {stats['n_clusters']} | "
        f"Toplam üyelik: {stats['total_memberships']} | "
        f"Singleton: {stats['singleton_clusters']} | "
        f"Multi: {stats['multi_clusters']}"
    )
    return rep2mem


# ──────────────────────────────────────────────────────────────────────────────
# FASTA / LABELS
# ──────────────────────────────────────────────────────────────────────────────

def load_fasta(fasta_path: Path) -> Dict[str, SeqRecord]:
    records: Dict[str, SeqRecord] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        nid = normalize_id(rec.id)
        rec.id = nid
        rec.name = nid
        rec.description = ""
        records[nid] = rec

    print(f"  FASTA: {len(records)} kayıt  <- {fasta_path.name}")
    return records


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)

    if "id" not in df.columns:
        for cand in ("uniprot_id", "accession", "protein_id", "seq_id"):
            if cand in df.columns:
                df = df.rename(columns={cand: "id"})
                break
        else:
            sys.exit("[HATA] labels.csv'de 'id' sütunu bulunamadı.")

    if "family" not in df.columns:
        for cand in ("cazy_family", "fam", "Family", "CAZy_family"):
            if cand in df.columns:
                df = df.rename(columns={cand: "family"})
                break
        else:
            sys.exit("[HATA] labels.csv'de 'family' sütunu bulunamadı.")

    df["id"] = df["id"].astype(str).apply(normalize_id)

    if "class" not in df.columns:
        df["class"] = df["family"].astype(str).apply(fam2cls)
        print("  'class' sütunu yoktu -> family'den otomatik türetildi")

    dupes = df["id"].duplicated().sum()
    if dupes:
        print(f"  [UYARI] {dupes} tekrar ID bulundu -> ilk görünüm korunuyor")
        df = df.drop_duplicates(subset="id", keep="first")

    df = df.reset_index(drop=True)
    print(
        f"  Labels: {len(df)} kayıt | "
        f"{df['family'].nunique()} family | {df['class'].nunique()} class"
    )
    return df


def write_fasta(records: List[SeqRecord], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        SeqIO.write(records, fh, "fasta")
    print(f"  <- {path.name}  ({len(records)} seq)")


# ──────────────────────────────────────────────────────────────────────────────
# SPLIT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _add_cluster_to_split(
    rep: str,
    split_name: str,
    rep2mem: Dict[str, List[str]],
    split_reps: Dict[str, Set[str]],
    split_ids: Dict[str, Set[str]],
    split_family_counts: Dict[str, Counter],
    rep2dominant_fam: Dict[str, str],
    split_sizes: Dict[str, int],
) -> None:
    if rep in split_reps["train"] or rep in split_reps["val"] or rep in split_reps["test"]:
        return

    members = rep2mem[rep]
    split_reps[split_name].add(rep)
    split_ids[split_name].update(members)
    split_sizes[split_name] += len(members)

    fam = rep2dominant_fam.get(rep, "__unk__")
    split_family_counts[split_name][fam] += 1


def _sequence_level_fallback_family(
    family_reps: List[str],
    rep2mem: Dict[str, List[str]],
    split_ids: Dict[str, Set[str]],
    split_sizes: Dict[str, int],
    target_sizes: Dict[str, int],
    rng: random.Random,
) -> None:
    """
    Cluster sayısı < 3 olan family'lerde sequence-level fallback.
    Amaç: mümkün olduğunca train/val/test'e coverage sağlamak.
    """
    members: List[str] = []
    for rep in family_reps:
        members.extend(rep2mem[rep])

    rng.shuffle(members)
    m = len(members)

    if m == 1:
        split_ids["train"].add(members[0])
        split_sizes["train"] += 1
        return

    if m == 2:
        split_ids["train"].add(members[0])
        split_ids["val"].add(members[1])
        split_sizes["train"] += 1
        split_sizes["val"] += 1
        return

    # Önce coverage için 1'er tane yerleştir
    base_order = ["train", "val", "test"]
    for idx, split_name in enumerate(base_order):
        split_ids[split_name].add(members[idx])
        split_sizes[split_name] += 1

    remaining = members[3:]

    for sid in remaining:
        deficits = {
            k: target_sizes[k] - split_sizes[k]
            for k in ("train", "val", "test")
        }
        chosen = max(deficits, key=deficits.get)
        split_ids[chosen].add(sid)
        split_sizes[chosen] += 1


def _greedy_split_choice(
    cluster_size: int,
    split_sizes: Dict[str, int],
    target_sizes: Dict[str, int],
) -> str:
    def score(split_name: str) -> Tuple[float, float, float]:
        new_sizes = split_sizes.copy()
        new_sizes[split_name] += cluster_size

        dist = sum(abs(new_sizes[k] - target_sizes[k]) for k in ("train", "val", "test"))
        overshoot = sum(max(0, new_sizes[k] - target_sizes[k]) for k in ("train", "val", "test"))
        fullness = new_sizes[split_name] / max(target_sizes[split_name], 1)

        return (dist, overshoot, fullness)

    options = [("train", score("train")), ("val", score("val")), ("test", score("test"))]
    options.sort(key=lambda x: x[1])
    return options[0][0]


def _singleton_rebalance(
    rep2mem: Dict[str, List[str]],
    split_reps: Dict[str, Set[str]],
    split_ids: Dict[str, Set[str]],
    split_family_counts: Dict[str, Counter],
    rep2dominant_fam: Dict[str, str],
    split_sizes: Dict[str, int],
    target_sizes: Dict[str, int],
) -> None:
    """
    Sadece singleton cluster'ları taşıyarak oranları hedefe yaklaştırır.
    Family coverage'i bozmamaya çalışır.
    """
    rep2split = {}
    for s in ("train", "val", "test"):
        for rep in split_reps[s]:
            rep2split[rep] = s

    singleton_reps = [
        rep for rep, mems in rep2mem.items()
        if len(mems) == 1 and rep in rep2split
    ]

    # Maksimum birkaç bin denemelik güvenli sınır
    for _ in range(200000):
        deficits = {k: target_sizes[k] - split_sizes[k] for k in ("train", "val", "test")}
        if deficits["train"] == 0 and deficits["val"] == 0 and deficits["test"] == 0:
            break

        deficit_splits = [k for k in ("train", "val", "test") if deficits[k] > 0]
        surplus_splits = [k for k in ("train", "val", "test") if deficits[k] < 0]

        if not deficit_splits or not surplus_splits:
            break

        moved = False
        for dst in sorted(deficit_splits, key=lambda x: deficits[x], reverse=True):
            for src in sorted(surplus_splits, key=lambda x: deficits[x]):
                candidate_reps = [
                    rep for rep in singleton_reps
                    if rep2split.get(rep) == src
                ]

                for rep in candidate_reps:
                    fam = rep2dominant_fam.get(rep, "__unk__")
                    # Kaynak splitte family tamamen kaybolmasın
                    if split_family_counts[src][fam] <= 1:
                        continue

                    member = rep2mem[rep][0]

                    split_reps[src].remove(rep)
                    split_ids[src].remove(member)
                    split_family_counts[src][fam] -= 1
                    split_sizes[src] -= 1

                    split_reps[dst].add(rep)
                    split_ids[dst].add(member)
                    split_family_counts[dst][fam] += 1
                    split_sizes[dst] += 1

                    rep2split[rep] = dst
                    moved = True
                    break

                if moved:
                    break
            if moved:
                break

        if not moved:
            break


def stratified_split(
    rep2mem: Dict[str, List[str]],
    id2family: Dict[str, str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    rng = random.Random(seed)

    rep2dominant_fam: Dict[str, str] = {}
    fam2reps: Dict[str, List[str]] = defaultdict(list)

    for rep, members in rep2mem.items():
        fam_counts: Dict[str, int] = defaultdict(int)
        for mid in members:
            fam = id2family.get(mid)
            if fam:
                fam_counts[fam] += 1
        dominant = max(fam_counts, key=fam_counts.get) if fam_counts else "__unk__"
        rep2dominant_fam[rep] = dominant
        fam2reps[dominant].append(rep)

    total_n = sum(len(v) for v in rep2mem.values())
    target_sizes = {
        "train": round(total_n * (1.0 - val_ratio - test_ratio)),
        "val": round(total_n * val_ratio),
    }
    target_sizes["test"] = total_n - target_sizes["train"] - target_sizes["val"]

    split_reps = {"train": set(), "val": set(), "test": set()}
    split_ids = {"train": set(), "val": set(), "test": set()}
    split_sizes = {"train": 0, "val": 0, "test": 0}
    split_family_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    remaining_reps: List[str] = []

    # Önce family-aware başlangıç dağılımı
    for fam, reps in fam2reps.items():
        reps_sorted = sorted(reps, key=lambda r: len(rep2mem[r]), reverse=True)
        n_rep = len(reps_sorted)

        # Küçük family -> sequence-level fallback
        if n_rep < 3:
            _sequence_level_fallback_family(
                family_reps=reps_sorted,
                rep2mem=rep2mem,
                split_ids=split_ids,
                split_sizes=split_sizes,
                target_sizes=target_sizes,
                rng=rng,
            )
            continue

        # Train coverage: en büyük cluster
        train_rep = reps_sorted[0]
        _add_cluster_to_split(
            train_rep, "train", rep2mem, split_reps, split_ids,
            split_family_counts, rep2dominant_fam, split_sizes
        )

        # Val/Test coverage: mümkünse en küçük iki cluster
        val_rep = reps_sorted[-1]
        test_rep = reps_sorted[-2]

        if val_rep != train_rep:
            _add_cluster_to_split(
                val_rep, "val", rep2mem, split_reps, split_ids,
                split_family_counts, rep2dominant_fam, split_sizes
            )

        if test_rep not in {train_rep, val_rep}:
            _add_cluster_to_split(
                test_rep, "test", rep2mem, split_reps, split_ids,
                split_family_counts, rep2dominant_fam, split_sizes
            )

        for rep in reps_sorted:
            if rep not in split_reps["train"] and rep not in split_reps["val"] and rep not in split_reps["test"]:
                remaining_reps.append(rep)

    # Kalan cluster'ları büyükten küçüğe greedy yerleştir
    remaining_reps = sorted(remaining_reps, key=lambda r: len(rep2mem[r]), reverse=True)

    for rep in remaining_reps:
        csize = len(rep2mem[rep])
        chosen = _greedy_split_choice(csize, split_sizes, target_sizes)
        _add_cluster_to_split(
            rep, chosen, rep2mem, split_reps, split_ids,
            split_family_counts, rep2dominant_fam, split_sizes
        )

    # Son oran düzeltmesi: singleton cluster'larla
    _singleton_rebalance(
        rep2mem=rep2mem,
        split_reps=split_reps,
        split_ids=split_ids,
        split_family_counts=split_family_counts,
        rep2dominant_fam=rep2dominant_fam,
        split_sizes=split_sizes,
        target_sizes=target_sizes,
    )

    # Disjointness garantisi
    split_ids["val"] -= split_ids["test"]
    split_ids["train"] -= split_ids["test"]
    split_ids["train"] -= split_ids["val"]

    return split_ids["train"], split_ids["val"], split_ids["test"]


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_split(
    tr_lbl: pd.DataFrame,
    va_lbl: pd.DataFrame,
    te_lbl: pd.DataFrame,
) -> None:
    tr = set(tr_lbl["id"])
    va = set(va_lbl["id"])
    te = set(te_lbl["id"])

    for name, s in [("train", tr), ("val", va), ("test", te)]:
        if len(s) == 0:
            print(f"  [HATA] {name} BOŞ!")
        else:
            print(f"  {name}: {len(s)} seq  OK")

    overlap = len(tr & va) + len(tr & te) + len(va & te)
    if overlap:
        print(f"  [UYARI] {overlap} çakışan ID var")
    else:
        print("  Disjointness: OK")

    all_fam = set(tr_lbl["family"]) | set(va_lbl["family"]) | set(te_lbl["family"])
    missing_in_train = all_fam - set(tr_lbl["family"])
    if missing_in_train:
        print(
            f"  [UYARI] {len(missing_in_train)} family train'de yok: "
            f"{sorted(missing_in_train)[:8]}{'...' if len(missing_in_train) > 8 else ''}"
        )
    else:
        print(f"  Family coverage: {len(all_fam)} family train'de mevcut  OK")

    total = len(tr) + len(va) + len(te)
    if total > 0:
        print(
            f"  Oranlar: train={100 * len(tr) / total:.1f}%  "
            f"val={100 * len(va) / total:.1f}%  "
            f"test={100 * len(te) / total:.1f}%"
        )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CAZy homology-aware train/val/test split v7",
    )
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out", default=None,
                    help="Çıktı klasörü (default: dataset_dir/homology_split)")
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--min_seq_id", type=float, default=0.30)
    ap.add_argument("--cov", type=float, default=0.80)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fasta", default="all.fasta")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--skip_mmseqs", action="store_true",
                    help="clusters.tsv zaten varsa MMseqs2'yi atla")
    args = ap.parse_args()

    train_ratio = validate_ratios(args.val_ratio, args.test_ratio)

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out) if args.out else dataset_dir / "homology_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fa = dataset_dir / args.fasta
    labels_path = dataset_dir / args.labels

    if not all_fa.exists():
        sys.exit(f"[HATA] FASTA bulunamadı: {all_fa}")
    if not labels_path.exists():
        sys.exit(f"[HATA] Labels bulunamadı: {labels_path}")

    print(f"\n{'─' * 72}")
    print("  CAZy Homology-Aware Split  v7")
    print(f"  Giriş  : {all_fa}")
    print(f"  Labels : {labels_path}")
    print(f"  Çıktı  : {out_dir}")
    print(f"  Eşik   : min_seq_id={args.min_seq_id}  cov={args.cov}")
    print(
        f"  Hedef  : train~{train_ratio:.0%}  "
        f"val~{args.val_ratio:.0%}  test~{args.test_ratio:.0%}"
    )
    print(f"{'─' * 72}\n")

    print("[1] FASTA yükleniyor...")
    id2rec = load_fasta(all_fa)

    print("\n[2] Labels yükleniyor...")
    labels_df = load_labels(labels_path)
    id2family = dict(zip(labels_df["id"], labels_df["family"]))

    fasta_ids = set(id2rec.keys())
    label_ids = set(labels_df["id"])
    common_ids = fasta_ids & label_ids

    print(
        f"  FASTA: {len(fasta_ids)} | "
        f"Labels: {len(label_ids)} | "
        f"Ortak: {len(common_ids)}"
    )

    if len(common_ids) == 0:
        sys.exit("[HATA] FASTA ile labels arasında ortak ID yok.")

    labels_df = labels_df[labels_df["id"].isin(common_ids)].reset_index(drop=True)
    id2family = dict(zip(labels_df["id"], labels_df["family"]))

    tsv_path = out_dir / "clusters.tsv"

    if args.skip_mmseqs and tsv_path.exists():
        print("\n[3] MMseqs2 atlandı, mevcut TSV kullanılıyor...")
        validate_existing_clusters_tsv(tsv_path, common_ids)
    else:
        print("\n[3] MMseqs2 linclust çalıştırıyor...")
        labeled_recs = [id2rec[i] for i in common_ids if i in id2rec]
        tmp_fa = out_dir / "_input_labeled.fasta"
        write_fasta(labeled_recs, tmp_fa)
        run_mmseqs2(tmp_fa, out_dir, args.min_seq_id, args.cov, args.threads)
        tmp_fa.unlink(missing_ok=True)

    print("\n[4] Cluster TSV okunuyor...")
    rep2mem_raw = read_cluster_tsv(tsv_path)

    rep2mem: Dict[str, List[str]] = {}
    for rep, mems in rep2mem_raw.items():
        rep_norm = normalize_id(rep)
        filtered = [normalize_id(m) for m in mems if normalize_id(m) in common_ids]
        if filtered:
            rep2mem[rep_norm] = filtered

    raw_stats = compute_cluster_stats(rep2mem_raw)
    clean_stats = compute_cluster_stats(rep2mem)

    dropped_clusters = raw_stats["n_clusters"] - clean_stats["n_clusters"]
    dropped_memberships = raw_stats["total_memberships"] - clean_stats["total_memberships"]

    print(
        f"  Temizleme sonrası: kümeler={clean_stats['n_clusters']} | "
        f"üyelik={clean_stats['total_memberships']}"
    )
    if dropped_clusters > 0 or dropped_memberships > 0:
        print(
            f"  [BİLGİ] Atılan: {dropped_clusters} küme | "
            f"{dropped_memberships} üyelik"
        )

    covered = {m for mems in rep2mem.values() for m in mems}
    not_covered = common_ids - covered

    if not_covered:
        print(f"  {len(not_covered)} kümelenmemiş dizi -> singleton eklendi")
        for sid in sorted(not_covered):
            rep2mem[sid] = [sid]
    else:
        print("  Tüm diziler cluster yapısında kapsanıyor  OK")

    final_stats = compute_cluster_stats(rep2mem)
    print(
        f"  Final cluster özeti: kümeler={final_stats['n_clusters']} | "
        f"singleton={final_stats['singleton_clusters']} | "
        f"multi={final_stats['multi_clusters']} | "
        f"ortalama_boyut={final_stats['mean_cluster_size']:.3f}"
    )

    fam_cluster_counts: Dict[str, int] = defaultdict(int)
    for rep, members in rep2mem.items():
        fam_counts: Dict[str, int] = defaultdict(int)
        for mid in members:
            fam = id2family.get(mid)
            if fam:
                fam_counts[fam] += 1
        if fam_counts:
            fam_cluster_counts[max(fam_counts, key=fam_counts.get)] += 1

    small_fams = sum(1 for v in fam_cluster_counts.values() if v < 3)
    print(
        f"  {small_fams} family için cluster sayısı < 3 "
        f"-> sequence-level fallback devreye girebilir"
    )

    print("\n[5] Stratified split yapılıyor...")
    train_ids, val_ids, test_ids = stratified_split(
        rep2mem=rep2mem,
        id2family=id2family,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  Ham: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    print("\n[6] Dosyalar yazılıyor...")

    def make_split(ids: Set[str], name: str) -> Tuple[pd.DataFrame, List[SeqRecord]]:
        lbl = labels_df[labels_df["id"].isin(ids)].copy()
        recs = [id2rec[i] for i in ids if i in id2rec]
        if not recs:
            print(f"  [UYARI] {name} BOŞ!")
        return lbl, recs

    tr_lbl, tr_recs = make_split(train_ids, "train")
    va_lbl, va_recs = make_split(val_ids, "val")
    te_lbl, te_recs = make_split(test_ids, "test")

    write_fasta(tr_recs, out_dir / "train.fasta")
    write_fasta(va_recs, out_dir / "val.fasta")
    write_fasta(te_recs, out_dir / "test.fasta")

    tr_lbl.to_csv(out_dir / "train_labels.csv", index=False)
    va_lbl.to_csv(out_dir / "val_labels.csv", index=False)
    te_lbl.to_csv(out_dir / "test_labels.csv", index=False)

    print(
        f"\n  train_labels.csv : {len(tr_lbl):>6} kayıt | "
        f"{tr_lbl['family'].nunique()} fam | {tr_lbl['class'].nunique()} cls"
    )
    print(
        f"  val_labels.csv   : {len(va_lbl):>6} kayıt | "
        f"{va_lbl['family'].nunique()} fam | {va_lbl['class'].nunique()} cls"
    )
    print(
        f"  test_labels.csv  : {len(te_lbl):>6} kayıt | "
        f"{te_lbl['family'].nunique()} fam | {te_lbl['class'].nunique()} cls"
    )

    print("\n[7] Doğrulanıyor...")
    validate_split(tr_lbl, va_lbl, te_lbl)

    total = len(tr_lbl) + len(va_lbl) + len(te_lbl)
    report = {
        "version": 7,
        "min_seq_id": args.min_seq_id,
        "cov": args.cov,
        "seed": args.seed,
        "n_clusters": int(final_stats["n_clusters"]),
        "cluster_total_memberships": int(final_stats["total_memberships"]),
        "singleton_clusters": int(final_stats["singleton_clusters"]),
        "singleton_ratio": round(float(final_stats["singleton_ratio"]), 6),
        "multi_clusters": int(final_stats["multi_clusters"]),
        "multi_ratio": round(float(final_stats["multi_ratio"]), 6),
        "max_cluster_size": int(final_stats["max_cluster_size"]),
        "mean_cluster_size": round(float(final_stats["mean_cluster_size"]), 6),
        "n_small_fam_fallback": int(small_fams),
        "id_overlap": {
            "fasta_ids": len(fasta_ids),
            "label_ids": len(label_ids),
            "common_ids": len(common_ids),
            "coverage_after_cleaning": round(len(covered) / max(len(common_ids), 1), 6),
        },
        "train": {
            "n_seq": len(tr_lbl),
            "n_families": int(tr_lbl["family"].nunique()),
            "n_classes": int(tr_lbl["class"].nunique()),
            "ratio": round(len(tr_lbl) / max(total, 1), 6),
        },
        "val": {
            "n_seq": len(va_lbl),
            "n_families": int(va_lbl["family"].nunique()),
            "n_classes": int(va_lbl["class"].nunique()),
            "ratio": round(len(va_lbl) / max(total, 1), 6),
        },
        "test": {
            "n_seq": len(te_lbl),
            "n_families": int(te_lbl["family"].nunique()),
            "n_classes": int(te_lbl["class"].nunique()),
            "ratio": round(len(te_lbl) / max(total, 1), 6),
        },
        "target_ratios": {
            "train": round(1.0 - args.val_ratio - args.test_ratio, 6),
            "val": round(args.val_ratio, 6),
            "test": round(args.test_ratio, 6),
        },
    }

    with open(out_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 72}")
    print("  Homology split tamamlandı (v7)")
    print(f"  Çıktı: {out_dir}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
