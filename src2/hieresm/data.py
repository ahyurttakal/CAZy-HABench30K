from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class LabelMaps:
    families: List[str]
    classes: List[str]
    family_to_id: Dict[str, int]
    class_to_id: Dict[str, int]
    family_to_class_id: Dict[int, int]


def build_label_maps(df: pd.DataFrame, family_col: str, class_col: str) -> LabelMaps:
    families = sorted(df[family_col].dropna().unique().tolist())
    classes = sorted(df[class_col].dropna().unique().tolist())
    family_to_id = {x: i for i, x in enumerate(families)}
    class_to_id = {x: i for i, x in enumerate(classes)}

    family_to_class_id = {}
    for fam, grp in df.groupby(family_col):
        cls = grp[class_col].mode().iloc[0]
        family_to_class_id[family_to_id[fam]] = class_to_id[cls]

    return LabelMaps(families, classes, family_to_id, class_to_id, family_to_class_id)


class CAZySequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        label_maps: LabelMaps,
        sequence_col: str = "sequence",
        family_col: str = "family",
        class_col: str = "class",
        protein_id_col: str = "protein_id",
        max_length: int = 512,
    ):
        self.df = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.sequence_col = sequence_col
        self.family_col = family_col
        self.class_col = class_col
        self.protein_id_col = protein_id_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row[self.sequence_col]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["family_label"] = torch.tensor(self.label_maps.family_to_id[row[self.family_col]], dtype=torch.long)
        item["class_label"] = torch.tensor(self.label_maps.class_to_id[row[self.class_col]], dtype=torch.long)
        item["protein_id"] = str(row[self.protein_id_col])
        return item


def load_split_frame(data_dir: str | Path, cfg: dict, split_name: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    id_col = cfg["data"].get("protein_id_column", "protein_id")
    labels = pd.read_csv(data_dir / cfg["data"].get("labels_file", "labels.csv"))

    split_dir = data_dir / cfg["data"].get("split_dir", "splits")
    split_file = cfg["data"].get(f"{split_name}_split", f"{split_name}.csv")
    split_ids = pd.read_csv(split_dir / split_file)

    return labels[labels[id_col].isin(split_ids[id_col])].copy()


def load_all_frames(data_dir: str | Path, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelMaps]:
    labels = pd.read_csv(Path(data_dir) / cfg["data"].get("labels_file", "labels.csv"))
    family_col = cfg["data"].get("family_column", "family")
    class_col = cfg["data"].get("class_column", "class")
    label_maps = build_label_maps(labels, family_col, class_col)
    train = load_split_frame(data_dir, cfg, "train")
    val = load_split_frame(data_dir, cfg, "val")
    test = load_split_frame(data_dir, cfg, "test")
    return train, val, test, label_maps
