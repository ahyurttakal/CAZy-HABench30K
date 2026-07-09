from pathlib import Path
import json, random
import numpy as np
import torch

def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_json(obj, path):
    path = Path(path); ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
