from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


VARIANTS = {
    "frozen_esm2": {"use_attention_pooling": False, "use_hch": False, "use_ldam_drw": False},
    "esm2_lora": {"use_attention_pooling": False, "use_hch": False, "use_ldam_drw": False},
    "esm2_lora_attention": {"use_attention_pooling": True, "use_hch": False, "use_ldam_drw": False},
    "esm2_lora_attention_hch": {"use_attention_pooling": True, "use_hch": True, "use_ldam_drw": False},
    "full_hieresm": {"use_attention_pooling": True, "use_hch": True, "use_ldam_drw": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", default="1,7,42")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, overrides in VARIANTS.items():
        cfg = dict(base_cfg)
        cfg["model"] = dict(base_cfg["model"])
        cfg["model"].update(overrides)

        cfg_path = out / f"{name}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"))

        for seed in args.seeds.split(","):
            cmd = [
                "python", "-m", "src2.hieresm.train",
                "--config", str(cfg_path),
                "--data_dir", args.data_dir,
                "--out_dir", str(out / name),
                "--seed", seed,
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
