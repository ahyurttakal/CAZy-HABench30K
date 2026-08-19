from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        scores = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class HierarchicalConditionalHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_families: int, family_to_class_id: Dict[int, int]):
        super().__init__()
        self.family_head = nn.Linear(hidden_dim, num_families)
        mask = torch.full((num_classes, num_families), -1e9)
        for fam_id, cls_id in family_to_class_id.items():
            mask[int(cls_id), int(fam_id)] = 0.0
        self.register_buffer("hierarchy_mask", mask)

    def forward(self, features: torch.Tensor, class_log_prob: torch.Tensor):
        raw_family_logits = self.family_head(features)
        masked = raw_family_logits.unsqueeze(1) + self.hierarchy_mask.unsqueeze(0)
        cond_log_prob = F.log_softmax(masked, dim=-1)
        family_log_prob = torch.logsumexp(class_log_prob.unsqueeze(-1) + cond_log_prob, dim=1)
        return family_log_prob, raw_family_logits


class HierESM(nn.Module):
    def __init__(self, cfg: dict, label_maps):
        super().__init__()
        model_name = cfg["model"].get("backbone", "facebook/esm2_t12_35M_UR50D")
        self.backbone = AutoModel.from_pretrained(model_name)

        lora_cfg = LoraConfig(
            r=cfg["model"].get("lora_r", 16),
            lora_alpha=cfg["model"].get("lora_alpha", 32),
            lora_dropout=cfg["model"].get("lora_dropout", 0.05),
            target_modules=cfg["model"].get("lora_targets", ["query", "value"]),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        backbone_dim = getattr(self.backbone.base_model.model.config, "hidden_size", 480)
        hidden_dim = cfg["model"].get("hidden_dim", 256)
        dropout = cfg["model"].get("dropout", 0.2)

        self.use_attention_pooling = cfg["model"].get("use_attention_pooling", True)
        self.pool = AttentionPooling(backbone_dim) if self.use_attention_pooling else None

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(label_maps.classes)),
        )

        self.use_hch = cfg["model"].get("use_hch", True)
        if self.use_hch:
            self.family_head = HierarchicalConditionalHead(
                hidden_dim,
                len(label_maps.classes),
                len(label_maps.families),
                label_maps.family_to_class_id,
            )
        else:
            self.family_head = nn.Linear(hidden_dim, len(label_maps.families))

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        if self.pool is not None:
            pooled, attn_weights = self.pool(hidden, attention_mask)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            attn_weights = None

        features = self.proj(pooled)
        class_logits = self.class_head(features)
        class_log_prob = F.log_softmax(class_logits, dim=-1)

        if self.use_hch:
            family_log_prob, raw_family_logits = self.family_head(features, class_log_prob)
            family_logits = family_log_prob
        else:
            family_logits = self.family_head(features)
            family_log_prob = F.log_softmax(family_logits, dim=-1)
            raw_family_logits = family_logits

        return {
            "family_logits": family_logits,
            "family_log_prob": family_log_prob,
            "class_logits": class_logits,
            "class_log_prob": class_log_prob,
            "attention_weights": attn_weights,
            "features": features,
            "raw_family_logits": raw_family_logits,
        }


class LDAMLoss(nn.Module):
    def __init__(self, class_counts, max_margin: float = 0.7, temperature: float = 1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = max_margin / torch.pow(counts.clamp_min(1.0), 0.25)
        self.register_buffer("margins", margins)
        self.temperature = temperature

    def forward(self, logits, targets):
        margins = self.margins.to(logits.device)[targets]
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), targets] -= margins
        return F.cross_entropy(adjusted / self.temperature, targets)


def hierarchy_aware_loss(outputs, family_labels, class_labels, family_counts, class_counts, cfg: dict):
    loss_cfg = cfg.get("loss", {})
    fam_loss_fn = LDAMLoss(
        family_counts,
        max_margin=loss_cfg.get("ldam_max_margin", 0.7),
        temperature=loss_cfg.get("temperature", 1.0),
    ).to(family_labels.device)
    cls_loss_fn = LDAMLoss(
        class_counts,
        max_margin=loss_cfg.get("ldam_max_margin", 0.7),
        temperature=loss_cfg.get("temperature", 1.0),
    ).to(class_labels.device)

    fam_loss = fam_loss_fn(outputs["family_logits"], family_labels)
    cls_loss = cls_loss_fn(outputs["class_logits"], class_labels)

    # KL consistency can be extended using a class-family aggregation mask.
    kl_loss = torch.tensor(0.0, device=family_labels.device)

    return (
        loss_cfg.get("lambda_family", 1.0) * fam_loss
        + loss_cfg.get("lambda_class", 0.3) * cls_loss
        + loss_cfg.get("lambda_kl", 0.1) * kl_loss
    )
