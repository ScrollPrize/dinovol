from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ComponentContrastiveLoss(nn.Module):
    """Supervised contrastive loss over sparse patch-token component labels.

    Negatives are scoped to a single source sample. This avoids asserting that
    arbitrary components from unrelated TIFFs represent different semantic classes.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        rows: torch.Tensor,
        patch_indices: torch.Tensor,
        group_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rows.numel() == 0:
            return patch_tokens.sum() * 0.0

        rows = rows.to(device=patch_tokens.device, dtype=torch.long)
        patch_indices = patch_indices.to(device=patch_tokens.device, dtype=torch.long)
        group_ids = group_ids.to(device=patch_tokens.device, dtype=torch.long)
        sample_ids = sample_ids.to(device=patch_tokens.device, dtype=torch.long)

        valid = (
            (rows >= 0)
            & (rows < patch_tokens.shape[0])
            & (patch_indices >= 0)
            & (patch_indices < patch_tokens.shape[1])
        )
        if masks is not None and bool(valid.any()):
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            unmasked = ~masks[rows[valid_indices], patch_indices[valid_indices]]
            valid[valid_indices] &= unmasked
        if int(valid.sum()) < 3:
            return patch_tokens.sum() * 0.0

        rows = rows[valid]
        patch_indices = patch_indices[valid]
        group_ids = group_ids[valid]
        sample_ids = sample_ids[valid]
        features = F.normalize(patch_tokens[rows, patch_indices].float(), dim=-1)

        sample_losses: list[torch.Tensor] = []
        for sample_id in torch.unique(sample_ids):
            in_sample = sample_ids == sample_id
            sample_features = features[in_sample]
            sample_groups = group_ids[in_sample]
            if sample_features.shape[0] < 3 or torch.unique(sample_groups).numel() < 2:
                continue

            logits = sample_features @ sample_features.T / self.temperature
            diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
            positive = sample_groups[:, None].eq(sample_groups[None, :]) & ~diagonal
            anchors = positive.any(dim=1)
            if not bool(anchors.any()):
                continue

            denominator_logits = logits.masked_fill(diagonal, float("-inf"))
            numerator_logits = logits.masked_fill(~positive, float("-inf"))
            per_anchor = -(
                torch.logsumexp(numerator_logits, dim=1)
                - torch.logsumexp(denominator_logits, dim=1)
            )
            sample_losses.append(per_anchor[anchors].mean())

        if not sample_losses:
            return patch_tokens.sum() * 0.0
        return torch.stack(sample_losses).mean().to(dtype=patch_tokens.dtype)
