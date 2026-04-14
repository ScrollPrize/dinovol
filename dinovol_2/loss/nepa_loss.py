"""NEPA: Next-Embedding Predictive Autoregression loss.

Implements the core NEPA objective from "From Representations to Models:
Next-Embedding Predictive Autoregression for Visual Self-Supervised Learning"
(arXiv:2512.16922).

The loss predicts the next patch embedding autoregressively: given the
transformer output at position t, predict the input embedding at position t+1.
A stop-gradient is applied to the target embeddings to prevent collapse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NEPALoss(nn.Module):
    """Next-embedding prediction loss with stop gradient on the target.

    Computes negative cosine similarity between the transformer output
    at position t (prediction) and the input embedding at position t+1
    (target, detached).
    """

    def __init__(self, shift: bool = True) -> None:
        super().__init__()
        self.shift = shift

    def forward(
        self,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the NEPA loss.

        Args:
            input_embeddings: Patch embeddings *before* the transformer,
                shape ``(B, N, D)``. Gradients are detached (stop-gradient).
            output_embeddings: Hidden states *after* the transformer (but
                before the DINO/iBOT heads), shape ``(B, N, D)``.

        Returns:
            Scalar loss (negative cosine similarity, averaged over all
            valid prediction positions and the batch).
        """
        target = input_embeddings.detach()

        if self.shift:
            pred = output_embeddings[:, :-1, :]
            target = target[:, 1:, :]
        else:
            pred = output_embeddings

        # Cast to fp32 for numerical stability under AMP: the default
        # F.normalize eps=1e-12 underflows in fp16/bf16, producing NaNs
        # when a patch embedding happens to have near-zero norm (common
        # after zero-initialised mask_token or early training).
        pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
        target = F.normalize(target.float(), dim=-1, eps=1e-6)

        loss = -(pred * target).sum(dim=-1).mean()
        return loss.to(output_embeddings.dtype)
