"""Shared utility functions for the NLP pipeline."""

from __future__ import annotations

import numpy as np
import torch


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean-pooled sentence representation from token hidden states.

    Masks padding tokens out of the average so only real tokens contribute
    to the final sentence vector.

    Args:
        hidden_states: Token hidden states with shape (batch, seq_len, hidden_dim)
        attention_mask: Binary mask with shape (batch, seq_len); 1 for real tokens

    Returns:
        Mean-pooled tensor with shape (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts
