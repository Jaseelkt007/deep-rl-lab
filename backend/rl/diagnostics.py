"""Measurements of what the networks have learned. Pure functions, no training logic.

These mirror the questions we ask of a large agent:
  - Is the critic any good?      explained variance of V against Monte Carlo returns
  - Do hidden units still work?  saturation (|h| > 0.95 for tanh), dead/constant units
  - Does the representation depend on the state?   effective rank of hidden features
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


def discounted_returns(rewards: Sequence[float], gamma: float, tail_value: float = 0.0) -> np.ndarray:
    """G_t = r_t + γ G_{t+1}, starting from G_T = tail_value.

    tail_value = 0 after termination. After truncation we fill in the unseen future with
    V(s_T), the same convention the TD target uses, so a perfect critic scores EV = 1.
    """
    g, out = tail_value, np.zeros(len(rewards))
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def explained_variance(pred: Sequence[float], target: Sequence[float]) -> float | None:
    """EV = 1 − Var(target − pred) / Var(target).

    1: the critic explains the returns. 0: no better than a constant. < 0: worse than a
    constant. None when the returns do not vary at all (nothing to explain).
    """
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    var = target.var()
    if len(target) < 2 or var < 1e-10:
        return None
    return float(1.0 - (target - pred).var() / var)


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-10 or b.std() < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def effective_rank(h: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of the centred feature matrix
    (Roy & Vetterli 2007). Ranges from ~1 (every state maps to the same direction)
    up to the number of units. Centred, so a state-independent layer gives 0."""
    h = h - h.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(h.double())
    total = float(s.sum())
    if total < 1e-10:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(math.exp(-(p * p.log()).sum()))


def hidden_stats(hs: list[torch.Tensor], activation: str) -> list[dict]:
    """Per hidden layer, over a batch of states [N, units]:

    saturated: share of activations with |h| > 0.95 (tanh: the gradient there is < 0.1)
    dead:      share of units that never activate (ReLU: h = 0 for every state)
    constant:  share of units whose output barely changes across states (std < 0.01)
    erank:     effective rank of the layer's features
    """
    out = []
    for h in hs:
        h = h.detach()
        std = h.std(dim=0) if h.shape[0] > 1 else torch.zeros(h.shape[1])
        out.append({
            "saturated": float((h.abs() > 0.95).float().mean()) if activation == "tanh" else 0.0,
            "dead": float((h.abs().amax(dim=0) == 0).float().mean()) if activation == "relu" else 0.0,
            "constant": float((std < 0.01).float().mean()),
            "erank": effective_rank(h) if h.shape[0] > 1 else 0.0,
            "units": int(h.shape[1]),
        })
    return out
