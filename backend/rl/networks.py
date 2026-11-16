"""Actor π(a|s) and critic V(s): two separate MLPs (no shared trunk).

Default: 2 hidden layers of 64 tanh units, PyTorch's default initialisation.
Both can return their hidden activations, so we can measure saturation and rank.
"""

from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str = "tanh"):
        super().__init__()
        dims = [in_dim, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = ACTIVATIONS[activation]()
        self.out = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hs = []
        for layer in self.hidden:
            x = self.act(layer(x))
            hs.append(x)
        y = self.out(x)
        return (y, hs) if return_hidden else y

    def linear_layers(self) -> list[tuple[str, nn.Linear]]:
        """(name, layer) for every weight layer, input to output. Used for per-layer stats."""
        named = [(f"h{i + 1}", layer) for i, layer in enumerate(self.hidden)]
        return named + [("out", self.out)]


class Actor(MLP):
    """Outputs logits; π(a|s) = softmax(logits)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64), activation="tanh"):
        super().__init__(obs_dim, tuple(hidden), n_actions, activation)


class GaussianActor(MLP):
    """Continuous actions: the network outputs the MEAN μ(s) of each action dimension; the spread σ is one
    learned number per dimension, the same in every state (log σ, so σ = exp(log σ) is always positive).
    π(a|s) = Π_j N(a_j; μ_j(s), σ_j²)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), activation="tanh", log_std_init: float = 0.0):
        super().__init__(obs_dim, tuple(hidden), act_dim, activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = super().forward(x)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


class Critic(MLP):
    """Outputs V(s), a scalar per state."""

    def __init__(self, obs_dim: int, hidden=(64, 64), activation="tanh"):
        super().__init__(obs_dim, tuple(hidden), 1, activation)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        if return_hidden:
            y, hs = super().forward(x, return_hidden=True)
            return y.squeeze(-1), hs
        return super().forward(x).squeeze(-1)
