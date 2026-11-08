"""Run configuration: one schema shared by the trainer, the server and runs/<id>/config.json.

Same config (including the seed) -> same curves.
Every helper is OFF by default, so RunConfig() is the plain algorithm on the official task.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EnvId = Literal["MountainCar-v0", "CartPole-v1"]


class RunConfig(BaseModel):
    # --- task -------------------------------------------------------------
    env_id: EnvId = "MountainCar-v0"
    seed: int = 0
    num_episodes: int = Field(1000, ge=1)

    # --- plain one-step actor-critic ---------------------------------------
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    actor_lr: float = Field(1e-3, gt=0.0)
    critic_lr: float = Field(5e-3, gt=0.0)
    hidden_sizes: tuple[int, ...] = (64, 64)

    # --- optional helpers (all off by default) -----------------------------
    # 1. entropy bonus: actor loss -= beta * H(pi(.|s)).  0 = off.
    entropy_beta: float = Field(0.0, ge=0.0)
    # 2. time limit: None = the official one (200 for MountainCar, 500 for CartPole).
    max_episode_steps: int | None = Field(None, ge=1)
    # 3. potential-based reward shaping F = gamma*Phi(s') - Phi(s)  (Ng et al. 1999).
    shaping: Literal["none", "energy", "height"] = "none"
    shaping_scale: float = Field(10.0, gt=0.0)
    # 4. n-step returns. 1 = the plain one-step algorithm.
    n_step: int = Field(1, ge=1, le=512)

    # --- logging -----------------------------------------------------------
    snapshot_every: int = Field(10, ge=1)  # phase-plane maps every K episodes
    grid_size: int = Field(50, ge=4, le=200)
    log_steps_every: int = Field(1, ge=1)  # write per-step records for every k-th episode
    checkpoint_every: int = Field(100, ge=1)

    # ------------------------------------------------------------------------
    def helpers(self) -> list[str]:
        """Short tags for every non-default switch. Empty list = the plain algorithm."""
        tags: list[str] = []
        if self.entropy_beta > 0:
            tags.append(f"entropy β={self.entropy_beta:g}")
        if self.max_episode_steps is not None:
            tags.append(f"T={self.max_episode_steps}")
        if self.shaping != "none":
            tags.append(f"shaping({self.shaping}×{self.shaping_scale:g})")
        if self.n_step > 1:
            tags.append(f"n={self.n_step}")
        return tags

    def label(self) -> str:
        tags = self.helpers()
        return " · ".join(tags) if tags else "plain"
