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

    # --- loss switches (defaults = plain) ---------------------------------
    # Truncated GAE(λ) over the n-step window: A = Σ_k (γλ)^k δ_{t+k}.
    # None = plain n-step return (same as λ = 1). Only has an effect when n_step > 1.
    gae_lambda: float | None = Field(None, ge=0.0, le=1.0)
    # Gradient clipping by global norm: per network, or one joint norm over both
    # networks (what PPO code does with a shared optimizer).
    grad_clip: Literal["none", "separate", "joint"] = "none"
    grad_clip_norm: float = Field(0.5, gt=0.0)
    # Advantage normalisation. Online updates have no batch (a 1-element batch would
    # normalise every advantage to 0), so this uses running EMA mean/std instead.
    adv_norm: bool = False
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"  # over episodes

    # --- RL formulation (defaults = plain) ---------------------------------
    # raw = unscaled state, scaled = [-1, 1] (the plain spec), rbf = Gaussian RBF grid.
    obs_mode: Literal["raw", "scaled", "rbf"] = "scaled"
    rbf_per_dim: int | None = Field(None, ge=2, le=20)  # None = 10 (MountainCar), 5 (CartPole)
    time_input: bool = False  # append time-left fraction (T - t) / T to the input
    # False = deliberate BUG for learning: treat the time limit as a terminal state.
    bootstrap_on_truncation: bool = True
    actions: Literal["all", "no_noop"] = "all"  # MountainCar only: drop "do nothing"
    activation: Literal["tanh", "relu"] = "tanh"

    # --- logging -----------------------------------------------------------
    snapshot_every: int = Field(10, ge=1)  # phase-plane maps every K episodes
    grid_size: int = Field(50, ge=4, le=200)
    log_steps_every: int = Field(1, ge=1)  # write per-step records for every k-th episode
    checkpoint_every: int = Field(100, ge=1)
    # Fixed-ranking probe every K episodes: does the actor use the state at all?
    state_probe_every: int = Field(10, ge=1)
    probe_episodes: int = Field(3, ge=1)

    # ------------------------------------------------------------------------
    def helpers(self) -> list[str]:
        """Short tags for every non-default switch. Empty list = the plain algorithm.

        Seed, learning rates and γ are not tags: they are hyperparameters, shown separately.
        """
        tags: list[str] = []
        if self.entropy_beta > 0:
            tags.append(f"entropy β={self.entropy_beta:g}")
        if self.max_episode_steps is not None:
            tags.append(f"T={self.max_episode_steps}")
        if self.shaping != "none":
            tags.append(f"shaping({self.shaping}×{self.shaping_scale:g})")
        if self.n_step > 1:
            tags.append(f"n={self.n_step}")
        if self.gae_lambda is not None and self.n_step > 1:
            tags.append(f"GAE λ={self.gae_lambda:g}")
        if self.grad_clip != "none":
            tags.append(f"clip {self.grad_clip} {self.grad_clip_norm:g}")
        if self.adv_norm:
            tags.append("adv-norm")
        if self.lr_schedule != "constant":
            tags.append(f"lr {self.lr_schedule}")
        if self.obs_mode != "scaled":
            tags.append(f"obs {self.obs_mode}")
        if self.time_input:
            tags.append("time input")
        if not self.bootstrap_on_truncation:
            tags.append("BUG: no bootstrap at time limit")
        if self.actions != "all" and self.env_id == "MountainCar-v0":
            tags.append("2 actions")
        if self.activation != "tanh":
            tags.append(self.activation)
        if tuple(self.hidden_sizes) != (64, 64):
            tags.append("net " + "×".join(str(h) for h in self.hidden_sizes))
        return tags

    def label(self) -> str:
        tags = self.helpers()
        return " · ".join(tags) if tags else "plain"
