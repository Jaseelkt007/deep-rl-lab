"""Build a continuous-action session from a saved config (the window viewer and play_walker2d.py use this)."""

from __future__ import annotations


def session_from_state(state: dict):
    """A session of the right kind (PPO or off-policy) with the saved weights loaded."""
    cfg = state["config"]
    if cfg.get("kind") == "offpolicy":
        from .offpolicy import OffPolicyConfig, OffPolicySession

        s = OffPolicySession(OffPolicyConfig(**{**cfg, "buffer_size": 1, "total_steps": None}))
    else:
        from .ppo_continuous import PPOContinuousConfig, PPOContinuousSession

        s = PPOContinuousSession(PPOContinuousConfig(**cfg))
    s.load_state_dict(state)
    return s
