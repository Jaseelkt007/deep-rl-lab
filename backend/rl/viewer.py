"""Play the current policy in a window WITHOUT pausing training.

The training script starts `play_greedy` in a separate process (multiprocessing, "spawn") with a COPY of
the actor's weights at that moment. Training goes on while the window plays; the window shows the policy
as it was when the copy was taken. It only reads weights, so it cannot change training.
"""

from __future__ import annotations


def play_greedy(env_id: str, env_kwargs: dict, actor_state: dict, hidden: int, n_actions: int, seed: int,
                label: str, score_judged: bool, title: str = "") -> None:
    import gymnasium as gym
    import torch

    from rl import envs  # noqa: F401  (registers the custom environments, e.g. the double cart-pole)
    from rl.config import RunConfig
    from rl.envs import ObsEncoder
    from rl.networks import Actor

    torch.set_num_threads(1)
    enc = ObsEncoder(RunConfig(env_id=env_id))
    actor = Actor(enc.dim, n_actions, (hidden,))
    actor.load_state_dict(actor_state)
    actor.eval()
    env = gym.make(env_id, render_mode="human", **env_kwargs)
    obs, _ = env.reset(seed=seed)  # the first frame opens the window
    if title:
        import pygame

        pygame.display.set_caption(f"{title} · {label}")  # which run this window belongs to
    steps, score = 0, 0.0
    try:
        with torch.no_grad():
            while True:
                a = int(actor(torch.as_tensor(enc(obs, 0))).argmax())  # the most likely action
                obs, r, term, trunc, _ = env.step(a)
                steps += 1
                score += float(r)
                if term or trunc:
                    break
    finally:
        env.close()
    result = f"score {score:.0f}" if score_judged else f"{steps} steps"
    print(f"     ↳ window ({label}) finished: {result}", flush=True)


def play_continuous(env_id: str, state: dict, label: str, seed: int = 10_000) -> None:
    """Continuous actions (MuJoCo): play the mean action of a PPOContinuousSession state in a window."""
    import gymnasium as gym
    import torch

    from rl.ppo_continuous import PPOContinuousConfig, PPOContinuousSession

    torch.set_num_threads(1)
    session = PPOContinuousSession(PPOContinuousConfig(**state["config"]))
    session.load_state_dict(state)
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=seed)
    steps, score = 0, 0.0
    try:
        while True:
            obs, r, term, trunc, info = env.step(session.act(obs))
            steps += 1
            score += float(r)
            if term or trunc:
                break
    finally:
        env.close()
    how = "fell" if term else "still up at the time limit"
    print(f"     ↳ window ({label}) finished: score {score:.0f}, {steps} steps, walked {info['x_position']:+.1f} m ({how})",
          flush=True)
