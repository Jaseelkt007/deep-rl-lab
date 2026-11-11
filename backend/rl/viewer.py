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
