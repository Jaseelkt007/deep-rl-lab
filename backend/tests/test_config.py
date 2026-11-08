from rl.config import RunConfig


def test_default_config_is_plain():
    cfg = RunConfig()
    assert cfg.helpers() == []
    assert cfg.label() == "plain"


def test_every_switch_shows_up_in_label():
    cfg = RunConfig(
        entropy_beta=0.01, max_episode_steps=1000, shaping="energy", n_step=5, gae_lambda=0.9,
        grad_clip="joint", adv_norm=True, lr_schedule="cosine", actor_discount_I=True, obs_mode="raw", time_input=True,
        bootstrap_on_truncation=False, actions="no_noop", activation="relu", hidden_sizes=(32,),
        actor_optimizer="sgd", adam_eps=1e-5, actor_adam_eps=1e-3,
    )
    assert len(cfg.helpers()) == 18


def test_gae_lambda_ignored_for_one_step():
    assert RunConfig(gae_lambda=0.9).label() == "plain"


def test_config_roundtrip_json():
    cfg = RunConfig(env_id="CartPole-v1", seed=3, hidden_sizes=(32, 32))
    assert RunConfig.model_validate_json(cfg.model_dump_json()) == cfg
