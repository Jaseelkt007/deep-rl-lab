// Shapes of the records the backend emits (see backend/rl/trainer.py).

export type EnvId = "MountainCar-v0" | "CartPole-v1";

export interface RunConfig {
  env_id: EnvId;
  seed: number;
  num_episodes: number;
  gamma: number;
  actor_lr: number;
  critic_lr: number;
  hidden_sizes: number[];
  entropy_beta: number;
  max_episode_steps: number | null;
  shaping: "none" | "energy" | "height";
  shaping_scale: number;
  n_step: number;
  gae_lambda: number | null;
  grad_clip: "none" | "separate" | "joint";
  grad_clip_norm: number;
  adv_norm: boolean;
  lr_schedule: "constant" | "linear" | "cosine";
  actor_discount_I: boolean;
  optimizer: "adam" | "sgd";
  reward_scale: number;
  value_norm: boolean;
  obs_mode: "raw" | "scaled" | "rbf";
  rbf_per_dim: number | null;
  time_input: boolean;
  bootstrap_on_truncation: boolean;
  actions: "all" | "no_noop";
  activation: "tanh" | "relu";
  snapshot_every: number;
  grid_size: number;
  log_steps_every: number;
  checkpoint_every: number;
  state_probe_every: number;
  probe_episodes: number;
}

export interface RunStart {
  type: "run_start";
  config: RunConfig;
  label: string;
  helpers: string[];
  obs_names: string[];
  action_names: string[];
  max_steps: number;
  phase: { dims: number[]; ranges: number[][]; names: string[] };
}

export interface StepRec {
  type: "step";
  ep: number;
  t: number;
  s: number[];
  s_next: number[];
  a: number;
  a_env: number;
  p: number[];
  r_env: number;
  r_shaped: number;
  r_train: number;
  window: number;
  bootstrap: boolean;
  term: boolean;
  trunc: boolean;
  v: number;
  v_next: number;
  delta: number;
  adv_raw: number;
  adv: number;
  target: number;
  actor_loss: number;
  critic_loss: number;
  entropy: number;
  logp: number;
  logp_after: number;
  p_a: number;
  p_a_after: number;
  p_after: number[];
  v_after: number;
  kl: number;
  gn_actor: number;
  gn_critic: number;
  gn_actor_clipped: number;
  gn_critic_clipped: number;
  clip_actor: number;
  clip_critic: number;
  critic_share: number;
  dw_actor: number;
  dw_critic: number;
  actor_weight: number;
  v_scale: number;
}

export interface LayerStat {
  name: string;
  grad: number;
  update: number;
  weight: number;
  ratio: number;
}

export interface HiddenStat {
  saturated: number;
  dead: number;
  constant: number;
  erank: number;
  units: number;
}

export interface EpisodeRec {
  type: "episode";
  ep: number;
  return: number;
  return_shaped: number;
  return_train: number;
  length: number;
  success: boolean;
  terminated: boolean;
  truncated: boolean;
  mean_abs_delta: number;
  adv_std: number;
  ev: number | null;
  corr: number | null;
  g_mean: number;
  lr_actor: number;
  lr_critic: number;
  total_steps: number;
  elapsed: number;
  max_pos: number | null;
  mean_delta: number;
  mean_entropy: number;
  mean_actor_loss: number;
  mean_critic_loss: number;
  mean_gn_actor: number;
  mean_gn_critic: number;
  mean_gn_actor_clipped: number;
  mean_gn_critic_clipped: number;
  mean_critic_share: number;
  mean_clip_actor: number;
  mean_clip_critic: number;
  mean_dw_actor: number;
  mean_dw_critic: number;
  mean_kl: number;
  mean_p_a: number;
  mean_adv_raw: number;
  mean_v: number;
  layers?: { actor: LayerStat[]; critic: LayerStat[] };
  hidden?: { actor: HiddenStat[]; critic: HiddenStat[] };
}

export interface SnapshotRec {
  type: "snapshot";
  ep: number;
  grid: number;
  x: number[];
  y: number[];
  value: number[];
  best_action: number[];
  p_right: number[];
  delta_bins: number;
  delta_mean: (number | null)[];
  delta_count: number[];
  traj: number[][];
}

export interface ProbeRec {
  type: "probe";
  ep: number;
  mean_logits: number[];
  fixed_probs: number[];
  fixed_action: number;
  state_dependence: number;
  returns: { policy: number; greedy: number; fixed: number; fixed_greedy: number };
}

export interface Status {
  type: "status";
  state: "idle" | "running" | "paused" | "finished" | "stopped" | "error";
  run_id?: string;
  label?: string;
  env_id?: EnvId;
  episode?: number;
  num_episodes?: number;
  total_steps?: number;
  speed?: number | null;
  steps_per_sec?: number;
  error?: string | null;
  jobs_running: number;
  jobs_total: number;
}

export interface RunMeta {
  run_id: string;
  label: string;
  helpers: string[];
  env_id: EnvId;
  seed: number;
  config_key: string;
  group: string | null;
  created: string;
  config?: RunConfig;
  summary: {
    status: string;
    episodes: number;
    total_steps?: number;
    recent_return?: number | null;
    recent_success?: number | null;
    best_return?: number | null;
  };
}

export interface RunDetail {
  meta: RunMeta & { config: RunConfig };
  run_start: RunStart | null;
  episodes: EpisodeRec[];
  snapshots: SnapshotRec[];
  probes: ProbeRec[];
  finished: boolean;
  episodes_with_steps: number[];
}
