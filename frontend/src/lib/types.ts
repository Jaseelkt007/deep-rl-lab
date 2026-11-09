// Shapes shared by the environment drawing (EnvView) and the Learn page.

export type EnvId = "MountainCar-v0" | "CartPole-v1";

/** One step as EnvView draws it: the state, the action taken and the action probabilities. */
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
  amp_actor: number;
  amp_critic: number;
  actor_weight: number;
  v_scale: number;
}
