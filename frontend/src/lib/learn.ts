// Types, API and small maths for the Learn page (batch actor-critic, one iteration at a time).
import { BACKEND } from "./api";

export type Mat = number[][]; // [out][in], torch convention
export interface NetWeights {
  W1: Mat;
  b1: number[];
  W2: Mat;
  b2: number[];
}

export interface LearnConfig {
  env_id: "CartPole-v1" | "MountainCar-v0";
  seed: number;
  batch_episodes: number | null;
  hidden: number;
  optimizer: "adam" | "sgd";
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  shaping: "none" | "energy";
  grid?: number;
}

export interface Maps {
  grid: number;
  x: number[];
  y: number[];
  value: number[];
  p_last: number[];
}

export interface Describe {
  session: string;
  config: LearnConfig;
  n_episodes: number;
  obs_names: string[];
  action_names: string[];
  obs_low: number[];
  obs_high: number[];
  reward: string;
  terminated: string;
  truncated: string;
  max_steps: number;
  value_meaning: string;
  weights: { actor: NetWeights; critic: NetWeights };
  maps: Maps;
  phase: { dims: number[]; ranges: number[][]; names: string[] };
}

export interface Episode {
  states: number[][];
  actions: number[];
  probs: number[][];
  rewards: number[];
  rewards_env: number[];
  final_state: number[];
  terminated: boolean;
  truncated: boolean;
  length: number;
  ret: number;
  final_value: number;
  tail: number;
}

export interface Summary {
  k: number;
  mean_return: number;
  mean_length: number;
  steps: number;
  critic_loss: number;
  critic_loss_after: number;
  actor_loss: number;
  entropy: number;
  ev: number | null;
  mean_abs_adv: number;
  mean_value: number;
  mean_return_G: number;
}

export interface NetStep {
  loss: number;
  loss_after?: number;
  grads: NetWeights;
  delta: NetWeights;
  weights_after: NetWeights;
  lr: number;
}

export interface Trace {
  k: number;
  summary: Summary;
  weights_before: { actor: NetWeights; critic: NetWeights };
  episodes: Episode[];
  features: number[][];
  ep_index: number[];
  t_index: number[];
  actions: number[];
  values: number[];
  returns: number[];
  advantages: number[];
  critic: NetStep & { values_after: number[] };
  actor: NetStep & { probs_before: number[][]; probs_after: number[][]; logp_before: number[]; logp_after: number[] };
  maps: { before: Maps; after: Maps };
  optimizer: "adam" | "sgd";
  gamma: number;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(BACKEND + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export const learnApi = {
  create: (cfg: LearnConfig) => post<Describe>("/api/learn/new", cfg),
  step: (sid: string) => post<{ trace: Trace; summaries: Summary[] }>(`/api/learn/${sid}/step`),
  forward: (sid: string, n: number) => post<{ trace: Trace; summaries: Summary[] }>(`/api/learn/${sid}/forward`, { n }),
};

// ------------------------------------------------------------------ forward pass
/** h = tanh(W1·x + b1); out = W2·h + b2. Exactly what the networks in rl/networks.py compute. */
export function forward(w: NetWeights, x: number[]) {
  const pre = w.W1.map((row, j) => row.reduce((s, wij, i) => s + wij * x[i], w.b1[j]));
  const h = pre.map(Math.tanh);
  const out = w.W2.map((row, k) => row.reduce((s, wkj, j) => s + wkj * h[j], w.b2[k]));
  return { pre, h, out };
}

export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Scaled features in [−1, 1], same as backend rl/envs.py scale_obs. */
export function scale(obs: number[], low: number[], high: number[]) {
  return obs.map((v, i) => (2 * (v - low[i])) / (high[i] - low[i]) - 1);
}

export const maxAbs = (m: Mat | number[]) =>
  Math.max(1e-12, ...(m as (number | number[])[]).flat().map((v) => Math.abs(v as number)));
