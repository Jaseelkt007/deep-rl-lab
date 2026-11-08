"use client";

import { ReactNode, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { storeActions } from "@/lib/store";
import type { RunConfig, Status } from "@/lib/types";

// speed slider: 0..100 -> 1..2000 steps/s (log), 100 = as fast as possible
const SPEED_MAX = 2000;
const toSpeed = (p: number) => (p >= 100 ? null : Math.round(10 ** ((p / 100) * Math.log10(SPEED_MAX))));
const toPos = (s: number | null) => (s == null ? 100 : (Math.log10(Math.max(1, s)) / Math.log10(SPEED_MAX)) * 100);

function Row({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex items-center justify-between gap-2 py-0.5 text-[12px]" title={hint}>
      <span className="text-neutral-600">{label}</span>
      {children}
    </label>
  );
}

const inputCls = "w-24 rounded border border-neutral-200 px-1.5 py-0.5 text-right text-[12px] tabular-nums focus:border-neutral-400 focus:outline-none";
const selectCls = "w-28 rounded border border-neutral-200 bg-white px-1 py-0.5 text-[12px] focus:outline-none";

function NumberInput({ value, onChange, step = "any", allowNull }: { value: number | null; onChange: (v: number | null) => void; step?: string; allowNull?: boolean }) {
  const [text, setText] = useState(value == null ? "" : String(value));
  const [prev, setPrev] = useState(value);
  if (value !== prev) {
    // the value changed from outside (e.g. defaults loaded): show it
    setPrev(value);
    setText(value == null ? "" : String(value));
  }
  return (
    <input
      className={inputCls}
      type="text"
      inputMode="decimal"
      step={step}
      value={text}
      placeholder={allowNull ? "official" : ""}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (allowNull && text.trim() === "") return onChange(null);
        const v = Number(text);
        if (Number.isFinite(v)) onChange(v);
        else setText(value == null ? "" : String(value));
      }}
    />
  );
}

function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      className={`relative h-4 w-7 rounded-full transition-colors ${on ? "bg-[#2a78d6]" : "bg-neutral-300"}`}
      aria-pressed={on}
    >
      <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${on ? "left-3.5" : "left-0.5"}`} />
    </button>
  );
}

function Section({ title, children, open = false }: { title: string; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group border-t border-neutral-100 pt-1.5">
      <summary className="cursor-pointer list-none text-[11px] font-medium uppercase tracking-wide text-neutral-500 group-open:mb-1">
        <span className="mr-1 inline-block transition-transform group-open:rotate-90">›</span>
        {title}
      </summary>
      {children}
    </details>
  );
}

const btn = "rounded-md border px-2.5 py-1 text-[12px] font-medium transition-colors disabled:opacity-40";

export default function Controls({ status }: { status: Status | null }) {
  const [cfg, setCfg] = useState<RunConfig | null>(null);
  const [speed, setSpeed] = useState<number | null>(30);
  const [msg, setMsg] = useState<string>("");

  useEffect(() => {
    api.defaults().then((d) => setCfg(d.config)).catch(() => setMsg("backend not reachable on :8000"));
  }, []);

  if (!cfg) return <div className="text-[12px] text-neutral-400">{msg || "loading…"}</div>;
  const set = <K extends keyof RunConfig>(k: K, v: RunConfig[K]) => setCfg({ ...cfg, [k]: v });
  const mc = cfg.env_id === "MountainCar-v0";
  const state = status?.state ?? "idle";
  const live = state === "running" || state === "paused";

  const run = async (f: () => Promise<unknown>) => {
    try {
      setMsg("");
      await f();
    } catch (e) {
      setMsg(String(e).slice(0, 160));
    }
  };

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1.5">
        <button
          className={`${btn} border-[#2a78d6] bg-[#2a78d6] text-white hover:bg-[#256abf]`}
          onClick={() =>
            run(async () => {
              storeActions.setMode("live");
              await api.start(cfg, speed);
            })
          }
        >
          {live ? "Restart" : "Start"}
        </button>
        <button
          className={`${btn} border-neutral-300 hover:bg-neutral-50`}
          disabled={!live}
          onClick={() => run(() => api.control(state === "running" ? "pause" : "resume"))}
        >
          {state === "running" ? "Pause" : "Resume"}
        </button>
        <button
          className={`${btn} border-neutral-300 hover:bg-neutral-50`}
          disabled={!live}
          title="One environment step (and its update), then pause"
          onClick={() => run(() => api.control("step"))}
        >
          Step
        </button>
        <button
          className={`${btn} border-neutral-300 hover:bg-neutral-50`}
          onClick={() =>
            run(async () => {
              await api.control("reset");
              storeActions.clearLive();
            })
          }
        >
          Reset
        </button>
      </div>

      <div className="py-1">
        <div className="flex justify-between text-[11px] text-neutral-500">
          <span>speed</span>
          <span className="tabular-nums">{speed == null ? "max" : `${speed} steps/s`}</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={toPos(speed)}
          className="w-full"
          onChange={(e) => {
            const s = toSpeed(Number(e.target.value));
            setSpeed(s);
            if (live) api.speed(s).catch(() => {});
          }}
        />
      </div>

      <Section title="Task" open>
        <Row label="environment">
          <select className={selectCls} value={cfg.env_id} onChange={(e) => set("env_id", e.target.value as RunConfig["env_id"])}>
            <option value="MountainCar-v0">MountainCar-v0</option>
            <option value="CartPole-v1">CartPole-v1</option>
          </select>
        </Row>
        <Row label="seed">
          <NumberInput value={cfg.seed} onChange={(v) => set("seed", Math.round(v ?? 0))} />
        </Row>
        <Row label="episodes">
          <NumberInput value={cfg.num_episodes} onChange={(v) => set("num_episodes", Math.max(1, Math.round(v ?? 1)))} />
        </Row>
      </Section>

      <Section title="Algorithm" open>
        <Row label="actor lr">
          <NumberInput value={cfg.actor_lr} onChange={(v) => set("actor_lr", v ?? cfg.actor_lr)} />
        </Row>
        <Row label="critic lr">
          <NumberInput value={cfg.critic_lr} onChange={(v) => set("critic_lr", v ?? cfg.critic_lr)} />
        </Row>
        <Row label="critic / actor lr" hint="Two timescales: the critic usually learns faster than the actor">
          <span className="text-[12px] tabular-nums text-neutral-500">{(cfg.critic_lr / cfg.actor_lr).toFixed(1)}×</span>
        </Row>
        <Row label="γ">
          <NumberInput value={cfg.gamma} onChange={(v) => set("gamma", Math.min(1, Math.max(0, v ?? cfg.gamma)))} />
        </Row>
      </Section>

      <Section title="Helpers (off = plain)" open>
        <Row label="entropy bonus β" hint="actor loss −= β·H(π). 0 = off">
          <NumberInput value={cfg.entropy_beta} onChange={(v) => set("entropy_beta", Math.max(0, v ?? 0))} />
        </Row>
        <Row label="time limit T" hint="Empty = the official limit (200 MountainCar, 500 CartPole)">
          <NumberInput value={cfg.max_episode_steps} allowNull onChange={(v) => set("max_episode_steps", v == null ? null : Math.max(1, Math.round(v)))} />
        </Row>
        <Row label="reward shaping" hint="F = γΦ(s′) − Φ(s): keeps the optimal policy (Ng et al. 1999)">
          <select className={selectCls} value={cfg.shaping} onChange={(e) => set("shaping", e.target.value as RunConfig["shaping"])}>
            <option value="none">none</option>
            <option value="energy">Φ = energy</option>
            <option value="height">Φ = height</option>
          </select>
        </Row>
        {cfg.shaping !== "none" && (
          <Row label="shaping scale">
            <NumberInput value={cfg.shaping_scale} onChange={(v) => set("shaping_scale", v ?? 10)} />
          </Row>
        )}
        <Row label="n-step" hint="1 = plain one-step">
          <NumberInput value={cfg.n_step} onChange={(v) => set("n_step", Math.max(1, Math.round(v ?? 1)))} />
        </Row>
      </Section>

      <Section title="Loss terms">
        <Row label="GAE λ" hint="Only with n-step > 1. Empty = plain n-step (λ = 1)">
          <NumberInput value={cfg.gae_lambda} allowNull onChange={(v) => set("gae_lambda", v == null ? null : Math.min(1, Math.max(0, v)))} />
        </Row>
        <Row label="gradient clip">
          <select className={selectCls} value={cfg.grad_clip} onChange={(e) => set("grad_clip", e.target.value as RunConfig["grad_clip"])}>
            <option value="none">none</option>
            <option value="separate">per network</option>
            <option value="joint">joint (PPO)</option>
          </select>
        </Row>
        {cfg.grad_clip !== "none" && (
          <Row label="clip norm">
            <NumberInput value={cfg.grad_clip_norm} onChange={(v) => set("grad_clip_norm", v ?? 0.5)} />
          </Row>
        )}
        <Row label="advantage norm" hint="Running mean/std (online learning has no batch)">
          <Toggle on={cfg.adv_norm} onChange={(v) => set("adv_norm", v)} />
        </Row>
        <Row label="lr schedule">
          <select className={selectCls} value={cfg.lr_schedule} onChange={(e) => set("lr_schedule", e.target.value as RunConfig["lr_schedule"])}>
            <option value="constant">constant</option>
            <option value="linear">linear → 0</option>
            <option value="cosine">cosine → 0</option>
          </select>
        </Row>
        <Row label="optimizer" hint="SGD needs ~10–100× larger learning rates">
          <select className={selectCls} value={cfg.optimizer} onChange={(e) => set("optimizer", e.target.value as RunConfig["optimizer"])}>
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </Row>
        <Row label="I = γᵗ on actor" hint="Sutton & Barto's discount on the actor step">
          <Toggle on={cfg.actor_discount_I} onChange={(v) => set("actor_discount_I", v)} />
        </Row>
        <Row label="value norm" hint="Critic regresses target / running RMS (simplified PopArt)">
          <Toggle on={cfg.value_norm} onChange={(v) => set("value_norm", v)} />
        </Row>
        <Row label="reward scale">
          <NumberInput value={cfg.reward_scale} onChange={(v) => set("reward_scale", v ?? 1)} />
        </Row>
      </Section>

      <Section title="Formulation">
        <Row label="observation">
          <select className={selectCls} value={cfg.obs_mode} onChange={(e) => set("obs_mode", e.target.value as RunConfig["obs_mode"])}>
            <option value="scaled">scaled [−1, 1]</option>
            <option value="raw">raw</option>
            <option value="rbf">RBF features</option>
          </select>
        </Row>
        <Row label="time-left input">
          <Toggle on={cfg.time_input} onChange={(v) => set("time_input", v)} />
        </Row>
        <Row label="bootstrap at time limit" hint="Off = the classic bug: treat the time limit as a terminal state">
          <Toggle on={cfg.bootstrap_on_truncation} onChange={(v) => set("bootstrap_on_truncation", v)} />
        </Row>
        {mc && (
          <Row label="actions">
            <select className={selectCls} value={cfg.actions} onChange={(e) => set("actions", e.target.value as RunConfig["actions"])}>
              <option value="all">← · → (3)</option>
              <option value="no_noop">← → (2)</option>
            </select>
          </Row>
        )}
        <Row label="activation">
          <select className={selectCls} value={cfg.activation} onChange={(e) => set("activation", e.target.value as RunConfig["activation"])}>
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
          </select>
        </Row>
        <Row label="hidden layers">
          <input
            className={inputCls}
            defaultValue={cfg.hidden_sizes.join(",")}
            onBlur={(e) => {
              const v = e.target.value.split(/[,\s×x]+/).map(Number).filter((n) => Number.isInteger(n) && n > 0);
              if (v.length) set("hidden_sizes", v);
            }}
          />
        </Row>
      </Section>

      <Section title="Logging">
        <Row label="maps every K episodes">
          <NumberInput value={cfg.snapshot_every} onChange={(v) => set("snapshot_every", Math.max(1, Math.round(v ?? 10)))} />
        </Row>
        <Row label="probe every K episodes">
          <NumberInput value={cfg.state_probe_every} onChange={(v) => set("state_probe_every", Math.max(1, Math.round(v ?? 10)))} />
        </Row>
      </Section>

      <div className="border-t border-neutral-100 pt-2">
        <button
          className={`${btn} w-full border-neutral-300 hover:bg-neutral-50`}
          title="Train this setup with 3 seeds in background processes (no live view); compare them later"
          onClick={() =>
            run(async () => {
              const r = await api.batch(cfg, [cfg.seed, cfg.seed + 1, cfg.seed + 2]);
              setMsg(`started ${r.run_ids.length} seeds in the background`);
            })
          }
        >
          Run 3 seeds in background
        </button>
      </div>
      {msg && <div className="text-[11px] text-neutral-500">{msg}</div>}
    </div>
  );
}
