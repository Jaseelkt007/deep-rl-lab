"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import { ACTION_COLORS, ACTION_GLYPH, C, divAction, rgbCss, seqOrange } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { Describe, Maps, NetWeights, Trace } from "@/lib/learn";
import { forward, maxAbs, softmax } from "@/lib/learn";
import type { StepRec } from "@/lib/types";
import EnvView from "../EnvView";
import { Heatmap } from "../Heatmap";
import BarStrip from "./BarStrip";
import NetDiagram, { EdgeMode, MatrixGrid } from "./NetDiagram";

// ----------------------------------------------------------------- small helpers
export function Box({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-lg border border-neutral-200 bg-white p-3 ${className}`}>{children}</div>;
}

export function Explain({ children }: { children: ReactNode }) {
  return <div className="space-y-1.5 text-[13px] leading-relaxed text-neutral-700">{children}</div>;
}

export function Formula({ children }: { children: ReactNode }) {
  return <div className="my-1 rounded bg-neutral-50 px-2 py-1 font-mono text-[12px] text-neutral-800">{children}</div>;
}

const sgn = (v: number) => (v > 0 ? "+" : v < 0 ? "−" : "");
const signColor = (v: number) => (v > 0 ? C.pos : v < 0 ? C.neg : C.muted);
const shortObs = (n: string) => n.replace("pole angular velocity", "pole ang. vel").replace("cart ", "cart ").replace("pole angle", "pole angle");

/** Animated sub-steps inside a stage: Back / Next / ▶ play. */
function useSubsteps(n: number, ms = 1800) {
  const [i, setI] = useState(0);
  const [want, setPlaying] = useState(false);
  const playing = want && i < n - 1; // stops by itself at the last sub-step
  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => setI((x) => Math.min(n - 1, x + 1)), ms);
    return () => window.clearTimeout(id);
  }, [playing, i, n, ms]);
  const bar = (labels: string[]) => (
    <div className="mb-3 flex flex-wrap items-center gap-1.5">
      {labels.map((l, k) => (
        <button
          key={l}
          onClick={() => (setPlaying(false), setI(k))}
          className={`rounded-full border px-2.5 py-0.5 text-[11px] ${k === i ? "border-neutral-800 bg-neutral-800 text-white" : k < i ? "border-neutral-300 bg-neutral-100 text-neutral-700" : "border-neutral-200 text-neutral-500"}`}
        >
          {k + 1}. {l}
        </button>
      ))}
      <button
        onClick={() => (i >= n - 1 ? (setI(0), setPlaying(true)) : setPlaying(!playing))}
        className="ml-1 rounded border border-neutral-300 px-2 py-0.5 text-[11px] hover:bg-neutral-50"
      >
        {playing ? "❚❚ pause" : "▶ animate"}
      </button>
    </div>
  );
  return { i, setI, bar };
}

function boundariesOf(tr: Trace) {
  const b: number[] = [];
  let acc = 0;
  for (const ep of tr.episodes) {
    b.push(acc);
    acc += ep.length;
  }
  return b;
}

function stepRec(state: number[], a: number, p: number[]): StepRec {
  return { s: state, a, p } as unknown as StepRec;
}

/** Forward pass of both networks for one state, ready for NetDiagram. */
function activations(w: NetWeights, x: number[], kind: "actor" | "critic", actionNames: string[]) {
  const f = forward(w, x);
  if (kind === "actor") {
    const p = softmax(f.out);
    return { x, h: f.h, out: p, outText: p.map((v) => `P=${v.toFixed(2)}`), names: actionNames.map((a) => `${ACTION_GLYPH[a]} ${a}`) };
  }
  return { x, h: f.h, out: f.out, outText: [`V=${f.out[0].toFixed(2)}`], names: ["V(s)"] };
}

function MapPair({ title, before, after, d, kind }: { title: string; before: Maps; after?: Maps; d: Describe; kind: "value" | "p_last" }) {
  const all = [...before[kind], ...(after?.[kind] ?? [])];
  const lo = kind === "p_last" ? 0 : Math.min(...all);
  const hi = kind === "p_last" ? 1 : Math.max(...all);
  const color = (v: number) => (kind === "p_last" ? rgbCss(divAction(v)) : rgbCss(seqOrange(hi - lo < 1e-9 ? 0.5 : (v - lo) / (hi - lo))));
  const last = d.action_names.at(-1)!;
  const tip = (v: number | null) => (v == null ? "" : kind === "p_last" ? `P(${last}) = ${v.toFixed(3)}` : `V = ${v.toFixed(3)}`);
  const one = (m: Maps, label: string) => (
    <div className="w-40">
      <div className="mb-0.5 text-[10px] text-neutral-500">{label}</div>
      <Heatmap nx={m.grid} ny={m.grid} values={m[kind]} color={color} xRange={d.phase.ranges[0]} yRange={d.phase.ranges[1]} names={d.phase.names} tip={tip} />
    </div>
  );
  return (
    <div>
      <div className="mb-1 text-[11px] font-medium text-neutral-700">{title}</div>
      <div className="flex gap-3">
        {one(before, after ? "before this update" : "now")}
        {after && one(after, "after this update")}
      </div>
      <div className="mt-1 text-[10px] tabular-nums text-neutral-500">
        {kind === "p_last" ? `colour: P(${last}) from 0 (violet) to 1 (amber), gray = 0.5` : `colour: V from ${fmt(lo)} (light) to ${fmt(hi)} (dark)`} · y axis: {d.phase.names[1]}
      </div>
    </div>
  );
}

// =================================================================== stage 0: problem
export function ProblemStage({ d }: { d: Describe }) {
  const mc = d.config.env_id === "MountainCar-v0";
  const rows: [string, string, string][] = [
    ["state s (what the agent sees)", "position, velocity (2 numbers)", "cart position, cart velocity, pole angle, pole angular velocity (4 numbers)"],
    ["actions a", "← push left · do nothing · → push right", "← push left · → push right"],
    ["reward r per step", "−1 every step (time is costly)", "+1 every step the pole is still up"],
    ["episode ends: terminated", "car reaches the flag (x ≥ 0.5): a real end, nothing comes after", "pole tips past ±12° or the cart leaves the track: a real end"],
    ["episode ends: truncated", "after 200 steps (time limit): the future would continue", "after 500 steps (time limit)"],
    ["best possible", "reach the flag fast (≈ −100 return)", "balance all 500 steps (return 500)"],
    ["what V(s) means", "≈ −(steps still needed, discounted); −100 if the flag is never reached", "≈ how many more steps the pole stays up (discounted, max ≈ 100)"],
    ["what makes it hard", "random pushing almost never reaches the flag: every step looks the same (−1), so there is nothing to learn from", "falling is immediate feedback: bad actions shorten the episode, so learning starts at once"],
  ];
  const example = mc ? [-0.5, 0.0] : [0.0, 0.0, 0.03, 0.0];
  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_360px]">
      <Box>
        <Explain>
          <p>
            <b>Reinforcement learning</b> = an <b>agent</b> repeatedly sees a <b>state</b>, picks an <b>action</b>, and the <b>environment</b> answers with a
            <b> reward</b> and the next state. The goal: pick actions that make the <b>total reward</b> of an episode as large as possible. Nobody tells the agent the
            correct action; it only sees rewards.
          </p>
          <p>The two environments use the same algorithm, but the problem is set up (formulated) very differently:</p>
        </Explain>
        <table className="mt-2 w-full text-left text-[12px]">
          <thead>
            <tr className="text-[11px] text-neutral-500">
              <th className="w-44 py-1 font-normal" />
              <th className={`py-1 pr-3 font-medium ${mc ? "text-neutral-900" : "text-neutral-400"}`}>MountainCar-v0</th>
              <th className={`py-1 font-medium ${!mc ? "text-neutral-900" : "text-neutral-400"}`}>CartPole-v1</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, a, b]) => (
              <tr key={k} className="border-t border-neutral-100 align-top">
                <td className="py-1.5 pr-3 text-neutral-500">{k}</td>
                <td className={`py-1.5 pr-3 ${mc ? "text-neutral-900" : "text-neutral-400"}`}>{a}</td>
                <td className={`py-1.5 ${!mc ? "text-neutral-900" : "text-neutral-400"}`}>{b}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Box>
      <Box>
        <div className="mb-1 text-[12px] font-medium text-neutral-800">{d.config.env_id}</div>
        <EnvView envId={d.config.env_id} step={stepRec(example, -1, d.action_names.map(() => 1 / d.action_names.length))} trail={[]} actionNames={d.action_names} />
        <div className="mt-2 text-[11px] text-neutral-500">The bars will show the actor&apos;s probability for each action.</div>
      </Box>
    </div>
  );
}

// ================================================================ stage 1: networks
export function NetworksStage({ d, weights, title }: { d: Describe; weights: { actor: NetWeights; critic: NetWeights }; title?: string }) {
  const [px, py] = d.phase.dims;
  const [sx, setSx] = useState(0.5);
  const [sy, setSy] = useState(0.5);
  const state = useMemo(() => {
    const s = d.obs_low.map((lo, i) => (lo + d.obs_high[i]) / 2);
    s[px] = d.phase.ranges[0][0] + sx * (d.phase.ranges[0][1] - d.phase.ranges[0][0]);
    s[py] = d.phase.ranges[1][0] + sy * (d.phase.ranges[1][1] - d.phase.ranges[1][0]);
    return s;
  }, [d, px, py, sx, sy]);
  const x = state.map((v, i) => (2 * (v - d.obs_low[i])) / (d.obs_high[i] - d.obs_low[i]) - 1);
  const act = activations(weights.actor, x, "actor", d.action_names);
  const cri = activations(weights.critic, x, "critic", d.action_names);
  const inputs = d.obs_names.map(shortObs);
  return (
    <div className="space-y-3">
      <Box>
        <Explain>
          <p>
            {title ?? "The agent is two small neural networks."} Both take the <b>same input</b>: the state, scaled to about [−1, 1]. Each has one hidden layer of{" "}
            {weights.actor.W1.length} tanh units.
          </p>
          <p>
            <b style={{ color: C.actor }}>Actor (policy π)</b>: state → one score per action → softmax → <b>probabilities</b>. It has the same shape as a classification
            network, but there are <b>no labels</b>. To act, it <b>samples</b> from these probabilities.
            <br />
            <b style={{ color: C.critic }}>Critic (value V)</b>: state → <b>one number</b> V(s) = &quot;how much reward do I expect from here on, if I keep acting like
            this?&quot;. This is a regression network.
          </p>
          <p className="text-neutral-500">
            Every line is one weight: <span style={{ color: C.pos }}>aqua = positive</span>, <span style={{ color: C.neg }}>red = negative</span>, thicker = larger.
            Circles show the values flowing through for the state you pick below. Right now the weights are random, so the outputs are meaningless.
          </p>
        </Explain>
        <div className="mt-2 flex flex-wrap items-center gap-4 text-[12px]">
          <label className="flex items-center gap-2">
            {d.phase.names[0]}
            <input type="range" min={0} max={1} step={0.01} value={sx} onChange={(e) => setSx(+e.target.value)} />
            <span className="w-14 tabular-nums">{fmt(state[px], 3)}</span>
          </label>
          <label className="flex items-center gap-2">
            {d.phase.names[1]}
            <input type="range" min={0} max={1} step={0.01} value={sy} onChange={(e) => setSy(+e.target.value)} />
            <span className="w-14 tabular-nums">{fmt(state[py], 3)}</span>
          </label>
          {d.obs_names.length > 2 && <span className="text-neutral-500">(cart position and velocity held at 0)</span>}
        </div>
      </Box>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Box>
          <NetDiagram title="Actor π(a|s)" color={C.actor} edges={weights.actor} mode="weights" inputs={inputs} outputs={act.names} activations={act} />
        </Box>
        <Box>
          <NetDiagram title="Critic V(s)" color={C.critic} edges={weights.critic} mode="weights" inputs={inputs} outputs={cri.names} activations={cri} />
        </Box>
      </div>
    </div>
  );
}

// ================================================================ stage 2: collect
export function CollectStage({ d, tr }: { d: Describe; tr: Trace }) {
  const steps = useMemo(() => tr.episodes.flatMap((ep, e) => ep.states.map((s, t) => ({ e, t, s, a: ep.actions[t], p: ep.probs[t], r: ep.rewards_env[t] }))), [tr]);
  const [i, setI] = useState(0);
  const [want, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(tr.k === 1 ? 15 : 60);
  const playing = want && i < steps.length - 1;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setI((x) => (x >= steps.length - 1 ? x : x + 1)), 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, steps.length]);
  const cur = steps[Math.min(i, steps.length - 1)];
  const ep = tr.episodes[cur.e];
  const sofar = ep.rewards_env.slice(0, cur.t + 1).reduce((a, b) => a + b, 0);
  const trail = d.config.env_id === "MountainCar-v0" ? ep.states.slice(0, cur.t + 1).map((s) => s[0]) : [];
  const name = d.action_names[cur.a];
  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_1fr]">
      <Box>
        <Explain>
          <p>
            <b>Step 1: collect a batch.</b> The current actor is <b>frozen</b>. It plays {tr.episodes.length} complete episodes, and we record every step: the state, the
            probabilities, the action it sampled, and the reward.
          </p>
          <p className="text-neutral-500">
            Nothing learns while playing. On-policy means the data must come from <i>this</i> policy, so after one update the batch is thrown away and a new one is
            played.
          </p>
        </Explain>
        <div className="mt-3 flex items-center gap-2 text-[12px]">
          <button className="rounded border border-neutral-300 px-2 py-0.5" onClick={() => (i >= steps.length - 1 ? (setI(0), setPlaying(true)) : setPlaying(!playing))}>
            {playing ? "❚❚" : "▶"}
          </button>
          <input type="range" min={0} max={steps.length - 1} value={i} onChange={(e) => (setPlaying(false), setI(+e.target.value))} className="flex-1" />
          <select className="rounded border border-neutral-300 px-1 py-0.5" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
            {[2, 5, 15, 60, 240].map((s) => (
              <option key={s} value={s}>
                {s} steps/s
              </option>
            ))}
          </select>
          <button className="rounded border border-neutral-300 px-2 py-0.5" onClick={() => (setPlaying(false), setI(steps.length - 1))}>
            skip to end
          </button>
        </div>
        <table className="mt-3 w-full text-[12px] tabular-nums">
          <tbody>
            <tr>
              <td className="text-neutral-500">episode</td>
              <td>
                {cur.e + 1} of {tr.episodes.length}, step t = {cur.t}
              </td>
            </tr>
            <tr>
              <td className="text-neutral-500">state s</td>
              <td>{cur.s.map((v) => v.toFixed(3)).join(", ")}</td>
            </tr>
            <tr>
              <td className="text-neutral-500">actor probabilities</td>
              <td>{cur.p.map((p, k) => `${ACTION_GLYPH[d.action_names[k]]} ${p.toFixed(2)}`).join("   ")}</td>
            </tr>
            <tr>
              <td className="text-neutral-500">sampled action a</td>
              <td style={{ color: ACTION_COLORS[name] }}>
                {ACTION_GLYPH[name]} {name}
              </td>
            </tr>
            <tr>
              <td className="text-neutral-500">reward r</td>
              <td>
                {cur.r > 0 ? "+" : ""}
                {cur.r} (episode so far: {sofar})
              </td>
            </tr>
          </tbody>
        </table>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {tr.episodes.map((e, k) => {
            const done = k < cur.e || (k === cur.e && cur.t === e.length - 1);
            return (
              <span key={k} className={`rounded border px-1.5 py-0.5 text-[11px] tabular-nums ${k === cur.e ? "border-neutral-800" : "border-neutral-200"} ${done ? "text-neutral-800" : "text-neutral-400"}`}>
                ep {k + 1}: {done ? `${e.length} steps, return ${e.ret}, ${e.terminated ? "terminated" : "truncated (time limit)"}` : "…"}
              </span>
            );
          })}
        </div>
      </Box>
      <Box>
        <EnvView envId={d.config.env_id} step={stepRec(cur.s, cur.a, cur.p)} trail={trail} actionNames={d.action_names} />
      </Box>
    </div>
  );
}

// ======================================================= stage 3: critic on the batch
export function CriticOutStage({ d, tr, focus, setFocus }: { d: Describe; tr: Trace; focus: number; setFocus: (i: number) => void }) {
  const b = boundariesOf(tr);
  const cri = activations(tr.weights_before.critic, tr.features[focus], "critic", d.action_names);
  return (
    <div className="space-y-3">
      <Box>
        <Explain>
          <p>
            <b>Step 2: pass every state of the batch through the critic.</b> For each of the {tr.values.length} recorded states, the critic outputs its guess V(s) of the
            reward still to come.
          </p>
          <p className="text-neutral-500">
            {tr.k === 1
              ? "The critic has never learned anything yet, so these numbers are just what random weights produce, close to 0 and meaningless."
              : `After ${tr.k - 1} update(s) the critic has started to learn. Compare its guesses with what really happened, in the next step.`}{" "}
            Click a bar to follow one state through the network.
          </p>
        </Explain>
        <div className="mt-2 text-[11px] text-neutral-500">orange line: V(s_t) for every step of the batch; gray lines: episode boundaries</div>
        <BarStrip values={tr.values.map(() => null)} boundaries={b} lines={[{ values: tr.values, color: C.critic, width: 2 }]} selected={focus} onSelect={setFocus} />
      </Box>
      <Box>
        <NetDiagram
          title={`Critic on the selected state (episode ${tr.ep_index[focus] + 1}, t = ${tr.t_index[focus]})`}
          color={C.critic}
          edges={tr.weights_before.critic}
          mode="weights"
          inputs={d.obs_names.map(shortObs)}
          outputs={cri.names}
          activations={cri}
        />
      </Box>
    </div>
  );
}

// ================================================================ stage 4: returns
export function ReturnsStage({ tr, d }: { tr: Trace; d: Describe }) {
  const [e, setE] = useState(0);
  return <ReturnsEpisode key={e} tr={tr} d={d} e={e} setE={setE} />;
}

function ReturnsEpisode({ tr, d, e, setE }: { tr: Trace; d: Describe; e: number; setE: (e: number) => void }) {
  const ep = tr.episodes[e];
  const off = boundariesOf(tr)[e];
  const T = ep.length;
  const [c, setC] = useState(T - 1); // reveal G from the end down to c
  const [want, setPlaying] = useState(true);
  const playing = want && c > 0;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setC((x) => Math.max(0, x - Math.max(1, Math.round(T / 120)))), 60);
    return () => window.clearInterval(id);
  }, [playing, T]);
  const G = tr.returns.slice(off, off + T);
  const V = tr.values.slice(off, off + T);
  const shown = G.map((g, t) => (t >= c ? g : null));
  const next = c + 1 < T ? G[c + 1] : ep.tail;
  return (
    <div className="space-y-3">
      <Box>
        <Explain>
          <p>
            <b>Step 3: what really happened, the return G.</b> For each step we add up the rewards that actually came after it, with each step further away counting a
            bit less (γ = {tr.gamma}). We compute it <b>backwards</b> from the end of the episode:
          </p>
          <Formula>G_t = r_t + γ · G_(t+1)</Formula>
          <p>
            Where it starts depends on how the episode ended:{" "}
            {ep.terminated ? (
              <>
                this episode was <b>terminated</b> ({d.terminated}). That is a real end, so <b>G after the last step = 0</b>.
              </>
            ) : (
              <>
                this episode was <b>truncated</b> ({d.truncated}). The future would have continued, so we fill it in with the critic&apos;s own guess:{" "}
                <b>G after the last step = V(s_last) = {fmt(ep.tail)}</b> (&quot;bootstrapping&quot;).
              </>
            )}
          </p>
        </Explain>
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
          episode:
          {tr.episodes.map((x, k) => (
            <button key={k} onClick={() => setE(k)} className={`rounded border px-1.5 py-0.5 ${k === e ? "border-neutral-800 bg-neutral-800 text-white" : "border-neutral-300"}`}>
              {k + 1} ({x.length} steps)
            </button>
          ))}
          <button className="ml-2 rounded border border-neutral-300 px-2 py-0.5" onClick={() => (c <= 0 ? (setC(T - 1), setPlaying(true)) : setPlaying(!playing))}>
            {playing ? "❚❚" : "▶ replay"}
          </button>
        </div>
        <Formula>
          t = {c}: G_{c} = r_{c} + {tr.gamma} × G_{c + 1} = {ep.rewards[c]} + {tr.gamma} × {fmt(next)} = <b>{fmt(G[c])}</b>
        </Formula>
        <div className="text-[11px] text-neutral-500">
          black: return G_t (filled in from the end) · orange: what the critic guessed, V(s_t)
        </div>
        <BarStrip values={G.map(() => null)} boundaries={[0]} lines={[{ values: V, color: C.critic, width: 2 }, { values: shown, color: C.ink, width: 1.75 }]} height={170} />
        <div className="mt-1 text-[11px] text-neutral-500">The gap between black and orange is how wrong the critic is. That gap drives everything that follows.</div>
      </Box>
    </div>
  );
}

// ============================================================== stage 5: advantages
export function AdvantageStage({ d, tr, focus, setFocus }: { d: Describe; tr: Trace; focus: number; setFocus: (i: number) => void }) {
  const b = boundariesOf(tr);
  const A = tr.advantages[focus];
  const a = d.action_names[tr.actions[focus]];
  return (
    <div className="space-y-3">
      <Box>
        <Explain>
          <p>
            <b>Step 4: the advantage A = G − V.</b> For each step: did things go <b style={{ color: C.pos }}>better (+)</b> or{" "}
            <b style={{ color: C.neg }}>worse (−)</b> than the critic expected? This single number is the teaching signal for the actor.
          </p>
          <p className="text-neutral-500">
            {tr.k === 1
              ? "Careful: in iteration 1 the critic is random, so A is mostly the critic's own error, not information about the actions. This is why the critic has to learn alongside the actor: if the critic is bad, the actor's teaching signal is bad."
              : d.config.env_id === "MountainCar-v0"
                ? "In MountainCar every step gives −1 whatever the action, so once the critic predicts this well, A ≈ 0 everywhere and the actor gets no signal."
                : "In CartPole a step whose episode went on longer than the critic expected gets A > 0, so that action gets reinforced."}
          </p>
        </Explain>
        <BarStrip values={tr.advantages} boundaries={b} selected={focus} onSelect={setFocus} height={150} />
        <div className="mt-2 grid grid-cols-2 gap-x-6 text-[12px] tabular-nums sm:grid-cols-4">
          <div>
            selected: episode {tr.ep_index[focus] + 1}, t = {tr.t_index[focus]}
          </div>
          <div>
            action: <span style={{ color: ACTION_COLORS[a] }}>{ACTION_GLYPH[a]} {a}</span>
          </div>
          <div>
            G = {fmt(tr.returns[focus])}, V = {fmt(tr.values[focus])}
          </div>
          <div style={{ color: signColor(A) }}>
            A = G − V = {sgn(A)}
            {fmt(Math.abs(A))} {A > 0 ? "↑ better than expected" : "↓ worse than expected"}
          </div>
        </div>
      </Box>
    </div>
  );
}

// ======================================================= stages 6 and 7: gradient steps
function Matrices({ before, grads, delta, after, net, d, outputs }: { before: NetWeights; grads: NetWeights; delta: NetWeights; after: NetWeights; net: string; d: Describe; outputs: string[] }) {
  const inputs = d.obs_names.map(shortObs);
  const hidden = before.W1.map((_, j) => `h${j + 1}`);
  const col = (label: string, w: NetWeights, which: "W1" | "W2", scale?: number) => (
    <MatrixGrid
      title={label}
      m={which === "W1" ? w.W1 : w.W2}
      scale={scale}
      rowLabel={(r) => (which === "W1" ? hidden[r] : outputs[r])}
      colLabels={which === "W1" ? inputs : hidden}
    />
  );
  const wScale1 = Math.max(maxAbs(before.W1), maxAbs(after.W1));
  const wScale2 = Math.max(maxAbs(before.W2), maxAbs(after.W2));
  return (
    <div className="space-y-3 overflow-x-auto">
      {(["W1", "W2"] as const).map((which) => (
        <div key={which}>
          <div className="mb-1 text-[11px] font-medium text-neutral-700">
            {net} {which === "W1" ? "W1: input → hidden" : "W2: hidden → output"}
          </div>
          <div className="flex flex-wrap items-start gap-3">
            {col("before", before, which, which === "W1" ? wScale1 : wScale2)}
            <span className="self-center text-neutral-400">gradient →</span>
            {col("∂L/∂w", grads, which)}
            <span className="self-center text-neutral-400">step →</span>
            {col("Δw", delta, which)}
            <span className="self-center text-neutral-400">=</span>
            {col("after", after, which, which === "W1" ? wScale1 : wScale2)}
          </div>
        </div>
      ))}
    </div>
  );
}

function optimizerText(opt: "adam" | "sgd", lr: number) {
  return opt === "sgd" ? (
    <>
      <b>SGD</b>: Δw = −lr × ∂L/∂w with lr = {lr}. Each weight moves against its gradient, in proportion to it.
    </>
  ) : (
    <>
      <b>Adam</b> (lr = {lr}): Δw = −lr × (running average of the gradient) / (running size of the gradient). Every weight moves about lr in the downhill direction,{" "}
      <i>whatever</i> the size of its gradient (in the first step exactly ±lr). Switch to SGD in the settings to see Δw = −lr × gradient instead.
    </>
  );
}

export function CriticStepStage({ d, tr, focus }: { d: Describe; tr: Trace; focus: number }) {
  const { i, bar } = useSubsteps(5);
  const b = boundariesOf(tr);
  const N = tr.values.length;
  const mode: EdgeMode = i === 0 ? "weights" : i <= 2 ? "grads" : "delta";
  const edges = i === 0 ? tr.weights_before.critic : i <= 2 ? tr.critic.grads : i === 3 ? tr.critic.delta : tr.critic.weights_after;
  const cri = activations(i >= 4 ? tr.critic.weights_after : tr.weights_before.critic, tr.features[focus], "critic", d.action_names);
  return (
    <div className="space-y-3">
      <Box>
        <div className="mb-1 text-[13px] font-semibold" style={{ color: C.critic }}>
          Step 5: one gradient step for the critic (regression)
        </div>
        {bar(["the loss", "gradient into W2", "…back into W1", "the weight change", "result"])}
        <Explain>
          {i === 0 && (
            <>
              <p>The critic should predict G. Its loss is the mean squared error over the whole batch:</p>
              <Formula>
                L_V = (1/N) Σ_t ½ (G_t − V(s_t))² = <b>{fmt(tr.critic.loss)}</b> (N = {N} steps)
              </Formula>
              <p>Each step pulls V(s_t) toward its own G_t: up where A &gt; 0, down where A &lt; 0. These are exactly the bars of the previous step.</p>
            </>
          )}
          {i === 1 && (
            <>
              <p>
                <b>Backpropagation, part 1.</b> How much does the loss change if we nudge each weight of the <b>last layer</b> W2 (hidden → V)? That is the gradient. For
                the output layer:
              </p>
              <Formula>∂L/∂W2[j] = −(1/N) Σ_t A_t · h_j(s_t)</Formula>
              <p className="text-neutral-500">A hidden unit that is active (h ≠ 0) in states where the critic was wrong gets a large gradient.</p>
            </>
          )}
          {i === 2 && (
            <>
              <p>
                <b>Backpropagation, part 2.</b> The error flows back through the tanh units into the <b>first layer</b> W1 (input → hidden), by the chain rule:
              </p>
              <Formula>∂L/∂W1[j][i] = −(1/N) Σ_t A_t · W2[j] · (1 − h_j²) · x_i</Formula>
              <p className="text-neutral-500">(1 − h²) is the slope of tanh. A saturated unit (h ≈ ±1) passes almost no gradient.</p>
            </>
          )}
          {i === 3 && <p>{optimizerText(tr.optimizer, tr.critic.lr)}</p>}
          {i === 4 && (
            <p>
              After one step, the critic&apos;s guesses on this batch moved toward the returns: loss <b>{fmt(tr.critic.loss)}</b> → <b>{fmt(tr.critic.loss_after)}</b>. One
              step only moves them a little. Over many batches, V learns.
            </p>
          )}
        </Explain>
      </Box>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Box>
          <NetDiagram
            title="Critic"
            color={C.critic}
            edges={edges}
            mode={i === 4 ? "weights" : mode}
            inputs={d.obs_names.map(shortObs)}
            outputs={cri.names}
            activations={cri}
            focusLayer={i === 1 ? "W2" : i === 2 ? "W1" : null}
          />
        </Box>
        <Box>
          {i < 3 && (
            <BarStrip values={tr.advantages} boundaries={b} lines={[]} height={190} />
          )}
          {i === 3 && (
            <Matrices before={tr.weights_before.critic} grads={tr.critic.grads} delta={tr.critic.delta} after={tr.critic.weights_after} net="Critic" d={d} outputs={["V"]} />
          )}
          {i === 4 && (
            <>
              <div className="text-[11px] text-neutral-500">black: G (target) · dashed orange: V before · orange: V after the step</div>
              <BarStrip
                values={tr.values.map(() => null)}
                boundaries={b}
                lines={[
                  { values: tr.returns, color: C.ink, width: 1.25 },
                  { values: tr.values, color: C.critic, width: 1.25, dash: [4, 3] },
                  { values: tr.critic.values_after, color: C.critic, width: 2 },
                ]}
                height={190}
              />
            </>
          )}
          {i < 3 && <div className="mt-1 text-[11px] text-neutral-500">the errors G − V the critic is trying to shrink (one bar per step)</div>}
        </Box>
      </div>
    </div>
  );
}

export function ActorStepStage({ d, tr, focus }: { d: Describe; tr: Trace; focus: number }) {
  const { i, bar } = useSubsteps(5);
  const N = tr.values.length;
  const names = d.action_names.map((a) => `${ACTION_GLYPH[a]} ${a}`);
  const mode: EdgeMode = i === 0 ? "weights" : i <= 2 ? "grads" : "delta";
  const edges = i === 0 ? tr.weights_before.actor : i <= 2 ? tr.actor.grads : i === 3 ? tr.actor.delta : tr.actor.weights_after;
  const act = activations(i >= 4 ? tr.actor.weights_after : tr.weights_before.actor, tr.features[focus], "actor", d.action_names);
  const examples = useMemo(() => {
    const idx = tr.advantages.map((a, k) => k).sort((p, q) => Math.abs(tr.advantages[q]) - Math.abs(tr.advantages[p]));
    const pos = idx.filter((k) => tr.advantages[k] > 0).slice(0, 3);
    const neg = idx.filter((k) => tr.advantages[k] < 0).slice(0, 3);
    return [...pos, ...neg];
  }, [tr]);
  const moved = useMemo(() => {
    const up = tr.advantages.map((a, k) => (a > 0 ? tr.actor.logp_after[k] - tr.actor.logp_before[k] : null)).filter((v): v is number => v != null);
    const dn = tr.advantages.map((a, k) => (a < 0 ? tr.actor.logp_after[k] - tr.actor.logp_before[k] : null)).filter((v): v is number => v != null);
    const mean = (xs: number[]) => (xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : 0);
    return { up: mean(up), dn: mean(dn), nUp: up.length, nDn: dn.length };
  }, [tr]);
  return (
    <div className="space-y-3">
      <Box>
        <div className="mb-1 text-[13px] font-semibold" style={{ color: C.actor }}>
          Step 6: one gradient step for the actor (classification, weighted by A)
        </div>
        {bar(["the loss", "gradient into W2", "…back into W1", "the weight change", "result"])}
        <Explain>
          {i === 0 && (
            <>
              <p>
                The actor has no labels, so we use <b>the action it actually took</b> as if it were the label, and weight each step by its advantage:
              </p>
              <Formula>
                L_π = −(1/N) Σ_t A_t · log π(a_t | s_t) = <b>{fmt(tr.actor.loss)}</b> (N = {N})
              </Formula>
              <p>
                Making this smaller means: where <b style={{ color: C.pos }}>A &gt; 0</b>, raise the probability of the action that was taken; where{" "}
                <b style={{ color: C.neg }}>A &lt; 0</b>, lower it. The bigger |A|, the harder the push. With all A = 1 this would be plain cross-entropy
                classification.
              </p>
            </>
          )}
          {i === 1 && (
            <>
              <p>
                <b>Backpropagation, part 1.</b> For the action scores (logits) z_k the gradient is
              </p>
              <Formula>∂L/∂z_k = −(1/N) Σ_t A_t · (1[k = a_t] − π_k(s_t))</Formula>
              <p className="text-neutral-500">
                The taken action&apos;s score goes up (A &gt; 0) or down (A &lt; 0), and the others move the opposite way. Then ∂L/∂W2[k][j] = Σ_t (∂L/∂z_k) · h_j(s_t).
              </p>
            </>
          )}
          {i === 2 && (
            <p>
              <b>Backpropagation, part 2.</b> As for the critic, through the tanh slope (1 − h²) into W1. The actor and critic are <b>separate networks</b>: these
              gradients never touch the critic&apos;s weights, and the other way round.
            </p>
          )}
          {i === 3 && <p>{optimizerText(tr.optimizer, tr.actor.lr)}</p>}
          {i === 4 && (
            <>
              <p>
                Result on this batch: for the {moved.nUp} steps with A &gt; 0 the taken action&apos;s log-probability changed by{" "}
                <b style={{ color: signColor(moved.up) }}>{fmt(moved.up, 4)}</b> on average
                {moved.nDn > 0 ? (
                  <>
                    ; for the {moved.nDn} steps with A &lt; 0 by <b style={{ color: signColor(moved.dn) }}>{fmt(moved.dn, 4)}</b>
                  </>
                ) : null}
                . The policy moved a little toward what worked better than expected.
              </p>
              {moved.nUp === 0 && (
                <p className="text-neutral-500">
                  Every A is negative here. The critic still predicts about 0 while every step cost −1, so <i>every</i> action taken gets pushed down,
                  whichever it was. That is noise, not information about good or bad actions: in MountainCar the reward is the same for every action until the
                  flag is reached.
                </p>
              )}
              {moved.nDn === 0 && (
                <p className="text-neutral-500">
                  Every A is positive here. The critic still predicts about 0 while every step earned reward, so <i>every</i> action taken gets reinforced. What
                  matters is <i>how much</i>: actions from long episodes (large G) get the biggest push.
                </p>
              )}
              <p className="text-neutral-500">
                One step changes weights that <b>all</b> states share. So for a single step the probability can move the &quot;wrong&quot; way, pulled by the pushes from
                the other states (see the table). Only the batch as a whole is guaranteed to move in the right direction.
              </p>
            </>
          )}
        </Explain>
      </Box>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Box>
          <NetDiagram
            title="Actor"
            color={C.actor}
            edges={edges}
            mode={i === 4 ? "weights" : mode}
            inputs={d.obs_names.map(shortObs)}
            outputs={names}
            activations={act}
            focusLayer={i === 1 ? "W2" : i === 2 ? "W1" : null}
          />
        </Box>
        <Box>
          {i === 3 ? (
            <Matrices before={tr.weights_before.actor} grads={tr.actor.grads} delta={tr.actor.delta} after={tr.actor.weights_after} net="Actor" d={d} outputs={names} />
          ) : (
            <>
              <div className="mb-1 text-[11px] text-neutral-500">
                the strongest pushes in this batch: the 3 most positive and 3 most negative A {i === 4 ? "(probabilities before → after)" : ""}
              </div>
              <table className="w-full text-[12px] tabular-nums">
                <thead className="text-[11px] text-neutral-500">
                  <tr>
                    <th className="text-left font-normal">ep, t</th>
                    <th className="text-left font-normal">action</th>
                    <th className="text-right font-normal">A</th>
                    <th className="text-right font-normal">π(a|s)</th>
                  </tr>
                </thead>
                <tbody>
                  {examples.map((k) => {
                    const a = d.action_names[tr.actions[k]];
                    const pb = Math.exp(tr.actor.logp_before[k]);
                    const pa = Math.exp(tr.actor.logp_after[k]);
                    return (
                      <tr key={k} className="border-t border-neutral-100">
                        <td>
                          {tr.ep_index[k] + 1}, {tr.t_index[k]}
                        </td>
                        <td style={{ color: ACTION_COLORS[a] }}>
                          {ACTION_GLYPH[a]} {a}
                        </td>
                        <td className="text-right" style={{ color: signColor(tr.advantages[k]) }}>
                          {sgn(tr.advantages[k])}
                          {fmt(Math.abs(tr.advantages[k]))}
                        </td>
                        <td className="text-right">
                          {pb.toFixed(4)}
                          {i === 4 && (
                            <>
                              {" → "}
                              <span style={{ color: signColor(pa - pb) }}>
                                {pa.toFixed(4)} {pa > pb ? "↑" : "↓"}
                              </span>
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}
        </Box>
      </div>
    </div>
  );
}

// ======================================================================= stage 8: next
export function NextStage({ d, tr }: { d: Describe; tr: Trace }) {
  return (
    <div className="space-y-3">
      <Box>
        <Explain>
          <p>
            <b>Step 7: throw the batch away and repeat.</b> Both networks changed a little. The batch was played by the <i>old</i> policy, so it no longer describes the
            new one (on-policy). The next iteration plays a <b>fresh batch with the updated actor</b>, judges it with the updated critic, and takes the next two
            gradient steps.
          </p>
          <p className="text-neutral-500">
            Iteration {tr.k}: mean return of this batch {fmt(tr.summary.mean_return, 1)} ({tr.episodes.length} episodes, {tr.summary.steps} steps). Below: what each
            network now computes for every state (the maps cover {d.phase.names.join(" × ")}).
          </p>
        </Explain>
      </Box>
      <Box>
        <div className="flex flex-wrap gap-8">
          <MapPair title="Critic V(s)" before={tr.maps.before} after={tr.maps.after} d={d} kind="value" />
          <MapPair title={`Actor P(${d.action_names.at(-1)} | s)`} before={tr.maps.before} after={tr.maps.after} d={d} kind="p_last" />
        </div>
      </Box>
    </div>
  );
}
