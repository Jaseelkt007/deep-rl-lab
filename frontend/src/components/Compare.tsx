"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { C } from "@/lib/colors";
import { firstCrossing, fmt, movingAverage } from "@/lib/math";
import type { EpisodeRec, RunConfig, RunDetail, RunMeta } from "@/lib/types";
import Panel, { ChartTitle } from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

type LiteEpisode = EpisodeRec & { critic_sat?: number; actor_sat?: number };

/** One setup = all runs with the same config except the seed. */
interface Setup {
  key: string;
  env: string;
  label: string;
  groups: string[];
  runs: RunMeta[];
}

const MA = 20;
const METRICS: { title: string; help: string; get: (e: LiteEpisode) => number | null | undefined; range?: [number | null, number | null]; log?: boolean }[] = [
  { title: "Return (original reward)", help: "20-episode moving average per seed; line = mean over seeds, band = min to max.", get: (e) => e.return },
  { title: "Success rate", help: "Share of successful episodes (20-episode average).", get: (e) => (e.success ? 1 : 0), range: [0, 1] },
  { title: "Policy entropy", help: "Mean entropy of π per episode.", get: (e) => e.mean_entropy },
  { title: "Critic explained variance", help: "EV of V against the return of the training reward (clipped at −1).", get: (e) => (e.ev == null ? null : Math.max(-1, e.ev)), range: [-1, 1] },
  { title: "Critic last-layer saturation", help: "Share of the critic's last hidden layer beyond |h| > 0.95.", get: (e) => e.critic_sat, range: [0, 1] },
  { title: "Critic share of the gradient", help: "|∇critic|² / (|∇actor|² + |∇critic|²).", get: (e) => e.mean_critic_share, range: [0, 1] },
  { title: "|δ| per step", help: "Mean |δ| per episode (log).", get: (e) => e.mean_abs_delta, log: true },
  { title: "Actor ‖Δw‖ per update", help: "How far the actor's weights move per step (log).", get: (e) => e.mean_dw_actor, log: true },
];

function setupColor(i: number) {
  return { color: C.groups[i % C.groups.length], dash: i >= C.groups.length ? [6, 3] : undefined };
}

/** Mean / min / max over seeds of each seed's moving average.
 *  The mean is split in two: `full` while EVERY seed still has data, `partial` (drawn dotted)
 *  once some seed has ended (diverged or shorter run), so the line never silently switches to
 *  averaging only the survivors. */
function aggregate(runs: LiteEpisode[][], get: (e: LiteEpisode) => number | null | undefined) {
  const curves = runs.map((eps) => movingAverage(eps.map((e) => get(e) ?? null), MA));
  const n = Math.max(0, ...curves.map((c) => c.length));
  const full: (number | null)[] = [];
  const partial: (number | null)[] = [];
  const lo: (number | null)[] = [];
  const hi: (number | null)[] = [];
  const count: number[] = [];
  for (let i = 0; i < n; i++) {
    const alive = curves.filter((c) => i < c.length).length;
    const vs = curves.map((c) => c[i]).filter((v): v is number => v != null && Number.isFinite(v));
    const mean = vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
    const all = alive === curves.length;
    full.push(all ? mean : null);
    // overlap one point so the dotted part joins the solid one
    partial.push(!all || (i + 1 < n && curves.some((c) => i + 1 >= c.length)) ? mean : null);
    lo.push(vs.length ? Math.min(...vs) : null);
    hi.push(vs.length ? Math.max(...vs) : null);
    count.push(alive);
  }
  return { full, partial, lo, hi, n, count };
}

function CompareChart({
  setups,
  data,
  metric,
  markers,
}: {
  setups: Setup[];
  data: Record<string, LiteEpisode[][]>;
  metric: (typeof METRICS)[number];
  markers: Marker[];
}) {
  const { x, ys, specs, bands } = useMemo(() => {
    const ys: (number | null)[][] = [];
    const specs: SeriesSpec[] = [];
    const bands: { hi: number; lo: number; fill: string }[] = [];
    let n = 0;
    const aggs = setups.map((s) => aggregate(data[s.key] ?? [], metric.get));
    aggs.forEach((a) => (n = Math.max(n, a.n)));
    setups.forEach((s, i) => {
      const a = aggs[i];
      const pad = (arr: (number | null)[]) => [...arr, ...new Array(n - arr.length).fill(null)];
      const { color, dash } = setupColor(i);
      const base = ys.length;
      const nSeeds = (data[s.key] ?? []).length;
      const fewest = a.count.length ? Math.min(...a.count) : nSeeds;
      ys.push(pad(a.hi), pad(a.lo), pad(a.full), pad(a.partial));
      specs.push({ label: `${s.label} max`, color, hidden: true }, { label: `${s.label} min`, color, hidden: true });
      specs.push({ label: `${s.label} (mean of ${nSeeds} seeds)`, color, width: 2, dash });
      specs.push({ label: `${s.label} (fewer seeds, down to ${fewest})`, color, width: 1.5, dash: [1.5, 3], noLegend: fewest === nSeeds });
      bands.push({ hi: base, lo: base + 1, fill: color + "22" });
    });
    return { x: Array.from({ length: n }, (_, i) => i + 1), ys, specs, bands };
  }, [setups, data, metric]);
  return (
    <div className="min-w-0">
      <ChartTitle help={metric.help}>{metric.title}</ChartTitle>
      <UChart x={x} ys={ys} series={specs} bands={bands} yLog={metric.log} yRange={metric.range} markers={markers} sync="compare" height={170} />
    </div>
  );
}

function switches(cfg: RunConfig | undefined) {
  if (!cfg) return "";
  return `actor ${cfg.actor_lr} · critic ${cfg.critic_lr} · γ ${cfg.gamma} · ${cfg.num_episodes} ep`;
}

export default function Compare() {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [env, setEnv] = useState<string>("all");
  const [selected, setSelected] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});

  useEffect(() => {
    api.runs().then(setRuns).catch(() => setRuns([]));
  }, []);

  const setups = useMemo(() => {
    const by = new Map<string, Setup>();
    for (const r of runs) {
      const key = `${r.env_id}:${r.config_key}`;
      const s = by.get(key) ?? { key, env: r.env_id, label: r.label, groups: [], runs: [] };
      s.runs.push(r);
      if (r.group && !s.groups.includes(r.group)) s.groups.push(r.group);
      by.set(key, s);
    }
    return [...by.values()];
  }, [runs]);

  const visible = setups.filter((s) => env === "all" || s.env === env);
  const chosen = selected.map((k) => setups.find((s) => s.key === k)).filter((s): s is Setup => !!s);

  // load the selected setups' runs (lite: scalar curves only)
  useEffect(() => {
    const need = chosen.flatMap((s) => s.runs.map((r) => r.run_id)).filter((id) => !details[id]);
    if (!need.length) return;
    Promise.all(need.map((id) => api.run(id, true).then((d) => [id, d] as const))).then((pairs) =>
      setDetails((old) => ({ ...old, ...Object.fromEntries(pairs) })),
    );
  }, [chosen, details]);

  const data = useMemo(() => {
    const out: Record<string, LiteEpisode[][]> = {};
    for (const s of chosen) out[s.key] = s.runs.map((r) => details[r.run_id]?.episodes as LiteEpisode[] | undefined).filter((e): e is LiteEpisode[] => !!e && e.length > 0);
    return out;
  }, [chosen, details]);

  // × where a run diverged (values became inf/NaN): its curve ends there
  const divergedMarkers: Marker[] = chosen.flatMap((s, i) =>
    s.runs.flatMap((r) => {
      const d = details[r.run_id]?.diverged;
      return d ? [{ x: d.ep, label: `× s${r.seed} diverged`, color: setupColor(i).color }] : [];
    }),
  );

  const toggle = (k: string) => setSelected((sel) => (sel.includes(k) ? sel.filter((x) => x !== k) : [...sel, k]));

  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-neutral-200 bg-white/90 px-4 py-2 backdrop-blur">
        <h1 className="text-[14px] font-semibold text-neutral-900">RL playground</h1>
        <nav className="flex gap-3 text-[12px]">
          <Link href="/learn" className="text-neutral-500 hover:text-neutral-900">
            Learn
          </Link>
          <Link href="/terminal" className="text-neutral-500 hover:text-neutral-900">
            Terminal run
          </Link>
          <Link href="/" className="text-neutral-500 hover:text-neutral-900">
            Live
          </Link>
          <Link href="/compare" className="font-medium text-neutral-900">
            Compare
          </Link>
        </nav>
      </header>
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[320px_1fr]">
        <aside className="space-y-3">
          <Panel title="Setups" help="Runs with the same config except the seed form one setup. Tick setups to overlay them: line = mean over seeds, band = min to max. Ablation = change one switch at a time and compare with the control on the same seeds.">
            <div className="mb-2 flex gap-1 text-[11px]">
              {["all", "MountainCar-v0", "CartPole-v1"].map((e) => (
                <button key={e} onClick={() => setEnv(e)} className={`rounded px-1.5 py-0.5 ${env === e ? "bg-neutral-200 text-neutral-900" : "text-neutral-500 hover:bg-neutral-100"}`}>
                  {e === "all" ? "all" : e.split("-")[0]}
                </button>
              ))}
            </div>
            <ul className="max-h-[70vh] space-y-1 overflow-y-auto pr-1">
              {visible.map((s) => {
                const i = selected.indexOf(s.key);
                const { color } = setupColor(Math.max(0, i));
                return (
                  <li key={s.key}>
                    <label className="flex cursor-pointer items-start gap-2 rounded px-1 py-1 text-[11px] hover:bg-neutral-50">
                      <input type="checkbox" checked={i >= 0} onChange={() => toggle(s.key)} className="mt-0.5" style={{ accentColor: i >= 0 ? color : undefined }} />
                      <span className="min-w-0">
                        <span className="block truncate font-medium text-neutral-800">
                          {s.env.split("-")[0]} · {s.label}
                        </span>
                        <span className="block text-neutral-500">
                          {s.runs.length} seed{s.runs.length > 1 ? "s" : ""}: {s.runs.map((r) => r.seed).sort((a, b) => a - b).join(", ")}
                        </span>
                        {s.groups.map((g) => (
                          <span key={g} className="block truncate text-neutral-400">
                            {g}
                          </span>
                        ))}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </Panel>
        </aside>
        <main className="min-w-0 space-y-3">
          {!chosen.length ? (
            <div className="rounded-lg border border-dashed border-neutral-300 p-10 text-center text-[13px] text-neutral-500">Tick one or more setups on the left.</div>
          ) : (
            <>
              <Panel title="Setups being compared" help="The switches that differ from the plain algorithm, plus the hyperparameters.">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[11px] text-neutral-500">
                    <tr>
                      <th className="py-1 pr-2 font-normal" />
                      <th className="py-1 pr-2 font-normal">switches</th>
                      <th className="py-1 pr-2 font-normal">hyperparameters</th>
                      <th className="py-1 pr-2 font-normal">seeds loaded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chosen.map((s, i) => {
                      const { color, dash } = setupColor(i);
                      const cfg = details[s.runs[0].run_id]?.meta.config;
                      return (
                        <tr key={s.key} className="border-t border-neutral-100">
                          <td className="py-1 pr-2">
                            <svg width="22" height="6">
                              <line x1="0" y1="3" x2="22" y2="3" stroke={color} strokeWidth="2.5" strokeDasharray={dash?.join(" ")} />
                            </svg>
                          </td>
                          <td className="py-1 pr-2 text-neutral-800">
                            {s.env.split("-")[0]} · {s.label}
                          </td>
                          <td className="py-1 pr-2 tabular-nums text-neutral-600">{switches(cfg)}</td>
                          <td className="py-1 pr-2 tabular-nums text-neutral-600">
                            {(data[s.key] ?? []).length}/{s.runs.length}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Panel>
              <Panel title="Curves" help="Per seed: 20-episode moving average. Line: mean over seeds. Band: min to max over seeds. Few seeds = wide uncertainty: trust only clear, consistent separations.">
                <div className="grid grid-cols-1 gap-x-4 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
                  {METRICS.map((m) => (
                    <CompareChart key={m.title} setups={chosen} data={data} metric={m} markers={divergedMarkers} />
                  ))}
                </div>
              </Panel>
              <ChainTable setups={chosen} details={details} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

/** Per run: when does the critic saturate, when does entropy collapse, and the gap between. */
function ChainTable({ setups, details }: { setups: Setup[]; details: Record<string, RunDetail> }) {
  const rows = setups.flatMap((s, i) =>
    s.runs
      .map((r) => {
        const eps = details[r.run_id]?.episodes as LiteEpisode[] | undefined;
        if (!eps?.length) return null;
        const sat = firstCrossing(eps, (e) => e.critic_sat, (v) => v > 0.9);
        const ent = firstCrossing(eps, (e) => e.mean_entropy, (v) => v < 0.1);
        const ma100 = movingAverage(eps.map((e) => e.return), 100);
        const best = Math.max(...ma100.slice(99).map((v) => v ?? -Infinity));
        return { i, s, r, sat, ent, gap: sat != null && ent != null ? ent - sat : null, best, final: ma100.at(-1) ?? null };
      })
      .filter((x) => x != null),
  );
  if (!rows.length) return null;
  return (
    <Panel
      title="Order of events per run"
      help="Episode where the critic's last hidden layer passes 90% saturation, episode where policy entropy drops below 0.1 (both 10-episode averages), and the gap. A positive gap in every run means the critic saturates first and the policy collapses after: the order the diagnosis predicts. On MountainCar, read saturation together with the V* = −1/(1−γ) note: a flat critic there is correct."
    >
      <table className="w-full text-left text-[12px] tabular-nums">
        <thead className="text-[11px] text-neutral-500">
          <tr>
            <th className="py-1 pr-3 font-normal">setup</th>
            <th className="py-1 pr-3 font-normal">seed</th>
            <th className="py-1 pr-3 font-normal">critic sat &gt; 90%</th>
            <th className="py-1 pr-3 font-normal">entropy &lt; 0.1</th>
            <th className="py-1 pr-3 font-normal">gap (episodes)</th>
            <th className="py-1 pr-3 font-normal">best MA100</th>
            <th className="py-1 pr-3 font-normal">final MA100</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ i, s, r, sat, ent, gap, best, final }) => (
            <tr key={r.run_id} className="border-t border-neutral-100">
              <td className="py-0.5 pr-3">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-sm" style={{ background: setupColor(i).color }} />
                {s.label}
              </td>
              <td className="py-0.5 pr-3">{r.seed}</td>
              <td className="py-0.5 pr-3">{sat ?? "–"}</td>
              <td className="py-0.5 pr-3">{ent ?? "–"}</td>
              <td className="py-0.5 pr-3">{gap == null ? "–" : gap > 0 ? `+${gap}` : gap}</td>
              <td className="py-0.5 pr-3">{Number.isFinite(best) ? fmt(best, 1) : "–"}</td>
              <td className="py-0.5 pr-3">{fmt(final, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
