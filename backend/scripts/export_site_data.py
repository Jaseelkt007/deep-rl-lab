"""Copy the latest terminal runs into the web app as static JSON, for the hosted (read-only) site.

The local web app asks the FastAPI server for its data. A host such as Vercel runs only the web
app, so this script asks the same server functions once and writes their answers to
frontend/public/data/<env>/..., one file per API call:

    /api/cli/algos?env=E                 -> data/E/algos.json
    /api/cli/run/A?env=E                 -> data/E/run/A.json
    /api/cli/run/A/episodes?env=E        -> data/E/run/A/episodes.json
    /api/cli/run/A/episode/N?env=E       -> data/E/run/A/episode/N.json

Numbers are rounded to 5 significant digits to keep the files small. GIFs are not copied (the
page draws every episode itself). Run it after training, then commit frontend/public/data:

    uv run python scripts/export_site_data.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException  # noqa: E402

import server  # noqa: E402
from rl.logger import clean  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "data"


def write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(obj, 5), separators=(",", ":"))
    path.write_text(text)
    return len(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: frontend/public/data)")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)  # a fresh copy: algorithms or episodes removed locally disappear here too
    total = 0
    for env in server.CLI_ENVS:
        algos = server.cli_algos(env) if server._cli_root(env).exists() else []
        total += write(args.out / env / "algos.json", algos)
        for a in algos:
            algo = a["algo"]
            try:
                run = server.cli_run(algo, env)
            except HTTPException:
                continue
            total += write(args.out / env / "run" / f"{algo}.json", run)
            total += write(args.out / env / "run" / algo / "episodes.json", run["episodes"])
            for ep in run["episodes"]:
                total += write(args.out / env / "run" / algo / "episode" / f"{ep['name']}.json",
                               server.cli_episode(algo, ep["name"], env))
        print(f"{env:24s} {len(algos)} algorithm(s)")
    print(f"wrote {total / 1e6:.1f} MB to {args.out}")


if __name__ == "__main__":
    main()
