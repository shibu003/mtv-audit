"""D-α headless run driver (addendum-A §B, replaces the manual 42-session protocol).

One `claude -p` per cell. Each invocation is a fresh session (clean context), so
rep independence is automatic. arm is selected by --mcp-config (not add/remove),
so V and S can never be mixed:
  arm V -> --strict-mcp-config --mcp-config <empty.json>   (no svm-probe at all)
  arm S -> --strict-mcp-config --mcp-config <svm-probe.json>

repo isolation: a throwaway git worktree of ~/Shibubu per cell, with node_modules
symlinked in from the main checkout so tsc/test can still run. The real repo is
never mutated; the worktree is removed after each cell.

Cost guards: --mock runs the whole pipeline at $0 (no claude call), --only runs a
single cell. Nothing here spends until you drop --mock and pass a real cell.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIBUBU = Path(os.path.expanduser("~/Shibubu"))
SEEDS = ROOT / "reports" / "seeds"
RUNS = ROOT / "runs"
SVM_CFG = ROOT / "reports" / "svm-probe.mcp.json"
EMPTY_CFG = ROOT / "reports" / "empty.mcp.json"
MODEL = "claude-opus-4-8"

TASKS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
VARIANT = {"T1": "2line", "T2": "0line", "T3": "2line", "T4": "0line",
           "T5": "2line", "T6": "0line", "T7": "2line"}
HANDLE = {"T1": "9UCuRTj9", "T2": "vVzZodw8", "T3": "acy27aU5", "T4": "QyeVsVSQ",
          "T5": "utHfv5Rj", "T6": "8H7XkXa6", "T7": "PnugfqUU"}


def cells(reps: int):
    for t in TASKS:
        for arm in ("V", "S"):
            for rep in range(1, reps + 1):
                yield t, arm, rep


def transcript_path(t: str, arm: str, rep: int) -> Path:
    if arm == "S":
        return RUNS / f"{t}_S_{VARIANT[t]}_{rep}.jsonl"
    return RUNS / f"{t}_V_{rep}.jsonl"


def make_worktree(tag: str) -> Path:
    wt = ROOT / ".worktrees" / tag
    if wt.exists():
        subprocess.run(["git", "-C", str(SHIBUBU), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
    wt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(SHIBUBU), "worktree", "add", "--detach", str(wt), "HEAD"],
                   check=True, capture_output=True)
    # share node_modules (gitignored, so worktree lacks them) via symlink
    for nm in SHIBUBU.rglob("node_modules"):
        if any(p.name == "node_modules" for p in nm.relative_to(SHIBUBU).parents):
            continue  # skip nested node_modules/*/node_modules
        rel = nm.relative_to(SHIBUBU)
        dst = wt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(nm)
    return wt


def drop_worktree(wt: Path):
    subprocess.run(["git", "-C", str(SHIBUBU), "worktree", "remove", "--force", str(wt)],
                   capture_output=True)
    shutil.rmtree(wt, ignore_errors=True)


def run_cell(t: str, arm: str, rep: int, mock: bool) -> dict:
    seed = (SEEDS / f"seed_{t}_{arm}.txt").read_text()
    cfg = SVM_CFG if arm == "S" else EMPTY_CFG
    out = transcript_path(t, arm, rep)
    RUNS.mkdir(exist_ok=True)
    tag = f"{t}_{arm}_{rep}"
    wt = make_worktree(tag)
    try:
        cmd = ["claude", "-p", seed,
               "--model", MODEL,
               "--strict-mcp-config", "--mcp-config", str(cfg),
               "--permission-mode", "bypassPermissions",
               "--output-format", "stream-json", "--verbose"]
        if mock:
            # $0 pipeline check: prove worktree+symlink+seed+config+capture work,
            # without spending. Emit a stand-in transcript line.
            payload = {"mock": True, "task": t, "arm": arm, "rep": rep,
                       "cwd": str(wt), "config": str(cfg),
                       "seed_chars": len(seed),
                       "node_modules_linked": (wt / "node_modules").is_symlink()}
            out.write_text(json.dumps(payload) + "\n")
            rc = 0
        else:
            with out.open("w") as fh:
                rc = subprocess.run(cmd, cwd=str(wt), stdout=fh).returncode
        return {"task": t, "arm": arm, "variant": VARIANT[t] if arm == "S" else "-",
                "rep": rep, "transcript": str(out.relative_to(ROOT)), "rc": rc,
                "success": None}  # success scored later by mtv_audit.probe
    finally:
        drop_worktree(wt)


def parse_only(spec: str):
    t, arm, rep = spec.split(":")
    return [(t, arm, int(rep))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dalpha-driver")
    ap.add_argument("--only", help="single cell, e.g. T2:S:1")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mock", action="store_true", help="$0 dry pipeline (no claude call)")
    ap.add_argument("--results", default=str(ROOT / "reports" / "dalpha_results.json"))
    a = ap.parse_args(argv)

    if not SHIBUBU.exists():
        print(f"target repo not found: {SHIBUBU}", file=sys.stderr); return 2

    todo = parse_only(a.only) if a.only else list(cells(a.reps))
    print(f"{'MOCK ' if a.mock else ''}running {len(todo)} cell(s)")
    runs = []
    for t, arm, rep in todo:
        r = run_cell(t, arm, rep, a.mock)
        runs.append(r)
        print(f"  {t} {arm} rep{rep} -> {r['transcript']} (rc={r['rc']})")

    res = {"tasks": {t: [HANDLE[t]] for t in TASKS}, "runs": runs}
    Path(a.results).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"wrote {a.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
