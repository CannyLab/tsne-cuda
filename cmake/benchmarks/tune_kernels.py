#!/usr/bin/env python3
"""Kernel launch-parameter tuning harness for tsne-cuda.

The FFT-interpolation kernels expose their block sizes through GpuOptions, and
those can be overridden at runtime via environment variables (see
src/include/options.h):

    TSNE_FFT_THREADS     nbodyfft interpolation/copy kernels   (default 128)
    TSNE_ATTR_THREADS    attractive-forces (Pij x Qij) kernel  (default 1024)
    TSNE_REP_THREADS     repulsive-forces / charges kernel     (default 1024)
    TSNE_INTEG_THREADS   integration (apply-forces) kernel     (default 1024)
    TSNE_INTEG_FACTOR    integration blocks-per-SM factor      (default 1)

This script sweeps each parameter independently (holding the others at their
default), running each configuration in a fresh subprocess so GPU state never
leaks between runs, and parses the per-phase timers that fit_tsne prints under
verbose=1. It reports, per kernel, the block size that minimises that kernel's
phase time on the current device.

Usage:
    PYTHONPATH=<build>/python python tune_kernels.py [--data mnist.npz]
        [--n 60000] [--iters 500] [--repeats 3]

With no --data it uses reproducible gaussian-blob synthetic data.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

# phase timer name (in fit_tsne output) that each env var most directly drives
PHASE_FOR = {
    "TSNE_FFT_THREADS": "_time_nbodyfft",
    "TSNE_ATTR_THREADS": "_time_attr",
    "TSNE_REP_THREADS": "_time_norm",
    "TSNE_INTEG_THREADS": "_time_apply_forces",
    "TSNE_INTEG_FACTOR": "_time_apply_forces",
}
GRID = {
    "TSNE_FFT_THREADS": [32, 64, 128, 256, 512, 1024],
    "TSNE_ATTR_THREADS": [64, 128, 256, 512, 1024],
    "TSNE_REP_THREADS": [128, 256, 512, 1024],
    "TSNE_INTEG_THREADS": [256, 512, 1024],
    # threads-per-block is hardware-capped at 1024; the integration factor is a
    # grid multiplier (grid = sm_count * factor) and can scale much higher, so we
    # sweep it well past the SM count to find where extra blocks stop helping.
    "TSNE_INTEG_FACTOR": [1, 2, 4, 8, 16, 32, 64, 128],
}
TIMER_RE = re.compile(r"^(_time_\w+|total_time):\s*([0-9.]+)s")


def load_data(path, n):
    import numpy as np
    if path and os.path.exists(path):
        d = np.load(path)
        key = "x_train" if "x_train" in d else list(d.keys())[0]
        X = d[key].reshape(d[key].shape[0], -1).astype(np.float32)
        return X[:n]
    rng = np.random.default_rng(0)
    # 10 well-separated blobs in 50-D, like a clustered real dataset
    per = n // 10
    parts = [rng.standard_normal((per, 50)).astype(np.float32) + 12.0 * i for i in range(10)]
    return np.vstack(parts)[:n]


def run_worker(args):
    """Single fit; prints fit_tsne's verbose timers + a BENCH_WALL line."""
    import numpy as np
    from tsnecuda import TSNE
    X = load_data(args.data, args.n)
    t0 = time.time()
    TSNE(n_iter=args.iters, verbose=1, num_neighbors=64).fit_transform(X)
    print("BENCH_WALL %.4f" % (time.time() - t0))


def parse_timers(text):
    out = {}
    for line in text.splitlines():
        m = TIMER_RE.match(line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
        if line.startswith("BENCH_WALL"):
            out["wall"] = float(line.split()[1])
    return out


def measure(env_overrides, args):
    """Run repeats+1 times; drop the first (warmup) and return per-run timer dicts."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    samples = []
    for i in range(args.repeats + 1):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                            "--data", args.data or "", "--n", str(args.n),
                            "--iters", str(args.iters)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
            raise RuntimeError("worker failed for %s" % env_overrides)
        if i == 0:
            continue  # warmup
        samples.append(parse_timers(r.stdout))
    return samples


def agg(samples, key):
    xs = [s.get(key, float("nan")) for s in samples]
    xs = [x for x in xs if x == x]  # drop NaN
    mean = statistics.mean(xs) if xs else float("nan")
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--data", default="")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list of env vars to sweep")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args)

    params = [p for p in GRID if not args.only or p in args.only.split(",")]
    print("device sweep: n=%d iters=%d repeats=%d data=%s\n"
          % (args.n, args.iters, args.repeats, args.data or "synthetic"))
    recommended = {}
    for p in params:
        phase = PHASE_FOR[p]
        print("== %-18s (optimising %s; mean+-std over %d runs) =="
              % (p, phase, args.repeats))
        best_val, best_mean = None, float("inf")
        for v in GRID[p]:
            samples = measure({p: v}, args)
            m_phase, s_phase = agg(samples, phase)
            m_wall, s_wall = agg(samples, "wall")
            print("   %-6s  %s=%7.3f+-%.3fs   wall=%7.3f+-%.3fs"
                  % (v, phase, m_phase, s_phase, m_wall, s_wall))
            if m_phase < best_mean:
                best_mean, best_val = m_phase, v
        recommended[p] = best_val
        print("   -> best %s = %s (%s mean=%.3fs)\n" % (p, best_val, phase, best_mean))
    print("RECOMMENDED:", json.dumps(recommended))


if __name__ == "__main__":
    main()
