"""Numerically verify Proposition 1 (slope is bounded by prior miscalibration).

Claim: for a monotone non-decreasing truth-aligned trajectory
(a_0, ..., a_L) in [0,1] with a_0 fixed and a_L <= 1, the OLS slope
beta is at most K_L * (1 - a_0), where
  K_L = 6 * k * (L - k + 1) / [L * (L+1) * (L+2)],   k = ceil((L+1)/2).

We check:
  (a) the closed-form K_L matches a numerical grid maximum over
      monotone trajectories with fixed a_0, for several values of
      L and a_0,
  (b) the bound is tight (attained at the specific argmax in the
      proof), and
  (c) the reviewer's purported counterexamples (a_0, a_0, 1, 1) and
      (a_0, 0, 1, 1) for L=3 do not violate the bound when we apply
      the right interpretation (the latter violates monotonicity and
      is checked against the looser unconstrained-trajectory bound).
"""
from __future__ import annotations
import math
import itertools

import numpy as np


def ols_slope(a):
    """OLS slope of a_ell on ell, where ell = 0..L."""
    a = np.asarray(a, dtype=float)
    L = len(a) - 1
    ell = np.arange(L + 1)
    return float(np.polyfit(ell, a, 1)[0])


def K_L_closed_form(L: int) -> float:
    k = math.ceil((L + 1) / 2)
    return 6 * k * (L - k + 1) / (L * (L + 1) * (L + 2))


def argmax_monotone(L: int, a0: float):
    """Closed-form argmax under monotone trajectory: a_ell = a0 for ell < k,
    a_ell = 1 for ell >= k, with k = ceil((L+1)/2)."""
    k = math.ceil((L + 1) / 2)
    return [a0] * k + [1.0] * (L + 1 - k)


def numerical_max_monotone(L: int, a0: float, grid: int = 60) -> float:
    """Exhaustive search over monotone non-decreasing trajectories on a grid.

    Each interior a_ell ranges over a0 .. 1 on `grid` discrete values, with
    a_{ell-1} <= a_ell.
    """
    best = -float("inf")
    levels = np.linspace(a0, 1.0, grid)
    # Enumerate non-decreasing sequences of length L+1 starting at a0.
    def _enum(prefix, remaining):
        nonlocal best
        if remaining == 0:
            s = ols_slope(prefix)
            if s > best:
                best = s
            return
        start_idx = np.searchsorted(levels, prefix[-1])
        for v in levels[start_idx:]:
            _enum(prefix + [float(v)], remaining - 1)
    _enum([a0], L)
    return best


def numerical_max_unconstrained(L: int, a0: float, grid: int = 60) -> float:
    """Exhaustive grid search over (a_1, ..., a_L) in [0,1]^L with a_0 fixed.
    Slow but exact at grid resolution."""
    if L > 4:  # 60^4 = 12.96M, large but tractable
        raise ValueError("grid too large")
    best = -float("inf")
    levels = np.linspace(0.0, 1.0, grid)
    iters = itertools.product(*[levels for _ in range(L)])
    for tail in iters:
        s = ols_slope([a0] + list(tail))
        if s > best:
            best = s
    return best


def main():
    print("Verification of Proposition 1: beta <= K_L * (1 - a_0) under monotone trajectories.")
    print()

    test_grid = [(L, a0) for L in (1, 2, 3, 4) for a0 in (0.0, 0.2, 0.5, 0.8)]
    print(f"{'L':>2} {'a_0':>5} {'K_L*(1-a0)':>12} {'closed-form':>12} {'grid max':>11}  {'verdict':<10}")
    for L, a0 in test_grid:
        K = K_L_closed_form(L)
        bound = K * (1 - a0)
        traj = argmax_monotone(L, a0)
        s_closed = ols_slope(traj)
        s_grid = numerical_max_monotone(L, a0, grid=40)
        ok = abs(s_closed - bound) < 1e-9 and s_grid <= bound + 1e-3
        print(f"{L:>2} {a0:>5.2f} {bound:>12.5f} {s_closed:>12.5f} {s_grid:>11.5f}  {'OK' if ok else 'FAIL'}")

    print()
    print("Reviewer's counterexamples (L=3, a_0=0.5):")
    cases = {
        "(a0, 1, 1, 1)  monotone": [0.5, 1.0, 1.0, 1.0],
        "(a0, a0, 1, 1) monotone (the argmax)": [0.5, 0.5, 1.0, 1.0],
        "(a0, 0, 1, 1)  NON-monotone": [0.5, 0.0, 1.0, 1.0],
    }
    bound = K_L_closed_form(3) * (1 - 0.5)
    print(f"  Monotone bound K_3 * (1 - 0.5) = 0.4 * 0.5 = {bound:.5f}")
    for label, traj in cases.items():
        s = ols_slope(traj)
        mono = all(traj[i] <= traj[i+1] for i in range(len(traj)-1))
        flag = "<= bound" if s <= bound + 1e-9 else "> bound (violates monotone-only bound)"
        if not mono:
            flag = f"NON-monotone; checked against unconstrained max"
            uc_max = (2 - 1.5 * 0.5) / 5  # closed form for L=3
            flag += f" {uc_max:.5f}, slope = {s:.5f}, {'OK' if s <= uc_max + 1e-9 else 'FAIL'}"
        print(f"  {label:<42}  slope = {s:+.5f}   {flag}")

    print()
    print("Unconstrained-trajectory check (L=3): max over (a_1, a_2, a_3) in [0,1]^3")
    for a0 in (0.0, 0.3, 0.5, 0.7):
        uc_grid = numerical_max_unconstrained(3, a0, grid=21)
        uc_closed = (2 - 1.5 * a0) / 5  # argmax at (a_0, 0, 1, 1)
        print(f"  a_0 = {a0:.2f}  grid max = {uc_grid:.5f}  closed form (2 - 1.5 a0)/5 = {uc_closed:.5f}  "
              f"{'OK' if abs(uc_grid - uc_closed) < 0.02 else 'FAIL'}")


if __name__ == "__main__":
    main()
