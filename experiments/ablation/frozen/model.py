"""Poisson pattern regression with complete enumeration and audited convergence.

The dictionary is explicitly either frequent closed itemsets (default) or all
frequent itemsets, subject to minimum support and maximum pattern length.
Old experiment snapshots are not modified by this implementation.
"""
from __future__ import annotations

import heapq
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, xlogy

IMPLEMENTATION_VERSION = "2026-09-23-adaptive-reference-v9"


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0


def intersect_sorted(a, b):
    return np.intersect1d(a, b, assume_unique=True).astype(np.int32)


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def _responses(y, require_positive_sum=False):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("y must be a nonempty finite nonnegative vector")
    total = float(np.sum(y))
    if not math.isfinite(total):
        raise ValueError("The response sum must be finite")
    if require_positive_sum and total == 0:
        raise ValueError("All-zero y has no finite unpenalized intercept optimum")
    return y


def _deviance_terms(y, mu, eps):
    # Scoring convention only: legacy linear baselines can predict below zero.
    # This flooring is never used in the optimization objective or its dual.
    y = _responses(y)
    mu = np.asarray(mu, dtype=float)
    if mu.shape != y.shape or not np.isfinite(mu).all() or eps <= 0:
        raise ValueError("Predictions must be finite and match y; eps must be positive")
    mu = np.maximum(mu, eps)
    return 2.0 * (xlogy(y, y) - xlogy(y, mu) - y + mu)


def poisson_deviance(y, mu, eps=1e-12):
    return float(np.maximum(_deviance_terms(y, mu, eps), 0.0).sum())


def poisson_deviance_residuals(y, mu, eps=1e-12):
    return np.sign(np.asarray(y) - mu) * np.sqrt(np.maximum(_deviance_terms(y, mu, eps), 0))


def poisson_pearson_residuals(y, mu, eps=1e-12):
    return (np.asarray(y) - mu) / np.sqrt(np.maximum(mu, eps))


def poisson_pseudo_r2_deviance_from_mu(y, mu, eps=1e-12):
    y = _responses(y)
    null = poisson_deviance(y, np.full_like(y, np.mean(y)), eps)
    return float(1 - poisson_deviance(y, mu, eps) / null) if null > 1e-30 else 0.0


@dataclass(frozen=True)
class Pattern:
    itemset: Tuple[int, ...]
    tidset: Tuple[int, ...]

    def support(self):
        return len(self.tidset)

    def key(self):
        return "{" + ",".join(map(str, self.itemset)) + "}"


@dataclass
class EnumStats:
    nodes_visited: int = 0
    pruned_support: int = 0
    pruned_v: int = 0
    pruned_canonical: int = 0
    screened_u: int = 0
    emitted_closed: int = 0
    pruned_dual: int = 0
    pruned_topk: int = 0


def _centered_norm_from_support(s, n):
    return math.sqrt(max(0.0, s - s * s / n))


def _subtree_centered_norm_bound(s, n):
    k = min(float(s), n / 2.0)
    return math.sqrt(max(0.0, k - k * k / n))


def _subtree_dual_mass_bound(tid, alpha):
    vals = alpha[tid]
    return max(float(vals[vals > 0].sum()), float(-vals[vals < 0].sum()))


def _roundoff_pad(value):
    return 1e-12 * max(1.0, abs(float(value)))


def _radius_term(r, norm):
    return 0.0 if norm == 0 else r * norm


def _centered_minus_proj_d_norm_from_tid(tid, d, d_norm2, s, n):
    norm2 = max(0.0, s - s * s / n)
    if d_norm2 > 0:
        norm2 -= float(d[tid].sum()) ** 2 / d_norm2
    return math.sqrt(max(0.0, norm2))


def theorem5_u0_binary_tidset(tid, alpha1, r1, alpha2, r2, n):
    """Support of two balls, evaluated separately for +x and -x.

    Both balls must concern the same optimization problem. Disjoint/invalid
    intersection geometry falls back to the CURRENT ball, never a zero radius.
    """
    if min(r1, r2) < 0 or math.isnan(r1) or math.isnan(r2):
        raise ValueError("Radii must be nonnegative")
    s = len(tid)
    norm = _centered_norm_from_support(s, n)
    a1, a2 = float(alpha1[tid].sum()), float(alpha2[tid].sum())
    current = abs(a1) + _radius_term(r1, norm)
    other = abs(a2) + _radius_term(r2, norm)
    if norm == 0:
        return max(abs(a1), abs(a2))
    if not math.isfinite(r1) or not math.isfinite(r2):
        return min(current, other)
    d = np.asarray(alpha2) - alpha1
    d2 = float(d @ d)
    dist = math.sqrt(d2)
    if dist + r1 <= r2:
        return current
    if dist + r2 <= r1:
        return other
    if dist == 0:
        return min(current, other)
    if dist > r1 + r2:
        return current
    dot = float(d[tid].sum())
    t = (r1 * r1 - r2 * r2 + d2) / (2 * d2)
    rho = math.sqrt(max(0.0, r1 * r1 - t * t * d2))
    orthogonal = math.sqrt(max(0.0, norm * norm - dot * dot / d2))

    def support(sign):
        # The direction test is different for +x and -x.
        if r1 * r1 + d2 - 2 * r1 * sign * dot / norm <= r2 * r2:
            return sign * a1 + r1 * norm
        if r2 * r2 + d2 + 2 * r2 * sign * dot / norm <= r1 * r1:
            return sign * a2 + r2 * norm
        return sign * (a1 + t * dot) + rho * orthogonal

    value = max(support(1), support(-1))
    return max(0.0, value) + _roundoff_pad(value)


def _offer(heap, score, pattern, top_k):
    if top_k <= 0:
        return
    # Reverse lexicographic order also at a prefix: (2,) must beat (2, 3).
    entry = (float(score), tuple(-v for v in pattern.itemset) + (float("inf"),), pattern)
    heapq.heappush(heap, entry)
    if len(heap) > top_k:
        heapq.heappop(heap)


def _ordered(heap):
    return [entry[2] for entry in sorted(heap, key=lambda e: (-e[0], e[2].itemset))]


class LCMEnumerator:
    """Complete canonical closure traversal; not a claim to implement LCM itself."""

    def __init__(self, item_tidsets, items_sorted, n_samples, min_support=2,
                 max_len=4, pattern_space="closed", propagate_support=False):
        if pattern_space not in ("closed", "all"):
            raise ValueError("pattern_space must be 'closed' or 'all'")
        if n_samples < 1 or min_support < 1 or max_len < 1:
            raise ValueError("n_samples, min_support and max_len must be positive")
        self.items_sorted = sorted(set(items_sorted))
        self.item_tidsets = {}
        for it in self.items_sorted:
            tids = np.unique(np.asarray(item_tidsets[it], dtype=np.int32))
            if np.any(tids < 0) or np.any(tids >= n_samples):
                raise ValueError("Invalid transaction indices")
            self.item_tidsets[it] = tids
        self.n = int(n_samples)
        self._mask_bytes = (self.n + 7) // 8
        self.item_masks = {}
        for it, tids in self.item_tidsets.items():
            present = np.zeros(self.n, dtype=np.uint8)
            present[tids] = 1
            self.item_masks[it] = int.from_bytes(np.packbits(present, bitorder="little").tobytes(), "little")
        self.min_support = int(min_support)
        self.max_len = int(max_len)
        self.pattern_space = pattern_space
        self.propagate_support = bool(propagate_support)
        self.stats = EnumStats()

    def reset_stats(self):
        self.stats = EnumStats()

    def get_stats(self):
        return self.stats

    @staticmethod
    def _is_subset_tid(child_tid, super_tid):
        return np.isin(child_tid, super_tid, assume_unique=True).all()

    def _closure_and_canonical(self, prefix, last, it, child_tid):
        pref = set(prefix)
        if self.pattern_space == "all":
            return tuple(sorted(pref | {it})), it
        for j in self.items_sorted:
            if j >= it:
                break
            if j not in pref and self._is_subset_tid(child_tid, self.item_tidsets[j]):
                return None
        closed = pref | {it}
        for j in self.items_sorted:
            if j > it and j not in pref and self._is_subset_tid(child_tid, self.item_tidsets[j]):
                closed.add(j)
        if len(closed) > self.max_len:
            return None
        # Continue after the generator item, NOT after max(closure).
        return tuple(sorted(closed)), it

    def iter_patterns(self, prune=None):
        # Integer bitsets preserve exact intersections/closures without allocating
        # an array for every subset test in the canonicality checks.
        def frequent_children(pref, mask, last, extensions):
            for it in extensions:
                if it in pref or (last is not None and it <= last):
                    continue
                child_mask = mask & self.item_masks[it]
                self.stats.nodes_visited += 1
                if child_mask.bit_count() < self.min_support:
                    self.stats.pruned_support += 1
                    continue
                yield it, child_mask

        stack = [((), (1 << self.n) - 1, None, tuple(self.items_sorted))]
        while stack:
            prefix, mask, last, extensions = stack.pop()
            if len(prefix) >= self.max_len:
                continue
            pref = set(prefix)
            children = frequent_children(pref, mask, last, extensions)
            if self.propagate_support:
                # Infrequent under this prefix implies infrequent under ANY
                # descendant. Keep all support-valid extensions, even those
                # later rejected by canonicality or a score-based filter.
                children = list(children)
                extensions = tuple(it for it, _ in children)
            for it, child_mask in children:
                child = np.flatnonzero(np.unpackbits(
                    np.frombuffer(child_mask.to_bytes(self._mask_bytes, "little"), dtype=np.uint8),
                    bitorder="little", count=self.n))
                if prune is not None and prune(child):
                    continue
                closed = pref | {it}
                if self.pattern_space == "closed":
                    canonical = True
                    for j in self.items_sorted:
                        if j in pref or j == it:
                            continue
                        if child_mask & self.item_masks[j] == child_mask:
                            if j < it:
                                canonical = False
                                break
                            closed.add(j)
                    if not canonical or len(closed) > self.max_len:
                        self.stats.pruned_canonical += 1
                        continue
                closed = tuple(sorted(closed))
                yield Pattern(closed, tuple(child.tolist()))
                stack.append((closed, child_mask, it, extensions))

    def dfs_with_spp(self, alpha, grad_eta, r, lam, already_have, top_k=200,
                     alpha2=None, r2=None, mode="both"):
        if mode not in ("none", "vprune", "uscreen", "both"):
            raise ValueError("Unknown enumeration mode")
        alpha, grad_eta = np.asarray(alpha), np.asarray(grad_eta)
        if alpha.shape != (self.n,) or grad_eta.shape != (self.n,):
            raise ValueError("Reference/gradient length does not match transactions")
        two = alpha2 is not None and r2 is not None
        heap = []

        def prune(tid):
            if mode not in ("vprune", "both"):
                return False
            norm = _subtree_centered_norm_bound(len(tid), self.n)
            bound = _subtree_dual_mass_bound(tid, alpha) + _radius_term(r, norm)
            if two:
                bound = min(bound, _subtree_dual_mass_bound(tid, alpha2) + _radius_term(r2, norm))
            if bound + _roundoff_pad(bound) < lam:
                self.stats.pruned_v += 1
                return True
            return False

        for pattern in self.iter_patterns(prune):
            tid = np.asarray(pattern.tidset, dtype=np.int32)
            if mode in ("uscreen", "both"):
                norm = _centered_norm_from_support(len(tid), self.n)
                bound = abs(float(alpha[tid].sum())) + _radius_term(r, norm)
                if two:
                    bound = theorem5_u0_binary_tidset(tid, alpha, r, alpha2, r2, self.n)
                if bound + _roundoff_pad(bound) < lam:
                    self.stats.screened_u += 1
                    continue
            if pattern.itemset not in already_have:
                self.stats.emitted_closed += 1
                _offer(heap, abs(float(grad_eta[tid].sum())), pattern, top_k)
        return _ordered(heap)

    def dual_scan(self, alpha, lam, kappa, already_have=(), top_k=200, violation_tol=0.0,
                  collect_ray=False, collect_patterns=False):
        """Exact full-dictionary conjugate sum and missing-column KKT search.

        A subtree is skipped only when its positive/negative mass proves that
        every omitted conjugate contribution and KKT violation is zero.
        """
        have, heap, correlations, violating_patterns = set(already_have), [], [], []
        maximum, outside, total = float(lam), 0.0, 0.0

        def prune(tid):
            bound = _subtree_dual_mass_bound(tid, alpha)
            if bound + _roundoff_pad(bound) < lam:
                self.stats.pruned_dual += 1
                return True
            return False

        for pattern in self.iter_patterns(prune):
            corr = abs(float(alpha[np.asarray(pattern.tidset, dtype=np.int32)].sum()))
            maximum = max(maximum, corr)
            excess = max(0.0, corr - lam)
            if (collect_ray or collect_patterns) and excess > 0:
                correlations.append(corr)
                if collect_patterns:
                    violating_patterns.append(pattern)
            if kappa > 0:
                total += excess * excess / (2 * kappa)
            if pattern.itemset not in have:
                outside = max(outside, excess)
                if excess > violation_tol:
                    _offer(heap, corr, pattern, top_k)
        return dict(penalty=total, max_correlation=maximum,
                    outside_violation=outside, candidates=_ordered(heap),
                    ray_correlations=np.asarray(correlations, dtype=float) if collect_ray or collect_patterns else None,
                    ray_patterns=violating_patterns if collect_patterns else None)

    def price_scan(self, alpha, lam, already_have=(), top_k=200, violation_tol=0.0,
                   kappa=None, collect_ray=False):
        """Exact top-K/max with a full conjugate sum ONLY when justified.

        The threshold rises only after K candidates exist. Skipped branches
        cannot improve either the candidate heap or the global maximum. They
        can contain nonzero EN conjugate terms. Such a skip invalidates the raw
        conjugate sum. If all skipped bounds are below lambda, the sum is exact
        and can be reused, avoiding a second traversal near convergence.
        """
        if top_k < 1:
            raise ValueError("Pricing needs a positive candidate budget")
        have, heap, correlations = set(already_have), [], []
        maximum, outside, total = float(lam), 0., 0.
        complete = kappa is not None

        def prune(tid):
            nonlocal complete
            cutoff = heap[0][0] if len(heap) == top_k else lam
            bound = _subtree_dual_mass_bound(tid, alpha)
            if bound + _roundoff_pad(bound) < min(maximum, lam + outside, cutoff):
                if bound + _roundoff_pad(bound) < lam:
                    self.stats.pruned_dual += 1
                else:
                    self.stats.pruned_topk += 1
                    complete = False
                return True
            return False

        for pattern in self.iter_patterns(prune):
            corr = abs(float(alpha[np.asarray(pattern.tidset, dtype=np.int32)].sum()))
            maximum = max(maximum, corr)
            excess = max(0., corr-lam)
            if kappa is not None:
                if kappa > 0:
                    total += excess * excess / (2*kappa)
                if collect_ray and excess > 0:
                    correlations.append(corr)
            if pattern.itemset not in have:
                outside = max(outside, excess)
                if excess > violation_tol:
                    _offer(heap, corr, pattern, top_k)
        return dict(max_correlation=maximum, outside_violation=outside,
                    candidates=_ordered(heap), penalty=total if complete else None,
                    penalty_complete=complete,
                    ray_correlations=np.asarray(correlations,float) if complete and collect_ray else None)


def build_X_from_patterns(n, patterns):
    X = np.zeros((n, len(patterns)), dtype=float)
    for j, p in enumerate(patterns):
        X[np.asarray(p.tidset, dtype=np.int32), j] = 1.0
    return X


def build_X_from_patterns_on_transactions(transactions, patterns):
    rows = [set(t) for t in transactions]
    X = np.zeros((len(rows), len(patterns)))
    for j, pattern in enumerate(patterns):
        p = set(pattern.itemset)
        X[:, j] = [p <= row for row in rows]
    return X


@dataclass
class ENParams:
    b: float
    w: np.ndarray


def poisson_objective(y, eta, w, lam, kappa):
    with np.errstate(over="ignore", invalid="ignore"):
        mu = np.exp(eta)
        value = np.sum(mu - y * eta) + lam * np.abs(w).sum() + 0.5 * kappa * (w @ w)
    return float(value) if np.isfinite(value) else float("inf")


def kkt_residual(X, y, params, lam, kappa):
    mu = np.exp(params.b + X @ params.w)
    residual = mu - y
    gradient = X.T @ residual + kappa * params.w
    violation = np.maximum(np.abs(gradient) - lam, 0.0)
    active = params.w != 0.0
    violation[active] = np.abs(gradient[active] + lam * np.sign(params.w[active]))
    return float(max(abs(residual.sum()), np.max(violation, initial=0.0)) / len(y))


class PoissonElasticNetSolver:
    """Exact loss with analytically profiled intercept and L-BFGS-B.

    The public fit/ENParams interface is unchanged. eta_clip is retained only
    for constructor compatibility; neither the loss nor its derivative clips.
    """

    def __init__(self, lam, kappa, max_iter=500, tol=1e-6, eta_clip=10.0):
        if not np.isfinite([lam, kappa, tol]).all() or min(lam, kappa) < 0 or tol <= 0 or max_iter < 1:
            raise ValueError("Invalid penalty, tolerance or iteration limit")
        self.lam, self.kappa = float(lam), float(kappa)
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.eta_clip = float(eta_clip)
        self.diagnostics = {}

    @staticmethod
    def profile(X, y, w):
        total = float(y.sum())
        linear = X @ w
        normalizer = logsumexp(linear)
        b = math.log(total) - normalizer
        eta = linear + b
        # Normalized evaluation avoids overflow without changing the objective.
        mu = total * np.exp(linear - normalizer)
        return float(b), eta, mu

    def fit(self, X, y, warm=None):
        y = _responses(y, require_positive_sum=True)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != len(y) or not np.isfinite(X).all():
            raise ValueError("X must be finite with one row per response")
        n, m = X.shape
        w0 = np.zeros(m) if warm is None or len(warm.w) != m else np.asarray(warm.w, float).copy()
        if not np.isfinite(w0).all():
            raise ValueError("Initial coefficients must be finite")
        if m == 0:
            result = ENParams(float(np.log(y.mean())), w0)
            residual = kkt_residual(X, y, result, self.lam, self.kappa)
            self.diagnostics = dict(converged=bool(residual <= self.tol), kkt=residual, iterations=0,
                                    objective=poisson_objective(y, np.full(n, result.b), w0, self.lam, self.kappa),
                                    backend="analytic-intercept")
            return result

        def value_gradient(z):
            w = z[:m] - z[m:]
            b, eta, mu = self.profile(X, y, w)
            # Drop only the saturated-model constant. The expm1 form avoids
            # cancellation near a fit, without clipping the objective/gradient.
            positive = y > 0
            delta = eta[positive] - np.log(y[positive])
            terms = mu.copy()
            positive_terms = mu[positive] - y[positive] * (1 + delta)
            close = np.abs(delta) < 1.0
            positive_terms[close] = y[positive][close] * (np.expm1(delta[close]) - delta[close])
            terms[positive] = positive_terms
            value = (np.sum(terms) + self.lam * z.sum() + 0.5 * self.kappa * (w @ w)) / n
            g = X.T @ (mu - y) + self.kappa * w
            return float(value), np.r_[g + self.lam, -g + self.lam] / n

        optimized = minimize(value_gradient, np.r_[np.maximum(w0, 0), np.maximum(-w0, 0)],
                             jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * (2 * m),
                             options=dict(maxiter=self.max_iter, maxfun=self.max_iter * 20,
                                          maxls=50, ftol=0.0, gtol=self.tol * 0.1, maxcor=20))
        w = optimized.x[:m] - optimized.x[m:]
        b, eta, _ = self.profile(X, y, w)
        result = ENParams(b, w)
        residual = kkt_residual(X, y, result, self.lam, self.kappa)
        # Function values can stop changing before the gradient is accurate.
        # Refine a nearly stationary active set using the profiled Hessian;
        # accept only non-increasing objective and strictly improved full KKT.
        polished = 0
        for _ in range(12):
            if residual <= self.tol or residual > 1e-3:
                break
            b, eta, mu = self.profile(X, y, w)
            gradient = X.T @ (mu - y) + self.kappa * w
            active = np.flatnonzero((w != 0) | (np.abs(gradient) > self.lam + n*self.tol*.1))
            if not len(active) or len(active) > 512:
                break
            signs = np.where(w[active] != 0, np.sign(w[active]), -np.sign(gradient[active]))
            xa = X[:, active]
            mean = (mu @ xa) / y.sum()
            centered = xa - mean
            hessian = centered.T @ (mu[:, None] * centered)
            hessian.flat[::len(active)+1] += self.kappa
            direction = -np.linalg.lstsq(hessian, gradient[active] + self.lam*signs, rcond=None)[0]
            cross = (w[active] != 0) & (w[active]*direction < 0)
            limits = np.full(len(active), np.inf)
            limits[cross] = -w[active][cross]/direction[cross]
            step = min(1., float(limits.min()))
            current_value = value_gradient(np.r_[np.maximum(w,0),np.maximum(-w,0)])[0]
            accepted = False
            for _ in range(16):
                candidate = w.copy()
                candidate[active] += step*direction
                candidate[active[limits <= step]] = 0.
                cb, _, _ = self.profile(X,y,candidate)
                cparams = ENParams(cb,candidate)
                error = kkt_residual(X,y,cparams,self.lam,self.kappa)
                value = value_gradient(np.r_[np.maximum(candidate,0),np.maximum(-candidate,0)])[0]
                if error < residual and value <= current_value + 8*np.finfo(float).eps*max(1.,abs(current_value)):
                    w, result, residual = candidate, cparams, error
                    polished += 1
                    accepted = True
                    break
                step *= .5
            if not accepted:
                break
        b, eta, _ = self.profile(X,y,w)
        result = ENParams(b,w)
        self.diagnostics = dict(converged=bool(residual <= self.tol), kkt=residual,
                                iterations=int(optimized.nit)+polished, polish_iterations=polished,
                                message=str(optimized.message),
                                objective=poisson_objective(y, eta, w, self.lam, self.kappa),
                                backend="profiled-intercept-lbfgsb")
        return result


class PoissonENGap:
    """Finite-dictionary dual. End-to-end fitting supplies a FULL conjugate sum."""

    def __init__(self, lam, kappa, eta_clip=10.0):
        if min(lam, kappa) < 0:
            raise ValueError("Penalties must be nonnegative")
        self.lam, self.kappa, self.eta_clip = float(lam), float(kappa), float(eta_clip)

    def primal(self, y, eta, w):
        return poisson_objective(y, eta, w, self.lam, self.kappa)

    def Lstar(self, alpha, y):
        t = np.asarray(y) - alpha
        if np.any(t < 0):
            return float("inf")
        return float(np.sum(xlogy(t, t) - t))

    def Ustar(self, v):
        excess = np.maximum(np.abs(v) - self.lam, 0)
        if self.kappa == 0:
            return 0.0 if np.all(excess == 0) else float("inf")
        return float(np.sum(excess ** 2) / (2 * self.kappa))

    def dual(self, alpha, y, X=None, full_penalty=None):
        alpha = np.asarray(alpha, float)
        if alpha.shape != np.asarray(y).shape or not np.isfinite(alpha).all():
            return -float("inf")
        if abs(float(alpha.sum())) > 1e-10 * max(1.0, float(np.sum(y))):
            return -float("inf")
        loss = self.Lstar(alpha, y)
        if full_penalty is None:
            if X is None:
                raise ValueError("Provide X or a verified full-dictionary conjugate sum")
            full_penalty = self.Ustar(X.T @ alpha)
        return float(-loss - full_penalty)

    def build_dual_feasible(self, y, eta):
        """Poisson-domain/intercept feasibility; L1 feasibility needs a full scan."""
        y = _responses(y, require_positive_sum=True)
        eta = np.asarray(eta, float)
        if eta.shape != y.shape or not np.isfinite(eta).all():
            raise ValueError("eta must be finite and match y")
        mass = float(y.sum()) * np.exp(eta - logsumexp(eta))
        # Absorb summation roundoff in a coordinate with ample positive mass.
        alpha = y - mass
        j = int(np.argmax(mass))
        alpha[j] -= float(alpha.sum())
        return alpha

    def optimize_ray(self, y, alpha, scan, zero_penalty=False):
        """Improve the FULL dual on {s * alpha: 0 <= s <= 1}, without a new scan.

        Unvisited/subthreshold correlations cannot become violations when s <= 1.
        Keep the original feasible endpoint as a fallback, including its audited
        sum, to avoid cancellation changing the near-optimal gap.
        """
        alpha = np.asarray(alpha, float)
        maximum = scan["max_correlation"]
        lasso_scale = min(1., self.lam / (maximum + _roundoff_pad(maximum)))
        upper = lasso_scale if self.kappa == 0 or zero_penalty else 1.
        correlations = scan["ray_correlations"]
        if self.kappa > 0 and not zero_penalty and correlations is None:
            raise ValueError("Ray optimization requires the complete violating correlations")

        def penalty_at(scale):
            if self.kappa == 0 or zero_penalty:
                return 0.
            if scale == 1.:
                return scan["penalty"]
            excess = np.maximum(scale * correlations - self.lam, 0.)
            return float(excess @ excess / (2 * self.kappa))

        def value(scale):
            return self.dual(scale * alpha, y, full_penalty=penalty_at(scale))

        candidates = [upper, 0., lasso_scale]
        if upper > 0:
            opt = minimize_scalar(lambda s: -value(s), bounds=(0., upper), method="bounded",
                                  options=dict(xatol=1e-12, maxiter=80))
            if np.isfinite(opt.x) and 0 <= opt.x <= upper:
                candidates.append(float(opt.x))
        scale = max(candidates, key=value)
        return alpha * scale, penalty_at(scale), scale

    def gap(self, y, eta, w, alpha, X=None, full_penalty=None):
        P = self.primal(y, eta, w)
        D = self.dual(alpha, y, X, full_penalty)
        if not math.isfinite(P) or not math.isfinite(D):
            return P, D, float("inf")
        G = P - D
        if G < -1e-9 * max(1.0, abs(P), abs(D)):
            raise ArithmeticError("Negative primal-dual gap; objective/certificate mismatch")
        return P, D, max(0.0, G)

    def optimize_segment(self, y, alpha1, penalty1, patterns1, alpha2, penalty2, patterns2):
        """Optimize the exact full dual between two audited feasible endpoints.

        patterns1/2 contain EVERY violating column at the respective endpoint.
        A column outside their union stays below lambda on the entire segment.
        This is a dual-reference update, not two-ball screening.
        """
        a1, a2 = np.asarray(alpha1, float), np.asarray(alpha2, float)
        if a1.shape != np.asarray(y).shape or a2.shape != a1.shape:
            raise ValueError("Segment endpoint shape mismatch")
        if not all(math.isfinite(self.dual(a, y, full_penalty=p))
                   for a, p in ((a1, penalty1), (a2, penalty2))):
            raise ValueError("Segment endpoints must be dual feasible")
        union = list({p.itemset: p for p in list(patterns1)+list(patterns2)}.values())
        c1 = np.asarray([a1[np.asarray(p.tidset, np.int32)].sum() for p in union])
        c2 = np.asarray([a2[np.asarray(p.tidset, np.int32)].sum() for p in union])

        def penalty_at(t):
            if t == 0.:
                return penalty1
            if t == 1.:
                return penalty2
            return self.Ustar((1.-t)*c1+t*c2)

        def value(t):
            return self.dual((1.-t)*a1+t*a2, y, full_penalty=penalty_at(t))

        choices = [0., 1.]
        if not np.array_equal(a1, a2):
            opt = minimize_scalar(lambda t: -value(t), bounds=(0., 1.), method="bounded",
                                  options=dict(xatol=1e-12, maxiter=80))
            if np.isfinite(opt.x) and 0 <= opt.x <= 1:
                choices.append(float(opt.x))
        t = max(choices, key=value)
        center = (1.-t)*a1+t*a2
        correlations = (1.-t)*c1+t*c2
        # Retain boundary columns as well; numerical endpoint ties must not
        # silently disappear from the complete support used by the next update.
        kept = [p for p, c in zip(union, correlations)
                if abs(c)+_roundoff_pad(c) >= self.lam]
        return center, penalty_at(t), kept, t


@dataclass
class FitResult:
    patterns: List[Pattern]
    params: ENParams
    eta: np.ndarray
    alpha: np.ndarray
    gap: float
    r: float
    total_time: float
    round_times: List[Dict[str, float]]
    enum_stats_per_round: List[EnumStats]
    ws_screen_removed_per_round: List[int]
    converged: bool = False
    termination_reason: str = "unknown"
    kkt_residual: float = float("inf")
    audit_stats_per_round: List[EnumStats] = field(default_factory=list)
    gap_scope: str = "full-dictionary"
    pattern_space: str = "closed"
    implementation_version: str = IMPLEMENTATION_VERSION


class PoissonEN_LCM_SPP:
    """Audited working-set search with full-dual ray optimization and audit reuse.

    every_round retains the default audit-then-search schedule for ablations. reuse takes the
    audit's top KKT violators directly. periodic additionally searches between
    audits using a previously audited dual center and its FULL conjugate sum.
    All modes require a fresh full audit to declare convergence.
    priced performs top-K branch-and-bound, using a zero-conjugate scaled
    reference between exact EN penalty audits. It never treats top-K-pruned
    patterns as permanently inactive.
    """

    def __init__(self, lam, kappa, min_support=2, max_len=4, top_k_add=200,
                 max_rounds=10, solver_max_iter=500, solver_tol=1e-6,
                 eta_clip=10.0, seed=0, do_ws_screen=True,
                 enum_mode_single="both", enum_mode_two="both",
                 radius_mode="certified", pattern_space="closed", gap_tol=1e-7,
                 audit_strategy="every_round", audit_interval=5, use_dynamic_ref=False,
                 dual_reference="ray", propagate_support=True):
        if radius_mode not in ("heuristic", "certified"):
            raise ValueError("Unknown radius_mode")
        if pattern_space not in ("closed", "all"):
            raise ValueError("Unknown pattern_space")
        if enum_mode_single not in ("none", "vprune", "uscreen", "both") or enum_mode_two not in ("none", "vprune", "uscreen", "both"):
            raise ValueError("Unknown enumeration mode")
        if min(min_support, max_len, top_k_add, max_rounds, solver_max_iter) < 1 or gap_tol <= 0:
            raise ValueError("Budgets, support, length and tolerance must be positive")
        if audit_strategy not in ("every_round", "reuse", "periodic", "priced"):
            raise ValueError("Unknown audit_strategy")
        if dual_reference not in ("raw", "ray", "adaptive"):
            raise ValueError("Unknown dual_reference")
        if dual_reference == "adaptive" and audit_strategy == "priced":
            raise ValueError("Adaptive references require complete endpoint supports; priced audits are not supported")
        if not isinstance(audit_interval, (int, np.integer)) or audit_interval < 1:
            raise ValueError("audit_interval must be a positive integer")
        PoissonElasticNetSolver(lam, kappa, solver_max_iter, solver_tol, eta_clip)
        self.lam, self.kappa = float(lam), float(kappa)
        self.min_support, self.max_len = int(min_support), int(max_len)
        self.top_k_add, self.max_rounds = int(top_k_add), int(max_rounds)
        self.solver_max_iter, self.solver_tol = int(solver_max_iter), float(solver_tol)
        self.eta_clip, self.rng = float(eta_clip), np.random.default_rng(seed)
        self.do_ws_screen = bool(do_ws_screen)
        self.enum_mode_single, self.enum_mode_two = enum_mode_single, enum_mode_two
        self.radius_mode, self.pattern_space, self.gap_tol = radius_mode, pattern_space, float(gap_tol)
        self.audit_strategy, self.audit_interval = audit_strategy, int(audit_interval)
        self.use_dynamic_ref = bool(use_dynamic_ref)
        self.dual_reference = dual_reference
        self.propagate_support = bool(propagate_support)
        if self.lam == 0 and self.kappa == 0:
            raise ValueError("Pattern discovery requires a positive penalty; use the fixed-dictionary solver for unpenalized refits")

    @staticmethod
    def _transactions_to_item_tidsets(transactions):
        mapping = {}
        for i, row in enumerate(transactions):
            for item in sorted(set(row)):
                if not isinstance(item, (int, np.integer)):
                    raise ValueError("Item identifiers must be integers")
                mapping.setdefault(int(item), []).append(i)
        return {item: np.asarray(tids, np.int32) for item, tids in mapping.items()}, sorted(mapping)

    def _poisson_radius_scale(self, y, alpha, mu, G=0.0):
        if self.radius_mode == "heuristic":
            return float(np.mean(mu))
        if not math.isfinite(G):
            return float("inf")
        M = max(0.0, float(np.max(y - alpha)))
        local = M + G + math.sqrt(max(0.0, G * (G + 2 * M)))
        # sum(y-alpha)=sum(y): a global curvature bound on the dual simplex.
        return min(float(np.sum(y)), local)

    def _radius(self, y, alpha, mu, gap):
        if not math.isfinite(gap):
            return float("inf")
        return math.sqrt(max(0.0, 2 * self._poisson_radius_scale(y, alpha, mu, gap) * gap))

    def _cheap_feasible_reference(self, raw_alpha, tidsets):
        """All itemsets lie under a singleton: its signed mass bounds correlation."""
        bound = max((_subtree_dual_mass_bound(tid, raw_alpha)
                     for tid in tidsets.values()), default=0.)
        scale = min(1., self.lam/(bound+_roundoff_pad(bound)))
        return scale*raw_alpha, scale, len(tidsets)

    def fit(self, transactions, y, verbose=True, ref2=None):
        started = time.perf_counter()
        y = _responses(y, require_positive_sum=True)
        if len(transactions) != len(y):
            raise ValueError("One response per transaction is required")
        n = len(y)
        tidsets, items = self._transactions_to_item_tidsets(transactions)

        def new_enum():
            return LCMEnumerator(tidsets, items, n, self.min_support, self.max_len,
                                 self.pattern_space, self.propagate_support)

        calc = PoissonENGap(self.lam, self.kappa)
        solver = PoissonElasticNetSolver(self.lam, self.kappa, self.solver_max_iter,
                                        self.solver_tol, self.eta_clip)
        patterns, warm = [], None
        times, search_stats, audit_stats, removals = [], [], [], []
        previous_center, previous_dual = None, None
        pending_reference_audit = None
        reference_seconds = 0.0
        if ref2 is not None:
            reference_start = time.perf_counter()
            previous_center = np.asarray(ref2[0], float).copy()
            # The supplied OLD radius is deliberately ignored.
            if previous_center.shape != y.shape or not math.isfinite(calc.dual(previous_center, y, full_penalty=0.0)):
                raise ValueError("The previous dual center is not feasible for this response vector")
            ref_enum = new_enum()
            scan = ref_enum.dual_scan(previous_center, self.lam, self.kappa, top_k=0)
            if self.kappa == 0:
                scale = min(1.0, self.lam / (scan["max_correlation"] + _roundoff_pad(scan["max_correlation"])))
                previous_center *= scale
                penalty = 0.0
            else:
                penalty = scan["penalty"]
            previous_dual = calc.dual(previous_center, y, full_penalty=penalty)
            pending_reference_audit = ref_enum.stats
            reference_seconds = time.perf_counter() - reference_start

        converged, reason = False, "max_rounds"
        full_kkt = float("inf")
        cached_alpha, cached_penalty, cached_patterns = None, None, []
        last_audit_round, force_audit = 0, False
        for rd in range(1, self.max_rounds + 1):
            round_start = time.perf_counter()
            X = build_X_from_patterns(n, patterns)
            tick = time.perf_counter()
            params = solver.fit(X, y, warm)
            solver_seconds = time.perf_counter() - tick
            eta = params.b + X @ params.w
            mu = np.exp(eta)
            raw_alpha = calc.build_dual_feasible(y, eta)
            have = {p.itemset for p in patterns}

            tick = time.perf_counter()
            audited = (self.audit_strategy != "periodic" or cached_alpha is None
                       or force_audit or rd == self.max_rounds
                       or rd - last_audit_round >= self.audit_interval)
            scan = None
            reference_opt_seconds, reference_scale = 0., None
            segment_weight, reference_bound_nodes, reference_update = None, 0, "cached"
            raw_gap = raw_radius = raw_penalty = None
            pricing_seconds, pricing_nodes = 0., 0
            conjugate_audit = False
            if audited:
                auditor = new_enum()
                if self.audit_strategy == "priced":
                    pricing_start = time.perf_counter()
                    scan = auditor.price_scan(raw_alpha, self.lam, have,
                                               self.top_k_add, n * self.solver_tol,
                                               kappa=self.kappa,
                                               collect_ray=self.dual_reference == "ray" and self.kappa > 0)
                    audit_stats.append(auditor.stats)
                    pricing_nodes = auditor.stats.nodes_visited
                    pricing_seconds = time.perf_counter() - pricing_start
                    conjugate_audit = scan["penalty_complete"]
                full_kkt = max(solver.diagnostics["kkt"], scan["outside_violation"] / n) if scan else float("inf")
                if scan is None or (not conjugate_audit and
                                    (full_kkt <= self.solver_tol or rd == self.max_rounds)):
                    auditor = new_enum()
                    scan = auditor.dual_scan(raw_alpha, self.lam, self.kappa, have,
                                             self.top_k_add, n * self.solver_tol,
                                             collect_ray=self.dual_reference != "raw" and self.kappa > 0,
                                             collect_patterns=self.dual_reference == "adaptive" and self.kappa > 0)
                    audit_stats.append(auditor.stats)
                    conjugate_audit = True
                if self.use_dynamic_ref and cached_alpha is not None:
                    previous_center = cached_alpha.copy()
                    previous_dual = calc.dual(previous_center, y, full_penalty=cached_penalty)
                alpha = raw_alpha.copy()
                if self.kappa == 0 or not conjugate_audit:
                    denominator = scan["max_correlation"] + _roundoff_pad(scan["max_correlation"])
                    alpha *= min(1.0, self.lam / denominator)
                    penalty = 0.0
                else:
                    penalty = scan["penalty"]
                if conjugate_audit:
                    raw_penalty = penalty
                    raw_gap = max(0., calc.primal(y, eta, params.w) - calc.dual(alpha, y, full_penalty=penalty))
                    raw_radius = self._radius(y, alpha, mu, raw_gap)
                reference_scale = 1. if self.kappa > 0 and conjugate_audit else min(1., self.lam / denominator)
                if self.dual_reference != "raw":
                    optimize_start = time.perf_counter()
                    alpha, penalty, reference_scale = calc.optimize_ray(y, raw_alpha, scan,
                                                                       zero_penalty=not conjugate_audit)
                    reference_opt_seconds = time.perf_counter() - optimize_start
                reference_update = "audited-"+self.dual_reference
                if self.dual_reference == "adaptive":
                    tick_segment = time.perf_counter()
                    current_patterns = [p for p, c in zip(scan["ray_patterns"] or [],
                                                         scan["ray_correlations"] if scan["ray_correlations"] is not None else [])
                                        if reference_scale*c+_roundoff_pad(c) >= self.lam]
                    if cached_alpha is not None:
                        alpha, penalty, current_patterns, segment_weight = calc.optimize_segment(
                            y, alpha, penalty, current_patterns, cached_alpha, cached_penalty, cached_patterns)
                    cached_patterns = current_patterns
                    reference_opt_seconds += time.perf_counter()-tick_segment
                cached_alpha, cached_penalty = alpha.copy(), penalty
                last_audit_round = rd
                full_kkt = max(solver.diagnostics["kkt"], scan["outside_violation"] / n)
            else:
                audit_stats.append(EnumStats())
                alpha, penalty = cached_alpha.copy(), cached_penalty
                if self.dual_reference == "adaptive":
                    tick_segment = time.perf_counter()
                    if self.use_dynamic_ref:
                        previous_center = cached_alpha.copy()
                        previous_dual = calc.dual(previous_center, y, full_penalty=cached_penalty)
                    fresh, reference_scale, reference_bound_nodes = self._cheap_feasible_reference(raw_alpha, tidsets)
                    alpha, penalty, cached_patterns, segment_weight = calc.optimize_segment(
                        y, fresh, 0., [], cached_alpha, cached_penalty, cached_patterns)
                    cached_alpha, cached_penalty = alpha.copy(), penalty
                    reference_update = "cheap-segment"
                    reference_opt_seconds = time.perf_counter()-tick_segment
                    audit_stats.append(EnumStats(nodes_visited=reference_bound_nodes))
                # The current omitted-column KKT residual has NOT been checked.
                full_kkt = float("inf")
            audit_seconds = time.perf_counter() - tick
            if pending_reference_audit is not None:
                audit_stats.append(pending_reference_audit)
                pending_reference_audit = None
            # A cached D(alpha) is still a lower bound for this same problem.
            # Recompute the gap/radius from the CURRENT primal, not the old gap.
            P, D, G = calc.gap(y, eta, params.w, alpha, full_penalty=penalty)
            r = self._radius(y, alpha, mu, G)
            converged = bool(audited and full_kkt <= self.solver_tol and G / n <= self.gap_tol)
            refine = (audited and not converged and not scan["candidates"]
                      and full_kkt <= self.solver_tol and G / n > self.gap_tol)
            if refine:
                # A gradient tolerance alone need not achieve the requested dual
                # gap, particularly for pure L1. Tighten the inner solve instead
                # of repeatedly enumerating the same unchanged working set.
                if solver.tol > 1e-13:
                    solver.tol = max(1e-13, solver.tol * .1)
                else:
                    reason = "gap_stagnation_at_inner_precision_limit"
            r2 = None
            if previous_center is not None:
                if P - previous_dual < -1e-9 * max(1.0, abs(P), abs(previous_dual)):
                    raise ArithmeticError("Previous reference exceeds the current primal value")
                old_gap = max(0.0, P - previous_dual)
                r2 = self._radius(y, previous_center, mu, old_gap)

            enum = new_enum()
            removed, new_patterns, rescued = 0, [], 0
            candidate_source = "none"
            search_seconds = ws_seconds = 0.0
            if not converged and not refine and rd < self.max_rounds:
                tick = time.perf_counter()
                # Only exact zeros are removed: no unrefitted change to predictions.
                keep = []
                for j, p in enumerate(patterns):
                    tid = np.asarray(p.tidset, np.int32)
                    u = abs(float(alpha[tid].sum())) + _radius_term(r, _centered_norm_from_support(len(tid), n))
                    if self.do_ws_screen and params.w[j] == 0 and u + _roundoff_pad(u) < self.lam:
                        removed += 1
                    else:
                        keep.append(j)
                if removed:
                    patterns = [patterns[j] for j in keep]
                    params = ENParams(params.b, params.w[keep])
                    have = {p.itemset for p in patterns}
                ws_seconds = time.perf_counter() - tick

                tick = time.perf_counter()
                if audited and self.audit_strategy != "every_round":
                    candidates = scan["candidates"]
                    candidate_source = "pricing" if self.audit_strategy == "priced" else "audit"
                else:
                    mode = self.enum_mode_two if previous_center is not None else self.enum_mode_single
                    candidates = enum.dfs_with_spp(alpha, mu - y, r, self.lam, have,
                                                   self.top_k_add, previous_center, r2, mode)
                    candidate_source = "spp"
                new_patterns = [p for p in candidates
                                if abs(float(raw_alpha[np.asarray(p.tidset, np.int32)].sum())) > self.lam + n * self.solver_tol]
                # A heuristic or numerically questionable bound must not silently
                # hide a KKT-violating column forever.
                if audited and not new_patterns and scan["candidates"]:
                    new_patterns = [p for p in scan["candidates"] if p.itemset not in have]
                    rescued = len(new_patterns)
                search_seconds = time.perf_counter() - tick

            # Neither an empty screened pool nor a restricted solve can certify
            # global convergence. Refresh the full audit at the next round.
            force_audit = refine or (not audited and (not new_patterns or G / n <= self.gap_tol))

            search_stats.append(enum.stats)
            removals.append(removed)
            times.append(dict(round=float(rd), m=float(len(patterns)), solver_time=solver_seconds,
                              gap_time=audit_seconds + (reference_seconds if rd == 1 else 0.0),
                              audit_time=audit_seconds + (reference_seconds if rd == 1 else 0.0),
                              reference_time=reference_seconds if rd == 1 else 0.0, ws_time=ws_seconds,
                              reference_opt_time=reference_opt_seconds,
                              dual_reference=self.dual_reference, reference_scale=reference_scale,
                              reference_update=reference_update, segment_weight=segment_weight,
                              reference_bound_nodes=reference_bound_nodes,
                              raw_gap=raw_gap, raw_radius=raw_radius, raw_penalty=raw_penalty,
                              reference_penalty=penalty,
                              conjugate_audit=conjugate_audit, pricing_time=pricing_seconds,
                              pricing_nodes=pricing_nodes,
                              mine_time=search_seconds,
                              round_total=time.perf_counter() - round_start + (reference_seconds if rd == 1 else 0.0),
                              gap=G, r=r, ref2_radius=r2, kkt=full_kkt,
                              full_audit=audited, reference_age=rd-last_audit_round,
                              audit_strategy=self.audit_strategy, candidate_source=candidate_source,
                              candidates_added=len(new_patterns), force_next_audit=force_audit,
                              solver_converged=solver.diagnostics["converged"],
                              solver_iterations=solver.diagnostics["iterations"],
                              next_inner_tolerance=solver.tol, gap_refinement=refine,
                              radius_scale=self._poisson_radius_scale(y, alpha, mu, G),
                              ws_removed=float(removed), rescued_candidates=rescued))
            if verbose:
                print(f"[round {rd}] p={len(patterns)} full_gap={G:.6g} "
                      f"KKT/n={full_kkt:.3g} candidates={len(new_patterns)}", flush=True)
            if converged:
                reason = "full_gap_and_kkt"
                break
            if reason == "gap_stagnation_at_inner_precision_limit":
                break
            if rd == self.max_rounds:
                break
            old_count = len(patterns)
            patterns.extend(new_patterns)
            warm = ENParams(params.b, np.r_[params.w, np.zeros(len(patterns) - old_count)])

        # No candidate is appended after the final fitted round.
        eta = params.b + build_X_from_patterns(n, patterns) @ params.w
        if not converged:
            warnings.warn(f"DIPS did not converge: {reason}; full gap/n={G/n:.3g}, "
                          f"KKT/n={full_kkt:.3g}", RuntimeWarning, stacklevel=2)
        return FitResult(patterns, params, eta, alpha, G, r, time.perf_counter() - started,
                         times, search_stats, removals, converged, reason, full_kkt,
                         audit_stats, pattern_space=self.pattern_space)


def fit_path(model_factory, transactions, y, lam_list, verbose=True, use_two_ref=True):
    results, previous = [], None
    for lam in lam_list:
        result = model_factory(float(lam)).fit(transactions, y, verbose=verbose,
                                               ref2=previous if use_two_ref else None)
        results.append(result)
        previous = (result.alpha, result.r)
    return results


def _kfold_indices(n, K, seed=0):
    if not 2 <= K <= n:
        raise ValueError("Require 2 <= K <= number of observations")
    folds = np.array_split(np.random.default_rng(seed).permutation(n), K)
    for k, val in enumerate(folds):
        yield np.concatenate([folds[j] for j in range(K) if j != k]), val


def _subset_transactions(trans, idx):
    return [trans[i] for i in idx.tolist()]


def fit_cv(model_factory, transactions, y, lam_list, K=5, seed=0, use_two_ref=True):
    y = _responses(y)
    lambdas = list(lam_list)
    if not lambdas or len(set(lambdas)) != len(lambdas):
        raise ValueError("Supply distinct candidate lambdas")
    records = {lam: [] for lam in lambdas}
    for train, val in _kfold_indices(len(y), K, seed):
        tr, va = _subset_transactions(transactions, train), _subset_transactions(transactions, val)
        previous = None
        for lam in lambdas:
            res = model_factory(float(lam)).fit(tr, y[train], verbose=False,
                                                ref2=previous if use_two_ref else None)
            previous = (res.alpha, res.r)
            if not res.converged:
                raise RuntimeError(f"CV fit at lambda={lam} failed full-space convergence")
            eta = res.params.b + build_X_from_patterns_on_transactions(va, res.patterns) @ res.params.w
            with np.errstate(over="ignore"):
                mu = np.exp(eta)
            if not np.isfinite(mu).all():
                raise FloatingPointError("Nonfinite validation predictions; no clipping is applied")
            records[lam].append(dict(NLL=float(np.mean(mu - y[val] * eta)),
                                     Deviance=poisson_deviance(y[val], mu),
                                     MAE=float(np.mean(abs(y[val] - mu))),
                                     RMSE=float(np.sqrt(np.mean((y[val] - mu) ** 2))),
                                     D_null=poisson_deviance(y[val], np.full(len(val), y[train].mean()))))
    out = {}
    for lam, rows in records.items():
        out[lam] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        out[lam]["PseudoR2"] = 1 - out[lam]["Deviance"] / max(1e-12, out[lam]["D_null"])
    return min(lambdas, key=lambda lam: out[lam]["NLL"]), out


def make_synthetic_transactions(n=300, n_items=50, avg_len=7, seed=1):
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(n_items, min(max(1, int(rng.poisson(avg_len))), n_items),
                              replace=False).tolist()) for _ in range(n)]


def add_poisson_labels_from_true_patterns(transactions, true_patterns, weights,
                                          b=-0.8, seed=2, eta_clip=None):
    eta = np.full(len(transactions), float(b))
    for pattern, weight in zip(true_patterns, weights):
        eta += weight * np.asarray([set(pattern) <= set(t) for t in transactions])
    if eta_clip is not None:
        warnings.warn("Explicit eta_clip changes the data-generating mean", UserWarning, stacklevel=2)
        eta = np.clip(eta, -eta_clip, eta_clip)
    mu = np.exp(eta)
    return np.random.default_rng(seed).poisson(mu).astype(float)


def make_binary_transactions_with_interactions(n=300, n_items=60, base_on_prob=0.10,
                                               planted_patterns=None, planted_prob=None, seed=0):
    patterns = [(3, 7), (10, 11, 12), (25,)] if planted_patterns is None else planted_patterns
    probs = [0.25, 0.18, 0.30] if planted_prob is None else planted_prob
    if len(patterns) != len(probs):
        raise ValueError("One probability per embedded pattern is required")
    rng, transactions = np.random.default_rng(seed), []
    for _ in range(n):
        row = set(np.flatnonzero(rng.random(n_items) < base_on_prob).tolist())
        for p, probability in zip(patterns, probs):
            if rng.random() < probability:
                row.update(p)
        transactions.append(sorted(row))
    return transactions


def support(transactions, patt):
    return sum(set(patt) <= set(row) for row in transactions)


def summarize_pruning(res):
    search = res.enum_stats_per_round
    audits = res.audit_stats_per_round
    stats = search + audits
    count = lambda attr: sum(getattr(s, attr) for s in stats)
    nodes = count("nodes_visited")
    hard = count("pruned_support") + count("pruned_canonical") + count("pruned_v") + count("pruned_dual") + count("pruned_topk")
    total_times = lambda attr: sum(t.get(attr, 0.0) for t in res.round_times)
    return dict(nodes_visited=float(nodes),
                search_nodes_visited=float(sum(s.nodes_visited for s in search)),
                audit_nodes_visited=float(sum(s.nodes_visited for s in audits)),
                emitted_closed=float(count("emitted_closed")),
                pruned_support=float(count("pruned_support")),
                pruned_canonical=float(count("pruned_canonical")),
                pruned_v=float(count("pruned_v")), pruned_dual=float(count("pruned_dual")),
                pruned_topk=float(count("pruned_topk")),
                pricing_nodes_visited=total_times("pricing_nodes"),
                screened_u=float(count("screened_u")), hard_pruned_total=float(hard),
                hard_prune_rate=hard / max(1, nodes), v_prune_rate=count("pruned_v") / max(1, nodes),
                u_screen_rate=count("screened_u") / max(1, nodes),
                ws_removed_total=float(sum(res.ws_screen_removed_per_round)),
                time_total_sec=float(res.total_time), time_solver_sec=total_times("solver_time"),
                time_gap_sec=total_times("gap_time"), time_ws_sec=total_times("ws_time"),
                time_mine_sec=total_times("mine_time"), converged=res.converged,
                full_gap=res.gap, full_kkt=res.kkt_residual)


def summarize_with_reference(res, reference_res):
    """Cold-start accounting including construction of an external reference."""
    current, previous = summarize_pruning(res), summarize_pruning(reference_res)
    combined = dict(current)
    additive = ("nodes_visited", "search_nodes_visited", "audit_nodes_visited",
                "emitted_closed", "pruned_support", "pruned_canonical", "pruned_v",
                "pruned_dual", "pruned_topk", "pricing_nodes_visited", "screened_u", "hard_pruned_total", "ws_removed_total",
                "time_total_sec", "time_solver_sec", "time_gap_sec", "time_ws_sec", "time_mine_sec")
    for key in additive:
        combined[key] += previous[key]
    denominator = max(1.0, combined["nodes_visited"])
    combined["hard_prune_rate"] = combined["hard_pruned_total"] / denominator
    combined["v_prune_rate"] = combined["pruned_v"] / denominator
    combined["u_screen_rate"] = combined["screened_u"] / denominator
    combined["cached_reference_time_sec"] = current["time_total_sec"]
    combined["reference_setup_time_sec"] = previous["time_total_sec"]
    combined["cached_reference_nodes"] = current["nodes_visited"]
    combined["reference_setup_nodes"] = previous["nodes_visited"]
    combined["reference_converged"] = reference_res.converged
    return combined


def print_round_report(res):
    for row in res.round_times:
        print(f"round {int(row['round'])}: m={int(row['m'])} full_gap={row['gap']:.4g} "
              f"KKT/n={row['kkt']:.3g} time={row['round_total']:.3f}s")


def print_pruning_report(res):
    sm = summarize_pruning(res)
    print(f"Total time: {sm['time_total_sec']:.4f}s; converged={res.converged}")
    print(f"Nodes visited (search + full audits): {int(sm['nodes_visited'])} "
          f"({int(sm['search_nodes_visited'])} + {int(sm['audit_nodes_visited'])})")
    print("hard_prune_rate is the fraction of VISITED extensions stopped, not "
          "the fraction of the full pattern universe avoided.")
    print_round_report(res)
