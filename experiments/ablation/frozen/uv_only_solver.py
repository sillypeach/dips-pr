"""Experimental SPP search with no separate full-dictionary KKT/pricing scan.

The same u/v traversal supplies correlations for the next full-dual reference.
No candidate is filtered by a KKT threshold or rescued by another tree search.
"""
from __future__ import annotations

import time

import numpy as np

import model as m


def fused_traversal(enum, raw, alpha, radius, lam, have, top_k, mode,
                    alpha2=None, radius2=None):
    """Return SPP candidates and sufficient data for an exact scaled dual.

    For each v-pruned subtree, record a bound on its RAW correlations. Scaling
    below lambda/that bound makes every omitted EN conjugate term zero. Raw
    bounds are bookkeeping only: they never decide which subtree to prune.
    """
    heap, correlations = [], []
    maximum, skipped_bound, visited_patterns, bound_checks = 0., 0., 0, 0
    two = alpha2 is not None and radius2 is not None
    cached = hasattr(enum, "cache_metrics")

    def prune(tid):
        nonlocal skipped_bound, bound_checks
        if mode not in ("vprune", "both"):
            return False
        if cached and enum.certified_v():
            enum.cache_metrics["v_certificate_reuses"] += 1
        else:
            if cached:
                enum.cache_metrics["v_score_evaluations"] += 1
            norm = m._subtree_centered_norm_bound(len(tid), enum.n)
            bound = m._subtree_dual_mass_bound(tid, alpha) + m._radius_term(radius, norm)
            if two:
                bound = min(bound, m._subtree_dual_mass_bound(tid, alpha2)
                            + m._radius_term(radius2, norm))
            if bound + m._roundoff_pad(bound) >= lam:
                return False
            enum.stats.pruned_v += 1
            if cached:
                enum.remember_v()
        skipped_bound = max(skipped_bound, m._subtree_dual_mass_bound(tid, raw))
        bound_checks += 1
        return True

    for pattern in enum.iter_patterns(prune):
        tid = enum.current_node.tid if cached else np.asarray(pattern.tidset, np.int32)
        corr = abs(float(raw[tid].sum()))
        maximum = max(maximum, corr)
        visited_patterns += 1
        if corr > lam:
            correlations.append(corr)
        if mode in ("uscreen", "both"):
            if cached and enum.certified_u(pattern):
                enum.cache_metrics["u_certificate_reuses"] += 1
                continue
            if cached:
                enum.cache_metrics["u_score_evaluations"] += 1
            norm = m._centered_norm_from_support(len(tid), enum.n)
            bound = abs(float(alpha[tid].sum())) + m._radius_term(radius, norm)
            if two:
                bound = m.theorem5_u0_binary_tidset(
                    tid, alpha, radius, alpha2, radius2, enum.n)
            if bound + m._roundoff_pad(bound) < lam:
                enum.stats.screened_u += 1
                if cached:
                    enum.remember_u(pattern)
                continue
        if pattern.itemset not in have:
            enum.stats.emitted_closed += 1
            m._offer(heap, corr, pattern, top_k)
    return dict(candidates=m._ordered(heap), correlations=np.asarray(correlations),
                maximum=max(maximum, skipped_bound), skipped_bound=skipped_bound,
                correlation_evaluations=visited_patterns, raw_bound_checks=bound_checks)


def reference_from_traversal(calc, y, raw, scan):
    bound = scan["maximum"] if calc.kappa == 0 else scan["skipped_bound"]
    scale = min(1., calc.lam/(bound+m._roundoff_pad(bound))) if bound > 0 else 1.
    correlations = scale*scan["correlations"]
    correlations = correlations[correlations > calc.lam]
    penalty = calc.Ustar(correlations)
    endpoint_scan = dict(max_correlation=scale*scan["maximum"],
                         ray_correlations=correlations, penalty=penalty)
    alpha, penalty, ray_scale = calc.optimize_ray(
        y, scale*raw, endpoint_scan, zero_penalty=calc.kappa == 0)
    return alpha, penalty, scale*ray_scale


def fit_uv_only(transactions, y, config, verbose=True):
    """Converge by a full dual gap, not an outside-column KKT search.

    The restricted numerical solver retains its ordinary internal convergence
    checks. An independent global KKT check belongs to the benchmark, after fit.
    """
    started = time.perf_counter()
    geo = m.PoissonEN_LCM_SPP(**config)
    if geo.radius_mode != "certified" or geo.do_ws_screen:
        raise ValueError("This experiment requires certified radii and WS removal off")
    y = m._responses(y, require_positive_sum=True)
    if len(transactions) != len(y):
        raise ValueError("One response per transaction is required")
    n = len(y)
    tids, items = geo._transactions_to_item_tidsets(transactions)
    calc = m.PoissonENGap(geo.lam, geo.kappa)
    solver = m.PoissonElasticNetSolver(geo.lam, geo.kappa, geo.solver_max_iter,
                                      geo.solver_tol, geo.eta_clip)
    center, penalty = np.zeros(n), 0.
    dual = calc.dual(center, y, full_penalty=penalty)
    previous, previous_dual = None, None
    patterns, warm, rounds, stats = [], None, [], []
    converged, reason = False, "max_rounds"
    for rd in range(1, geo.max_rounds+1):
        tick_round = time.perf_counter()
        X = m.build_X_from_patterns(n, patterns)
        tick = time.perf_counter()
        params = solver.fit(X, y, warm)
        solver_seconds = time.perf_counter()-tick
        eta = params.b+X@params.w
        mu = np.exp(eta)
        raw = calc.build_dual_feasible(y, eta)
        P, _, prior_gap = calc.gap(y, eta, params.w, center, full_penalty=penalty)
        radius = geo._radius(y, center, mu, prior_gap)
        radius2 = None
        if previous is not None:
            radius2 = geo._radius(y, previous, mu, max(0., P-previous_dual))
        enum = m.LCMEnumerator(tids, items, n, geo.min_support, geo.max_len,
                               geo.pattern_space, geo.propagate_support)
        tick = time.perf_counter()
        mode = geo.enum_mode_two if previous is not None else geo.enum_mode_single
        scan = fused_traversal(enum, raw, center, radius, geo.lam,
                               {p.itemset for p in patterns}, geo.top_k_add, mode,
                               previous, radius2)
        mine_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        fresh, fresh_penalty, scale = reference_from_traversal(calc, y, raw, scan)
        fresh_dual = calc.dual(fresh, y, full_penalty=fresh_penalty)
        old_center, old_dual = center.copy(), dual
        if fresh_dual >= dual:
            center, penalty, dual = fresh, fresh_penalty, fresh_dual
        if geo.use_dynamic_ref:
            previous, previous_dual = old_center, old_dual
        P, dual, gap = calc.gap(y, eta, params.w, center, full_penalty=penalty)
        final_radius = geo._radius(y, center, mu, gap)
        gap_seconds = time.perf_counter()-tick
        converged = bool(gap/n <= geo.gap_tol and solver.diagnostics["converged"])
        candidates = scan["candidates"] if not converged and rd < geo.max_rounds else []
        if not converged and not scan["candidates"]:
            if solver.tol > 1e-13:
                solver.tol = max(1e-13, solver.tol*.1)
            else:
                reason = "no_candidates_and_gap_stagnation"
        stats.append(enum.stats)
        rounds.append(dict(round=rd, m=len(patterns), solver_time=solver_seconds,
                           gap_time=gap_seconds, mine_time=mine_seconds,
                           round_total=time.perf_counter()-tick_round,
                           gap=gap, pre_search_gap=prior_gap, r=radius,
                           updated_r=final_radius, ref2_radius=radius2,
                           kkt=None, restricted_kkt=solver.diagnostics["kkt"],
                           full_audit=False, candidate_source="uv-only",
                           candidates_added=len(candidates), candidate_pool=enum.stats.emitted_closed,
                           reference_scale=scale, raw_bound_checks=scan["raw_bound_checks"],
                           correlation_evaluations=scan["correlation_evaluations"],
                           skipped_raw_bound=scan["skipped_bound"],
                           solver_converged=solver.diagnostics["converged"],
                           solver_iterations=solver.diagnostics["iterations"],
                           next_inner_tolerance=solver.tol))
        if verbose:
            print(f"[UV round {rd}] W={len(patterns)} gap/n={gap/n:.6g} "
                  f"pool={enum.stats.emitted_closed} add={len(candidates)} "
                  f"u={enum.stats.screened_u} v={enum.stats.pruned_v}", flush=True)
        if converged:
            reason = "full_gap_and_restricted_solver"
            break
        if reason == "no_candidates_and_gap_stagnation" or rd == geo.max_rounds:
            break
        old_count = len(patterns)
        patterns.extend(candidates)
        warm = m.ENParams(params.b, np.r_[params.w, np.zeros(len(patterns)-old_count)])
    return m.FitResult(patterns, params, eta, center, gap, final_radius,
                       time.perf_counter()-started, rounds, stats, [0]*len(stats),
                       converged=converged, termination_reason=reason,
                       kkt_residual=float("inf"), audit_stats_per_round=[],
                       pattern_space=geo.pattern_space,
                       implementation_version="experimental-uv-only-fused-v1")
