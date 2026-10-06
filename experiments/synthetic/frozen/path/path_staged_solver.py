"""Bounded-working-set Poisson regularization paths with safe cross-lambda transfer.

A fused traversal bounds the correlation of its raw residual over the ENTIRE
allowed dictionary: explicitly visited patterns plus signed-mass bounds for
pruned subtrees. Ray scaling is included in the saved bound. At a new lambda,
shrinking that same vector so its bound is <= lambda makes every EN conjugate
term zero. This is a new target-lambda dual certificate, not reuse of an old
radius or reuse of an old inactive set. All radii use the new primal objective.

Only model.py and uv_only_solver.py are runtime source dependencies. The
working-set schedule is copied from uv_staged_solver.py; that file is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

import model as m
from uv_only_solver import fused_traversal, reference_from_traversal

VERSION = "path-staged-certified-v2"


def _fingerprint(transactions, y, min_support, max_len, pattern_space):
    h = hashlib.sha256()
    h.update(json.dumps([sorted(set(map(int, row))) for row in transactions],
                        separators=(",", ":")).encode())
    h.update(np.asarray(y, dtype="<f8").tobytes())
    h.update(json.dumps([int(min_support), int(max_len), pattern_space],
                        separators=(",", ":")).encode())
    return h.hexdigest()


def _previous_state(previous, fingerprint, arm):
    if previous is None:
        return None
    state = previous.get("state", previous)
    if state.get("version") != VERSION or state.get("fingerprint") != fingerprint:
        raise ValueError("Previous state must have identical data, row order and dictionary")
    if state.get("arm") != arm:
        raise ValueError("Each path arm must use only its own previous state")
    if not state.get("converged"):
        raise ValueError("A nonconverged point cannot initialize the next path point")
    if state["transfer_certificate"].get("fingerprint") != fingerprint:
        raise ValueError("Transfer certificate fingerprint mismatch")
    if len(state["patterns"]) != len(state["w"]):
        raise ValueError("Previous working set and coefficients disagree")
    return state


def lambda_max_exact(transactions, y, min_support, max_len, pattern_space="closed",
                     propagate_support=True):
    """Exact training-dictionary maximum, streamed with valid subtree bounds.

    At the intercept-only optimum b=log(mean(y)), the l1 threshold is
    max_P |z_P.T @ (y-mean(y))|; the elastic-net quadratic term vanishes.
    Pruning uses an upper bound over all descendants, never a fitted model or
    planted truths. The returned maximum is not merely a conservative start.
    An empty/constant-response dictionary returns zero (caller handles that
    degenerate path explicitly). There is no time/node cap or partial result.
    """
    started = time.perf_counter()
    y = m._responses(y, require_positive_sum=True)
    if len(transactions) != len(y):
        raise ValueError("One response per transaction is required")
    tids, items = m.PoissonEN_LCM_SPP._transactions_to_item_tidsets(transactions)
    enum = m.LCMEnumerator(tids, items, len(y), min_support, max_len,
                           pattern_space, propagate_support)
    alpha = y-float(y.mean())
    largest, argmax, evaluated = 0., None, 0
    def prune(tid):
        upper = m._subtree_dual_mass_bound(tid, alpha)
        if upper+m._roundoff_pad(upper) < largest:
            enum.stats.pruned_dual += 1
            return True
        return False
    for pattern in enum.iter_patterns(prune):
        correlation = abs(float(alpha[np.asarray(pattern.tidset, np.int32)].sum()))
        evaluated += 1
        if correlation > largest:
            largest, argmax = correlation, list(pattern.itemset)
    return dict(lambda_max=largest, exact=True, argmax=argmax, patterns_evaluated=evaluated,
                fingerprint=_fingerprint(transactions, y, min_support, max_len, pattern_space),
                stats=asdict(enum.stats), seconds=time.perf_counter()-started,
                intercept=float(np.log(y.mean())), role="training-only path calibration")


def node_score(pattern, center, radius, n, second=None, second_radius=None):
    tid = np.asarray(pattern.tidset, np.int32)
    if second is not None:
        return m.theorem5_u0_binary_tidset(tid, center, radius, second, second_radius, n)
    return abs(float(center[tid].sum())) + m._radius_term(
        radius, m._centered_norm_from_support(len(tid), n))


def fit_path_point(transactions, y, config, previous=None, mode="PATH", verbose=False,
                   initial_tol=1e-3, tolerance_factor=.1, ready_fraction=.8,
                   same_round_selection=False, calibration=None):
    """Solve one target lambda, with independent BASE/LOCAL/PATH state.

    BASE uses no screening. LOCAL uses only references constructed at this
    target. PATH fixes a safely rescaled previous-lambda reference as its
    second ball throughout this target. All arms warm-start the primal using
    their own previous solution. No pruned tree state crosses lambda values.
    With exact lambda-max calibration, BASE/LOCAL use the scaled intercept-only
    residual at every target; PATH uses it when no previous fitted point exists.
    This reference is ready before the FIRST dictionary traversal at that target.
    The analytic solution at/above lambda_max performs no u/v screening.
    Return ``fit`` (ordinary FitResult), JSON-safe ``state`` for resumption,
    ``transfer_certificate`` and ``path_transfer`` accounting.
    """
    if initial_tol <= 0 or not 0 < tolerance_factor < 1 or not 0 < ready_fraction < 1:
        raise ValueError("Invalid refinement policy")
    if mode not in ("BASE", "LOCAL", "PATH"):
        raise ValueError("mode must be BASE, LOCAL, or PATH")
    arm = mode
    config = dict(config, enum_mode_single="none" if arm == "BASE" else "both",
                  enum_mode_two="none" if arm == "BASE" else "both",
                  do_ws_screen=arm != "BASE", use_dynamic_ref=arm != "BASE")
    started = time.perf_counter()
    geo = m.PoissonEN_LCM_SPP(**config)
    if geo.radius_mode != "certified":
        raise ValueError("This driver requires certified radii")
    y = m._responses(y, require_positive_sum=True)
    if len(transactions) != len(y):
        raise ValueError("One response per transaction is required")
    n = len(y)
    tids, items = geo._transactions_to_item_tidsets(transactions)
    calc = m.PoissonENGap(geo.lam, geo.kappa)
    solver = m.PoissonElasticNetSolver(geo.lam, geo.kappa, geo.solver_max_iter,
                                      max(initial_tol, geo.solver_tol), geo.eta_clip)
    def enumerator():
        return m.LCMEnumerator(tids, items, n, geo.min_support, geo.max_len,
                               geo.pattern_space, geo.propagate_support)

    fingerprint = _fingerprint(transactions, y, geo.min_support, geo.max_len, geo.pattern_space)
    old_state = _previous_state(previous, fingerprint, arm)
    if calibration is not None:
        if calibration.get("fingerprint") != fingerprint or calibration.get("exact") is not True:
            raise ValueError("Initialization requires exact calibration for this training dictionary")
        maximum = float(calibration["lambda_max"])
        if not np.isfinite(maximum) or maximum < 0:
            raise ValueError("Invalid lambda_max calibration")
        if old_state is None and geo.lam >= maximum:
            eta = np.full(n, np.log(y.mean()))
            params = m.ENParams(float(eta[0]), np.zeros(0))
            center = y-float(y.mean())
            _, _, gap = calc.gap(y, eta, params.w, center, full_penalty=0.)
            radius = geo._radius(y, center, np.exp(eta), gap)
            fit = m.FitResult([], params, eta, center, gap, radius, time.perf_counter()-started,
                              [], [], [], converged=bool(gap/n <= geo.gap_tol),
                              termination_reason="exact_lambda_max_intercept_certificate",
                              kkt_residual=0., pattern_space=geo.pattern_space,
                              implementation_version=VERSION)
            transfer = dict(used=False, source_lambda=None, target_lambda=geo.lam,
                            scale=None, source_global_bound=None, target_global_bound=None,
                            target_full_penalty=0., tree_traversals=0, seconds=0.,
                            null_initialization=True, calibration_lambda_max=maximum,
                            calibration_cost_included=False,
                            initial_reference_source="analytic_null_solution",
                            initial_reference_global_bound=maximum,
                            initial_reference_scale=1.,
                            initial_reference_used_for_screening=False,
                            first_traversal=None,
                            analytic_zero_solution_is_uv_pruning=False)
            return _package(fit, fingerprint, arm, geo.lam, maximum, 0., transfer)
    center, penalty = np.zeros(n), 0.
    dual = calc.dual(center, y, full_penalty=penalty)
    second, second_dual = None, None
    center_bound = 0.
    patterns, warm = [], None
    transfer = dict(used=False, source_lambda=None, target_lambda=geo.lam,
                    scale=None, source_global_bound=None, target_global_bound=None,
                    target_full_penalty=None, tree_traversals=0, seconds=0.,
                    null_initialization=False, initial_reference_source="zero_reference",
                    initial_reference_global_bound=0., initial_reference_scale=None,
                    initial_reference_used_for_screening=arm != "BASE",
                    first_traversal=None,
                    analytic_zero_solution_is_uv_pruning=False)
    if calibration is not None and (old_state is None or arm != "PATH"):
        # This reference is certified for the target lambda before the first
        # search. Its source is the analytically optimal intercept-only model,
        # not a fitted pattern model or a target-lambda KKT search. Every allowed
        # correlation is bounded by scale*lambda_max, so the full EN conjugate
        # sum is exactly zero for either kappa=0 or kappa>0.
        initial_tick = time.perf_counter()
        scale = min(1., geo.lam/(maximum+m._roundoff_pad(maximum)))
        center, center_bound = scale*(y-float(y.mean())), scale*maximum
        dual = calc.dual(center, y, full_penalty=0.)
        if not np.isfinite(dual):
            raise ValueError("Calibrated cold reference is outside the Poisson/intercept domain")
        warm = m.ENParams(float(np.log(y.mean())), np.zeros(0))
        transfer.update(initial_reference_source="calibrated_intercept_residual",
                        initial_reference_global_bound=center_bound,
                        initial_reference_scale=scale,
                        initial_reference_full_penalty=0.,
                        initial_reference_norm=float(np.linalg.norm(center)),
                        calibration_lambda_max=maximum, calibration_cost_included=False,
                        initial_reference_seconds=time.perf_counter()-initial_tick)
    if old_state is not None:
        patterns = [m.Pattern(tuple(p["itemset"]), tuple(p["tidset"]))
                    for p in old_state["patterns"]]
        warm = m.ENParams(float(old_state["b"]), np.asarray(old_state["w"], float))
        if arm == "PATH":
            transfer_tick = time.perf_counter()
            cert = old_state["transfer_certificate"]
            old_alpha = np.asarray(cert["alpha"], float)
            bound = float(cert["global_correlation_bound"])
            if old_alpha.shape != (n,) or not np.isfinite(old_alpha).all() or not np.isfinite(bound) or bound < 0:
                raise ValueError("Invalid previous global correlation certificate")
            scale = min(1., geo.lam/(bound+m._roundoff_pad(bound))) if bound > 0 else 1.
            center, center_bound = scale*old_alpha, scale*bound
            dual = calc.dual(center, y, full_penalty=0.)
            if not np.isfinite(dual):
                raise ValueError("Transferred reference is outside the Poisson/intercept domain")
            second, second_dual = center.copy(), dual
            transfer.update(used=True, source_lambda=cert["lambda"], scale=scale,
                            source_global_bound=bound, target_global_bound=center_bound,
                            target_full_penalty=0., seconds=time.perf_counter()-transfer_tick,
                            initial_reference_source="previous_lambda_transfer",
                            initial_reference_global_bound=center_bound,
                            initial_reference_scale=scale,
                            initial_reference_norm=float(np.linalg.norm(center)))
    times, stats, removals = [], [], []
    screen_phase = False
    previous_primal = None
    converged, reason = False, "max_rounds"
    for rd in range(1, geo.max_rounds+1):
        tick_round = time.perf_counter()
        X = m.build_X_from_patterns(n, patterns)
        tick = time.perf_counter()
        used_tol = solver.tol
        params = solver.fit(X, y, warm)
        solver_seconds = time.perf_counter()-tick
        eta = params.b+X@params.w
        mu = np.exp(eta)
        raw = calc.build_dual_feasible(y, eta)
        P, _, pre_gap = calc.gap(y, eta, params.w, center, full_penalty=penalty)
        before_radius = geo._radius(y, center, mu, pre_gap)
        before_second_radius = None if second is None else geo._radius(
            y, second, mu, max(0., P-second_dual))
        mode = geo.enum_mode_two if second is not None else geo.enum_mode_single
        enum = enumerator()
        tick = time.perf_counter()
        scan = fused_traversal(enum, raw, center, before_radius, geo.lam,
                               {p.itemset for p in patterns}, geo.top_k_add, mode,
                               second, before_second_radius)
        mine_seconds = time.perf_counter()-tick
        if rd == 1:
            transfer["first_traversal"] = dict(
                source=transfer["initial_reference_source"], mode=mode,
                nodes_visited=enum.stats.nodes_visited,
                screened_u=enum.stats.screened_u, pruned_v=enum.stats.pruned_v,
                radius=float(before_radius), center_norm=float(np.linalg.norm(center)),
                global_bound=float(center_bound))
        stats.append(enum.stats)
        tick = time.perf_counter()
        fresh, fresh_penalty, scale = reference_from_traversal(calc, y, raw, scan)
        fresh_dual = calc.dual(fresh, y, full_penalty=fresh_penalty)
        old_center, old_dual = center.copy(), dual
        if fresh_dual >= dual:
            center, penalty, dual = fresh, fresh_penalty, fresh_dual
            center_bound = scale*scan["maximum"]
        if arm == "LOCAL" or (arm == "PATH" and old_state is None):
            second, second_dual = old_center, old_dual
        P, dual, gap = calc.gap(y, eta, params.w, center, full_penalty=penalty)
        radius = geo._radius(y, center, mu, gap)
        second_radius = None if second is None else geo._radius(y, second, mu, max(0., P-second_dual))
        gap_seconds = time.perf_counter()-tick
        # Do not force a screening demonstration after already meeting the target.
        converged = bool(gap/n <= geo.gap_tol and solver.diagnostics["kkt"] <= geo.solver_tol)
        ready = radius*np.sqrt(n)/2 <= ready_fraction*geo.lam
        # Coarse stationarity can otherwise add zero columns indefinitely.
        stalled = previous_primal is not None and P >= previous_primal - (
            64*np.finfo(float).eps*max(1., abs(P), abs(previous_primal)))
        candidates, keep = [], list(range(len(patterns)))
        selection_seconds = ws_seconds = 0.
        selection_stats = None
        filtered_candidates = nonzero_removed = 0
        entering = False
        if not converged and rd < geo.max_rounds:
            candidates = scan["candidates"]
            entering = bool(ready and not screen_phase)
            screen_phase = screen_phase or ready
            if entering and mode != "none" and same_round_selection:
                tick = time.perf_counter()
                selected = enumerator()
                candidates = selected.dfs_with_spp(
                    center, mu-y, radius, geo.lam, {p.itemset for p in patterns},
                    geo.top_k_add, second, second_radius, mode)
                selection_seconds = time.perf_counter()-tick
                selection_stats = selected.stats
                stats.append(selected.stats)
            if screen_phase and mode in ("uscreen", "both"):
                tick = time.perf_counter()
                if geo.do_ws_screen:
                    keep = []
                    for j, pattern in enumerate(patterns):
                        bound = node_score(pattern, center, radius, n, second, second_radius)
                        if bound + m._roundoff_pad(bound) < geo.lam:
                            nonzero_removed += int(params.w[j] != 0)
                        else:
                            keep.append(j)
                retained = []
                for pattern in candidates:
                    bound = node_score(pattern, center, radius, n, second, second_radius)
                    if bound + m._roundoff_pad(bound) < geo.lam:
                        filtered_candidates += 1
                    else:
                        retained.append(pattern)
                candidates = retained
                ws_seconds = time.perf_counter()-tick
            if screen_phase or not candidates or stalled:
                if solver.tol > geo.solver_tol:
                    solver.tol = max(geo.solver_tol, tolerance_factor*solver.tol)
                elif not candidates:
                    solver.tol = max(1e-13, tolerance_factor*solver.tol)
        removed = len(patterns)-len(keep)
        removals.append(removed)
        times.append(dict(round=rd, m=len(patterns), solver_time=solver_seconds,
                          gap_time=gap_seconds, mine_time=mine_seconds+selection_seconds,
                          ws_time=ws_seconds, round_total=time.perf_counter()-tick_round,
                          gap=gap, pre_search_gap=pre_gap, r=before_radius,
                          updated_r=radius, ref2_radius=second_radius, kkt=None,
                          restricted_kkt=solver.diagnostics["kkt"], full_audit=False,
                          candidate_source="path-staged-"+arm,
                          path_reference_fixed=bool(transfer["used"]),
                          center_global_correlation_bound=center_bound,
                          candidates_added=len(candidates),
                          candidate_pool=enum.stats.emitted_closed,
                          reference_scale=scale, raw_bound_checks=scan["raw_bound_checks"],
                          correlation_evaluations=scan["correlation_evaluations"],
                          skipped_raw_bound=scan["skipped_bound"],
                          used_inner_tolerance=used_tol, next_inner_tolerance=solver.tol,
                          primal_stalled=bool(stalled),
                          stagnation_refinement=bool(stalled and solver.tol < used_tol),
                          ready=bool(ready), entered_screen_phase=entering,
                          same_round_selection_seconds=selection_seconds,
                          same_round_selection=bool(same_round_selection),
                          selection_nodes=0 if selection_stats is None else selection_stats.nodes_visited,
                          u_fused=enum.stats.screened_u, v_fused=enum.stats.pruned_v,
                          u_selection=0 if selection_stats is None else selection_stats.screened_u,
                          v_selection=0 if selection_stats is None else selection_stats.pruned_v,
                          postfiltered_candidates=filtered_candidates,
                          ws_removed=removed, nonzero_ws_removed=nonzero_removed,
                          solver_iterations=solver.diagnostics["iterations"]))
        if verbose:
            print(f"[path-staged {arm} {rd}] W={len(patterns)} tol={used_tol:.2g} gap/n={gap/n:.5g} "
                  f"r={before_radius:.5g}->{radius:.5g} ready={ready} stalled={stalled} "
                  f"add={len(candidates)} drop={removed} "
                  f"u={enum.stats.screened_u+(0 if selection_stats is None else selection_stats.screened_u)} "
                  f"v={enum.stats.pruned_v+(0 if selection_stats is None else selection_stats.pruned_v)}", flush=True)
        if converged:
            reason = "full_gap_and_requested_inner_tolerance"
            break
        if rd == geo.max_rounds:
            break
        patterns = [patterns[j] for j in keep]+candidates
        warm = m.ENParams(params.b, np.r_[params.w[keep], np.zeros(len(candidates))])
        previous_primal = P
    fit = m.FitResult(patterns, params, eta, center, gap, radius, time.perf_counter()-started,
                      times, stats, removals, converged=converged, termination_reason=reason,
                      kkt_residual=float("inf"), audit_stats_per_round=[],
                      pattern_space=geo.pattern_space, implementation_version=VERSION)
    return _package(fit, fingerprint, arm, geo.lam, center_bound, penalty, transfer)


def _package(fit, fingerprint, arm, lam, center_bound, penalty, transfer):
    certificate = dict(version=VERSION, fingerprint=fingerprint, lambda_=lam,
                       global_correlation_bound=float(center_bound), alpha=fit.alpha.tolist(),
                       full_penalty=float(penalty))
    certificate["lambda"] = certificate.pop("lambda_")
    # All arms transfer the same exact-nonzero primal policy. Zero columns are
    # rediscoverable at every target; the complete FitResult remains auditable.
    active = np.flatnonzero(fit.params.w != 0.)
    state = dict(version=VERSION, fingerprint=fingerprint, arm=arm,
                 converged=fit.converged,
                 patterns=[dict(itemset=list(fit.patterns[j].itemset),
                                tidset=list(fit.patterns[j].tidset)) for j in active],
                 b=float(fit.params.b), w=fit.params.w[active].tolist(),
                 transfer_certificate=certificate)
    return dict(fit=fit, state=state, transfer_certificate=certificate, path_transfer=transfer)
