from model import *

# ══════════════════════════════════════════════════════
# § 1  Poisson Pseudo-R²（调用 cell0 的 poisson_deviance）
# ══════════════════════════════════════════════════════

def poisson_pseudo_r2(y: np.ndarray, mu: np.ndarray) -> float:
    mu0 = np.full_like(y, np.mean(y))
    D_null = poisson_deviance(y, mu0)
    return 0.0 if D_null <= 1e-30 else float(1.0 - poisson_deviance(y, mu) / D_null)

# ══════════════════════════════════════════════════════
# § 2  特征矩阵构建（跨 fold/holdout 的通用版）
# ══════════════════════════════════════════════════════

def build_X(n, patterns):
    X = np.zeros((n, len(patterns)), float)
    for j, p in enumerate(patterns): X[np.fromiter(p.tidset, np.int32), j] = 1.0
    return X

def build_X_trans(transactions, patterns):
    n, m = len(transactions), len(patterns)
    X = np.zeros((n, m), float)
    ts = [set(t) for t in transactions]
    for j, p in enumerate(patterns):
        ps = set(p.itemset)
        for i in range(n):
            if ps.issubset(ts[i]): X[i, j] = 1.0
    return X

# ─────────────────────────────────────────────────────────────
# Feature Recovery (Extended)
# ─────────────────────────────────────────────────────────────

def _norm(p): return tuple(sorted(p))

def _is_sub(a, b):
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]: i += 1; j += 1
        elif a[i] > b[j]: j += 1
        else: return False
    return i == len(a)

def _closure_patt(patt, item_tidsets, items_sorted, n):
    patt = _norm(patt)
    if not patt: return patt
    tid = None
    for it in patt:
        if it not in item_tidsets: return patt
        tid = item_tidsets[it] if tid is None else intersect_sorted(tid, item_tidsets[it])
        if tid.size == 0: return patt
    extra = [k for k in items_sorted if k not in set(patt) and
             LCMEnumerator._is_subset_tid(tid, item_tidsets[k])]
    return _norm(tuple(sorted(list(patt) + extra)))

def _prf(hit, pc, tc):
    p = hit / max(1, pc); r = hit / max(1, tc)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return float(p), float(r), float(f)


# ══════════════════════════════════════════════════════
# § 3  模式识别评估（exact / closure / superset recall）
# ══════════════════════════════════════════════════════

def eval_feature_recovery_extended(res: FitResult, true_patterns, transactions,
                                   eps=1e-8, topk_list=(10, 20, 50)):
    item_tidsets, items_sorted = PoissonEN_LCM_SPP._transactions_to_item_tidsets(transactions)
    n = len(transactions)
    true_norm  = [_norm(p) for p in true_patterns]
    true_set   = set(true_norm)
    true_clos  = [_closure_patt(p, item_tidsets, items_sorted, n) for p in true_norm]
    true_clos_set = set(true_clos)
    m = len(res.patterns)

    if m == 0:
        out = {}
        for px in ("exact", "closure", "superset"):
            for k2 in ("precision", "recall", "f1"): out[f"{px}_{k2}"] = 0.0
            for k in topk_list: out[f"{px}_top{k}_recall"] = 0.0
        for k3 in ("num_selected_nz", "num_hits_exact", "num_hits_closure", "num_hits_superset"):
            out[k3] = 0.0
        return out

    nz = [j for j in range(m) if abs(res.params.w[j]) > eps]
    pred_set = {_norm(res.patterns[j].itemset) for j in nz}

    h_e = len(pred_set & true_set)
    h_c = len(pred_set & true_clos_set)
    ps  = sorted(pred_set)
    h_s = sum(1 for tp in true_norm if any(_is_sub(tp, sp) for sp in ps))

    ep, er, ef = _prf(h_e, len(pred_set), len(true_set))
    cp, cr, cf = _prf(h_c, len(pred_set), len(true_clos_set))
    sr = h_s / max(1, len(true_norm)); sp2 = h_s / max(1, len(pred_set))
    sf = 0.0 if sp2 + sr == 0 else 2 * sp2 * sr / (sp2 + sr)

    out = {"exact_precision": ep, "exact_recall": er, "exact_f1": ef,
           "closure_precision": cp, "closure_recall": cr, "closure_f1": cf,
           "superset_precision": float(sp2), "superset_recall": float(sr), "superset_f1": float(sf),
           "num_selected_nz": float(len(pred_set)),
           "num_hits_exact": float(h_e), "num_hits_closure": float(h_c), "num_hits_superset": float(h_s)}

    order = np.argsort(-np.abs(res.params.w))
    for k in topk_list:
        top_set = {_norm(res.patterns[j].itemset) for j in order[:min(k, m)]}
        out[f"exact_top{k}_recall"]   = float(len(top_set & true_set)      / max(1, len(true_set)))
        out[f"closure_top{k}_recall"] = float(len(top_set & true_clos_set) / max(1, len(true_clos_set)))
        ts2 = sorted(top_set)
        out[f"superset_top{k}_recall"] = float(
            sum(1 for tp in true_norm if any(_is_sub(tp, sp) for sp in ts2)) / max(1, len(true_norm)))
    return out

def print_recovery_report(acc: Dict, label: str = ""):
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"  Feature Recovery{tag}")
    print(f"{'='*60}")
    print(f"  Nonzero patterns selected: {int(acc['num_selected_nz'])}")
    print(f"  {'Metric':<20}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  "
          f"{'top10R':>7}  {'top20R':>7}  {'top50R':>7}")
    print(f"  {'-'*65}")
    for px in ("exact", "closure", "superset"):
        p = acc.get(f"{px}_precision", 0)
        r = acc.get(f"{px}_recall",    0)
        f = acc.get(f"{px}_f1",        0)
        t10 = acc.get(f"{px}_top10_recall", 0)
        t20 = acc.get(f"{px}_top20_recall", 0)
        t50 = acc.get(f"{px}_top50_recall", 0)
        print(f"  {px:<20}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {t10:>7.3f}  {t20:>7.3f}  {t50:>7.3f}")

def eval_train_metrics(model, res, transactions, y):
    y = np.asarray(y, float)
    X = build_X_trans(transactions, res.patterns) if res.patterns else np.zeros((len(transactions), 0), float)
    eta = np.clip(res.params.b + (X @ res.params.w if X.shape[1] else 0.0), -model.eta_clip, model.eta_clip)
    mu = np.exp(eta)
    return {"NLL": float(np.sum(mu - y * eta)), "Deviance": poisson_deviance(y, mu),
            "PseudoR2": poisson_pseudo_r2(y, mu),
            "MAE": float(np.mean(np.abs(y - mu))), "RMSE": float(np.sqrt(np.mean((y - mu)**2)))}

# ─────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────

def run_ablation(transactions, y, true_patterns, lam, model_factory_base, use_two_ref=True):
    configs = [
        ("BASE(LCM only)", False, "none"),
        ("+v",             False, "vprune"),
        ("+u",             False, "uscreen"),
        ("+v+u",           False, "both"),
        ("+v+u+WS",        True,  "both"),
    ]
    base_res = model_factory_base(lam, False, "none").fit(transactions, y, verbose=False)
    base_sm  = summarize_pruning(base_res)
    rows = [("BASE", base_sm, eval_feature_recovery_extended(base_res, true_patterns, transactions), None)]

    for name, do_ws, mode in configs[1:]:
        res = model_factory_base(lam, do_ws, mode).fit(transactions, y, verbose=False)
        sm  = summarize_pruning(res); acc = eval_feature_recovery_extended(res, true_patterns, transactions)
        speed = {"speedup_total": base_sm["time_total_sec"] / max(1e-12, sm["time_total_sec"])}
        rows.append((name, sm, acc, speed))

    if use_two_ref:
        res_hi = model_factory_base(lam * 1.2, True, "both").fit(transactions, y, verbose=False)
        res2   = model_factory_base(lam, True, "both").fit(transactions, y, verbose=False,
                                                           ref2=(res_hi.alpha, res_hi.r))
        sm2  = summarize_pruning(res2); acc2 = eval_feature_recovery_extended(res2, true_patterns, transactions)
        rows.append(("+v+u+WS+2ref", sm2, acc2,
                     {"speedup_total": base_sm["time_total_sec"] / max(1e-12, sm2["time_total_sec"])}))

    print(f"\n{'='*120}")
    print(f"  Ablation @ lambda={lam}")
    print(f"{'='*120}")
    hdr = f"  {'Name':<16} {'Emit':>7} {'Visit':>7} {'Hard%':>6} {'v%':>5} {'u%':>5} {'WS_rm':>6} {'T_tot':>7} {'T_mn':>7} {'Spdup':>6} {'ExacR':>6} {'ClosR':>6} {'SupR':>6} {'SupT10':>7}"
    print(hdr); print("  " + "-"*110)
    for name, sm, acc, speed in rows:
        sp = speed["speedup_total"] if speed else 1.0
        print(f"  {name:<16} {int(sm['emitted_closed']):>7} {int(sm['nodes_visited']):>7} "
              f"{sm['hard_prune_rate']*100:>5.1f}% {sm['v_prune_rate']*100:>4.1f}% "
              f"{sm['u_screen_rate']*100:>4.1f}% {int(sm['ws_removed_total']):>6} "
              f"{sm['time_total_sec']:>6.3f}s {sm['time_mine_sec']:>6.3f}s "
              f"{sp:>5.2f}x {acc['exact_recall']:>6.3f} {acc['closure_recall']:>6.3f} "
              f"{acc['superset_recall']:>6.3f} {acc.get('superset_top10_recall',0):>7.3f}")

# ══════════════════════════════════════════════════════
# § 4  数据诊断（var/mean、零比例、support检查）
# ══════════════════════════════════════════════════════

def data_diagnosis(transactions, y, true_patterns=None):
    import math as _m
    if true_patterns is None: true_patterns = TRUE_PATTERNS
    n = len(transactions); my = float(np.mean(y)); vy = float(np.var(y))
    print(f"\n{'='*55}"); print("  数据诊断"); print(f"{'='*55}")
    print(f"  n={n}  mean(y)={my:.4f}  var(y)={vy:.4f}")
    disp = vy / max(my, 1e-9)
    print(f"  var/mean = {disp:.4f}  {'✓ Poisson 良好' if disp < 1.6 else '⚠ 超散布'}")
    zr = float(np.mean(y == 0)); pz = _m.exp(-my)
    print(f"  零值比例 = {zr:.3f}  (Poisson理论 = {pz:.3f})  {'✓' if abs(zr-pz)<0.1 else '⚠ 零膨胀'}")
    print(f"  事务平均长度 = {np.mean([len(t) for t in transactions]):.1f}")
    print(f"\n  模式 Support:")
    for p in true_patterns:
        s = sum(1 for t in transactions if set(p).issubset(t))
        print(f"    {str(p):<22}  sup={s:5d}  rate={s/n:.2%}")
    print(f"{'='*55}")
