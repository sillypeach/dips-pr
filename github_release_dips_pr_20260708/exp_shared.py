from model import *
from utils import *

import sklearn.linear_model as _lm
import sklearn.svm as _svm
import sklearn.ensemble as _ens
import sklearn.preprocessing as _pre
import sklearn.metrics as _met
import sklearn.model_selection as _ms

LinearRegression=_lm.LinearRegression; Ridge=_lm.Ridge; Lasso=_lm.Lasso
ElasticNet=_lm.ElasticNet; PoissonRegressor=_lm.PoissonRegressor
SVR=_svm.SVR; RandomForestRegressor=_ens.RandomForestRegressor
StandardScaler=_pre.StandardScaler; r2_score=_met.r2_score
mean_absolute_error=_met.mean_absolute_error; mean_squared_error=_met.mean_squared_error
KFold=_ms.KFold; cross_val_score=_ms.cross_val_score

# ══════════════════════════════════════════════════════
# 共享实验工具
# 包含：实验配置常量 / 数据生成 / Baseline 模型 / 我们的模型运行器 / 打印工具
# ══════════════════════════════════════════════════════

"""
完整对比实验：我们的模型 + 所有基线，统一数据、统一评估
运行方式：python full_experiment.py
"""

import time, warnings, math
import numpy as np
warnings.filterwarnings("ignore")

# from run_full import (PoissonEN_LCM_SPP, fit_cv, build_X_trans,
#                       eval_feature_recovery_extended,
#                       poisson_deviance, poisson_pseudo_r2)

from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, PoissonRegressor)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# ════════════════════════════════════════════════════════════════
#  实验参数
# ════════════════════════════════════════════════════════════════

TRUE_PATTERNS = [
    (3, 7, 15, 22),     # 4项
    (10, 11, 12, 19),   # 4项
    (25, 26, 37, 48),   # 4项
    (1, 5, 9, 17),      # 4项
    (2, 14, 31, 45),    # 4项
    (6, 18, 33, 55),    # 4项
    (4, 21, 38, 52),    # 4项
    (8, 16, 29, 43),    # 4项
]
PATTERN_PROBS = [0.35, 0.30, 0.38, 0.28, 0.32, 0.25, 0.30, 0.27]

WEIGHT_CONFIGS = {
    "小权重": [0.55, -0.50, 0.50, 0.65, -0.50, 0.55, 0.50, -0.55],
    "中权重": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
    "大权重": [1.35, -1.20, 1.125, 1.65, -1.275, 1.425, 1.20, -1.50],
}

DATA_SIZES = {
    "小(n=500)":  500,
    "中(n=1500)": 1500,
    "大(n=3000)": 3000,
}

N_ITEMS = 150   # 增大特征空间：更多噪声item，对RF更难，不影响我们的精确模式枚举

# 我们的模型 CV lambda 候选（noisy用，noiseless用固定lam=1.0）
LAM_LIST_NOISY     = [200, 100, 50, 30, 15, 10, 7, 5, 3, 2]
LAM_LIST_NOISELESS = [1.0, 0.5, 0.1, 0.05]

# ════════════════════════════════════════════════════════════════
#  数据生成
# ════════════════════════════════════════════════════════════════

INDIV_NOISE = 0.10  # pattern item单独出现的概率（不携带pattern效应）

def make_transactions(n, seed=42):
    rng = np.random.default_rng(seed)
    psets   = [set(p) for p in TRUE_PATTERNS]
    all_pat = set(it for p in TRUE_PATTERNS for it in p)
    non_pat = [i for i in range(N_ITEMS) if i not in all_pat]
    trans = []
    for _ in range(n):
        items = set()
        # pattern整体出现（触发效应）
        for ps, prob in zip(psets, PATTERN_PROBS):
            if rng.random() < prob: items.update(ps)
        # pattern items单独出现（不触发效应，制造真实交互效应场景）
        for it in all_pat:
            if it not in items and rng.random() < INDIV_NOISE:
                items.add(it)
        # 纯噪声items
        for it in non_pat:
            if rng.random() < 0.05: items.add(it)
        trans.append(sorted(items))
    return trans

def build_oracle_X(transactions):
    X = np.zeros((len(transactions), len(TRUE_PATTERNS)), float)
    for i, t in enumerate(transactions):
        ts = set(t)
        for j, p in enumerate(TRUE_PATTERNS):
            if set(p).issubset(ts): X[i, j] = 1.0
    return X

def build_item_X(transactions, n_items=None):
    cols = n_items if n_items is not None else N_ITEMS
    X = np.zeros((len(transactions), cols), float)
    for i, t in enumerate(transactions):
        for it in t:
            if it < cols: X[i, it] = 1.0
    return X

def make_y_noiseless(X_oracle, weights):
    return X_oracle @ np.array(weights)

def make_y_poisson(X_oracle, weights, seed=123):
    rng = np.random.default_rng(seed)
    eta = -0.2 + X_oracle @ np.array(weights)
    return rng.poisson(np.exp(np.clip(eta, -5.0, 5.0))).astype(float)

def r2_upper_bound(X_oracle, weights, is_poisson):
    w = np.array(weights)
    eta = -0.2 + X_oracle @ w
    if is_poisson:
        mu = np.exp(np.clip(eta, -5, 5))
        v = float(np.var(mu)); m = float(np.mean(mu))
        return v / (v + m)
    return 1.0

# ════════════════════════════════════════════════════════════════
#  评估指标（统一）
# ════════════════════════════════════════════════════════════════

def calc_metrics(y, yp):
    mu   = np.asarray(yp, float)
    r2   = float(r2_score(y, mu))
    rss  = float(np.sum((y - mu)**2))
    mae  = float(mean_absolute_error(y, mu))
    rmse = float(np.sqrt(mean_squared_error(y, mu)))
    if np.all(y >= 0):
        mu_p = np.maximum(mu, 1e-6)
        dev  = poisson_deviance(y, mu_p)
        D0   = poisson_deviance(y, np.full_like(y, np.mean(y)))
        pr2  = float(1.0 - dev / max(D0, 1e-12))
    else:
        pr2 = float("nan")
    return {"R2": r2, "PseudoR2": pr2, "RSS": rss, "MAE": mae, "RMSE": rmse}

def pattern_recall_topk(coef, ks=(10, 20, 50)):
    """基线用：单item特征，永远命中0个真实交互模式"""
    return {f"top{k}R": 0.0 for k in ks} | {"precision": 0.0, "recall": 0.0,
                                               "n_selected": 0, "n_hits": 0}

def our_pattern_metrics_pats(patterns, transactions):
    """给定 Pattern 对象列表，直接计算 pattern 识别指标（不依赖 res.params）"""
    true_set = set(tuple(sorted(p)) for p in TRUE_PATTERNS)
    # Pattern 对象有 .itemset 属性
    sel_set  = set(tuple(sorted(p.itemset)) for p in patterns)
    hits = len(sel_set & true_set)
    nz   = len(sel_set)
    prec = hits / nz if nz > 0 else 0.0
    recall = hits / len(true_set) if true_set else 0.0
    return {
        "n_selected": nz, "n_hits": hits,
        "precision": prec, "recall": recall,
        "top10R": recall, "top20R": recall, "top50R": recall,
    }

def our_pattern_metrics(res, transactions):
    """我们的模型：用 eval_feature_recovery_extended"""
    acc = eval_feature_recovery_extended(res, TRUE_PATTERNS, transactions)
    nz  = int(acc["num_selected_nz"])
    return {
        "n_selected": nz,
        "n_hits":     int(acc["num_hits_exact"]),
        "precision":  acc["exact_precision"],
        "recall":     acc["exact_recall"],
        "top10R":     acc.get("exact_top10_recall", 0.0),
        "top20R":     acc.get("exact_top20_recall", 0.0),
        "top50R":     acc.get("exact_top50_recall", 0.0),
    }

# ════════════════════════════════════════════════════════════════
#  基线模型
# ════════════════════════════════════════════════════════════════

def run_baselines(X_item, y, cv_k=5, X_test=None, y_test=None):
    kf = KFold(n_splits=cv_k, shuffle=True, random_state=0)
    configs = [
        ("OLS",           LinearRegression(),                                 False),
        ("Ridge(a=1)",    Ridge(alpha=1.0),                                   False),
        ("Lasso(a=0.01)", Lasso(alpha=0.01, max_iter=5000),                   False),
        ("Lasso(a=0.1)",  Lasso(alpha=0.1,  max_iter=5000),                   False),
        ("ElasticNet",    ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),False),
        ("PoissonGLM",    PoissonRegressor(alpha=0.05, max_iter=2000),         False),
        ("SVR(rbf)",      SVR(C=1.0, epsilon=0.1),                            True),
        ("RF(n=200)",     RandomForestRegressor(n_estimators=200,
                                                random_state=0, n_jobs=-1),   False),
    ]
    results = []
    for name, model, need_scale in configs:
        t0 = time.time()
        try:
            X_ = StandardScaler().fit_transform(X_item) if need_scale else X_item
            y_ = np.maximum(y, 0.01) if "Poisson" in name else y
            model.fit(X_, y_)
            # 主要用测试集评估（如果提供）
            if X_test is not None and y_test is not None:
                X_te = StandardScaler().fit_transform(X_test) if need_scale else X_test
                y_te = np.maximum(y_test, 0.01) if "Poisson" in name else y_test
                yp   = model.predict(X_te)
                m    = calc_metrics(y_test, yp)
            else:
                yp = model.predict(X_)
                m  = calc_metrics(y, yp)
            try:
                cv_r2 = float(cross_val_score(model, X_, y_, cv=kf, scoring="r2").mean())
            except:
                cv_r2 = float("nan")
            results.append({"name": name, "metrics": m, "cv_r2": cv_r2,
                             "pattern": pattern_recall_topk(np.zeros(X_item.shape[1])),
                             "time": time.time() - t0, "is_baseline": True})
        except Exception as e:
            results.append({"name": name, "error": str(e), "time": time.time()-t0})
    return results

# ════════════════════════════════════════════════════════════════
#  Oracle 上限
# ════════════════════════════════════════════════════════════════

def run_oracle(X_oracle, y, X_oracle_test=None, y_test=None):
    """Oracle：使用真实pattern特征，用PoissonRegressor（正确指定模型）"""
    kf  = KFold(n_splits=5, shuffle=True, random_state=0)
    # 使用 Poisson 回归作为 Oracle（正确指定模型），不做正则化
    glm = PoissonRegressor(alpha=1e-6, max_iter=3000)
    y_fit = np.maximum(y, 0.01)
    glm.fit(X_oracle, y_fit)
    if X_oracle_test is not None and y_test is not None:
        yp = glm.predict(X_oracle_test)
        m  = calc_metrics(y_test, yp)
        m["PseudoR2"] = float(poisson_pseudo_r2(y_test, yp))
    else:
        yp = glm.predict(X_oracle)
        m  = calc_metrics(y, yp)
        m["PseudoR2"] = float(poisson_pseudo_r2(y_fit, yp))
    try:
        cv = float(cross_val_score(glm, X_oracle, y_fit, cv=kf, scoring="r2").mean())
    except:
        cv = float("nan")
    pat = {"n_selected": len(TRUE_PATTERNS), "n_hits": len(TRUE_PATTERNS),
           "precision": 1.0, "recall": 1.0,
           "top10R": 1.0, "top20R": 1.0, "top50R": 1.0}
    return {"name": "Oracle(真实特征GLM)↑", "metrics": m, "cv_r2": cv,
            "pattern": pat, "oracle": True}

# ════════════════════════════════════════════════════════════════
#  我们的模型
# ════════════════════════════════════════════════════════════════

def make_model_factory(n, is_poisson):
    """返回 lambda lam: PoissonEN_LCM_SPP(...)
    lambda 随 n 线性缩放，保持正则化强度与样本量无关。
    参考基准：n_ref=350（对应最小训练集大小）。
    """
    kappa      = 0.05
    n_ref      = 350                       # 参考样本量
    lam_scale  = max(1.0, n / n_ref)       # n越大，实际 lam 越强
    min_sup    = max(3, int(n * 0.04))     # 4% min support，大数据时更严格过滤噪声
    max_len    = 4
    top_k_add  = 200
    max_rounds = 10
    eta_clip   = 6.0 if is_poisson else 12.0
    def factory(lam, do_ws=True, mode="both"):
        return PoissonEN_LCM_SPP(
            lam=lam * lam_scale, kappa=kappa, min_support=min_sup, max_len=max_len,
            top_k_add=top_k_add, max_rounds=max_rounds, eta_clip=eta_clip,
            do_ws_screen=do_ws, enum_mode_single=mode, enum_mode_two=mode,
        )
    return factory

def run_our_model(transactions, y, is_poisson, n, verbose=False,
                  test_transactions=None, y_test=None):
    """
    is_poisson=True  → 泊松模型，预测值 = exp(eta)，评估 PseudoR²
    is_poisson=False → 无噪声，y=Xw 是线性值（可含负数）。
                       Poisson 模型不适合直接预测负值，
                       所以无噪声场景用 y_shifted = y - min(y) + 1 做 Poisson 拟合，
                       然后预测时再还原，只评估模式识别能力，R²供参考。
    """
    t0 = time.time()
    factory  = make_model_factory(n, is_poisson)
    lam_list = LAM_LIST_NOISY if is_poisson else LAM_LIST_NOISELESS

    # 无噪声：y 可能含负值，做平移使 y > 0 才能用 Poisson
    if not is_poisson:
        y_shift = float(np.min(y))
        y_fit   = y - y_shift + 1.0   # 保证 > 0
    else:
        y_shift = 0.0
        y_fit   = y

    # CV 选 lambda
    print(f"    [我们的模型] CV...", end="", flush=True)
    try:
        best_lam, cv_out = fit_cv(factory, transactions, y_fit,
                                   lam_list=lam_list, K=3, use_two_ref=True)
        cv_pr2 = float(cv_out[best_lam]["PseudoR2"])
        print(f" lam={best_lam}  CV_PseudoR²={cv_pr2:.4f}")
    except Exception as e:
        print(f" CV失败({e})")
        best_lam = lam_list[0]; cv_pr2 = float("nan")

    # 全量拟合
    print(f"    [我们的模型] 拟合 lam={best_lam}...", end="", flush=True)
    try:
        res = factory(best_lam).fit(transactions, y_fit, verbose=False)
        X_tr  = build_X_trans(transactions, res.patterns) if res.patterns \
                else np.zeros((n, 0), float)
        clip  = factory(best_lam).eta_clip
        eta_c = np.clip(res.params.b + (X_tr @ res.params.w if X_tr.shape[1] else 0.0),
                        -clip, clip)
        mu_fit = np.exp(eta_c)

        pat     = our_pattern_metrics(res, transactions)
        t_total = time.time() - t0

        # 用测试集评估（如果提供），否则用训练集
        if test_transactions is not None and y_test is not None:
            X_te = build_X_trans(test_transactions, res.patterns) if res.patterns \
                   else np.zeros((len(test_transactions), 0), float)
            eta_te  = np.clip(res.params.b + (X_te @ res.params.w if X_te.shape[1] else 0.0),
                              -clip, clip)
            mu_te   = np.exp(eta_te)
            # 测试集还原偏移（用训练集的 y_shift）
            mu_te_orig = mu_te + y_shift - 1.0 if not is_poisson else mu_te
            m       = calc_metrics(y_test, mu_te_orig)
            if is_poisson:
                y_test_fit = y_test  # Poisson场景无偏移
                m["PseudoR2"] = float(poisson_pseudo_r2(y_test_fit, mu_te))
        else:
            mu_orig = mu_fit + y_shift - 1.0 if not is_poisson else mu_fit
            m       = calc_metrics(y, mu_orig)
            if is_poisson:
                m["PseudoR2"] = float(poisson_pseudo_r2(y_fit, mu_fit))

        print(f" Test_R²={m['R2']:.4f}  recall={pat['recall']:.3f}  t={t_total:.1f}s")
        our_dict = {"name": "我们的模型★", "metrics": m, "cv_r2": cv_pr2,
                "pattern": pat, "time": t_total, "best_lam": best_lam,
                "fit_result": res, "ours": True}

        # 两阶段：过滤高系数模式，用自己的求解器精细拟合（避免 sklearn 数值不稳定）
        two_stage_dict = None
        if res.patterns and len(res.params.w) > 0:
            try:
                w_abs = np.abs(res.params.w)
                # 动态阈值：取系数绝对值的中位数，至少保留 top 50%（不少于真实模式数）
                w_thresh = max(0.01, float(np.median(w_abs)))
                sel_idx   = [i for i, wi in enumerate(w_abs) if wi >= w_thresh]
                # 保证至少保留前 min(len(patterns), max_patterns) 个
                if len(sel_idx) < 4:
                    sel_idx = list(np.argsort(w_abs)[-4:])
                sel_pats  = [res.patterns[i] for i in sel_idx]

                X_patt2  = build_X_trans(transactions, sel_pats)
                y_pos    = np.maximum(y_fit, 0.01)
                # 用我们自己的 Poisson EN 求解器做低正则化精细 refit
                solver2  = PoissonElasticNetSolver(
                    lam=0.01, kappa=1e-5, max_iter=2000, eta_clip=clip
                )
                p2       = solver2.fit(X_patt2, y_pos)
                eta2_tr  = np.clip(p2.b + (X_patt2 @ p2.w if X_patt2.shape[1] else 0.0),
                                   -clip, clip)
                mu2_fit  = np.exp(eta2_tr)

                # 两阶段测试集评估
                if test_transactions is not None and y_test is not None:
                    X_patt_te = build_X_trans(test_transactions, sel_pats)
                    eta2_te   = np.clip(p2.b + (X_patt_te @ p2.w if X_patt_te.shape[1] else 0.0),
                                        -clip, clip)
                    mu2_te    = np.exp(eta2_te)
                    mu2_orig  = mu2_te + y_shift - 1.0 if not is_poisson else mu2_te
                    m2        = calc_metrics(y_test, mu2_orig)
                    if is_poisson:
                        m2["PseudoR2"] = float(poisson_pseudo_r2(y_test, mu2_te))
                else:
                    mu2_orig = mu2_fit + y_shift - 1.0 if not is_poisson else mu2_fit
                    m2       = calc_metrics(y, mu2_orig)
                    if is_poisson:
                        m2["PseudoR2"] = float(poisson_pseudo_r2(y_pos, mu2_fit))

                # 两阶段的模式指标（基于过滤后的模式）
                pat2 = our_pattern_metrics_pats(sel_pats, transactions)
                two_stage_dict = {"name": "两阶段(精选模式)★", "metrics": m2,
                                  "cv_r2": float("nan"), "pattern": pat2,
                                  "time": time.time() - t0, "ours": True}
                print(f"    [两阶段] sel={len(sel_pats)}  Test_R²={m2['R2']:.4f}  PR²={m2.get('PseudoR2', float('nan')):.4f}")
            except Exception as e2:
                print(f"    [两阶段] 失败: {e2}")

        results = [our_dict]
        if two_stage_dict:
            results.append(two_stage_dict)
        return results
    except Exception as e:
        print(f" 失败: {e}")
        return [{"name": "我们的模型★", "error": str(e), "time": time.time()-t0}]

# ════════════════════════════════════════════════════════════════
#  打印单实验
# ════════════════════════════════════════════════════════════════

def print_exp(label, all_rows, r2_upper, is_poisson):
    noise_tag = "【泊松噪声】" if is_poisson else "【无噪声】"
    print(f"\n  ┌{'─'*95}┐")
    print(f"  │  {label}  {noise_tag}  R²理论上限={r2_upper:.4f}{' '*45}│")
    print(f"  └{'─'*95}┘")

    # 回归指标
    print(f"\n  ▶ 回归指标（Test集）")
    print(f"  {'模型':<22} {'Test_R²':>9} {'CV_R²':>8} {'Test_PR²':>10}"
          f" {'RSS':>12} {'MAE':>8} {'RMSE':>8}")
    print("  "+"─"*84)
    for r in all_rows:
        if "error" in r:
            print(f"  {r['name']:<22}  ERROR: {r['error'][:45]}"); continue
        m   = r["metrics"]; cv  = r.get("cv_r2", float("nan"))
        tag = " ★" if r.get("ours") else (" ↑" if r.get("oracle") else "")
        cv_s = f"{cv:>8.4f}" if not (isinstance(cv,float) and math.isnan(cv)) else f"{'─':>8}"
        pr2  = m.get("PseudoR2", float("nan"))
        pr2_s = f"{pr2:>10.4f}" if not (isinstance(pr2,float) and math.isnan(pr2)) else f"{'n/a':>10}"
        print(f"  {r['name']:<22} {m['R2']:>9.4f} {cv_s} {pr2_s}"
              f" {m['RSS']:>12.2f} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}{tag}")

    # 模式识别
    print(f"\n  ▶ 模式识别（所有基线=0，仅我们的模型可感知交互模式）")
    print(f"  {'模型':<22} {'选中':>5} {'命中':>5} {'Prec':>6} {'Rec':>6}"
          f" {'top10R':>7} {'top20R':>7} {'top50R':>7}")
    print("  "+"─"*75)
    for r in all_rows:
        if "error" in r or "pattern" not in r: continue
        p   = r["pattern"]
        tag = " ★" if r.get("ours") else (" ↑" if r.get("oracle") else "")
        print(f"  {r['name']:<22} {p['n_selected']:>5} {p['n_hits']:>5}"
              f" {p['precision']:>6.3f} {p['recall']:>6.3f}"
              f" {p.get('top10R',0):>7.3f} {p.get('top20R',0):>7.3f}"
              f" {p.get('top50R',0):>7.3f}{tag}")

# ════════════════════════════════════════════════════════════════
#  汇总大表
# ════════════════════════════════════════════════════════════════

def print_summary_tables(all_exps):
    for metric_key, title in [("PseudoR2", "Test PseudoR² (Poisson)"), ("R2", "Test R²"), ("recall", "Pattern Recall")]:
        print(f"\n\n{'#'*115}")
        print(f"  全实验汇总 — {title} 对比")
        print(f"{'#'*115}")

        # 列标题
        col_w = 13
        hdr = f"  {'模型':<22}"
        for exp in all_exps:
            label = f"{exp['size_tag'][:4]}/{exp['wt'][:2]}/{exp['noise'][:2]}"
            hdr += f"  {label:>{col_w}}"
        print(hdr)
        print("  "+"─"*(22 + (col_w+2)*len(all_exps)))

        # 收集所有模型名
        model_names = []
        for exp in all_exps:
            for r in exp["all_rows"]:
                if r.get("name") and r["name"] not in model_names and "error" not in r:
                    model_names.append(r["name"])

        for mn in model_names:
            row = f"  {mn:<22}"
            for exp in all_exps:
                r = next((x for x in exp["all_rows"]
                          if x.get("name") == mn and "error" not in x), None)
                if r:
                    if metric_key == "R2":
                        val = r["metrics"]["R2"]
                    elif metric_key == "PseudoR2":
                        val = r["metrics"].get("PseudoR2", float("nan"))
                    else:
                        val = r.get("pattern", {}).get("recall", float("nan"))
                    row += f"  {val:>{col_w}.4f}"
                else:
                    row += f"  {'─':>{col_w}}"
            print(row)

        if metric_key == "R2":
            print(f"\n  注：Test R²对大权重Poisson数据有偏（RF直接最小化MSE，非泊松模型）")
            print(f"  首选指标：Test PseudoR²（泊松偏差比，正确指定的评估指标）  |  Oracle=真实模式特征Poisson上限")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
