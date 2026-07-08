"""
Heuristic vs Certified radius 对比实验
目标：证明即使使用启发式半径，也不会遗漏重要特征组

实验设计：
  1. 相同数据 × 多个随机种子（5个）
  2. 两种 radius_mode: heuristic / certified
  3. 比较：选中 pattern 集合、true pattern recall、pattern 系数
  4. 覆盖 5 种数据配置 × D1-D5
"""

from model import *
from utils import *
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  与消融实验一致的数据配置
# ════════════════════════════════════════════════════════════════

TRUE_PATTERNS = [
    (3,7), (10,11,12), (25,26), (1,5,9),
    (2,14), (6,18,33), (4,21), (8,16),
]
PATTERN_PROBS = [0.35, 0.30, 0.40, 0.28, 0.32, 0.25, 0.30, 0.27]
N_ITEMS = 80

def make_transactions(n, seed=42):
    rng = np.random.default_rng(seed)
    psets   = [set(p) for p in TRUE_PATTERNS]
    all_pat = set(it for p in TRUE_PATTERNS for it in p)
    non_pat = [i for i in range(N_ITEMS) if i not in all_pat]
    trans = []
    for _ in range(n):
        items = set()
        for ps, prob in zip(psets, PATTERN_PROBS):
            if rng.random() < prob: items.update(ps)
        for it in non_pat:
            if rng.random() < 0.05: items.add(it)
        trans.append(sorted(items))
    return trans

def make_y_poisson(X_oracle, weights, eta_clip=5.0, seed=123):
    rng = np.random.default_rng(seed)
    eta = -0.2 + X_oracle @ np.array(weights)
    return rng.poisson(np.exp(np.clip(eta, -eta_clip, eta_clip))).astype(float)

def make_y_noiseless(X_oracle, weights):
    y = X_oracle @ np.array(weights)
    return y - float(np.min(y)) + 1.0

def build_oracle_X(transactions):
    X = np.zeros((len(transactions), len(TRUE_PATTERNS)), float)
    for i, t in enumerate(transactions):
        tset = set(t)
        for j, p in enumerate(TRUE_PATTERNS):
            if set(p).issubset(tset): X[i, j] = 1.0
    return X

DATA_CONFIGS = {
    "D1": {"n": 1500, "is_poisson": True,
            "weights": [0.35,-0.30,0.25,0.40,-0.30,0.35,0.30,-0.35],
            "lam": 15, "eta_clip": 6.0, "label": "小权重/泊松"},
    "D2": {"n": 1500, "is_poisson": True,
            "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
            "lam": 30, "eta_clip": 6.0, "label": "中权重/泊松"},
    "D3": {"n": 1500, "is_poisson": True,
            "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
            "lam": 60, "eta_clip": 6.0, "label": "中权重/高λ"},
    "D4": {"n": 1500, "is_poisson": False,
            "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
            "lam": 15, "eta_clip": 12.0, "label": "无噪声"},
    "D5": {"n": 3000, "is_poisson": True,
            "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
            "lam": 60, "eta_clip": 6.0, "label": "中权重/n=3000"},
}

SEEDS = [42, 99, 137, 256, 512]   # 5 个随机种子

# ════════════════════════════════════════════════════════════════
#  运行单次对比
# ════════════════════════════════════════════════════════════════

def run_one(transactions, y, n, lam, eta_clip, radius_mode):
    """用完整配置（+v+u+WS+2ref）运行，返回 FitResult"""
    model = PoissonEN_LCM_SPP(
        lam=lam, kappa=0.05,
        min_support=2, max_len=4,
        top_k_add=200, max_rounds=10,
        eta_clip=eta_clip,
        do_ws_screen=True,
        enum_mode_single="both",
        enum_mode_two="both",
        radius_mode=radius_mode,
    )
    return model.fit(transactions, y, verbose=False)

def patterns_to_set(res):
    """返回选中 pattern 的 frozenset（只含 nonzero weight 的）"""
    selected = set()
    for j, p in enumerate(res.patterns):
        if j < len(res.params.w) and abs(res.params.w[j]) > 1e-8:
            selected.add(frozenset(p.itemset))
    return selected

def true_recall(res, transactions):
    """计算 true patterns 的 recall（superset 容忍闭包扩展）"""
    acc = eval_feature_recovery_extended(res, TRUE_PATTERNS, transactions)
    return acc["superset_recall"]

def coef_dict(res):
    """返回 pattern -> coefficient 的字典"""
    d = {}
    for j, p in enumerate(res.patterns):
        if j < len(res.params.w):
            w = float(res.params.w[j])
            if abs(w) > 1e-8:
                d[frozenset(p.itemset)] = w
    return d

# ════════════════════════════════════════════════════════════════
#  主对比
# ════════════════════════════════════════════════════════════════

def run_comparison():
    SEP = "=" * 110

    print(SEP)
    print("  Heuristic vs Certified 半径对比实验")
    print("  目标：证明启发式半径不遗漏重要特征组")
    print(SEP)

    # 汇总统计
    total_runs = 0
    total_recall_match = 0    # 两者 recall 相同
    total_patset_match = 0    # 选中 pattern 集合完全一致
    total_coef_close  = 0     # 系数最大绝对差 < 0.05

    grand_rows = []  # (cfg, seed, recall_h, recall_c, match, n_only_h, n_only_c, max_coef_diff)

    for cfg_name, cfg in DATA_CONFIGS.items():
        n         = cfg["n"]
        lam       = cfg["lam"]
        eta_clip  = cfg["eta_clip"]
        is_pois   = cfg["is_poisson"]
        weights   = cfg["weights"]
        label     = cfg["label"]

        print(f"\n{'─'*110}")
        print(f"  {cfg_name}: {label}  (n={n}, lam={lam})")
        print(f"{'─'*110}")
        print(f"  {'种子':>6}  {'recall_H':>9}  {'recall_C':>9}  {'|P_H|':>6}  {'|P_C|':>6}  "
              f"{'仅H有':>6}  {'仅C有':>6}  {'max|Δw|':>9}  {'pattern集合':>10}")

        for seed in SEEDS:
            trans = make_transactions(n, seed=seed)
            X_oracle = build_oracle_X(trans)
            if is_pois:
                y = make_y_poisson(X_oracle, weights, eta_clip=eta_clip, seed=seed+1000)
            else:
                y = make_y_noiseless(X_oracle, weights)

            res_h = run_one(trans, y, n, lam, eta_clip, "heuristic")
            res_c = run_one(trans, y, n, lam, eta_clip, "certified")

            recall_h = true_recall(res_h, trans)
            recall_c = true_recall(res_c, trans)

            pset_h = patterns_to_set(res_h)
            pset_c = patterns_to_set(res_c)

            only_h = pset_h - pset_c     # heuristic 有但 certified 没有
            only_c = pset_c - pset_h     # certified 有但 heuristic 没有
            patset_match = (pset_h == pset_c)

            # 系数对比（只比较两者都选的）
            coef_h = coef_dict(res_h)
            coef_c = coef_dict(res_c)
            common = set(coef_h.keys()) & set(coef_c.keys())
            max_coef_diff = max(
                (abs(coef_h[k] - coef_c[k]) for k in common), default=0.0
            )

            match_str = "✓一致" if patset_match else "△不同"
            print(f"  {seed:>6}  {recall_h:>9.3f}  {recall_c:>9.3f}  "
                  f"{len(pset_h):>6}  {len(pset_c):>6}  "
                  f"{len(only_h):>6}  {len(only_c):>6}  "
                  f"{max_coef_diff:>9.4f}  {match_str:>10}")

            total_runs += 1
            if abs(recall_h - recall_c) < 1e-6:
                total_recall_match += 1
            if patset_match:
                total_patset_match += 1
            if max_coef_diff < 0.05:
                total_coef_close += 1

            grand_rows.append({
                "cfg": cfg_name, "seed": seed,
                "recall_h": recall_h, "recall_c": recall_c,
                "n_h": len(pset_h), "n_c": len(pset_c),
                "only_h": len(only_h), "only_c": len(only_c),
                "max_coef_diff": max_coef_diff,
                "patset_match": patset_match,
                "only_h_items": only_h,
                "only_c_items": only_c,
            })

        # 若有不一致的 pattern，展示哪些 true patterns 只有 heuristic 找到
        ds_rows = [r for r in grand_rows if r["cfg"] == cfg_name]
        all_only_h = set().union(*[r["only_h_items"] for r in ds_rows])
        all_only_c = set().union(*[r["only_c_items"] for r in ds_rows])

        if all_only_h or all_only_c:
            # 找出 only_h 和 only_c 中是否包含 true patterns
            true_sets = [frozenset(p) for p in TRUE_PATTERNS]
            th_h = [fs for fs in all_only_h if any(fs >= ts for ts in true_sets)]
            th_c = [fs for fs in all_only_c if any(fs >= ts for ts in true_sets)]
            print(f"\n  注：出现 pattern 集合不完全一致")
            print(f"    仅 Heuristic 选出的 true-related patterns({len(th_h)}): "
                  + (str([set(x) for x in th_h[:5]]) if th_h else "无"))
            print(f"    仅 Certified 选出的 true-related patterns({len(th_c)}): "
                  + (str([set(x) for x in th_c[:5]]) if th_c else "无"))

    # ────────────────────────── 汇总 ──────────────────────────
    print(f"\n{SEP}")
    print("  汇总统计")
    print(SEP)
    print(f"  总运行次数（数据集×种子）：{total_runs}")
    print(f"  Recall 完全一致（heuristic==certified）：{total_recall_match}/{total_runs}  "
          f"({total_recall_match/total_runs:.1%})")
    print(f"  选中 pattern 集合完全一致：{total_patset_match}/{total_runs}  "
          f"({total_patset_match/total_runs:.1%})")
    print(f"  系数最大差 < 0.05：{total_coef_close}/{total_runs}  "
          f"({total_coef_close/total_runs:.1%})")

    # ── 按数据集聚合 recall ──
    print(f"\n  各数据集 recall 汇总（mean ± std，5种子）")
    print(f"  {'数据集':<30}  {'Heuristic recall':>18}  {'Certified recall':>18}  {'recall一致':>10}")
    for cfg_name in DATA_CONFIGS:
        rows = [r for r in grand_rows if r["cfg"] == cfg_name]
        rh = [r["recall_h"] for r in rows]
        rc = [r["recall_c"] for r in rows]
        same = sum(abs(a-b) < 1e-6 for a, b in zip(rh, rc))
        print(f"  {cfg_name:<30}  {np.mean(rh):.3f} ± {np.std(rh):.3f}        "
              f"  {np.mean(rc):.3f} ± {np.std(rc):.3f}        "
              f"  {same}/{len(rows)}")

    # ── 核查：heuristic 是否会遗漏 true pattern ──
    print(f"\n{SEP}")
    print("  核心问题：Heuristic 是否遗漏了 Certified 能找到的 true patterns？")
    print(SEP)
    missed_by_h = 0
    for r in grand_rows:
        # certified 独有的 pattern 中，是否包含 true pattern
        true_sets = [frozenset(p) for p in TRUE_PATTERNS]
        only_c_true = [fs for fs in r["only_c_items"]
                       if any(fs >= ts for ts in true_sets)]
        if only_c_true:
            missed_by_h += 1
            print(f"  !! {r['cfg']} seed={r['seed']}: Heuristic 漏掉 true-pattern {[set(x) for x in only_c_true]}")
    if missed_by_h == 0:
        print(f"  结论：在全部 {total_runs} 次运行中，Heuristic 从未遗漏 Certified 能找到的 true patterns ✓")

    print(f"\n{SEP}")
    print("  实验完成")
    print(SEP)

if __name__ == "__main__":
    run_comparison()
