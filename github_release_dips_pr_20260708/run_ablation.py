from model import *
from utils import *
from exp_shared import *

# ══════════════════════════════════════════════════════
# 消融实验：各剪枝机制在不同数据集下的效果
# 依赖：cell0 (model), cell1 (utils)
# ══════════════════════════════════════════════════════

"""
消融研究：跨数据集下各剪枝机制的效果对比
测试 5 种数据配置 × 6 种消融组合

数据配置：
  D1: 小权重 + 泊松（高散布，r大，预期剪枝弱）
  D2: 中权重 + 泊松（标准场景，预期剪枝中等）
  D3: 大权重 + 泊松（低散布，r小，预期剪枝强）
  D4: 无噪声（y=Xw，r≈0，预期剪枝极强）
  D5: 中权重 + 泊松 + 大数据(n=3000)（规模效应）

消融组合（累积加法）：
  BASE    : 仅支持度剪枝（LCM 基础）
  +v      : + v-pruning (SPP single ref)
  +u      : + u-screening (SPP single ref)
  +v+u    : + 两者同时
  +v+u+WS : + Warm-Start screening
  +v+u+WS+2ref: + 双参考点
"""

import time, warnings
import numpy as np
warnings.filterwarnings("ignore")

# from run_full import (PoissonEN_LCM_SPP, summarize_pruning,
#                       eval_feature_recovery_extended)

# ════════════════════════════════════════════════════════════════
#  数据生成（与 full_experiment.py 一致）
# ════════════════════════════════════════════════════════════════

TRUE_PATTERNS = [
    (3,7), (10,11,12), (25,26), (1,5,9),
    (2,14), (6,18,33), (4,21), (8,16),
]
PATTERN_PROBS = [0.35, 0.30, 0.40, 0.28, 0.32, 0.25, 0.30, 0.27]
N_ITEMS = 80

# ── 消融实验专用数据生成（独立于 exp_shared.py，保持原始80items设置）──

def make_transactions(n, seed=42):
    """消融实验专用版本：80 items，混合长度pattern，无INDIV_NOISE"""
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

def build_oracle_X(transactions):
    X = np.zeros((len(transactions), len(TRUE_PATTERNS)), float)
    for i, t in enumerate(transactions):
        tset = set(t)
        for j, p in enumerate(TRUE_PATTERNS):
            if set(p).issubset(tset): X[i, j] = 1.0
    return X

def make_y_poisson(X_oracle, weights, eta_clip=5.0, seed=123):
    rng = np.random.default_rng(seed)
    eta = -0.2 + X_oracle @ np.array(weights)
    return rng.poisson(np.exp(np.clip(eta, -eta_clip, eta_clip))).astype(float)

def make_y_noiseless(X_oracle, weights):
    y = X_oracle @ np.array(weights)
    return y - float(np.min(y)) + 1.0   # 平移到 >0，Poisson模型可用

# 5 种数据配置
# lam 已手工校准：保证 lam * 1.2 < 真实pattern梯度阈值，使 2ref 不会过剪
# D1 阈值≈20-50；D2/D3/D4/D5（中权重）阈值≈65-155（按噪声大小）
DATA_CONFIGS = {
    "D1:小权重/泊松(n=1500)": {
        "n": 1500, "is_poisson": True,
        "weights": [0.35,-0.30,0.25,0.40,-0.30,0.35,0.30,-0.35],
        "lam": 15, "desc": "弱信号基准，var/mean≈1.3，lam_hi=18<<阈值"
    },
    "D2:中权重/泊松(n=1500)": {
        "n": 1500, "is_poisson": True,
        "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
        "lam": 30, "desc": "标准场景，var/mean≈9，lam_hi=36<<阈值"
    },
    "D3:中权重/高λ/泊松(n=1500)": {
        "n": 1500, "is_poisson": True,
        "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
        "lam": 60, "desc": "高正则场景，lam_hi=72<<阈值，剪枝更强"
    },
    "D4:无噪声(n=1500)": {
        "n": 1500, "is_poisson": False,
        "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
        "lam": 15, "desc": "无噪声，gap快速收敛，lam_hi=18<<阈值，u-screening有效"
    },
    "D5:中权重/泊松(n=3000)": {
        "n": 3000, "is_poisson": True,
        "weights": [0.90,-0.80,0.75,1.10,-0.85,0.95,0.80,-1.00],
        "lam": 60, "desc": "大样本规模效应，lam=60对应n=3000阈值，2ref效果强"
    },
}

# ════════════════════════════════════════════════════════════════
#  消融运行器
# ════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    ("BASE",         False, "none"),
    ("+v",           False, "vprune"),
    ("+u",           False, "uscreen"),
    ("+v+u",         False, "both"),
    ("+v+u+WS",      True,  "both"),
    ("+v+u+WS+2ref", True,  "both"),    # 特殊处理
]


# ── 消融专用模型工厂与运行函数 ──

def make_factory(n, eta_clip, lam, do_ws, mode, min_sup=2, radius_mode="certified"):
    return PoissonEN_LCM_SPP(
        lam=lam, kappa=0.05,
        min_support=min_sup, max_len=4,
        top_k_add=200, max_rounds=10,
        eta_clip=eta_clip,
        do_ws_screen=do_ws,
        enum_mode_single=mode,
        enum_mode_two=mode,
        radius_mode=radius_mode,
    )

def run_single_ablation(transactions, y, n, eta_clip, lam, name, do_ws, mode,
                        base_sm=None, ref2=None):
    """运行单个消融配置，返回 (summary_dict, recovery_dict, speedup)"""
    model = make_factory(n, eta_clip, lam, do_ws, mode)
    res   = model.fit(transactions, y, verbose=False, ref2=ref2)
    sm    = summarize_pruning(res)
    acc   = eval_feature_recovery_extended(res, TRUE_PATTERNS, transactions)
    spdup = (base_sm["time_total_sec"] / max(1e-6, sm["time_total_sec"])
             if base_sm else 1.0)
    return sm, acc, spdup, res

def run_dataset_ablation(cfg_name, cfg):
    n          = cfg["n"]
    is_poisson = cfg["is_poisson"]
    weights    = cfg["weights"]
    lam        = cfg["lam"]
    desc       = cfg["desc"]
    eta_clip   = 6.0 if is_poisson else 12.0

    print(f"\n{'='*110}")
    print(f"  数据集：{cfg_name}")
    print(f"  说明：{desc}")
    print(f"  n={n}  lam={lam}  (lam_hi={lam*1.2:.1f})  eta_clip={eta_clip}")
    print(f"{'='*110}")

    # 生成数据
    trans    = make_transactions(n, seed=42)
    X_oracle = build_oracle_X(trans)
    y = (make_y_poisson(X_oracle, weights, eta_clip)
         if is_poisson else make_y_noiseless(X_oracle, weights))

    # 数据诊断
    mu_mean  = float(np.mean(y))
    vm_ratio = float(np.var(y) / max(mu_mean, 1e-9))
    zero_r   = float(np.mean(y == 0))
    print(f"  y诊断: mean={mu_mean:.2f}  var/mean={vm_ratio:.2f}"
          f"  零值率={zero_r:.3f}")
    print()

    rows = []
    base_sm = None

    for i, (name, do_ws, mode) in enumerate(ABLATION_CONFIGS):
        is_2ref = (name == "+v+u+WS+2ref")
        print(f"  运行 {name:<16}...", end="", flush=True)
        t0 = time.time()

        if is_2ref:
            # 先跑高 lambda 得到参考点
            hi_model = make_factory(n, eta_clip, lam * 1.2, True, "both")
            res_hi   = hi_model.fit(trans, y, verbose=False)
            ref2     = (res_hi.alpha, res_hi.r)
            sm, acc, spdup, res = run_single_ablation(
                trans, y, n, eta_clip, lam, name, True, "both",
                base_sm=base_sm, ref2=ref2)
        else:
            sm, acc, spdup, res = run_single_ablation(
                trans, y, n, eta_clip, lam, name, do_ws, mode,
                base_sm=base_sm)

        if i == 0:
            base_sm = sm

        # 打印进度行
        r_est = float(np.sqrt(max(0, 2 * mu_mean * max(sm.get("gap_est",0), 1))))
        print(f" {sm['time_total_sec']:>6.2f}s  "
              f"hard={sm['hard_prune_rate']*100:>5.1f}%  "
              f"v={sm['v_prune_rate']*100:>4.1f}%  "
              f"u={sm['u_screen_rate']*100:>4.1f}%  "
              f"spdup={spdup:>5.2f}x  "
              f"recall={acc['exact_recall']:.3f}")

        rows.append((name, sm, acc, spdup))

    # ── 详細表 ─────────────────────────────────────────────────
    print(f"\n  {'─'*105}")
    print(f"  {'配置':<18} {'Emit':>7} {'Visit':>8} {'HrdPrn%':>8} "
          f"{'v%':>6} {'u%':>6} {'WS_rm':>6} "
          f"{'T_tot':>8} {'T_mine':>8} {'Spdup':>7} "
          f"{'ExR':>6} {'ClR':>6} {'SupR':>6} {'sT10':>6}")
    print(f"  {'─'*105}")

    for name, sm, acc, spdup in rows:
        delta_t = ""
        if name != "BASE":
            base_t  = rows[0][1]["time_total_sec"]
            saved   = base_t - sm["time_total_sec"]
            delta_t = f"({saved:+.2f}s)"
        print(f"  {name:<18} "
              f"{int(sm['emitted_closed']):>7} "
              f"{int(sm['nodes_visited']):>8} "
              f"{sm['hard_prune_rate']*100:>7.1f}% "
              f"{sm['v_prune_rate']*100:>5.1f}% "
              f"{sm['u_screen_rate']*100:>5.1f}% "
              f"{int(sm['ws_removed_total']):>6} "
              f"{sm['time_total_sec']:>7.2f}s "
              f"{sm['time_mine_sec']:>7.2f}s "
              f"{spdup:>6.2f}x "
              f"{acc['exact_recall']:>6.3f} "
              f"{acc['closure_recall']:>6.3f} "
              f"{acc['superset_recall']:>6.3f} "
              f"{acc.get('superset_top10_recall',0):>6.3f}"
              f"  {delta_t}")

    return rows

# ════════════════════════════════════════════════════════════════
#  汇总：各剪枝机制的边际贡献
# ════════════════════════════════════════════════════════════════

def print_marginal_contribution(all_dataset_rows):
    """
    计算每个剪枝机制的边际加速（相对于前一步的增量）
    BASE → +v → +u → +v+u → +WS → +2ref
    """
    print(f"\n\n{'#'*110}")
    print("  各剪枝机制边际加速贡献（相对前一步）")
    print(f"{'#'*110}")
    print(f"  {'数据集':<28} {'BASE':>8} {'+v边际':>8} {'+u边际':>8} "
          f"{'+v+u':>8} {'WS边际':>8} {'2ref边际':>9} {'总加速':>8}")
    print("  "+"─"*95)

    for cfg_name, rows in all_dataset_rows:
        times = [r[1]["time_total_sec"] for r in rows]
        base  = times[0]
        # 边际加速 = 前一步时间 / 当前步时间
        marginals = []
        for i in range(1, len(times)):
            marginals.append(times[i-1] / max(1e-6, times[i]))
        total = base / max(1e-6, times[-1])
        cfg_short = cfg_name[:28]
        m_str = "".join(f"{m:>8.2f}x" for m in marginals)
        print(f"  {cfg_short:<28} {base:>6.1f}s {m_str}  {total:>6.2f}x")

    print(f"\n  注：边际加速 > 1.0 表示该步有效，≈1.0 表示无效")

def print_pruning_rate_summary(all_dataset_rows):
    """各数据集下 v% 和 u% 的汇总"""
    print(f"\n\n{'#'*110}")
    print("  各数据集下 v-pruning 和 u-screening 比率（+v+u 配置）")
    print(f"{'#'*110}")
    print(f"  {'数据集':<34} {'v%':>8} {'u%':>8} {'WS_rm':>8} "
          f"{'节点减少':>10} {'时间(s)':>8} {'加速':>8}")
    print("  "+"─"*90)

    for cfg_name, rows in all_dataset_rows:
        # 找 +v+u 行
        r_vu   = next((r for r in rows if r[0] == "+v+u"),   None)
        r_base = rows[0]
        r_full = next((r for r in rows if r[0] == "+v+u+WS+2ref"), None)
        if not r_vu: continue
        sm_vu   = r_vu[1]
        sm_base = r_base[1]
        node_red = (sm_base["nodes_visited"] - sm_vu["nodes_visited"])
        node_red_pct = node_red / max(1, sm_base["nodes_visited"]) * 100
        full_spdup = r_full[3] if r_full else float("nan")
        print(f"  {cfg_name[:34]:<34} "
              f"{sm_vu['v_prune_rate']*100:>7.1f}% "
              f"{sm_vu['u_screen_rate']*100:>7.1f}% "
              f"{int(sm_vu['ws_removed_total']):>8} "
              f"{node_red_pct:>8.1f}% "
              f"{sm_vu['time_total_sec']:>7.2f}s "
              f"{r_vu[3]:>7.2f}x")

    print(f"\n  关键洞察：")
    print(f"  · v-pruning 效果取决于 r（越小越好）")
    print(f"  · u-screening 效果取决于 gap（越小越好）")
    print(f"  · 超散布数据 r 大 → 两者都弱")
    print(f"  · 无噪声/中权重数据 gap 快速收敛 → 两者都强")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*110)
    print("  消融研究：各剪枝机制在不同数据集下的效果")
    print("="*110)

    all_dataset_rows = []

    for cfg_name, cfg in DATA_CONFIGS.items():
        rows = run_dataset_ablation(cfg_name, cfg)
        all_dataset_rows.append((cfg_name, rows))

    # 汇总表
    print_marginal_contribution(all_dataset_rows)
    print_pruning_rate_summary(all_dataset_rows)

    print(f"\n\n{'='*110}")
    print("  消融研究完成 ✓")
    print(f"  · 论文中建议展示 D2/D3/D4 三种场景的消融表")
    print(f"  · D1（高散布）作为 limitation 分析")
    print(f"  · 边际贡献表清晰展示每个机制的独立价值")
    print(f"{'='*110}")
