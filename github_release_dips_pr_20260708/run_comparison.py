from model import *
from utils import *
from exp_shared import *
import math, time

if __name__ == "__main__":
    print("="*75)
    print("  完整对比实验：我们的模型 + 所有基线")
    print("="*75)

    all_exps = []
    TRAIN_FRAC = 0.7  # 70% 训练，30% 测试

    for size_tag, n in DATA_SIZES.items():
        print(f"\n{'─'*75}")
        print(f"  数据规模：{size_tag}  (n={n}，train={int(n*TRAIN_FRAC)}，test={n-int(n*TRAIN_FRAC)})")
        print(f"{'─'*75}")
        # 生成全量数据，再分割
        trans_all    = make_transactions(n, seed=42)
        X_oracle_all = build_oracle_X(trans_all)

        n_train = int(n * TRAIN_FRAC)
        trans_train    = trans_all[:n_train]
        trans_test     = trans_all[n_train:]
        X_oracle_train = X_oracle_all[:n_train]
        X_oracle_test  = X_oracle_all[n_train:]
        X_item_train   = build_item_X(trans_train)
        X_item_test    = build_item_X(trans_test)

        for wt, weights in WEIGHT_CONFIGS.items():

            for noise_type in ["noisy"]:  # 仅保留Poisson噪声场景
                is_poisson = (noise_type == "noisy")
                label      = f"{size_tag} | {wt}"
                noise_str  = "泊松" if is_poisson else "无噪声"
                print(f"\n  ── {label}  [{noise_str}] ──")

                # 分别生成训练集和测试集的 y（同一批数据切分）
                y_all   = (make_y_poisson(X_oracle_all, weights)
                           if is_poisson
                           else make_y_noiseless(X_oracle_all, weights))
                y_train = y_all[:n_train]
                y_test  = y_all[n_train:]
                r2_ub   = r2_upper_bound(X_oracle_test, weights, is_poisson)

                # 基线（在训练集上训练，测试集上评估）
                bl_rows = run_baselines(X_item_train, y_train,
                                        X_test=X_item_test, y_test=y_test)
                oracle  = run_oracle(X_oracle_train, y_train,
                                     X_oracle_test=X_oracle_test, y_test=y_test)

                # 我们的模型（在训练集上训练，测试集上评估）
                our_list = run_our_model(trans_train, y_train, is_poisson, n_train,
                                         verbose=False,
                                         test_transactions=trans_test, y_test=y_test)

                all_rows = bl_rows + [oracle] + our_list

                print_exp(label, all_rows, r2_ub, is_poisson)

                all_exps.append({
                    "size_tag": size_tag, "wt": wt,
                    "noise": noise_type, "n": n,
                    "all_rows": all_rows, "r2_upper": r2_ub,
                })

    # 汇总
    print_summary_tables(all_exps)

    print(f"\n\n{'='*75}")
    print("  全部实验完成 ✓")
    print(f"{'='*75}")
