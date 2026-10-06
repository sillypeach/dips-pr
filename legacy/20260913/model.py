# %%

# ══════════════════════════════════════════════════════
# § 1  导入与基础工具
# ══════════════════════════════════════════════════════
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set, Callable
import time

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


# -------------------------
# Utils: tidset ops
# -------------------------

def intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # BUG8 fix: np.intersect1d is C-level, 10~50x faster than pure-Python loop
    return np.intersect1d(a, b, assume_unique=True).astype(np.int32)

def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))
def poisson_deviance_residuals(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> np.ndarray:

# ══════════════════════════════════════════════════════
# § 2  Poisson 损失函数
# ══════════════════════════════════════════════════════
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), eps)
    # term = y*log(y/mu) define as 0 when y=0
    term = np.where(y > 0, y * (np.log(np.maximum(y, eps)) - np.log(mu)), 0.0)
    dev_i = 2.0 * (term - (y - mu))
    dev_i = np.maximum(dev_i, 0.0)
    return np.sign(y - mu) * np.sqrt(dev_i)

def poisson_pearson_residuals(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), eps)
    return (y - mu) / np.sqrt(mu)

def poisson_deviance(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), eps)
    term = np.where(y > 0, y * (np.log(np.maximum(y, eps)) - np.log(mu)), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))

def poisson_pseudo_r2_deviance_from_mu(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), eps)
    mu0 = np.full_like(y, np.mean(y))
    D_model = poisson_deviance(y, mu)
    D_null  = poisson_deviance(y, mu0)
    if D_null <= 1e-30:
        return 0.0
    return float(1.0 - D_model / D_null)

# -------------------------

# ══════════════════════════════════════════════════════
# § 3  模式与统计数据结构
# ══════════════════════════════════════════════════════
# Pattern representation
# -------------------------

@dataclass(frozen=True)
class Pattern:
    itemset: Tuple[int, ...]
    tidset: Tuple[int, ...]  # tuple for hashing

    def support(self) -> int:
        return len(self.tidset)

    def key(self) -> str:
        return "{" + ",".join(map(str, self.itemset)) + "}"

@dataclass
class EnumStats:
    nodes_visited: int = 0          # 访问过的候选扩展节点数（child_tid 生成后算一次）
    pruned_support: int = 0         # child support < min_support
    pruned_v: int = 0               # v-bound < lam => 子树剪枝
    pruned_canonical: int = 0       # canonical test fail
    screened_u: int = 0             # u-bound < lam => 该节点不加入候选（但子树仍可能继续）
    emitted_closed: int = 0         # 输出的 closed patterns 数

# -------------------------

# ══════════════════════════════════════════════════════
# § 4  SPP 剪枝辅助函数（Theorem 5）
# ══════════════════════════════════════════════════════
# Theorem-5 helpers for binary columns without materializing x
# -------------------------

def _centered_norm_from_support(s: int, n: int) -> float:
    # ||x - Π1(x)||_2 for binary x with support s
    return float(math.sqrt(max(0.0, s - (s * s) / n)))

def _subtree_centered_norm_bound(s: int, n: int) -> float:
    # For any descendant support k <= s, maximize ||x - mean(x)1||_2.
    # The centered norm is justified by the intercept constraint on dual
    # differences and is tighter than the uncentered SPP-style sqrt(s) term.
    if s <= 0:
        return 0.0
    if s <= n / 2:
        k = float(s)
    else:
        k = float(n) / 2.0
    return float(math.sqrt(max(0.0, k - (k * k) / n)))

def _subtree_dual_mass_bound(tid: np.ndarray, alpha: np.ndarray) -> float:
    # For any descendant tid' subset tid, |sum_{tid'} alpha| is bounded by
    # the larger of the positive and negative mass inside the parent tidset.
    vals = alpha[tid]
    pos = float(np.sum(vals[vals > 0.0]))
    neg = float(-np.sum(vals[vals < 0.0]))
    return max(pos, neg)

def _centered_minus_proj_d_norm_from_tid(
    tid: np.ndarray, d: np.ndarray, d_norm2: float, s: int, n: int
) -> float:
    """
    || x - Π1(x) - Π_d( x - Π1(x) ) ||_2
    where x is binary indicator of tid.
    Using:
      ||xc||^2 = s - s^2/n
      xc^T d = x^T d  because 1^T d = 0 (if alphas are sum0, then d sum0)
      => ||xc - proj||^2 = ||xc||^2 - (xc^T d)^2 / ||d||^2
    """
    normc2 = max(0.0, s - (s * s) / n)
    if d_norm2 <= 1e-30 or normc2 <= 1e-30:
        return float(math.sqrt(normc2))
    dotd = float(np.sum(d[tid]))
    val2 = max(0.0, normc2 - (dotd * dotd) / d_norm2)
    return float(math.sqrt(val2))

def theorem5_u0_binary_tidset(
    tid: np.ndarray,
    alpha1: np.ndarray, r1: float,
    alpha2: np.ndarray, r2: float,
    n: int
) -> float:
    """
    Tight multi-reference u0 (Theorem-5 style) for binary x using tidset.
    Returns u0 = max(u_plus, u_minus) with piecewise cone test, otherwise uses (alpha0,r0).
    Assumes alphas satisfy sum(alpha)=0 (hyperplane H). This is true with our dual builder.
    """
    s = int(tid.size)
    if s == 0:
        return 0.0

    d = alpha1 - alpha2
    d_norm2 = float(d @ d)
    normc = _centered_norm_from_support(s, n)
    if d_norm2 <= 1e-30 or normc <= 1e-30:
        # fallback (still safe, slightly weaker)
        dot1 = float(np.sum(alpha1[tid]))
        dot2 = float(np.sum(alpha2[tid]))
        u1 = abs(dot1) + r1 * normc
        u2 = abs(dot2) + r2 * normc
        return min(u1, u2)

    # cone tests use (x^T d) / ||x-Π1x||
    xTd = float(np.sum(d[tid]))
    ratio = xTd / normc

    th1 = (r2 * r2 - r1 * r1 - d_norm2) / (2.0 * r1) if r1 > 0 else -float("inf")
    th2 = (r2 * r2 - r1 * r1 + d_norm2) / (2.0 * r2) if r2 > 0 else float("inf")

    in_c1 = (ratio <= th1)
    in_c2 = (ratio >= th2)

    xTa1 = float(np.sum(alpha1[tid]))
    xTa2 = float(np.sum(alpha2[tid]))

    if in_c1:
        u_plus = xTa1 + r1 * normc
        u_minus = -xTa1 + r1 * normc
    elif in_c2:
        u_plus = xTa2 + r2 * normc
        u_minus = -xTa2 + r2 * normc
    else:
        t = 0.5 * (1.0 + (r2 * r2 - r1 * r1) / d_norm2)
        # alpha0 and r0
        alpha0 = t * alpha1 + (1.0 - t) * alpha2
        r0_sq = r2 * r2 - (t * t) * d_norm2
        r0 = float(math.sqrt(max(0.0, r0_sq)))

        xTa0 = float(np.sum(alpha0[tid]))
        norm_cd = _centered_minus_proj_d_norm_from_tid(tid, d, d_norm2, s, n)

        u_plus = xTa0 + r0 * norm_cd
        u_minus = -xTa0 + r0 * norm_cd

    return max(u_plus, u_minus)

# -------------------------

# ══════════════════════════════════════════════════════
# § 5  LCM 闭合频繁集枚举器
#      支持：support剪枝 / v-剪枝 / u-screening / 双参考
# ══════════════════════════════════════════════════════
# LCM enumerator (LCM-style DFS, no closure)
#   Added: single/two-ref pruning+screening, Theorem-5 u0
# -------------------------

class LCMEnumerator:
    """
    LCM-style DFS with:
      - closure (closed itemset)
      - canonical test (avoid duplicates)
      - optional SPP pruning (v-bound) + screening (u-bound / theorem5 u0)
    """

    def __init__(
        self,
        item_tidsets: Dict[int, np.ndarray],
        items_sorted: List[int],
        n_samples: int,
        min_support: int = 2,
        max_len: int = 4,
    ):
        self.item_tidsets = item_tidsets
        self.items_sorted = items_sorted
        self.n = n_samples
        self.min_support = int(min_support)
        self.max_len = int(max_len)
        self.stats = EnumStats()
    def reset_stats(self):
        self.stats = EnumStats()

    def get_stats(self) -> EnumStats:
        return self.stats


    # ---------- tidset subset test ----------
    @staticmethod
    def _is_subset_tid(child_tid: np.ndarray, super_tid: np.ndarray) -> bool:
        """
        Return True iff child_tid ⊆ super_tid. Both sorted int arrays.
        Two-pointer linear scan.
        """
        i = j = 0
        a, b = child_tid, super_tid
        na, nb = a.size, b.size
        while i < na and j < nb:
            if a[i] == b[j]:
                i += 1; j += 1
            elif a[i] > b[j]:
                j += 1
            else:
                return False
        return i == na

    # ---------- closure computation + canonical test ----------
    def _closure_and_canonical(
        self,
        prefix: Tuple[int, ...],
        last: int,
        it: int,
        child_tid: np.ndarray
    ) -> Optional[Tuple[Tuple[int, ...], int]]:
        """
        For candidate extension 'it', with tidset = child_tid:

        1) Canonical test (LCM):
           If exists j < it, j not in prefix, s.t. child_tid ⊆ tidset(j),
           then this node is NOT canonical -> skip (return None).

        2) Closure:
           Add all items k > it such that child_tid ⊆ tidset(k).
           Return closed_itemset, new_last = max item in closed itemset.

        Notes:
        - We keep item ordering by items_sorted.
        - We cap max_len by CLOSED itemset length (common in practice).
        """
        # Build a fast "in prefix" test (prefix is small)
        pref_set = set(prefix)

        # --- canonical test: any j < it (not in prefix) covers tid? then skip
        for j in self.items_sorted:
            if j >= it:
                break
            if j in pref_set:
                continue
            if self._is_subset_tid(child_tid, self.item_tidsets[j]):
                return None  # not canonical

        # --- closure: add all k > it that cover child_tid
        extra = []
        for k in self.items_sorted:
            if k <= it:
                continue
            # optional early stop by max_len upper bound:
            # (closure could exceed max_len; we handle after building)
            if self._is_subset_tid(child_tid, self.item_tidsets[k]):
                extra.append(k)

        closed = tuple(sorted(prefix + (it,) + tuple(extra)))
        new_last = closed[-1] if closed else it

        # enforce max_len on closed itemset
        if len(closed) > self.max_len:
            return None

        return closed, new_last

    # ---------- main DFS ----------
    def dfs_with_spp(
        self,
        alpha: np.ndarray,
        grad_eta: np.ndarray,
        r: float,
        lam: float,
        already_have: Set[Tuple[int, ...]],
        top_k: int = 200,
        alpha2: Optional[np.ndarray] = None,
        r2: Optional[float] = None,
        mode: str = "both",
    ) -> List[Pattern]:

        grad_eta = np.asarray(grad_eta, float)
        assert grad_eta.shape == (self.n,)

        alpha = np.asarray(alpha, float)
        n = self.n
        assert alpha.shape == (n,)

        two_ref = (alpha2 is not None) and (r2 is not None)
        if two_ref:
            alpha2 = np.asarray(alpha2, float)
            assert alpha2.shape == (n,)

        cand: List[Tuple[float, Pattern]] = []
        root_tid = np.arange(n, dtype=np.int32)

        def corr_score(tid: np.ndarray) -> float:
            return float(abs(np.sum(grad_eta[tid])))

        def v_bound(tid: np.ndarray) -> float:
            s = int(tid.size)
            if s == 0:
                return 0.0

            def one_ref(a: np.ndarray, rr: float) -> float:
                mass = _subtree_dual_mass_bound(tid, a)
                norm_bound = _subtree_centered_norm_bound(s, n)
                return mass + rr * norm_bound

            v1 = one_ref(alpha, r)
            if not two_ref:
                return v1
            v2 = one_ref(alpha2, r2)
            return min(v1, v2)

        def u_single(tid: np.ndarray, a: np.ndarray, rr: float) -> float:
            s = int(tid.size)
            if s == 0:
                return 0.0
            dot = float(np.sum(a[tid]))
            normc = _centered_norm_from_support(s, n)
            return abs(dot) + rr * normc

        def u_bound(tid: np.ndarray) -> float:
            if not two_ref:
                return u_single(tid, alpha, r)
            return theorem5_u0_binary_tidset(tid, alpha, r, alpha2, r2, n)

        # stack element: (current_closed_itemset, current_tidset, last_item)
        stack: List[Tuple[Tuple[int, ...], np.ndarray, int]] = [(tuple(), root_tid, -1)]

        while stack:
            itemset, tid, last = stack.pop()
            if len(itemset) >= self.max_len:
                continue

            # iterate extension items greater than last
            for it in self.items_sorted:
                if it <= last:
                    continue

                child_tid = intersect_sorted(tid, self.item_tidsets[it])
                self.stats.nodes_visited += 1

                if child_tid.size < self.min_support:
                    self.stats.pruned_support += 1
                    continue


                # v-pruning: can prune subtree safely
                if mode in ("vprune", "both"):
                    if v_bound(child_tid) < lam:
                        self.stats.pruned_v += 1
                        continue


                # closure + canonical test (LCM core)
                out = self._closure_and_canonical(itemset, last, it, child_tid)
                if out is None:
                    self.stats.pruned_canonical += 1
                    continue
                closed_itemset, new_last = out

                # screening decision for THIS node (u / u0)
                keep_this = True
                if mode in ("uscreen", "both"):
                    keep_this = (u_bound(child_tid) >= lam)
                    if not keep_this:
                        self.stats.screened_u += 1

                # note: tidset corresponds to CLOSED pattern now
                if keep_this and closed_itemset not in already_have:
                    self.stats.emitted_closed += 1
                    p = Pattern(closed_itemset, tuple(child_tid.tolist()))
                    cand.append((corr_score(child_tid), p))

                # expand to children from this CLOSED node
                stack.append((closed_itemset, child_tid, new_last))

        if not cand:
            return []
        cand.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in cand[:top_k]]



# ══════════════════════════════════════════════════════
# § 6  特征矩阵构建
# ══════════════════════════════════════════════════════
def build_X_from_patterns(n: int, patterns: List[Pattern]) -> np.ndarray:
    m = len(patterns)
    X = np.zeros((n, m), dtype=np.float64)
    for j, p in enumerate(patterns):
        tid = np.fromiter(p.tidset, dtype=np.int32)
        X[tid, j] = 1.0
    return X

def build_X_from_patterns_on_transactions(transactions: List[List[int]], patterns: List[Pattern]) -> np.ndarray:
    """
    Build dense binary matrix X for arbitrary transactions using pattern.itemset.
    This is required for CV/holdout evaluation because tidset indices are fold-specific.
    """
    n = len(transactions)
    m = len(patterns)
    X = np.zeros((n, m), dtype=np.float64)

    # preconvert each transaction to set for fast subset check
    trans_sets = [set(t) for t in transactions]

    for j, p in enumerate(patterns):
        pset = set(p.itemset)
        for i in range(n):
            if pset.issubset(trans_sets[i]):
                X[i, j] = 1.0
    return X


# ══════════════════════════════════════════════════════
# § 7  Poisson 弹性网求解器（FISTA + backtracking）
# ══════════════════════════════════════════════════════
# -------------------------
# Poisson + Elastic-Net solver (FISTA), with eta clipping (for Lipschitz)
# -------------------------

@dataclass
class ENParams:
    b: float
    w: np.ndarray

class PoissonElasticNetSolver:
    def __init__(self, lam: float, kappa: float, max_iter: int = 500, tol: float = 1e-6, eta_clip: float = 10.0):
        self.lam = float(lam)
        self.kappa = float(kappa)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eta_clip = float(eta_clip)

    def fit(self, X: np.ndarray, y: np.ndarray, warm: Optional[ENParams] = None) -> ENParams:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, m = X.shape

        if warm is None or warm.w.shape[0] != m:
            b = float(np.log(np.mean(y) + 1e-12))
            w = np.zeros(m, float)
        else:
            b, w = float(warm.b), warm.w.copy()

        z_b, z_w = b, w.copy()
        t = 1.0
        b_old, w_old = b, w.copy()
        L = 1.0

        def smooth_value_and_grad(bb: float, ww: np.ndarray):
            eta = bb + (X @ ww if m else 0.0)
            eta_c = np.clip(eta, -self.eta_clip, self.eta_clip)
            mu = np.exp(eta_c)
            f = float(np.sum(mu - y * eta_c) + 0.5 * self.kappa * (ww @ ww))
            g_eta = mu - y
            grad_b = float(np.sum(g_eta))
            grad_w = (X.T @ g_eta if m else np.zeros_like(ww)) + self.kappa * ww
            return f, eta, eta_c, grad_b, grad_w

        def full_obj(bb: float, ww: np.ndarray):
            eta = bb + (X @ ww if m else 0.0)
            eta_c = np.clip(eta, -self.eta_clip, self.eta_clip)
            return float(np.sum(np.exp(eta_c) - y * eta_c) + self.lam * np.sum(np.abs(ww)) + 0.5 * self.kappa * (ww @ ww))

        last_obj = float("inf")
        for _ in range(self.max_iter):
            fz, _, _, gb, gw = smooth_value_and_grad(z_b, z_w)

            while True:
                step = 1.0 / L
                b_new = z_b - step * gb
                w_tmp = z_w - step * gw
                w_new = soft_threshold(w_tmp, self.lam * step)

                f_new, _, _, _, _ = smooth_value_and_grad(b_new, w_new)
                db = b_new - z_b
                dw = w_new - z_w
                quad = fz + gb * db + float(gw @ dw) + 0.5 * L * float(db * db + dw @ dw)

                if f_new <= quad + 1e-12:
                    break
                L *= 2.0

            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z_b = b_new + ((t - 1.0) / t_new) * (b_new - b_old)
            z_w = w_new + ((t - 1.0) / t_new) * (w_new - w_old)
            t = t_new
            b_old, w_old = b_new, w_new

            o = full_obj(b_new, w_new)
            if abs(last_obj - o) / max(1.0, abs(last_obj)) < self.tol:
                b, w = b_new, w_new
                break
            last_obj = o
            b, w = b_new, w_new

        return ENParams(b=b, w=w)



# ══════════════════════════════════════════════════════
# § 8  对偶间隙计算（用于 SPP 参考向量 alpha/r）
# ══════════════════════════════════════════════════════
# -------------------------
# Dual + gap machinery (feasible alpha, intercept constraint)
#   Patch: stable build_dual_feasible (alternating projections)
# -------------------------

class PoissonENGap:
    """
    Dual used (same structure as your base code):
      D(alpha) = -( L^*(alpha) + U^*(X^T alpha) )
    with constraints:
      sum(alpha)=0  (intercept)
      alpha <= y    (domain of L^* for Poisson)
    """

    def __init__(self, lam: float, kappa: float, eta_clip: float = 10.0):
        self.lam = float(lam)
        self.kappa = float(kappa)
        self.eta_clip = float(eta_clip)

    def primal(self, y: np.ndarray, eta: np.ndarray, w: np.ndarray) -> float:
        eta_c = np.clip(eta, -self.eta_clip, self.eta_clip)
        return float(np.sum(np.exp(eta_c) - y * eta_c) + self.lam * np.sum(np.abs(w)) + 0.5 * self.kappa * (w @ w))

    def Lstar(self, alpha: np.ndarray, y: np.ndarray) -> float:
        t = y - alpha
        if np.any(t < 0):
            return float("inf")
        return float(np.sum(t * safe_log(t) - t))

    def Ustar(self, v: np.ndarray) -> float:
        t = np.maximum(np.abs(v) - self.lam, 0.0)
        if self.kappa == 0.0:
            return 0.0 if np.all(t == 0.0) else float("inf")
        return float(np.sum((t * t) / (2.0 * self.kappa)))

    def dual(self, alpha: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
        if abs(float(np.sum(alpha))) > 1e-8:
            return -float("inf")
        if np.any(alpha > y + 1e-12):
            return -float("inf")

        Ls = self.Lstar(alpha, y)
        if not np.isfinite(Ls):
            return -float("inf")

        Us = self.Ustar(X.T @ alpha)
        return float(-(Ls + Us))

    def build_dual_feasible(self, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
        # Start from alpha_raw = y - mu, center it, then shrink toward a
        # strictly feasible centered anchor so that sum(alpha)=0 and alpha<y.
        eta_c = np.clip(eta, -self.eta_clip, self.eta_clip)
        mu = np.exp(eta_c)
        alpha_raw = y - mu
        alpha_c = alpha_raw - float(np.mean(alpha_raw))

        dom_eps = 1e-10
        if np.all(alpha_c < y - dom_eps):
            return alpha_c

        zero = y <= dom_eps
        pos = ~zero

        # Degenerate all-zero responses have no useful interior centered anchor.
        # Fall back to a tiny centered vector; such data are excluded in the paper.
        if not np.any(pos):
            alpha = alpha_c - float(np.mean(alpha_c))
            return np.minimum(alpha, y - dom_eps)

        alpha0 = np.zeros_like(y, dtype=float)
        if np.any(zero):
            zc = int(np.sum(zero))
            pc = int(np.sum(pos))
            min_pos_y = float(np.min(y[pos]))
            delta = min(1e-8, 0.5 * min_pos_y * pc / max(1, zc))
            delta = max(delta, 1e-12)
            alpha0[zero] = -delta
            alpha0[pos] = delta * zc / pc
        # If all y_i > 0, alpha0=0 is already strictly feasible and centered.

        diff = alpha_c - alpha0
        upper = y - dom_eps
        mask = diff > 0.0
        rho = 1.0
        if np.any(mask):
            rho = min(rho, float(np.min((upper[mask] - alpha0[mask]) / diff[mask])))
        rho = min(1.0, max(0.0, rho))
        alpha = alpha0 + rho * diff
        alpha = alpha - float(np.mean(alpha))

        # Numerical guard: if roundoff touches the boundary, shrink once more.
        if np.any(alpha >= y):
            alpha = alpha0 + 0.999999 * rho * diff
            alpha = alpha - float(np.mean(alpha))
        return alpha

    def gap(self, y: np.ndarray, eta: np.ndarray, w: np.ndarray, alpha: np.ndarray, X: np.ndarray):
        P = self.primal(y, eta, w)
        D = self.dual(alpha, y, X)
        if not np.isfinite(D):
            return P, D, float("inf")
        return P, D, float(P - D)


# -------------------------
# End-to-end trainer
#   Patch: working-set safe screening (drop columns),
#          two-reference pruning+screening,
#          optional path/CV drivers below.
# -------------------------


# ══════════════════════════════════════════════════════
# § 9  主模型：PoissonEN_LCM_SPP
#      迭代：求解 → 计算对偶间隙 → WS剪枝 → 枚举新模式
# ══════════════════════════════════════════════════════
@dataclass
class FitResult:
    patterns: List[Pattern]
    params: ENParams
    eta: np.ndarray
    alpha: np.ndarray
    gap: float
    r: float

    # new:
    total_time: float
    round_times: List[Dict[str, float]]
    enum_stats_per_round: List[EnumStats]
    ws_screen_removed_per_round: List[int]


class PoissonEN_LCM_SPP:
    def __init__(
        self,
        lam: float,
        kappa: float,
        min_support: int = 2,
        max_len: int = 4,
        top_k_add: int = 200,
        max_rounds: int = 10,
        solver_max_iter: int = 500,
        solver_tol: float = 1e-6,
        eta_clip: float = 10.0,
        seed: int = 0,
        # pruning/screening controls
        do_ws_screen: bool = True,
        enum_mode_single: str = "both",   # "vprune"/"uscreen"/"both"
        enum_mode_two: str = "both",      # same, when two-ref is used
        radius_mode: str = "heuristic",   # "heuristic" or "certified"
    ):
        self.lam = float(lam)
        self.kappa = float(kappa)
        self.min_support = int(min_support)
        self.max_len = int(max_len)
        self.top_k_add = int(top_k_add)
        self.max_rounds = int(max_rounds)
        self.solver_max_iter = int(solver_max_iter)
        self.solver_tol = float(solver_tol)
        self.eta_clip = float(eta_clip)
        self.rng = np.random.default_rng(seed)
        self.do_ws_screen = bool(do_ws_screen)
        self.enum_mode_single = enum_mode_single
        self.enum_mode_two = enum_mode_two
        if radius_mode not in ("heuristic", "certified"):
            raise ValueError("radius_mode must be 'heuristic' or 'certified'")
        self.radius_mode = radius_mode

    def _poisson_radius_scale(self, y: np.ndarray, alpha: np.ndarray, mu: np.ndarray, G: float = 0.0) -> float:
        """
        Return gamma in r = sqrt(2 * gamma * G).

        Heuristic mode: gamma = mean(mu).  Fast and tight in practice but not a
        formal certificate (gamma may not satisfy y_i - alpha_i <= gamma for all
        alpha in the dual-gap ball).

        Certified mode: self-consistent safe gamma derived as follows.
        For any alpha in Ball(ã, r), component-wise: alpha_i >= ã_i - r, so
            y_i - alpha_i <= max_i(y_i - ã_i) + r  =:  M + r.
        Requiring M + sqrt(2*gamma*G) <= gamma yields the quadratic
            gamma^2 - (2M + 2G)*gamma + M^2 >= 0,
        whose smallest admissible root is
            gamma_cert = M + G + sqrt(G*(G + 2*M)).
        At this gamma, M + r = gamma holds exactly (tight self-consistent bound),
        guaranteeing y_i - alpha_i <= gamma for every alpha in the ball, which
        makes the strong-concavity argument and thus the pruning rules certified.
        """
        if self.radius_mode == "heuristic":
            return float(np.mean(mu))
        M = float(np.max(np.asarray(y, float) - np.asarray(alpha, float)))
        M = max(M, 1e-12)
        if G <= 0.0 or not math.isfinite(G):
            return M
        return float(M + G + math.sqrt(G * (G + 2.0 * M)))

    @staticmethod
    def _transactions_to_item_tidsets(transactions: List[List[int]]):
        item_to_tids: Dict[int, List[int]] = {}
        for tid, items in enumerate(transactions):
            for it in items:
                item_to_tids.setdefault(it, []).append(tid)
        item_tidsets = {it: np.array(sorted(tids), dtype=np.int32) for it, tids in item_to_tids.items()}
        items_sorted = sorted(item_tidsets.keys())
        return item_tidsets, items_sorted

    def fit(
        self,
        transactions: List[List[int]],
        y: np.ndarray,
        verbose: bool = True,
        # optional two-reference input: (alpha2, r2) from neighbor solution (Alg.3/4)
        ref2: Optional[Tuple[np.ndarray, float]] = None,
    ) -> FitResult:
        t_all = Timer()
        round_times: List[Dict[str, float]] = []
        enum_stats_round: List[EnumStats] = []
        ws_removed_round: List[int] = []

        y = np.asarray(y, float)
        n = len(transactions)
        assert y.shape == (n,)

        item_tidsets, items_sorted = self._transactions_to_item_tidsets(transactions)
        enum = LCMEnumerator(item_tidsets, items_sorted, n, min_support=self.min_support, max_len=self.max_len)

        patterns: List[Pattern] = []
        have: Set[Tuple[int, ...]] = set()
        warm: Optional[ENParams] = None

        solver = PoissonElasticNetSolver(
            self.lam, self.kappa,
            max_iter=self.solver_max_iter, tol=self.solver_tol,
            eta_clip=self.eta_clip
        )
        gapcalc = PoissonENGap(self.lam, self.kappa, eta_clip=self.eta_clip)

        last_params = None
        last_eta = None
        last_alpha = None
        last_gap = None
        last_r = None

        for rd in range(1, self.max_rounds + 1):
            t_round = Timer()

            X = build_X_from_patterns(n, patterns) if patterns else np.zeros((n, 0), float)

            t_solver = Timer()
            params = solver.fit(X, y, warm=warm)
            solver_time = t_solver.elapsed()

            eta = (params.b + X @ params.w) if X.shape[1] else np.full(n, params.b, dtype=float)

            t_gap = Timer()
            alpha = gapcalc.build_dual_feasible(y, eta)
            P, D, G = gapcalc.gap(y, eta, params.w, alpha, X)
            gap_time = t_gap.elapsed()


            eta_c = np.clip(eta, -self.eta_clip, self.eta_clip)
            mu_cur = np.exp(eta_c)
            # Radius scale for the dual-gap ball.
            _G_for_radius = float(G) if np.isfinite(G) else 0.0
            g = self._poisson_radius_scale(y, alpha, mu_cur, G=_G_for_radius)
            r = math.sqrt(max(0.0, 2.0 * g * _G_for_radius))

            if verbose:
                print(f"[round {rd}] m={len(patterns)}  P={P:.6g}  D={D:.6g}  gap={G:.6g}  g={g:.6g}  r={r:.6g}")

            _gap_tol = 1e-4
            gap_converged = np.isfinite(G) and G < _gap_tol and rd > 1
            if verbose and gap_converged:
                print(f"  Restricted gap {G:.2e} < {_gap_tol:.0e}; checking for new patterns before stopping.")


            last_alpha = alpha.copy()
            last_gap = float(G)
            last_r = float(r)
            ws_time = 0.0
            removed = 0

            if self.do_ws_screen and patterns and np.isfinite(G):
                t_ws = Timer()
                old_len = len(patterns)

                keep: List[Pattern] = []
                keep_idx: List[int] = []

                for j, p in enumerate(patterns):
                    tid = np.fromiter(p.tidset, dtype=np.int32)
                    s = int(tid.size)
                    if s == 0:
                        continue
                    dot = float(np.sum(alpha[tid]))
                    normc = _centered_norm_from_support(s, n)
                    u = abs(dot) + r * normc
                    if u >= self.lam:
                        keep.append(p)
                        keep_idx.append(j)

                new_len = len(keep)
                removed = old_len - new_len

                if removed > 0:
                    patterns = keep
                    have = {p.itemset for p in patterns}
                     # ✅ shrink current solution params to match the reduced working set
                    params = ENParams(b=params.b, w=params.w[keep_idx].copy())

                    # ✅ recompute eta under the reduced set (optional but keeps things consistent)
                    # (no need to rebuild X now; just rebuild next round. Here we keep eta consistent for printing/gap.)
                    X_red = build_X_from_patterns(n, patterns) if patterns else np.zeros((n, 0), float)
                    eta = (params.b + (X_red @ params.w if X_red.shape[1] else 0.0))


                    # shrink warm-start weights to kept indices
                    if warm is not None and warm.w.size:
                        warm = ENParams(b=warm.b, w=warm.w[keep_idx].copy())

                ws_time = t_ws.elapsed()

            # 记得把 removed/ws_time 记录到你的 ws_removed_round / round_times 里

            ws_removed_round.append(int(removed))
                    # rebuild X size next round via patterns list
            # refresh last snapshots (whether or not ws-screen happened)


            # -------------------------
            # enumerate new patterns with SPP pruning/screening
            # -------------------------
            mu = mu_cur  # BUG6: reuse mu already computed above
            grad_eta = mu - y
            enum.reset_stats()
            t_mine = Timer()
            if ref2 is None:
                new_patts = enum.dfs_with_spp(
                    alpha=alpha,
                    grad_eta=grad_eta,
                    r=r,
                    lam=self.lam,
                    already_have=have,
                    top_k=self.top_k_add,
                    mode=self.enum_mode_single,
                )
            else:
                alpha2, r2 = ref2
                new_patts = enum.dfs_with_spp(
                    alpha=alpha,
                    grad_eta=grad_eta,
                    r=r,
                    lam=self.lam,
                    already_have=have,
                    top_k=self.top_k_add,
                    alpha2=alpha2,
                    r2=r2,
                    mode=self.enum_mode_two,
                )
            mine_time = t_mine.elapsed()
            enum_stats_round.append(enum.get_stats())


            if not new_patts:
                last_params = params
                last_eta = eta.copy()
                last_alpha = alpha.copy()
                last_gap = float(G)
                last_r = float(r)
                if verbose:
                    print("No new patterns found (after SPP pruning/screening). Stop.")
                break

            added = 0
            for p in new_patts:
                if p.itemset in have:
                    continue
                have.add(p.itemset)
                patterns.append(p)
                added += 1

            if verbose:
                print(f"  added {added} patterns; working set size -> {len(patterns)}")
            # ✅ ensure params matches current working-set length (patterns)
            if params.w.shape[0] != len(patterns):
                w_aligned = np.zeros(len(patterns), dtype=float)
                k = min(params.w.shape[0], w_aligned.shape[0])
                if k > 0:
                    w_aligned[:k] = params.w[:k]
                params = ENParams(b=params.b, w=w_aligned)
            # refresh last snapshots AFTER patterns/params are consistent
            last_params = params
            # keep eta consistent with aligned params (optional but safer)
            X_now = build_X_from_patterns(n, patterns) if patterns else np.zeros((n, 0), float)
            eta = params.b + (X_now @ params.w if X_now.shape[1] else 0.0)
            last_eta = eta.copy()
            round_times.append({
                "round": float(rd),
                "m": float(len(patterns)),
                "solver_time": float(solver_time),
                "gap_time": float(gap_time),
                "ws_time": float(ws_time),
                "mine_time": float(mine_time),
                "round_total": float(t_round.elapsed()),
                "gap": float(G),
                "r": float(r),
                "radius_scale": float(g),
                "ws_removed": float(removed),
             })

            if added == 0:
                break

            # expand warm-start weights
            old_w = params.w
            new_w = np.zeros(len(patterns), float)
            if old_w.size:
                new_w[:old_w.size] = old_w
            warm = ENParams(b=params.b, w=new_w)

       # fallback (should rarely trigger)
        if last_params is None:
            params0 = ENParams(b=float(np.log(np.mean(y) + 1e-12)), w=np.zeros(0))
            eta0 = np.full(n, params0.b)
            alpha0 = np.zeros(n)
            return FitResult(
                patterns=[],
                params=params0,
                eta=eta0,
                alpha=alpha0,
                gap=float("inf"),
                r=float("inf"),
                total_time=t_all.elapsed(),
                round_times=round_times,
                enum_stats_per_round=enum_stats_round,
                ws_screen_removed_per_round=ws_removed_round,
            )
        # ===== FINAL ALIGNMENT (must be right before return) =====
        m = len(patterns)
        if last_params is None:
            raise RuntimeError("last_params is None unexpectedly")

        if last_params.w.shape[0] != m:
            w_aligned = np.zeros(m, dtype=float)
            k = min(last_params.w.shape[0], m)
            if k > 0:
                w_aligned[:k] = last_params.w[:k]
            last_params = ENParams(b=last_params.b, w=w_aligned)

        # keep eta consistent with the returned params/patterns
        X_last = build_X_from_patterns(n, patterns) if patterns else np.zeros((n, 0), float)
        last_eta = last_params.b + (X_last @ last_params.w if X_last.shape[1] else 0.0)
        # (optional) alpha/gap refresh if you care
        assert last_params.w.shape[0] == len(patterns), (last_params.w.shape[0], len(patterns))

        # normal return
        return FitResult(
            patterns=patterns,
            params=last_params,
            eta=last_eta,
            alpha=last_alpha,
            gap=last_gap,
            r=last_r,
            total_time=t_all.elapsed(),
            round_times=round_times,
            enum_stats_per_round=enum_stats_round,
            ws_screen_removed_per_round=ws_removed_round,
        )



# ══════════════════════════════════════════════════════
# § 10  路径与交叉验证驱动（fit_path / fit_cv）
# ══════════════════════════════════════════════════════
# -------------------------
# Path & CV drivers (Alg.2 / Alg.4 style)
# -------------------------

def fit_path(
    model_factory: Callable[[float], PoissonEN_LCM_SPP],
    transactions: List[List[int]],
    y: np.ndarray,
    lam_list: List[float],
    verbose: bool = True,
    use_two_ref: bool = True,
    ):
    """
    Pathwise (lambda decreasing) with optional two-reference from previous solution.
    Returns list of FitResult (one per lambda).
    """
    results: List[FitResult] = []
    prev_ref = None
    for lam in lam_list:
        model = model_factory(float(lam))
        res = model.fit(transactions, y, verbose=verbose, ref2=prev_ref if use_two_ref else None)
        results.append(res)
        prev_ref = (res.alpha, res.r)
    return results

def _kfold_indices(n: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, K)
    for k in range(K):
        val = folds[k]
        trn = np.concatenate([folds[i] for i in range(K) if i != k])
        yield trn, val

def _subset_transactions(trans: List[List[int]], idx: np.ndarray) -> List[List[int]]:
    return [trans[i] for i in idx.tolist()]

# def fit_cv(
#     model_factory: Callable[[float], PoissonEN_LCM_SPP],
#     transactions: List[List[int]],
#     y: np.ndarray,
#     lam_list: List[float],
#     K: int = 5,
#     seed: int = 0,
#     verbose: bool = False,
#     use_two_ref: bool = True,
# ):
#     """
#     K-fold CV over lambda list, returns (best_index, mean_scores).
#     Score uses Poisson NLL on validation folds.
#     """
#     y = np.asarray(y, float)
#     n = len(transactions)
#     scores = np.zeros(len(lam_list), float)
#
#     for fold, (trn_idx, val_idx) in enumerate(_kfold_indices(n, K, seed)):
#         trans_trn = _subset_transactions(transactions, trn_idx)
#         y_trn = y[trn_idx]
#         trans_val = _subset_transactions(transactions, val_idx)
#         y_val = y[val_idx]
#
#         prev_ref = None
#         for t, lam in enumerate(lam_list):
#             model = model_factory(float(lam))
#             res = model.fit(trans_trn, y_trn, verbose=verbose, ref2=prev_ref if use_two_ref else None)
#
#             # evaluate NLL on validation set using learned patterns (binary columns)
#             Xv = build_X_from_patterns_on_transactions(trans_val, res.patterns) if res.patterns else np.zeros((len(trans_val), 0), float)
#             eta_v = res.params.b + (Xv @ res.params.w if Xv.shape[1] else 0.0)
#             eta_v = np.clip(eta_v, -model.eta_clip, model.eta_clip)
#             mu_v = np.exp(eta_v)
#             nll = float(np.sum(mu_v - y_val * eta_v))
#             scores[t] += nll
#
#             prev_ref = (res.alpha, res.r)
#
#     scores /= K
#     best = int(np.argmin(scores))
#     return best, scores
def fit_cv(
    model_factory, transactions, y, lam_list, K=5, seed=0, use_two_ref=True
):
    y = np.asarray(y, float)
    n = len(transactions)

    # 每个 lambda 累计各指标
    acc = {lam: {"NLL": 0.0, "Deviance": 0.0, "MAE": 0.0, "RMSE": 0.0, "D_null": 0.0} for lam in lam_list}

    for fold, (trn_idx, val_idx) in enumerate(_kfold_indices(n, K, seed)):
        trans_trn = _subset_transactions(transactions, trn_idx)
        y_trn = y[trn_idx]
        trans_val = _subset_transactions(transactions, val_idx)
        y_val = y[val_idx]

        prev_ref = None
        mu0 = np.full_like(y_val, np.mean(y_trn))   # 用训练均值作 null（更标准）
        D_null = poisson_deviance(y_val, mu0)

        for lam in lam_list:
            model = model_factory(float(lam))
            res = model.fit(trans_trn, y_trn, verbose=False, ref2=prev_ref if use_two_ref else None)

            Xv = build_X_from_patterns_on_transactions(trans_val, res.patterns) if res.patterns else np.zeros((len(trans_val), 0), float)
            eta_v = res.params.b + (Xv @ res.params.w if Xv.shape[1] else 0.0)
            eta_v = np.clip(eta_v, -model.eta_clip, model.eta_clip)
            mu_v = np.exp(eta_v)

            nll = float(np.sum(mu_v - y_val * eta_v))
            dev = poisson_deviance(y_val, mu_v)
            mae = float(np.mean(np.abs(y_val - mu_v)))
            rmse = float(np.sqrt(np.mean((y_val - mu_v) ** 2)))

            acc[lam]["NLL"] += nll
            acc[lam]["Deviance"] += dev
            acc[lam]["MAE"] += mae
            acc[lam]["RMSE"] += rmse
            acc[lam]["D_null"] += D_null

            prev_ref = (res.alpha, res.r)

    # 平均并算 pseudoR2
    out = {}
    for lam in lam_list:
        out_lam = {k: acc[lam][k] / K for k in ["NLL", "Deviance", "MAE", "RMSE", "D_null"]}
        out_lam["PseudoR2"] = float(1.0 - out_lam["Deviance"] / max(1e-12, out_lam["D_null"]))
        out[lam] = out_lam

    # 选最优（比如按 NLL 最小）
    best_lam = min(lam_list, key=lambda l: out[l]["NLL"])
    return best_lam, out

# 用法


# ══════════════════════════════════════════════════════
# § 11  合成数据生成器
# ══════════════════════════════════════════════════════
# -------------------------
# Synthetic data (your original + planted interactions)
# -------------------------

def make_synthetic_transactions(n: int = 300, n_items: int = 50, avg_len: int = 7, seed: int = 1):
    rng = np.random.default_rng(seed)
    trans = []
    for _ in range(n):
        k = max(1, int(rng.poisson(avg_len)))
        items = rng.choice(n_items, size=min(k, n_items), replace=False)
        trans.append(sorted(items.tolist()))
    return trans

def add_poisson_labels_from_true_patterns(
    transactions: List[List[int]],
    true_patterns: List[Tuple[int, ...]],
    weights: List[float],
    b: float = -0.8,
    seed: int = 2,
    eta_clip: float = 10.0
):
    rng = np.random.default_rng(seed)
    n = len(transactions)

    patt_sets = [set(p) for p in true_patterns]
    eta = np.full(n, b, float)
    for pset, w in zip(patt_sets, weights):
        x = np.array([1.0 if pset.issubset(set(t)) else 0.0 for t in transactions], float)
        eta += w * x
    eta = np.clip(eta, -eta_clip, eta_clip)
    mu = np.exp(eta)
    return rng.poisson(mu).astype(float)

def make_binary_transactions_with_interactions(
    n: int = 300,
    n_items: int = 60,
    base_on_prob: float = 0.10,
    planted_patterns: Optional[List[Tuple[int, ...]]] = None,
    planted_prob: Optional[List[float]] = None,
    seed: int = 0,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    if planted_patterns is None:
        planted_patterns = [(3,7), (10,11,12), (25,)]
    if planted_prob is None:
        planted_prob = [0.25, 0.18, 0.30]
    assert len(planted_patterns) == len(planted_prob)

    trans: List[List[int]] = []
    patt_sets = [set(p) for p in planted_patterns]

    for _ in range(n):
        items: Set[int] = set()
        base_mask = rng.random(n_items) < base_on_prob
        items.update(np.where(base_mask)[0].tolist())
        for pset, p in zip(patt_sets, planted_prob):
            if rng.random() < p:
                items.update(pset)
        trans.append(sorted(items))
    return trans

def support(trans: List[List[int]], patt: Tuple[int, ...]) -> int:
    s = set(patt)
    return sum(1 for t in trans if s.issubset(t))


# ══════════════════════════════════════════════════════
# § 12  日志与剪枝报告工具
# ══════════════════════════════════════════════════════
def print_round_report(res: FitResult):
    for rt, st, rm in zip(res.round_times, res.enum_stats_per_round, res.ws_screen_removed_per_round):
        denom = max(1, st.nodes_visited)
        print(
            f"round {int(rt['round'])}: "
            f"m={int(rt['m'])} gap={rt['gap']:.3g} r={rt['r']:.3g} | "
            f"time(solver={rt['solver_time']:.3f}s, mine={rt['mine_time']:.3f}s, ws={rt['ws_time']:.3f}s, total={rt['round_total']:.3f}s) | "
            f"visited={st.nodes_visited} "
            f"v={st.pruned_v}({st.pruned_v/denom:.1%}) "
            f"can={st.pruned_canonical} "
            f"sup={st.pruned_support} "
            f"u={st.screened_u}({st.screened_u/denom:.1%}) "
            f"emit={st.emitted_closed} ws_removed={rm}"
        )
def summarize_pruning(res: FitResult) -> Dict[str, float]:
    tot_nodes = sum(s.nodes_visited for s in res.enum_stats_per_round)
    tot_sup  = sum(s.pruned_support for s in res.enum_stats_per_round)
    tot_v    = sum(s.pruned_v for s in res.enum_stats_per_round)
    tot_can  = sum(s.pruned_canonical for s in res.enum_stats_per_round)
    tot_u    = sum(s.screened_u for s in res.enum_stats_per_round)
    tot_emit = sum(s.emitted_closed for s in res.enum_stats_per_round)
    tot_ws_removed = sum(res.ws_screen_removed_per_round) if res.ws_screen_removed_per_round else 0

    denom = max(1, tot_nodes)
    hard_pruned = tot_sup + tot_can + tot_v

    # 时间拆分
    sum_solver = sum(rt.get("solver_time", 0.0) for rt in res.round_times)
    sum_gap    = sum(rt.get("gap_time", 0.0) for rt in res.round_times)
    sum_ws     = sum(rt.get("ws_time", 0.0) for rt in res.round_times)
    sum_mine   = sum(rt.get("mine_time", 0.0) for rt in res.round_times)

    return {
        "nodes_visited": float(tot_nodes),
        "emitted_closed": float(tot_emit),

        "pruned_support": float(tot_sup),
        "pruned_canonical": float(tot_can),
        "pruned_v": float(tot_v),
        "screened_u": float(tot_u),

        "hard_pruned_total": float(hard_pruned),
        "hard_prune_rate": float(hard_pruned / denom),

        "v_prune_rate": float(tot_v / denom),
        "u_screen_rate": float(tot_u / denom),

        "ws_removed_total": float(tot_ws_removed),

        "time_total_sec": float(res.total_time),
        "time_solver_sec": float(sum_solver),
        "time_gap_sec": float(sum_gap),
        "time_ws_sec": float(sum_ws),
        "time_mine_sec": float(sum_mine),
    }


def print_pruning_report(res: FitResult):
    s = summarize_pruning(res)
    print("\n=== Pruning/Timing Summary ===")
    print(f"Total time: {s['time_total_sec']:.4f} s  (solver={s['time_solver_sec']:.4f}, gap={s['time_gap_sec']:.4f}, ws={s['time_ws_sec']:.4f}, mine={s['time_mine_sec']:.4f})")
    print(f"Nodes visited: {int(s['nodes_visited'])}   Closed emitted: {int(s['emitted_closed'])}")
    print(f"Hard pruned: {int(s['hard_pruned_total'])}  (rate={s['hard_prune_rate']:.2%})")
    print(f"  - support pruned:   {int(s['pruned_support'])}")
    print(f"  - canonical pruned: {int(s['pruned_canonical'])}")
    print(f"  - v-pruned:         {int(s['pruned_v'])}  (rate={s['v_prune_rate']:.2%})")
    print(f"Screened (u/u0): {int(s['screened_u'])}  (rate={s['u_screen_rate']:.2%})")
    print(f"WS-screen removed cols total: {int(s['ws_removed_total'])}")

    print("\n=== Per-round breakdown ===")
    for rt, st, rm in zip(res.round_times, res.enum_stats_per_round, res.ws_screen_removed_per_round):
        denom = max(1, st.nodes_visited)
        print(
            f"[round {int(rt['round'])}] m={int(rt['m'])} gap={rt['gap']:.3g} r={rt['r']:.3g} | "
            f"time: solver={rt['solver_time']:.3f}s gap={rt['gap_time']:.3f}s ws={rt['ws_time']:.3f}s mine={rt['mine_time']:.3f}s total={rt['round_total']:.3f}s | "
            f"visited={st.nodes_visited} v={st.pruned_v}({st.pruned_v/denom:.1%}) "
            f"can={st.pruned_canonical} sup={st.pruned_support} u={st.screened_u}({st.screened_u/denom:.1%}) "
            f"emit={st.emitted_closed} ws_removed={rm}"
        )
