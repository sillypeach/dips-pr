from model import *
from utils import *
from exp_shared import *

# ══════════════════════════════════════════════════════
# 数据诊断：分布图 + var/mean + 零比例
# ══════════════════════════════════════════════════════

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

# 实际分布
plt.subplot(1,2,1)
plt.hist(y, bins=30, density=True, alpha=0.7, label='actual y')

# 理论 Poisson 分布
from scipy.stats import poisson
mu = np.mean(y)
ks = np.arange(0, int(y.max())+1)
plt.plot(ks, poisson.pmf(ks, mu), 'r-o', markersize=4, label=f'Poisson(μ={mu:.2f})')
plt.legend()
plt.title('Distribution check')

# Q-Q 风格：实际分位数 vs Poisson 理论分位数
plt.subplot(1,2,2)
from scipy.stats import poisson
theoretical = poisson.ppf(np.linspace(0.01, 0.99, 200), mu)
actual = np.quantile(y, np.linspace(0.01, 0.99, 200))
plt.plot(theoretical, actual, 'o', markersize=3)
plt.plot([theoretical.min(), theoretical.max()],
         [theoretical.min(), theoretical.max()], 'r--')
plt.xlabel('Theoretical Poisson quantiles')
plt.ylabel('Actual quantiles')
plt.title('Q-Q plot')

plt.tight_layout()
plt.show()

# var/mean 比率（> 2 建议用负二项分布）
# var/mean ratio 先确认一下
print(np.var(y) / np.mean(y))  # 如果 >> 1 就用 NB

# 零比例 vs Poisson 理论零比例
print(f"零的比例: {np.mean(y==0):.3f}")
print(f"Poisson理论零比例: {np.exp(-np.mean(y)):.3f}")