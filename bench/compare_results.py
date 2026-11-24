"""比较两次 infer_bench 运行存下的表情系数（.npz），判断精度或设备改变后动作有没有变形。

    python bench/compare_results.py bench/results/A.npz bench/results/B.npz
注意：扩散采样有随机性，随机数在不同设备/精度下不是同一串噪声，所以即使实现完全正确，两次结果也不会逐帧相同。
这里报告的相关系数应与“同设备不同随机种子”的相关系数对照着看：差距明显更大才说明精度有问题。
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print(f"{'组合':28s} {'逐帧RMSE':>9s} {'前5主成分相关':>14s} {'表情标准差比':>12s}")
for k in sorted(set(a.files) & set(b.files)):
    x, y = a[k], b[k]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    rmse = float(np.sqrt(((x - y) ** 2).mean()))
    u, _, vt = np.linalg.svd(np.concatenate([x, y]) - np.concatenate([x, y]).mean(0), full_matrices=False)
    px, py = x @ vt[:5].T, y @ vt[:5].T
    corr = np.mean([np.corrcoef(px[:, i], py[:, i])[0, 1] for i in range(5)])
    print(f"{k:28s} {rmse:9.4f} {corr:14.3f} {x.std() / (y.std() + 1e-9):12.3f}")
