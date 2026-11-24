# bench：DiffPoseTalk 时延基准

在仓库根目录、`conda activate dpt` 之后运行。

```bash
python bench/infer_bench.py                                    # 全部示例音频 × 步数 5/20/50/100/500，fp32，GPU
python bench/infer_bench.py -a demo/input/audio/to_be_or.wav --steps 20 100 --repeats 10
python bench/infer_bench.py --dtype bfloat16                   # 低精度对照
python bench/infer_bench.py --batch 4                          # 一次并行 4 个样本
python bench/infer_bench.py --device cpu --repeats 2           # CPU 对照
python bench/compare_results.py bench/results/<fp32>.npz bench/results/<bf16>.npz
```

结果写入 `bench/results/`（已在 `.gitignore` 里排除）：一个 JSON（耗时、`nvidia-smi` 前后快照、环境版本）和一个同名 `.npz`（各组合的表情系数）。

## 指标

| 字段 | 含义 |
|---|---|
| `total_s` | `infer_coeffs` 端到端中位数 |
| `audio_feat_s` | HuBERT 特征提取，整段音频一次 |
| `denoise_s` | 所有窗口、所有步里去噪网络前向的总和；`denoise_per_call_ms` 是单次前向（CFG 的 3 份输入拼成一个 batch） |
| `rtf` | `total_s` / 音频时长，小于 1 表示快于实时 |

离线模式要等整段音频都到了才开始，所以 RTF 小于 1 不等于流式延迟小；流式延迟还要加上窗口长度（4 秒）。

## 参考基线

Mac M5 Pro CPU、4.2 秒音频（2 个窗口）：500 步约 26 秒，100 步 6.3 秒，50 步 3.9 秒，20 步 2.2 秒，5 步 1.4 秒。

## 测量注意

- spark 可能多人共用：每次结果里有 `gpu_before` / `gpu_after`，别人有任务在跑时数据不可信，重测。
- GB10 是统一内存，`nvidia-smi` 的显存字段可能是 N/A，显存用结果里的 `peak_memory_mb`（torch 分配峰值）。
- 首次 CUDA 调用有编译和上下文初始化开销，所以默认预热 1 次。
