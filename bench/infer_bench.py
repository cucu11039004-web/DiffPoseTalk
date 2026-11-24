"""DiffPoseTalk 离线推理时延基准（独立脚本，不依赖 learn-3dmm）。

在仓库根目录运行（需要 conda activate dpt）：
    python bench/infer_bench.py                                   # 默认：demo/input/audio 下全部音频，步数 5/20/50/100/500
    python bench/infer_bench.py -a demo/input/audio/to_be_or.wav --steps 20 100 --repeats 5
    python bench/infer_bench.py --device cpu --repeats 2          # 对照 CPU

每个（音频，步数）组合：先预热 --warmup 次，再计时 --repeats 次，报告中位数。输出 JSON 到 bench/results/。
可调参数与预期：
    --steps     反向扩散步数；耗时近似与步数成正比，< 20 步动作会变糊。
    --dtype     float32 / bfloat16 / float16：用 torch.autocast 只降低矩阵运算精度（权重与扩散调度表仍是 float32）。
                GB10 上低精度通常更快；每次运行把各组合的表情系数存成同名 .npz，用 compare_results.py 与
                float32 的结果对照，判断低精度有没有让动作变形。
    --batch     n_repetitions，一次并行生成多少个样本（不同随机噪声）。GPU 上吞吐应优于串行。
    --seed      固定随机种子；同设备同种子结果可复现。
分段计时：
    audio_feat_s  HuBERT 特征提取（一次完整音频）
    denoise_s     所有窗口、所有步里去噪网络前向的总和（CFG 每步把 3 份输入拼成一个 batch）
    total_s       infer_coeffs 端到端，含音频预处理、采样、后处理
    rtf           total_s / 音频时长；< 1 表示快于实时。注意离线模式必须等整段音频到达，不等于流式延迟。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # 官方代码用相对路径（models/data、demo/output）
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
from demo import Demo  # noqa: E402

EXP_NAME, ITER = "head-SA-hubert-WM", 110000
STYLE_DIR = ROOT / "demo/input/style/head-L4H4-T0.1-BS32/iter_0026000"


def respace(model, n_steps: int) -> None:
    """时间步重排（Nichol & Dhariwal 2021）：在原 T 步里均匀取 n_steps 个时间步，网络仍接收原始时间步。
    DiffPoseTalk 预测干净样本 x0（target=sample），所以跳步后每步后验依然精确。重复调用请先重新加载模型。"""
    sched = model.diffusion_sched
    T = sched.num_steps
    if n_steps >= T:
        return
    tau = torch.round(torch.linspace(0, T, n_steps + 1)).long().to(sched.alpha_bars.device)
    ab = sched.alpha_bars[tau]
    alphas = torch.ones_like(ab)
    alphas[1:] = ab[1:] / ab[:-1]
    betas = 1 - alphas
    inflex = torch.zeros_like(betas)
    inflex[1:] = (1 - ab[:-1]) / (1 - ab[1:]) * betas[1:]
    sched.num_steps = n_steps
    sched.alpha_bars, sched.alphas, sched.betas = ab, alphas, betas
    sched.sigmas_flex, sched.sigmas_inflex = torch.sqrt(betas), torch.sqrt(inflex)

    net = model.denoising_net

    class RemapSteps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, motion, audio, person, prev_motion, prev_audio, step, indicator=None):
            return self.net(motion, audio, person, prev_motion, prev_audio, tau[step], indicator)

    model.denoising_net = RemapSteps()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def gpu_snapshot() -> str:
    """nvidia-smi 当前占用，用来判断机器是否被别人的任务占用（GB10 统一内存，显存字段可能是 N/A）。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{out} | apps: {apps or 'none'}"
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def load_demo(device: str, dtr: float) -> Demo:
    demo_args = SimpleNamespace(
        exp_name=EXP_NAME, iter=ITER, coef_stats=str(ROOT / "datasets/HDTF_TFHP/lmdb/stats_train.npz"),
        device=device, no_context_audio_feat=False, black_bg=False, save_coef=False,
        dynamic_threshold_ratio=dtr, dynamic_threshold_min=1.0, dynamic_threshold_max=4.0,
    )
    return Demo(demo_args, load_flame=False, load_renderer=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", "-a", type=Path, nargs="+", default=sorted((ROOT / "demo/input/audio").glob("*")))
    p.add_argument("--shape", "-c", type=Path, default=ROOT / "demo/input/coef/TH217.npy")
    p.add_argument("--style", "-s", type=Path, default=STYLE_DIR / "TH217.npy")
    p.add_argument("--scale_style", "-ss", type=float, default=3.0)
    p.add_argument("--scale_audio", "-sa", type=float, default=1.15)
    p.add_argument("--dtr", type=float, default=0.99)
    p.add_argument("--steps", type=int, nargs="+", default=[5, 20, 50, 100, 500])
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--batch", type=int, default=1, help="n_repetitions：并行生成的样本数")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="结果 JSON 路径，默认 bench/results/<时间>_<设备>_<dtype>.json")
    args = p.parse_args()

    import librosa

    dtype = getattr(torch, args.dtype)
    result = {
        "time": datetime.now().isoformat(timespec="seconds"), "device": args.device, "dtype": args.dtype,
        "batch": args.batch, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "gpu_before": gpu_snapshot(), "runs": [],
    }
    style = np.load(args.style).astype(np.float32)
    shape = np.load(args.shape).astype(np.float32).reshape(-1)[:100]

    for n_steps in args.steps:
        # 每个步数重新加载一次模型：respace 会替换调度表，不能叠加
        t0 = time.perf_counter()
        demo = load_demo(args.device, args.dtr)
        sync(args.device)
        load_s = time.perf_counter() - t0
        respace(demo.model, n_steps)
        cfg_cond = demo.model.guiding_conditions
        cfg_scale = [args.scale_audio if c == "audio" else args.scale_style for c in cfg_cond]

        # 分段计时钩子：在去噪网络前后同步并累计
        acc = {"denoise": 0.0, "calls": 0}
        state = {}

        def pre(_m, _inp):
            sync(args.device)
            state["t"] = time.perf_counter()

        def post(_m, _inp, _out):
            sync(args.device)
            acc["denoise"] += time.perf_counter() - state["t"]
            acc["calls"] += 1
            # 采样里的 torch.quantile 不支持 bf16/fp16，把低精度输出转回 float32 再交给扩散采样
            return _out.float() if torch.is_tensor(_out) and _out.is_floating_point() else _out

        demo.model.denoising_net.register_forward_pre_hook(pre)
        demo.model.denoising_net.register_forward_hook(post)

        for audio in args.audio:
            wav, _ = librosa.load(str(audio), sr=16000, mono=True)
            dur = len(wav) / 16000
            totals, feats, denoises, last = [], [], [], None

            for it in range(args.warmup + args.repeats):
                torch.manual_seed(args.seed)
                if args.device.startswith("cuda"):
                    torch.cuda.manual_seed_all(args.seed)
                acc["denoise"], acc["calls"] = 0.0, 0
                sync(args.device)
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32
                                                                              and args.device.startswith("cuda"))):
                    coef = demo.infer_coeffs(str(audio), shape, style, args.batch, cfg_mode=None,
                                             cfg_cond=cfg_cond, cfg_scale=cfg_scale)
                sync(args.device)
                total = time.perf_counter() - t0
                if it >= args.warmup:
                    totals.append(total)
                    denoises.append(acc["denoise"])
                last = coef

            # 单独测一次 HuBERT 特征提取（整段音频）
            n_motions = demo.n_motions
            n_sub = max(1, int(np.ceil(int(dur * demo.fps) / n_motions)))
            wav_t = torch.from_numpy(wav).to(args.device)
            wav_t = (wav_t - wav_t.mean()) / (wav_t.std() + 1e-5)
            pad = demo.n_audio_samples * n_sub - len(wav_t)
            if pad > 0:
                wav_t = torch.nn.functional.pad(wav_t, (0, pad))
            for it in range(args.warmup + args.repeats):
                sync(args.device)
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype,
                                                      enabled=(dtype != torch.float32
                                                               and args.device.startswith("cuda"))):
                    demo.model.extract_audio_feature(wav_t.unsqueeze(0), n_motions * n_sub)
                sync(args.device)
                if it >= args.warmup:
                    feats.append(time.perf_counter() - t0)

            med = statistics.median(totals)
            run = {
                "audio": str(audio), "audio_seconds": round(dur, 3), "frames": int(last["exp"].shape[1]),
                "windows": n_sub, "steps": n_steps, "load_s": round(load_s, 2),
                "total_s": round(med, 4), "total_all_s": [round(x, 4) for x in totals],
                "audio_feat_s": round(statistics.median(feats), 4),
                "denoise_s": round(statistics.median(denoises), 4),
                "denoise_per_call_ms": round(1000 * statistics.median(denoises) / max(1, n_steps * n_sub), 3),
                "rtf": round(med / dur, 4),
                "exp": last["exp"][0].float().cpu().numpy(),
            }
            print(f"[{args.device}/{args.dtype}] {audio.name} {dur:.1f}s  steps={n_steps:3d}  total={med:.3f}s  "
                  f"hubert={run['audio_feat_s']:.3f}s  denoise={run['denoise_s']:.3f}s  rtf={run['rtf']:.3f}")
            result["runs"].append(run)
        del demo
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    result["gpu_after"] = gpu_snapshot()
    if args.device.startswith("cuda"):
        result["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or out_dir / f"{stamp}_{args.device.replace(':', '')}_{args.dtype}_b{args.batch}.json"
    npz = out.with_suffix(".npz")
    np.savez_compressed(npz, **{f"{Path(r['audio']).stem}_s{r['steps']}": r["exp"] for r in result["runs"]})
    for r in result["runs"]:
        del r["exp"]
    result["exp_npz"] = npz.name  # 各组合的表情系数，用来跨精度/跨设备比较结果有没有变形
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
