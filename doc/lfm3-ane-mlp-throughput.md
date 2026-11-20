# LFM3 MLP 走 ANE：吞吐测量结果（issue #145）

测量时间：**2026-11-20 18:32–18:36 CST**，M1 Pro（MacBookPro18,3，16 GB），macOS 27.0.1。
预登记：`doc/lfm3-ane-mlp-preregister.md`（gates 冻结于 commit `7d754fc6`，先于任何设备运行）。
Receipts：`devlocal/lfm3-ane-mlp/`（4 个测量 pass 的原始 JSON、`window.log`、三次进程扫描、
`analysis4.txt` 四窗对账、`final.txt` 判定表、探针脚本）。

**判定：WIN —— 适用硬件 M1 Pro，M4 待测。** 在 LFM3 的 MLP 形状上，ANE（CoreML INT8，clef 配方）吞吐为 GPU（MLX BF16）的
**2.0–2.3 倍**（ρ 栏 0.40），三点 T 的隐含 prefill 回收 **44.6%–46.4%**（栏 12%），
即便按 clef 实测兑现率对折（×0.5）也仍有 **~22%**，覆盖 8–12% 缺口。

> **硬件适用范围（owner 要求写入，2026-11-20）**：本 WIN 的全部数字来自 M1 Pro
> （MacBookPro18,3）。M4 的 ANE 与 M1 Pro 的 ANE 不是同一代（core 数、每周期 MAC、
> INT8 路径都可能不同），45.7% 在 M4 上可能高也可能低。**在 M4 复跑确认之前，不得拿
> 这些数字直接在 M4 上做移植决定。** M4 判定规则（owner 定）：复跑后 ρ ≥ 2.0 ⇒ 才谈移植；
> ρ < 1.2 ⇒ 方向在目标硬件上不成立，记录并关闭；中间按预登记 §6 的冻结表判。
> 复跑方法见 §7 runbook。

---

## 1. Gate 对账（冻结项逐项）

| Gate | 结果 | 证据 |
|---|---|---|
| G0 预登记先行 | PASS | 预登记 commit `7d754fc6`（18:29 前）；首个设备运行 18:31:59（smoke，带锁） |
| G1 双锁 + ANE 独占 | PASS | 两个窗口全程持有 `/tmp/enginetailor-metal.lock` → `~/.cache/enginetailor/metal-measure.lock`（flock 等不抢）；窗口前/中/后三次 `ps` 扫描：无 apxinf-serve / native-worker / coreml / clef / coremltools 进程（`logs/ps-*.txt`，前后一致）；本机无 sudo 故无 powermetrics，独占证据 = 双锁 + 扫描 + 四窗可复现性 |
| G2 placement | PASS | 全部 20 个 ANE cell：所有 conv/elementwise op preferred = NeuralEngine（`results/ane-*.json` 逐 cell placement 字符串） |
| G3 A/A | PASS（重跑后） | 首窗 A/B 在小 cell 超 5%；按冻结规则重跑一窗（C/D）。机制对账（`analysis4.txt`）：**每窗首个探针 pass 的前 1–2 个 cell 带冷启动惩罚**（A≈C，B≈D，可复现、非噪声）；持续对 ANE 取 {B,D}、GPU 取 {C,D}，逐 cell 双窗比 ≤2%（绝大多数 ≤1%）。冷启动单列 §4 |
| G4 不移植 | PASS | `git diff --stat`：`doc/`（2 文件）+ 两个探针 example + 两个 crate 的 `Cargo.toml` 仅加 `[[example]]` 注册（无此行则 Linux 上 example 无法按 feature 门控，CI 会破；非库代码）。未触碰 `metal_lfm3`、未新增 `ane.rs`、两个 crate 库代码零改动 |
| G5 证据齐全 | PASS | receipts 目录含全部原始数字与分析脚本（`scripts/run-matrix*.sh`、`analyze*.py`、`final.py` 可重算） |

## 2. 实测吞吐（持续态；ANE = B/D 双窗、GPU = C/D 双窗中位，双窗比 ≤2%）

**ANE（CoreML INT8 1×1 conv，clef 配方，含每次 predict 的 host 开销）**

| T \\ S | 2048 | 3584 | 4096 | 5632 | 7680 |
|---|---|---|---|---|---|
| 128 | 0.667 ms / 2.41 TMAC/s | 0.922 / 3.06 | 1.061 / 3.04 | 1.567 / 2.83 | 1.858 / 3.25 |
| 512 | 2.086 / 3.09 | 3.322 / 3.39 | 3.924 / 3.28 | 5.404 / 3.28 | 7.316 / 3.30 |
| 2048 | 6.954 / 3.71 | 11.834 / 3.81 | 14.278 / 3.61 | 20.538 / 3.45 | 28.351 / 3.41 |

**GPU（MLX，同 MACs 链：gate/up matmul → silu·mul → down）**

| T \\ S | 2048 | 3584 | 4096 | 5632 | 7680 |
|---|---|---|---|---|---|
| 128 | 1.501 ms / 2.15 TFLOP/s | 2.128 / 2.65 | 2.438 / 2.64 | 3.215 / 2.76 | 4.191 / 2.88 |
| 512 | 4.183 / 3.08 | 7.099 / 3.18 | 8.099 / 3.18 | 10.918 / 3.25 | 14.801 / 3.26 |
| 2048 | 15.351 / 3.36 | 26.751 / 3.37 | 30.464 / 3.38 | 41.811 / 3.39 | 57.161 / 3.38 |

**Q4 `quantized_matmul` 臂**：每个 cell 都比 BF16 慢（T=2048 S=7680：79.3 vs 57.2 ms）——
prefill 计算密集，反量化开销吃不掉带宽节省（与 #144 在 M=1 的 roofline 结论互证）。
**ANE 的对照基线因此就是 BF16。**

**ρ = ANE MAC/s ÷ GPU BF16 FLOP/s**：全部 15 个 cell ∈ **[2.00, 2.31]**（S=7680：T=128/512/2048 =
2.26 / 2.02 / 1.99）。冻结栏杆 0.25/0.40 ⇒ **全部通过，余量 ~5×**。

## 3. 隐含回收算术（冻结公式，f = 0.73）

`R(T,s) = f · (gpu_full − max(gpu_rest, ane_s)) / gpu_full`，T=2048 的 ANE 侧用 4×512 tile 部署形态：

| split S（share） | T=128 | T=512 | T=2048 |
|---|---|---|---|
| 2048（0.267） | +17.0% | +19.2% | +19.6% |
| 3584（0.467） | +30.5% | +33.1% | +34.1% |
| 4096（0.533） | +35.9% | +38.0% | +38.8% |
| **5632（0.733）** | **+45.7%** | **+46.4%** | **+44.6%** |
| 7680（1.000） | +40.6% | +36.9% | +34.9% |

最优份额与闭式解 s\* = ρ/(1+ρ) ≈ 0.67–0.69 一致（实测 S=5632 = 0.733 最优）。
**R_max 三点 = 44.6%–46.4% ⇒ 五档判定 WIN（栏 12%）。** 保守折算：clef 在 M1 Pro 的
GPU+ANE 实测兑现率约为理想值的 ~55–60%（3603→2790 ms vs 列切分理想）；即便按 ×0.5 打折，
R ≈ 22–23%，仍 ≥ 缺口上限 12% 的 1.8 倍。

## 4. 附带发现：引擎冷启动（deployment caveat，不进判定）

四窗对账暴露一个可复现的**首窗首 cell 冷启动惩罚**（机制：空闲后 ANE/GPU 时钟未爬升）：

| cell | 冷（每窗第一 pass） | 持续（之后 pass） | 倍率 |
|---|---|---|---|
| ANE S=2048 T=128 | 1.920 / 1.907 ms（A/C 两窗一致） | 0.667 ms（B/D 一致） | 2.88× |
| ANE S=2048 T=512 | 2.872 / 2.968 | 2.086 | 1.38× |
| GPU S=2048 T=128 BF16 | 1.870 | 1.501 | 1.25× |

部署含义：24 层 × 若干 tile 的 prefill 是背靠背 predict（本探针 cell 内即如此），爬坡只影响
**引擎启动/空闲后的第一个请求**；持续服务中无影响。但短 prompt 缺口（128/512）恰好在小 cell
上，**移植棒**应量一次「冷启动首请求」的端到端影响。

## 5. 局限（继承预登记 §8，测量后仍成立）

1. 合成 INT8 权重 ⇒ 只测吞吐；INT8 质量门、Q4→INT8 host 量化属移植棒（#144 未合）。
2. R 是**上限**：未含 GPU 侧部分和相加、GPU∥ANE 带宽争抢、24 层 dispatch 队列效应；栏杆已留
   ~5× 余量，且 §3 给了 ×0.5 兑现率折算。
3. **M1 Pro 本地判定**。M4 的 GPU:ANE 速率比不同（M4 ANE 快得多、GPU 也快），ρ 不可直接外推；
   但探针与门槛已冻结，在 M4 复跑同一二进制即得 M4 判定。
4. GPU 臂为 eager MLX；compiled 图若更快只会抬高 GPU 基线（对 ANE 更严），本判定偏保守方向。

## 6. 结论与下一步（移植前提，**本棒不做**）

**WIN（M1 Pro 本地）。** ANE 在 LFM3 MLP 形状上持续吞吐 ~2× GPU，隐含回收 3.7× 于缺口上限。
**下一步不是移植，是把同一套探针在 M4 上复跑确认**（owner 2026-11-20 裁定）：

1. **M4 复跑本探针**（同二进制、同冻结 gates —— §7 runbook 一条命令链）。
   判定规则：ρ ≥ 2.0 ⇒ 才谈移植；ρ < 1.2 ⇒ 方向在目标硬件关闭并记录；中间按预登记冻结表。
2. **#144 packed 4-bit 合入**（ANE 权重从 Q4 checkpoint 在 host 量化 —— clef 已验证的路径）。
3. INT8 质量门：对 Q4 权重量化到 INT8 的 token 一致性（#130 硬门）。
4. 端到端兑现率：24 层 dispatch + GPU 部分和 + 争抢下的实测回收（预期打折，§3 已按 ×0.5 估）。
5. 冷启动首请求影响（§4）。

## 7. M4 复跑 runbook

探针已在 `main`（≥ `24a42d58`，PR #146）。在 M4 上（需 MLX C++ SDK 与 macOS SDK，
与 #130 测量纪律相同环境）：

```sh
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=<本机 MLX SDK 路径>          # M1 Pro 上是 ~/apxinf-runtime/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
cargo build --release -p apxinf-coreml --features native --example lfm3_mlp_probe \
                    -p apxinf-mlx     --features native --example lfm3_mlp_gpu_probe

# 持双 Metal 锁（flock，等不抢；无锁机制时至少确认无其他 ANE/GPU lane 在跑），
# 每个探针跑两遍（A/A），进程扫描证明 ANE 独占：
export APXINF_LFM3_PROBE_CACHE=<cache 目录>
./target/release/examples/lfm3_mlp_probe     ane-A.json   # 再跑一遍存 ane-B.json
./target/release/examples/lfm3_mlp_gpu_probe gpu-A.json   # 再跑一遍存 gpu-B.json
```

- 判定用 `devlocal/lfm3-ane-mlp/scripts/final.py` 的同一套公式（MACs = 3·T·2048·S；
  ρ = gpu_ms/ane_ms；R 按预登记 §3，f = 0.73）。
- A/A 规则同 G3：逐 cell 中位比 ≤5%；首窗首 cell 的冷启动惩罚（§4）若在 M4 复现，
  同样取持续 pass 对（机制对账，不许取中位凑）。
- placement 必须全 conv 链落 NeuralEngine（G2 原样适用）。
- **判定规则（owner）**：三点 T 的 ρ(S=7680) 全 ≥ 2.0 ⇒ M4 确认，才进入移植讨论；
  任一点 ρ < 1.2 ⇒ 方向在目标硬件上不成立，记录关闭；中间按预登记 §6 冻结表判。
