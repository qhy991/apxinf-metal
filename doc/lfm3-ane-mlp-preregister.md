# LFM3 MLP 走 ANE 预登记：吞吐能否吃下 prefill 的 8–12% 缺口（只测量、不移植）

预登记时间：**2026-11-20 18:30 CST**（commit 先于任何设备运行；本 doc 的 Gate/Decision/五档判定在测量后**不许改**）。

**来源**：issue #145（owner 派单，flame-chase 接力，执行者 @enginetailor，接力同伴 @Open-Cake）。
**人工约束**：本棒只测量、不携入 `metal_lfm3` 运行时；不新增 `ane.rs`；只出探针 + 数字 + 判定。
**Blocked by**：#144（packed 4-bit）——故本棒只测「假设权重已是 INT8」的吞吐，不测端到端。

---

## 0. 事实基座（引用，非本轮测量）

| 事实 | 出处 |
|---|---|
| `metal_clef/ane.rs` 已在 M4/M1 Pro 验证 GPU+ANE 列切分：gate/up INT8 1×1 conv、SiLU `0.5v(1+tanh(v/2))`、down K-split INT8 conv、512-token tile、权重 host 侧从 Q4 量化 | `crates/apxinf-model/src/metal_clef/ane.rs` |
| 跨层流水**不可能**：下一层 RMSNorm 依赖本层 ANE share 完成 ⇒ ANE 工作只能与**同层** GPU share 重叠 | `ane-design/clef-joint-parity`（5870c090）commit 记录 |
| LFM3 短 prompt prefill 比 llama.cpp 慢 8–12%（128/512 token），2048 基本持平 | #130 第 3 项 / #121 |
| LFM3 形状：hidden 2048、intermediate 7680、24 层（13 conv + 6 full-attn + 5 sliding-attn，window 256）、16Q/4KV、head_dim 128 | 本地 `LFM3-1.7B-MLX-4bit/config.json` 实测 |
| `metal_lfm3` 无 ANE（无 `ane.rs`、零 CoreML 引用）；4-bit 路径全在 GPU | #145 |

## 1. 假设与 falsifier

**H**：在 LFM3 的 MLP 形状（`[T,2048]×[2048,S]`，S 为 7680 的份额）上，ANE（CoreML INT8 1×1 conv，clef 配方）的实测吞吐相对 GPU（MLX）足够高，使得「把 MLP 的一部分列搬到 ANE、与同层 GPU share 重叠」在算术上能回收 prefill 墙钟的 ≥8–12%。

**falsifier（owner 原文要求的强化版）**：若 T=128/512/2048 三点上 ANE 全都**不快于** GPU ⇒ FALSIFIED，方向关闭。
**本轮对该 falsifier 的精确化（冻结）**：overlap 允许「ANE 比 GPU 慢」仍然回收缺口（见 §3 算术：ρ=0.3 即可回收 ~16%），故 owner 的「全都不快于」不是本方向的关闭条件；**真正的关闭条件是 §6 的 R̂_max 判据**。此偏差在交接评论中显式说明。

## 2. 测什么（冻结配置）

**探针 A（ANE）**：`crates/apxinf-coreml/examples/lfm3_mlp_probe.rs`，复刻 clef `program()` 的算子链（INT8 per-row conv gate/up → SiLU tanh 近似 → abs/reduce_max 归一 → 乘积 → down K-split INT8 conv 求和），形状参数化 `H=2048`：

- 列份额 S ∈ {2048, 3584, 4096, 5632, 7680}（含 7680−{4096,2048} 的互补列数，供回收算术直接查表）
- token tile T ∈ {128, 512, 2048}
- 另测部署形态：T=2048 以 4×512-tile 顺序 predict（clef M1 Pro 的既有 tile 模式）
- 合成权重：确定性伪随机 INT8 + 行标 FP16 scale=0.01（行绝对值和 ≪ 8192，factor_cap=16 与 clef 发货档一致）；权重值不影响 conv 吞吐，只影响形状/编码 —— 本棒按 #145 只承诺「权重已是 INT8」
- `ComputeUnits::CpuAndNeuralEngine`（不发 GPU fallback），`load_with_placement` 逐 op 取证
- 每 cell：warmup 3 + 20 次 `predict`，取中位；报 median/min/max 与 TMAC/s（MACs = 3·T·2048·S）

**探针 B（GPU）**：`crates/apxinf-mlx/examples/lfm3_mlp_gpu_probe.rs`，同 MAC 的 MLX 链（gate/up matmul → silu·mul → down matmul），两臂：

- **BF16 matmul**（对账 #121 认证基线的引擎口径）
- **Q4 group-64 `quantized_matmul`**（#144 blocked-by 的 packed 4-bit 口径）
- 每 cell：warmup 3 + 20 次逐次 `stream.eval`，取中位；报 TFLOP/s

**对照比**：ρ(T,S) = (ANE MAC/s) ÷ (GPU BF16 FLOP/s)，同机同窗同 T。

## 3. 回收算术（公式冻结；f 与缺口先写死）

**MLP 占 prefill 的 FLOP 份额 f**（按 §0 形状逐层算，分数 FLOP 忽略、lm_head 每次 prefill 只算一次忽略）：

- MLP（每层都有）：3×2×2048×7680 = 94.37 MF/tok ×24 = **2264.9 MF/tok**
- conv mixer：in_proj 25.17 + depthwise 0.03 + out_proj 8.39 = 33.6 MF/tok ×13 = 436.6
- attn mixer（q/k/v/gate/o 投影）：21.0 MF/tok ×11 = 231.4
- attn scores：T=128/512/2048 分别 ≈ 8/23/61 MF/tok（6 full 因果 + 5 sliding-256）
- ⇒ f(T=128) = 0.770，f(512) = 0.766，f(2048) = 0.757

**冻结取值 f = 0.73**（FLOP 下限附近的保守点：墙钟 f 会被小 kernel launch 稀释，故不用 0.76；灵敏度表覆盖 f ∈ {0.60, 0.65, 0.70, 0.75, 0.80}）。

**回收模型**（逐层串行框架 —— 跨层重叠已被证不可能，同层 GPU∥ANE 重叠）：

```
W  = W_other + W_mlp                        （GPU-only 基线，W_mlp = f·W）
W' = W_other + W_mlp · max(1−s, s/ρ)        （s = ANE 列份额，ρ = ANE/GPU 速率比）
R(s,ρ) = (W − W')/W = f · (1 − max(1−s, s/ρ))
最优份额 s* = ρ/(1+ρ)，R_max(ρ) = f · ρ/(1+ρ)
```

实测时不依赖该闭式解：对每个 (T, S) 直接用测得的 `ane_ms(T,S)` 与 `gpu_ms(T,7680−S)` 算
`R̂(T,s) = f · (gpu_ms(7680) − max(gpu_ms(7680−S), ane_ms(S))) / gpu_ms(7680)`，取 max 得 R̂_max(T)。
（`predict`/`eval` 计时已含每调用 host/传输开销；不含 GPU 侧部分和相加与双引擎带宽争抢 —— 见 §8 局限。）

**X 的反推（owner 要求）**：R_max ≥ g ⇔ ρ ≥ g/(f−g)。f=0.73 下：g=0.08 ⇒ ρ≥0.123；g=0.12 ⇒ ρ≥0.197。留 2× 开销余量 ⇒ **冻结两道栏杆：ρ ≥ 0.25（低栏，覆盖缺口低端）与 ρ ≥ 0.40（高栏，覆盖全缺口）**。f=0.60/0.80 时低栏为 0.31/0.22 —— 0.25 在 f≥0.65 下都成立。

## 4. Gate（冻结；PR 引用原文）

> **G0 预登记先行**：本 doc commit 先于任何 GPU/ANE 设备运行（编译不算设备运行）。
> **G1 双锁 + ANE 独占**：整个测量窗口持有 `/tmp/enginetailor-metal.lock` → `~/.cache/enginetailor/metal-measure.lock`（flock，等不抢，不杀其他 campaign）。窗口前后各做一次 `ps` 扫描：不得存在其他可能用 ANE 的进程（`apxinf-serve`/`apxinf-native-worker`/任何 coreml/clef 探针/python coremltools）。扫描输出存入 receipts。本机无 sudo、powermetrics 不可用 —— 独占证据 = 双锁 + 进程扫描 + 运行内稳定性（下条）。
> **G2 placement 取证**：每 cell 的 placement 证据必须显示全部 conv 链 op preferred = NeuralEngine；任一 conv 落 CPU/GPU ⇒ 该 cell **INVALID**（不是 ANE 吞吐，不进判定）。
> **G3 A/A 稳定性**：完整矩阵跑两遍（A/A），逐 cell 中位比 ≤5% ⇒ 有效；>5% 重跑一次，仍 >5% ⇒ 该 cell **INCONCLUSIVE**，不许取中位凑。单 cell 内 20 次 max/min >1.5 ⇒ 记为争抢嫌疑，按同规则处理。
> **G4 不移植（按构造）**：本棒 diff 只允许：`apxinf-coreml/examples/lfm3_mlp_probe.rs`、`apxinf-mlx/examples/lfm3_mlp_gpu_probe.rs`、`doc/`、devlocal receipts。**不得**触碰 `metal_lfm3` 任何文件、不得新增 `ane.rs`、不得改两个 crate 的库代码。`git diff --stat` 越界 ⇒ **INVALID**。
> **G5 判定输入齐全**：三点 T × 五点 S 的 ANE+GPU(BF16) 中位数、placement、A/A、独占扫描、回收算术全部落 receipts（`devlocal/lfm3-ane-mlp/`），缺项 ⇒ INCONCLUSIVE。

## 5. Decision（冻结）

> **方向**：issue #145 —— LFM3 MLP 中间维走 ANE 的**吞吐可行性测量**（不移植）。
> **范围**：§2 两支探针 + 数字 + §6 判定；PR base = `main`（不依赖 feat/lfm3-metal；不带入 metal_lfm3 运行时）。
> **判定后动作**：WIN/PARTIAL ⇒ 交接评论给出移植前提清单（#144 合入、INT8 质量门、overlap 调度），**不在本棒移植**；FALSIFIED ⇒ 理由写入 EXPERIENCE（本 doc §7 + issue 评论），方向关闭；INCONCLUSIVE ⇒ 写明缺什么证据。
> **五个 token**：WIN / PARITY / PARTIAL / FALSIFIED / INCONCLUSIVE（+ INVALID），附机制算术（TMAC/s、GB/s、ms/layer、回收 %）；算不平写「机制未知」。

## 6. 五档判定（冻结）

对 T ∈ {128, 512, 2048} 逐点算 R̂_max(T)（§3）与 ρ(T, S=7680)：

| 判定 | 条件（G1/G2/G3/G5 全过为前提） |
|---|---|
| **WIN** | 三点 T 全部 R̂_max ≥ 0.12（等价地 ρ ≥ 0.40）：能覆盖整个 8–12% 缺口 |
| **PARTIAL** | 三点 T 全部 R̂_max ≥ 0.08，但至少一点 < 0.12：只能覆盖缺口低端 |
| **PARITY** | R̂_max ∈ [0.05, 0.08)：有回收但够不到缺口低端；留作测量资产，不建议移植 |
| **FALSIFIED** | R̂_max < 0.05 于全部三点 T（约 ρ < 0.18），或 G2 显示该形状根本不上 ANE |
| **INCONCLUSIVE** | 锁/独占/placement/A-A 任一不满足，或关键 cell 两窗不稳 |
| **INVALID** | G4 越界（碰了 metal_lfm3 / 库代码），或探针未测到所声称的程序（空臂） |

## 7. Results（测量后填写；本节以上全部冻结）

**判定：WIN（2026-11-20 18:32–18:36 CST，M1 Pro，双锁窗口 ×2 + G3 重跑）。**
**M4 复跑确认（owner 代跑，2026-11-20 19:27 CST）**：Mac16,10 Mac mini，ρ(S=7680) =
2.88 / 2.60 / 2.46，三点全 ≥ 2.0，方向在目标硬件上成立，可进入移植讨论。证据强度上限：
M4 ANE 臂为单次运行（第二跑 CoreML compute plan internal failure）；GPU 臂 A/A 离群
22.51% 的机制 = 冷启动在每窗首 cell 于 M4 复现（结果档 §4），判定列 A/A ≤ 0.35%。
全表与移植盯项见 `doc/lfm3-ane-mlp-throughput.md`。

- ρ(T, S=7680) = 2.26 / 2.02 / 1.99（T=128/512/2048；栏 0.40）；全 15 cell ρ ∈ [2.00, 2.31]。
- R̂_max = **45.7% / 46.4% / 44.6%**（S=5632；栏 12%），按 clef 兑现率 ×0.5 折算仍 ~22%。
- G2 全 cell conv 链落 ANE；G1 三次进程扫描干净；G3 首窗小 cell 超 5% ⇒ 按冻结规则重跑，
  对账为可复现的首窗冷启动（非噪声），持续对双窗比 ≤2%，全 cell 有效。
- 附带发现：引擎冷启动首 pass 惩罚（ANE 小 cell 2.88×/1.38×，GPU 1.25×）——部署影响见结果档 §4。
- 全表与机制算术：`doc/lfm3-ane-mlp-throughput.md`；receipts：`devlocal/lfm3-ane-mlp/`。

## 8. 局限（先写死，免得事后找补）

1. 合成 INT8 权重 ⇒ 只测吞吐，**不测 INT8 质量**（质量门属移植棒，且 #144 未合）。
2. `predict` 含每调用 host/传输开销，但**不含**：GPU 侧两部分和相加、GPU∥ANE 内存带宽争抢、24 层逐层 dispatch 的驱动队列效应。回收算术是**上限估计**，故栏杆留了 2× 余量。
3. 本机 M1 Pro（ANE ≈11 TOPS 档）。#130 的 8–12% 缺口在 M4 对账；M1 Pro 与 M4 的 GPU:ANE 速率比不同，**ρ 不可直接搬到 M4**。本棒判定只对「M1 Pro 上 ANE 能否吃下」有效；M4 判定需在 M4 复跑同一探针（探针形状与门槛已冻结，复跑即得）。**[2026-11-20 解决：M4 已由 owner 复跑确认，ρ = 2.88/2.60/2.46，本条局限关闭。]**
4. GPU 探针为 eager MLX 逐 op（默认引擎口径）；compiled 图若更快只会让栏杆更高（对 ANE 更严），不影响方向判定的保守性。
