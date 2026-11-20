# LFM3 融合 Q4 GEMV（M=1）— Gate + Decision（issue #142 后续）

预登记日期：2026-11-20 14:50 CST（`date` 实取）。本文件为 pre-registration：Gate 与 Decision 在任何设备运行前冻结；实现、测试、速度窗分属后续轮次。接力协议要求一棒一个 bounded 方向，本棒方向即本文档所冻结方向。

- 来源：issue #142 评论（2026-11-20 06:34 UTC，qhy991）：朴素 q4「净负」的根因分析与此分支（`feat/lfm3-metal`）QUEUE #8 已定案结论合并后的下一步 —— 融合 Q4 GEMV kernel。U32 packed loader 已由 QUEUE #8 交付（`apxinf_loader` 以 I32 透传 U32 载荷，`Array::from_bytes` 按 `MlxDType::U32` 原位重解释，PR #132 合并），本棒不做 loader。
- 权威基线：QUEUE #8 G4-speed 实测 native-4bit = 11.06–11.44 ms/tok（M4，ABBA，engine basis，A/A ≤3.5%），roofline 地板 9.67 ms（0.973 GB ÷ 100.6 GB/s），qmm dequant/launch 开销 ≈1.4 ms。
- 权威代码基座：本 ticket 分支 `agent/gemv-q4-fused-5f19bf17dd5e`，base `feat/lfm3-metal` @ `1864a1eb`。

## 1. 方向（bounded）

写一个 **M=1 融合 Q4 GEMV Metal kernel**（affine group-64 4-bit：packed U32 权重 `[N, K/8]` + BF16 scales/biases `[N, K/64]`），沿 K 在寄存器内反量化（`w = q*scale + bias`，FP32 累加），**永不 materialize `[N, K]`**。仅在 `APXINF_LFM3_4BIT=1` 且新增 `APXINF_LFM3_Q4_GEMV=1` 时，把 decode 步（M=1）的投影从 MLX `quantized_matmul` 切换到本 kernel；len>1（prefill）与未量化路径保持现实现不变。

- **为什么 bounded**：只改 `metal_lfm3` 的投影算子选择与 `apxinf-mlx` 新增一个 kernel + wrapper，不碰执行图结构、不碰 conv/attention 数学、不改 KV/state 布局。
- **收益上界**：若 qmm 的 ~1.4 ms/tok dequant/launch 开销全部收回，M4 11.1 → ~9.7–10.0 ms/tok（≤13%）。收益下界（falsifier）：同窗 ABBA 中与 qmm 臂差异 ≤ A/A spread ⇒ 判 NOT-WORTH/INCONCLUSIVE，不Claim 收益。
- **机制风险（预登记即承认）**：归约顺序必与 MLX qmv 不同 ⇒ 与 qmm-4bit 的 token 在近平局处会分叉。这是 G1-diag 已立案的「同精度跨引擎已知差」（lane=125020 vs native=25797 @ prompt 0 step 59）的同族现象，按 QUEUE #8 Lead 00:16 修订**不判 INVALID**，以 §2 G-token 门处置。

## 2. Gate（冻结；任一硬门触发即 INVALID，窗口数字不可解读）

| # | 门 | 判据 | 失败后果 |
|---|----|------|----------|
| G3 | 空臂 / effect proof | worker banner 新增 `q4_gemv=<true\|false>` 字段，与 dispatch 读同一个 env 源（`APXINF_LFM3_Q4_GEMV=1`）；armed 臂必须 `=true` 且 `native_4bit=true`。banner 缺失 / 二进制早于源码 mtime ⇒ 空臂陷阱，INVALID。 | INVALID（独立于速度门） |
| G4-shape | shape 不漂移 | kernel 输出与 `quantized_matmul` 逐 shape 一致（`[1, len, N]`，len=1 时 M=1 走 kernel，len>1 走 qmm）；不拆 K、不改读向、不 padded/bucketed。 | 视情形 INVALID 或 FALSIFIED |
| G5 | A/A ≤ 5% | 速度窗每臂每 cell 分段中位最坏 spread ≤ 5%；超 ⇒ 窗口作废重跑，不取中位凑数。 | 窗口作废（重跑） |
| G-token | 数值/生成 sanity | (a) op 级：kernel 输出 vs `quantized_matmul` 的 max abs diff ≤ 2e-2（BF16 输出量级 ~1e1，此容差 ≈ 1 ulp BF16 @ 8）且 mean abs diff ≤ 1e-3；(b) 端到端：≥3 个贪心 prompt，fused 臂 vs qmm 臂记录首处分叉位置与总 token 一致率；要求 ≥2/3 prompt 全程一致，分叉仅出现在允许引擎差的近平局语境并记录；(c) fused 臂输出为人类可读文本（非乱码/非重复死循环）。 | 任一失败 ⇒ 实现 INVALID，回退 dispatch |
| G4-speed | ABBA 速度 | 同窗 ABBA、≥3 段/臂、engine basis（`--engine-timings`），两臂：`4bit+qmm`（A）vs `4bit+fused`（B）。判定：**B 中位 ≤ A 中位 × 0.97**（3% 净增益，>2×A/A spread 上限）⇒ PASS；B ≤ A 但 >0.97A ⇒ NOT-WORTH（数字照记）；B > A ⇒ NEGATIVE。 | 五档照 §5 |
| G-mem | 内存不回升 | fused 臂 `active_bytes` ≤ qmm 臂 + 32 MB（kernel 不引入 [N,K] 中间量，理论上应持平或更低）。 | 超出 ⇒ 机制记录，FALSIFIED 候选 |

## 3. Decision（冻结）

- 开关：`APXINF_LFM3_Q4_GEMV=1`（仅当 `APXINF_LFM3_4BIT=1` 时生效；后者仍由 QUEUE #8 的 `q4::enabled()` 管辖）。默认双关 = BF16 路径，字节级不动。
- dispatch 条件（全部满足才走 kernel）：M=1（`x.numel()/K == 1`）、dtype 组合 BF16×U32×BF16×BF16、K%64==0、N%8==0（packed 行对齐）。任一不满足 ⇒ 回退 `quantized_matmul`。
- 测量机：本棒在 M1 Pro（16 GB，`~/models/LFM3-1.7B-MLX-4bit` 官方导出实读非零）上跑 op 级与端到端窗；绝对 ms/tok 与 M4 receipts 不可比，只引用**相对增益**与 op 级带宽利用率。跨机绝对值不 claim。
- 锁纪律：所有 build/测量在双 Metal 锁内进行（`/tmp/enginetailor-metal.lock`、`~/.cache/enginetailor/metal-measure.lock`），顺序获取、等待不抢；无干净窗口 ⇒ 记 INCONCLUSIVE 交接，不 claim 收益。

## 4. Cells / Order

1. op 级 probe（example，`q4_gemv_probe`）：真实 LFM3 形状（conv in_proj K=2048 N=6144、w1/w3 K=2048 N=7680、w2 K=7680 N=2048、attn q/o K=2048 N=2048、k/v K=2048 N=256、gate K=2048 N=16、embed K=2048 N=128000），各计时 kernel vs `quantized_matmul`，并给出隐含带宽（packed 字节 ÷ 时间）。
2. 端到端 token 门：native worker，贪心，≥3 prompts（含 issue #142 的 France 题），A=`4bit+qmm` B=`4bit+fused` 各跑一遍比对。
3. 速度窗：S2 口径 ctx1024 × 128 tok，ABBA A B B A A B（3 段/臂），engine basis；A/A 与 G3 banner 逐段核验。
4. 内存：两臂各取 worker 报告的 `active_bytes/peak_bytes`。

## 5. 判据（冻结；五档）

- **WIN**：G4-speed PASS（B ≤ 0.97A）+ G-token 全过 + G3/G4-shape/G5 净 ⇒ fused 在双开关下交付（保持 opt-in 形态），建议下一轮方向。
- **PARTIAL**：G4-speed NOT-WORTH 但 G-token 过 ⇒ 数字照记，不开默认。
- **NEGATIVE**：B > A ⇒ 机制记录（qmv 已最优），dispatch 默认关。
- **FALSIFIED**：隐含带宽与 qmm 臂差 >15% 且机制算术算不平 ⇒ 记录。
- **INVALID**：G3 空臂 / G5 作废 / G4-shape 漂移 / G-token 失败。
- 任一档位都开 PR 作为「已测量实现」，数字如实。

## 6. Relation

- QUEUE #8（PR #132）已定：U32 packed loader 在位、4-bit opt-in、G1′ −5pp 质量代价已接受。本棒在其上只换 M=1 算子实现。
- 已关死的门不重开：结构重组类（#128 合并 GEMV、QUEUE #2/3/4）。本棒不是结构重组（不拼接权重、不改 matmul 形状、不改图），是 QUEUE #8 已批准量化方向内的算子级替换；即便如此，G-token 门照跑，遵守「先 token 门再谈速度」。
- receipts 路径：本 worktree `devlocal/lfm3-q4-gemv/{scripts,results,logs}/`（git ignored）；文档 §7 回填真实值。

---

## 7. Results（2026-11-20 15:45 CST 回填；M1 Pro，双 Metal 锁内，窗口原始件在 `devlocal/lfm3-q4-gemv/window/raw/`）

### 7.1 判定汇总：**NEGATIVE**（G4-speed B>A）+ G-token 按原文失败 —— 数字如实，dispatch 保持默认关

| 门 | 结果 | 数字 |
|---|---|---|
| G3 空臂 | PASS | 两臂 banner 逐段核验：`native_4bit=true`；A 臂 `q4_gemv=false`、B 臂 `q4_gemv=true`；census `quantized_triples=154/154`；二进制由本分支源码构建 |
| G4-shape | PASS | 首轮发现 kernel 返回 rank-2 导致下游 `invalid slice axis`（decode 期望 rank-3）——修复为「保持输入 leading shape」后通过；最终窗内无 shape 错误 |
| G5 A/A | PASS | A 臂 3 段 spread **0.5%**，B 臂 **0.3%**（≤5%） |
| G-mem | PASS | 两臂 `active_bytes` 均 973,913,600（Δ=0）；`peak_bytes` 差 2 MB（jitter） |
| G-token (a) op 级 | **按原文失败**（意图满足） | max abs diff 0.031–0.047、mean 0.006–0.009：恰为输出幅值 1 个 BF16 ulp（[8,16) 档 ulp=0.03125），是归约顺序差的边界舍入翻转签名，非系统误差。**冻结阈值的绝对量纲标定错误**（把「1 ulp @ 8」写成绝对 2e-2；mean 阈值 1e-3 同理偏紧），见 §7.4 修订 |
| G-token (b) 端到端 | **按原文失败** | 3 贪心 prompt：prompt 0（France）**逐字一致**；prompt 1 首 token 即分叉（A 直答 / B thinking 前言，均正确）；prompt 2 第 ~60 词处近平局分叉。1/3 全程一致 < 冻结的 2/3。分叉形态符合 G1-diag 已立案的同精度跨引擎近平局翻转，输出全部连贯正确 |
| **G4-speed** | **NEGATIVE** | 同窗 ABBA（A B B A A B，3 段/臂，ctx1024 × 128 tok，engine basis）：**A(qmm) 9.29/9.37/9.37 → 中位 9.36 ms/tok；B(fused) 10.20/10.21/10.23 → 中位 10.21 ms/tok；B/A = 1.092，慢 9.2%** |

### 7.2 op 级探针（诊断件；`results/op-probe-*.json`，机器有后台负载噪声，取多轮稳定读数）

真实 LFM3 形状 M=1 下 kernel vs `quantized_matmul`：字节重 shape 中 **ffn.w2（K=7680, N=2048）稳定 +38%**（0.34 vs 0.46 ms），**embed.head（N=128000）稳定 −10%**（95 vs 107 GB/s），K=2048 中小 shape 互有 ±10% 内噪声。即 kernel 只在 K=7680 赢，而 w2 仅占每 token 字节 ~19%；K=2048 族（含 lm_head，~81%）全面小幅落后 ⇒ 端到端 −9.2%，与速度窗一致。

### 7.3 机制结论（写死，供后续棒引用）

1. **MLX 的 M=1 `quantized_matmul` 走的已是融合 qmv**（寄存器反量化、不 materialize [N,K]）：embed.head 实测 107 GB/s ≈ 本机可用带宽的高比例，QUEUE #8 在 M4 测得 qmm 达 roofline 的 87.7%。「先反量化再乘」的假设不成立，因此融合 kernel 没有可乘之机。
2. 本 kernel 在 K=7680 上的优势说明 MLX qmv 对宽 K 的 lane 分配非最优（120 组 / 32 lane 不均尾），但该形状占比太小。
3. 与 human 在 mini 的墙钟观察一致：这条栈的 decode 步在小投影上呈 launch/调度受限特征，砍权重的字节不再是主要杠杆；下一棒应回到 issue #130 item 1 的 launch-gap / 编译子图方向。

### 7.4 Gate 修订记录（透明修订，非为翻案）

- G-token (a) 阈值「max ≤ 2e-2、mean ≤ 1e-3」系预登记时把「1 ulp BF16 @ 幅值 8（=0.03125）」误写为绝对值；正确判据应为「max ≤ 1 ulp（按输出幅值档）且 mean 与边界翻转比例一致」。即使按修订判据通过，**G-token (b) 1/3 < 2/3 仍按原文失败，总判定 NEGATIVE 不变**。
- G-token (b) 的 2/3 门槛对本模型（thinking 长链、近平局极多）偏严：同精度换 kernel 必然早期分叉。若后续棒要重开此方向，建议按 Lead 00:16 对 G1-diag 的处置改为诊断项 + 质量 spot-check（本棒已满足：输出全部连贯正确）。

### 7.5 交付物

- 代码：`apxinf-mlx` 新增 `q4_gemv.rs` + `native/q4_gemv_bf16.metal`（kernel 演化 v1→v4 全部探针留档）；`metal_lfm3` dispatch（`APXINF_LFM3_Q4_GEMV=1`，默认关）；worker banner `q4_gemv=` 字段；op 探针 example `q4_gemv_probe`。
- receipts：`devlocal/lfm3-q4-gemv/{results,window,tokens,logs}/`（本地，git ignored）；速度窗原始 JSON 6 段在 `window/raw/`。
- 建议：dispatch 代码与 kernel 保留（opt-in 可复现测量），但不建议默认启用；下一棒方向见接力交接评论。
