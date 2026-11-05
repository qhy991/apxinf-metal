# 新优化方向候选（A 已批准并实施，结果见 FUSION-MATMUL.md；B 未批准）

任务 2 的交付物：候选清单。**批准前不写实现**——已履行：A 的实现与
campaign 都在收到批准（PR #5 评论 `08b361d…`）之后才开始。测量纪律沿用
[HANDOVER.md](HANDOVER.md)：3 轮反序、双锁单窗口、Swapouts ≠ 0 不进 KEEP。

已关闭方向不重做：fusion 组合、deeper-ANE、wavefront K=2、megakernel、
`mx.compile`、vis014、vision Metal batch、Core AI 硬件轴。

依据：[TIMELINE-OPERATOR.md](TIMELINE-OPERATOR.md) 的单请求算子时间线
（3453.2 ms；ANE predict 46.3 %、ANE wait 38.3 %、layer host 图构建
9.2 %、host stages 6.1 %；GPU busy 55.2 %，379 个 command buffer）。

## 候选 A：attention / GDN 的 matmul 级融合

把同一层内相邻的多个 Q4 投影 matmul 合并成更少的 GEMM。这与已否定的
elementwise fusion 不同（合并的是 matmul 本身），此前未评估过。

- **预计收益**：偏小，需实测。天花板是 layer host 图构建 319.3 ms
  （9.2 %）与 GPU command buffer 的发射/排队开销；合并 GEMM **不减
  FLOPs**，收益来自更大 N 的利用率提升与更少的 decode+GEMM 往返。
  预期 ≤2 %。
- **实现范围**：`metal_clef/model.rs` 的 `GpuKernels::{split_linear,
  single_linear, mlp}` + `weights.rs` 的 Q4 拼接辅助；单模块 ~150 行。
  分两步：
  - **A1** attention k/v 拼接：两次 `single_linear` → 一次（输出维均为
    64 的倍数）。
  - **A2** GDN `z` / `a` / `b` 合并：`split_projection(x, z)` + 两个
    1 列 `qmm(x, a|b)` → 一个 GEMM。注意 GPU 组合的二分
    `split_linear` 是为 kernel 约束而设，合并前需确认 N 上限。
- **parity 风险**：**A1 低**——k/v 输出维都是 64 的倍数，按 64 列块拼接
  保持每块的 scales/biases 不变，预期 bit 一致；**A2 中**——`a`/`b`
  各 1 列，合并进同一个 64 列块会改变 scale 分组 → 反量化值不同，
  **必须走 parity gate**（max|Δp| 阈值 + 42 案决策），不可声称 bit 一致。
- **测量方案**：reference（GPU-only）tier 先行：3 轮反序配对、双锁单窗、
  Swapouts 门；42 案 parity（决策 + max|Δp|）；KEEP 后再测 auto tier。

### 状态（A）
- **已批准**（2026-11-05 PR #5 评论 `08b361d…`）后才开始实施。
- **两处技术更正**（实测，见 [FUSION-MATMUL.md](FUSION-MATMUL.md) §2）：
  本节“`a`/`b` 各 1 列”不成立——实为 32 行（`linear_num_value_heads`）；
  “合并改变 scale 分组”不成立——Q4 scales 为**每输出行**
  （`[out, in/64]`），沿输出轴拼接不共享、不改变任何分组。
  另有未预料的发现：**A1 也不是 bit 一致**（GEMM shape 变化改变 tiling
  与 FP32 累加顺序），所以 A1 与 A2 都走了完整 parity 门。
- **结果**：**NOT-WORTH**——reference tier median **+0.109 %**
  （CI95 [−0.100 %, +0.303 %] 含 0），且 parity 门失败
  （vs 冻结 MLX max|Δp| **0.04896 > 0.035**，42/42 决策相同）。
  gate 保持默认关闭，证据见 [FUSION-MATMUL.md](FUSION-MATMUL.md) §3.2。

## 候选 B：多图逐图 image-row 前缀链（Tier-2 扩展）

当前 Tier-2 的前缀 = system + header + **全部**图像的 rows（`media_end`），
key = 前缀 hash + **整组**图像 key。它已天然覆盖“同图多问 / joint 多问”
（key 不含 state / questions，已实测单图多问 +45.3 %）。缺口是：
**图像集合只要有一个不同就整体 miss**——共享图像的 vision rows 与
backbone 前缀被重算。

扩展方式：在每个图像边界缓存 backbone 前缀状态，key = 图像 key 链 +
边界 hash；请求复用最长匹配链，只为新增图像跑 vision + backbone rows。

- **预计收益**：取决于 joint 流量中“图像子集复用”的实际比例，**必须
  先量化再决定是否实现**。当前 42 单图案例与 same-image 套件里子集
  变化为 0 → 收益 0。demo 的 joint 模式每帧把 3 张裁图一起发；只有当
  跨请求出现部分共享（例如 `[A]` vs `[A,B,C]`，或问题子集变化导致裁图
  集合变化）才有收益。实现前先做 **traffic preflight**，无复用则不做。
- **实现范围**：`model.rs`（前缀缓存结构与查找）、`encode.rs`（逐图
  media 边界）、`preprocess.rs`（逐图 image key）；~250 行，触及缓存
  schema，需界定容量上限。
- **parity 风险**：低——同一计算，仅把已完成的前缀复用；风险在 key
  精确性（图像 key 链 + 边界 hash）与缓存容量。Tier-2 的复用机制本身
  已通过双重 parity 门验证。
- **测量方案**：(0) **traffic preflight**：只读计数图像子集复用率，
  一次运行、不改模型；(1) preflight 通过才构造“多图共享子集”套件；
  (2) 3 轮反序配对 + 42 案 parity + no-hit control（照 Tier-2 的双重门）。

### 状态（B）
- **未批准**（“先不做，等以后有明确子集复用流量再说”）→ 未实施，
  preflight 也未做。本方向保留在这里，等出现明确子集复用流量时再申请。

## 建议

先批准 **A**（有界、可在现有套件上立即测量）；**B** 只批准 **preflight**
（只读计数，不改模型），确认存在子集复用后再回来申请实现批准。
