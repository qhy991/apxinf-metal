# LFM3 原生推理优化 campaign 总结（QUEUE #1–#8，2026-11-19 → 2026-11-20）

Campaign: `lfm3-dig`, base `feat/lfm3-metal`.
Written offline on M1; **no device, no Metal lock, no new measurement.**
Sources: STATUS.md round-1–78, `doc/lfm3-{structural-reorg-negative,native-4bit,batch-throughput}.md`,
PR #132 (merged `e353eda5`), QUEUE #6 receipt branch `agent/lfm3-dig-batch`.

## 1. 最终状态表

| QUEUE | 项 | Verdict | 关键数字 / 挡下的门 |
|-------|----|---------|---------------------|
| #1 | 整步 `mx.compile` | **INCONCLUSIVE** | 臂从未跑成（闭包集成缺陷 + M4 锁被无关 campaign 长期占用）⇒ 无 token 结果，不可 FALSIFIED |
| #2 | 自写 lm_head GEMV + silu 融合 | **INVALID** | 首个生成 token 即分叉（128× 37179 vs 20, 229, …）；同形状、不同归约形式 ⇒ logit 漂移 |
| #3 | 短 prompt prefill 图复用 | **INVALID** | P1 prefill token gate 失败 |
| #4 | launch-gap 探针 | **INVALID** | 空臂 + A/A 6.95% >窗口作废 |
| #5 | 结构重组负结果收口报告 | **报告** | 五次尝试（含 #128）institutional record，离线成文，无窗口 |
| #6 | 批处理吞吐 | **PARITY / NOT-WORTH**（算术 receipt，**未测量**） | shipped 串行 dispatch = **1.000× by construction**；≥1.5× WIN 门需共享 ≥38.1% 权重字节 ⇒ 只能由真批量 forward 实现 |
| #7 | startup/replay census + serving 小修 | **PASS**（效果门） | L1：4 prompts × 128 tok instrumented == baseline 逐 token 相同；打点不触碰单轨迹数值 |
| #8 | 原生 4-bit 量化 | **PARTIAL**（G4-speed PASS + G1′ PARTIAL n=20） | 见 §2 |

## 2. 主成果：原生 4-bit（QUEUE #8，唯一经用户批准的方向）

**速度（G4-speed PASS，ABBA，engine basis，A/A ≤3.5% ≤ 5%，G3 banner/census 无空臂）**：

| cell | native-4bit | native-bf16 | 加速 |
|------|-------------|-------------|------|
| ctx 1024 × 128 tok | **11.06–11.13 ms/tok** | 36.59 | **3.26–3.31×** |
| ctx 4096 × 128 tok | **11.39–11.44 ms/tok** | 37.13–37.26 | **3.26×** |

- 认证 BF16 基线 36.07/36.40（vs llama 36.42/36.67，parity 0.99×）⇒ **4-bit ≈ 3.3× vs BF16**。
- 快于 llama.cpp Q4_K_M（12.39/12.71）**8–11%**（0.89–0.92×）。
- 机制算术：0.973 GB/token（BF16 的 28.1%）÷ 100.6 GB/s ⇒ floor **9.67 ms**；实测 11.10 ⇒ 有效带宽 **87.7 GB/s**（roofline 87%），多出的 ~1.4 ms 为 qmm dequant/launch 开销，机制对得平。
- 可交付形态：`APXINF_LFM3_4BIT=1` opt-in，**BF16 保持默认**。

**质量（G1′ PARTIAL，n=20，ABBA 3 段/臂，suite task-set sha `13a96e89…` 冻结）**：
both 口径 4-bit 3/20 = 15.0% vs BF16 4/20 = 20.0%（**−5.00pp**）；either 口径 6/20 vs 6/20（**0.00pp**）。
落在 3–10pp 区间 ⇒ PARTIAL 候选，**不支持 STOP**（n=6 时的 −16.67pp STOP 已按 Lead 02:18 改判为 INCONCLUSIVE 并扩测）。
Caveat（不掩盖）：两臂 3 段 solved spread 10.00 / 5.00pp 均 >5%，A/A DISAGREE；结论是 PARTIAL 候选而非精确 −5pp 点估计，20 题中 14 题双方全失败、有效分辨题仅 6/20。是否接受 −5pp 作为 opt-in 质量代价，或扩 ≥40 题收敛，留 Lead 裁决。
G1-diag（与 Python lane 同精度首处分叉 prompt 0 step 59：lane=125020 vs native=25797）为诊断项，FP16 scale 复现未跑。

## 3. 这条栈优化的边界（可复用结论）

1. **代数等价 ≠ token 等价。** 本 campaign 五次结构改动（#128 合并 GEMV、#2 自写 kernel、#3 prefill 图、#4 eval-split、#128 同族）全部被 token 门挡下：拼接权重 / 改 matmul 形状 ⇒ MLX 换 tiling ⇒ 累积顺序变 ⇒ 数值漂移 ⇒ argmax 翻 token；同形状换 kernel 也因归约形式不同翻 token。**任何结构改动必须先过 token 门，再谈速度。**（详见 `doc/lfm3-structural-reorg-negative.md`）
2. **单请求延迟 roofline 已近触顶。** 100.6 GB/s × 3.4576 GB/token ⇒ **34.36 ms floor**，实测 36.07 ⇒ 已用 **95–97%**，剩余 ~1.7 ms 且结构重组路径已封盘 ⇒ BF16 路径按当前范式无进一步大收益。
3. **破 roofline 的路是改字节数，不是改图。** 4-bit 把每 token 字节降到 28.1%，不改执行图结构（走 MLX 现成 qmm），因此绕开了 5 次被 token 门挡下的形状类风险，拿到确定的 3.3×。这验证了「先量化每 token 字节，再谈 kernel」的路线选择。
4. **吞吐指标不进入本 campaign 验收。** 验收工作负载（agent 端到端）是 batch=1 串行 ⇒ 批处理吞吐改善不影响验收指标；真批量 forward 属高风险改形状类，故 QUEUE #6 收口为 NOT-WORTH（`/readyz` 广告已 fail-closed 修复，`batching: true` 以真实 batched forward 为前提）。
5. **Honesty 与 receipt 纪律是有代价但必须付的。** 九次 L1 启动前八次死于 harness 层缺陷（stderr 分块/转义/跨帧、500 body 被吞、契约白名单漏字段），全部如实记录作废；无判定效力的运行不进五档。算术可证的结论（#6 1.00×）标注「unmeasured by construction」，不开窗冒充实测。

## 4. 交付状态

- PR #132（4-bit + census/banner 插桩 + 文档 + `scripts/` harness）已合并 `e353eda5`。
- QUEUE #6：分支 `agent/lfm3-dig-batch`（8 commits）留作 receipt，**不合并**。
- 下一步：Lead 裁决 G1′（−5pp opt-in vs 扩 ≥40 题）与 G1-diag 是否补跑。
