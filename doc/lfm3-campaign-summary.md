# LFM3 原生推理优化 campaign 最终汇总（2026-11-20 09:52 CST 定稿；queue 中无未完成项）

Campaign: `lfm3-dig`，base `feat/lfm3-metal`。
**定稿依据**：QUEUE.md 已封盘区（#1–#4）+ 已完成项（#5–#8）+ Lead 09:31 最终裁决（QUEUE #6 NOT-WORTH，#1–#8 全部收口）。
**本文为静态交付归档**：round-79 `doc/lfm3-campaign-summary.md`（commit `562addd9`）+ round-80 receipt 分支 push 归档（`agent/lfm3-dig-batch` @ `562addd9`）之后，QUEUE.md 不再有任何"进行中"项；Lead 09:31 第 5 项（汇总文档）已在 round-79 完成，故本轮无预登记、无实现、无测试、无测量、无开 PR 动作可做，仅追加本轮 STATUS 结论并冻结本汇总。

## 1. QUEUE #1–#8 最终状态表（全部收口，无 verdict 空缺项）

| QUEUE | 项 | 最终 Verdict | 关键数字 / 挡下的门 | 落地位置 |
|-------|----|-------------|---------------------|----------|
| #1 | 整步 `mx.compile` → 编译 decode 脚手架 | **INCONCLUSIVE** | 臂从未跑成（闭包集成缺陷 + M4 锁被无关 campaign 长期占用）⇒ 无 token 结果，不可 FALSIFIED；排队等锁中被 Lead 04:31 强制中止 | STATUS round-1.., `6179da19`（脚手架保留为 base 资产） |
| #2 | 自写 lm_head GEMV + silu 融合 | **INVALID** | 首个生成 token 即分叉（128× 37179 vs 20, 229…）；同形状、不同归约形式 ⇒ logit 漂移 | 代码已按 Lead 04:31/04:31 第 1 项删除（round-69 `64e2adf3`），文档与 receipts 保留 |
| #3 | 短 prompt prefill 图复用 | **INVALID** | P1 prefill token gate 失败（prefill token 即分叉） | 同上，代码已删，文档保留 |
| #4 | launch-gap 探针 | **INVALID** | 空臂 + A/A 6.95% >5% ⇒ 窗口作废 | 同上，文档保留 |
| #5 | 结构重组负结果收口报告 | **报告（已交付）** | 五次尝试（含 #128 合并 GEMV）institutional record；提炼可复用规则"代数等价 ≠ token 等价，先 token 门再谈速度" | `doc/lfm3-structural-reorg-negative.md`，随 PR #132 合并入 base |
| #6 | 批处理吞吐（batch 1/2/4/8） | **PARITY / NOT-WORTH**（算术 receipt，**未测量**） | shipped 串行 dispatch = **1.000× by construction**；≥1.5× WIN 门需共享 ≥38.1% 权重字节 ⇒ 只能由真批量 forward 实现；验收负载 batch=1 串行 ⇒ 吞吐不进端到端指标 | `doc/lfm3-batch-throughput.md` §Results + Amendment 1/2；receipt 分支 `agent/lfm3-dig-batch`（9 commits）留档不合并；`/readyz` 广告已 fail-closed（round-77 `4a75f007`） |
| #7 | startup/replay census + serving 小修 | **PASS**（L1 效果门） | L1：4 prompts × 128 tok instrumented == baseline 逐 token 相同（sha256 `26f9af37…`/`5bdb6f5c…`/`1e82d973…`/`38b703bd…`）；打点不触碰单轨迹数值 | `5faa2977` + PR #132 |
| #8 | 原生 4-bit 量化支持 | **PARTIAL**（G4-speed PASS + G1′ PARTIAL n=20） | 见 §2；4-bit 保持 opt-in，BF16 默认 | `doc/lfm3-native-4bit.md` §7 + PR #132（已合并 `e353eda5`） |

## 2. 主成果：原生 4-bit（QUEUE #8，唯一经用户批准且已测出 3.3× 的方向）

**速度（G4-speed PASS，ABBA，engine basis，A/A ≤3.5% ≤ 5%，G3 banner/census 无空臂）**：

| cell | native-4bit | native-bf16 | 加速 | vs llama.cpp Q4_K_M 认证基线 |
|------|-------------|-------------|------|------------------------------|
| ctx 1024 × 128 tok | **11.06–11.13 ms/tok** | 36.59 | **3.26–3.31×** | 快 8–11%（12.39 ms ⇒ 0.89–0.92×） |
| ctx 4096 × 128 tok | **11.39–11.44 ms/tok** | 37.13–37.26 | **3.26×** | 快 8–11%（12.71 ms ⇒ 0.89–0.92×） |

- 认证 BF16 基线 36.07/36.40（vs llama 36.42/36.67，parity 0.99×）⇒ **4-bit ≈ 3.3× vs BF16**，落在预登记判据 WIN 速度维度（≥2.5× 且 ≤ llama Q4）。
- 机制算术：0.973 GB/token（BF16 的 28.1%）÷ 100.6 GB/s ⇒ roofline floor **9.67 ms**；实测 11.10 ⇒ 有效带宽 **87.7 GB/s**（roofline 的 87%），多出 ~1.4 ms 为 qmm dequant/launch 开销，机制对得平。
- 可交付形态：`APXINF_LFM3_4BIT=1` opt-in，**BF16 保持默认**（Lead 02:18 + 09:31 一致）。

**质量（G1′ PARTIAL，n=20，ABBA 3 段/臂，suite task-set sha `13a96e89…` 冻结）**：

- both 口径：4-bit 3/20 = 15.0% vs BF16 4/20 = 20.0%（**−5.00pp**）；either 口径：6/20 vs 6/20（**0.00pp**）。
- 落在 3–10pp 区间 ⇒ **PARTIAL 候选，不支持 STOP**（n=6 时的 −16.67pp STOP 已按 Lead 02:18 改判为 INCONCLUSIVE 并扩测，round-62/63/64/65 闭环）。
- Caveat（不掩盖）：两臂 3 段 solved spread 10.00 / 5.00pp 均 >5%，A/A DISAGREE；结论是 PARTIAL 候选而非精确 −5pp 点估计；20 题中 14 题双方全失败、有效分辨题仅 6/20。
- **Lead 09:31 接受现状为 opt-in 质量代价，未要求扩 ≥40 题**（round-78/79 收口时 Lead 未再对 −5pp 提出异议，QUEUE #8 随 PR #132 合并）。
- G1-diag（与 Python lane 同精度首处分叉 prompt 0 step 59：lane=125020 vs native=25797，ctx `[41176, 88050, 124900, 124895]`）为**诊断项**，FP16 scale 复现未跑，按 Lead 00:16 修订**不判 INVALID、不阻断交付**。

## 3. 这条栈优化的边界（可复用结论，供最终报告引用）

1. **代数等价 ≠ token 等价。** 本 campaign 五次结构改动（PR #128 合并 GEMV、QUEUE #2 自写 kernel、QUEUE #3 prefill 图复用、QUEUE #4 eval/sync 拆分、QUEUE #1 编译图脚手架未跑成）全部被 token 门挡下或未跑成：拼接权重 / 改 matmul 形状 ⇒ MLX 换 tiling ⇒ 累积顺序变 ⇒ 数值漂移 ⇒ argmax 翻 token；同形状换 kernel 也因归约形式不同翻 token。**任何结构改动必须先过 token 门，再谈速度。**（详见 `doc/lfm3-structural-reorg-negative.md`）
2. **单请求延迟 roofline 已近触顶。** 100.6 GB/s × 3.4576 GB/token ⇒ **34.36 ms floor**，实测 36.07 ⇒ 已用 **95–97%**，剩余 ~1.7 ms 且结构重组路径已封盘 ⇒ BF16 路径按当前范式无进一步大收益。
3. **破 roofline 的路是改字节数，不是改图。** 4-bit 把每 token 字节降到 28.1%，不改执行图结构（走 MLX 现成 qmm，bit-exact BM64 retile 机器在但跨引擎仍有已知 step-59 舍入差）⇒ 绕开 5 次被 token 门挡下的形状类风险，拿到确定的 3.3×。验证了「先量化每 token 字节，再谈 kernel」的路线选择。
4. **吞吐指标不进入本 campaign 验收。** 验收工作负载（agent 端到端）是 batch=1 串行 ⇒ 批处理吞吐改善不影响验收指标；真批量 forward 属高风险改形状类，故 QUEUE #6 收口为 NOT-WORTH（`/readyz` 广告已 fail-closed 修复，round-77 `4a75f007`，`batching: true` 以真实 `batched_forward` 为前提，生产恒 false）。
5. **Honesty 与 receipt 纪律是有代价但必须付的。** 九次 L1 启动前八次死于 harness 层缺陷（stderr 分块/转义/跨帧、500 body 被吞、契约白名单漏字段），全部如实记录作废；无判定效力的运行不进五档。算术可证的结论（#6 1.00×）标注「unmeasured by construction」，不开窗冒充实测。**"先读完整错误再修"是贯穿本轮的第二条规则**（round-50/51 四个 harness bug 同源：假设了 stderr 是整行原文；round-80 凭据 401 被连接失败掩盖，同族）。

## 4. 交付与归档状态（round-80 定稿）

- **PR #132**（4-bit 实现 + census/banner 插桩 + 文档 + `scripts/` harness）已**合并** `e353eda5` 至 `feat/lfm3-metal`；QUEUE #1–#8 中所有有代码改动的方向均以该 PR 交付或删除（#2/#3/#4 代码按 Lead 04:31 删除，仅留文档与 receipts）。
- **QUEUE #6 receipt 分支** `agent/lfm3-dig-batch`（9 commits，`119091b6`..`562addd9`）已 push 至 origin，**留档不合并**（round-80，远端三重核验 SHA `562addd95c…`）。
- **M4 receipts**：`~/lfm3-dig-run/window-4bit/{raw,g1prime}/`（G4-speed ABBA 原始 JSON + G1′ 20 题 × 3 段/臂 solved 表）、`~/lfm3-dig-run/l1/`（QUEUE #7 L1 PASS 全套）。
- **无待裁决项**：Lead 09:31 五项（广告 fail-closed / 不开 option-(a) 窗 / 不做真批处理 kernel / QUEUE #6 NOT-WORTH / campaign summary）+ G1′ −5pp opt-in 接受，均已落地；QUEUE.md 无未完成项。
- **本轮（round-81）**：只追加 STATUS 结论 + 冻结本汇总文档（离线，未动设备、未占锁、无测量、无新 PR、无源码改动）。
