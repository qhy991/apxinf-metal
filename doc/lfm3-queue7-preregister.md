# QUEUE #7（自选）预登记：native-worker 加载/启动分段计时 census

预登记时间：**2026-11-19 23:50 CST**（`date` 实取；commit 须先于任何设备运行）。

**状态**：预登记。本轮不含任何实现、测试或测量；后续按一轮一步纪律分别执行。

---

## 0. Queue 归属

- **QUEUE.md v2 第 7 项（自选）**：方向**必须**满足两个义务 ——
  1. 事先量化收益残差（不是"听起来可能有收益"）；
  2. 先证明该改动**不触碰生成路径数值**（不改 kernel / matmul 形状 / 归约序 / 权重读序 / decode schedule）。
- 本轮选定方向：**native-worker 加载 / 启动路径的分段计时 census**（测量类，纯 host `Instant`）。
- 若 census 确认有可观残差，**优化方向与实现另起一轮并先写优化档自己的 Gate**；本 ticket 只负责"值不值得做"的判定。

## 1. 关门清单（不重开）

| 关门项 | 事实 |
|---|---|
| #115 | 编译/整步 `mx.compile` + 位置分桶：整步 compiled decode +0.7% 无收益；门关死 |
| #547 | prompt-lookup 投机：门关死 |
| #128 | 合并 GEMV（改 matmul 形状 ⇒ 累积顺序变）：INVALID |
| QUEUE #1 | 整步 compiled decode：**INCONCLUSIVE（未测量）**；launch-gap drain 假设保持开放，但本 ticket 不依赖它 |
| QUEUE #2 | 自写 lm_head/silu kernel：**INVALID**（K3 token 硬门失败） |
| QUEUE #3–#5 | prefill 复用 / eval 分离 / 结构重组收口：**INCONCLUSIVE / FALSIFIED** |
| QUEUE #6 | batched decode：**INCONCLUSIVE**（批内 token 门事前未能证明） |
| QUEUE #8 | 原生 4-bit：**INVALID**（G1 跨引擎 token 在 step 59 分歧，机制未知） |
| #121 | S1 短生成 cell A/A = 5.02% —— 本 ticket 的 A/A 不可引用其值，须自测 |

**与结构重组封盘的关系**：本 ticket **不改执行图、不改 kernel、不改 matmul/归约/读序**；它只在 host 侧 `Instant` 打点。因此它**不在**"任何结构改动都会扰动 token"的禁区内 —— 但按 RUNNER §5-1，**仍然必须先过 token 一致性硬门**，不能因为"按构造不碰 GPU"就豁免。

## 2. 方向选择：候选量化（QUEUE #7 义务 1）

| 候选 | 已知残差 | 判定 |
|---|---|---|
| gateway 往返 | 0.34 / 0.39 ms ≈ **0.004%**（34.36 ms/token 基准） | 已量化且极小 —— **不选**（无优化空间） |
| tokenizer / detokenizer | 稳态 decode 中 detokenize 已由 `profile.excluded_time` 从 TPOT 中扣除（见 Gate 原文依据）；round-8 gate-0 census 已有 `detokenization_ns` 数据 | **待本 ticket 补齐分摊量级后判定**（若 <0.01% ⇒ 明确 FALSIFIED 候选） |
| 模型加载 / 进程 worker 启动（本 ticket 主项） | **已知无 receipt**：现有 ready 帧只报 `resident memory`，**不报 load 各阶段耗时**；round-45/46 实测 `4bit` 加载"异常快 0.165 s"曾误导根因方向，说明**缺失该数据会付出取证成本** | **选中**：先补 census，量化残差后再决定是否有优化价值 |
| launch-gap drain（~0.46–0.51 ms/token 假设） | QUEUE #1 INCONCLUSIVE，**假设保持开放** | **不选**（结构/调度类，属已封盘区；且 QUEUE #1 未测量） |

**为什么选加载/启动而非"更显然"的方向**：QUEUE #7 明文排除生成路径；gateway/tokenizer 的残差**已被量化到 0.004% 量级**，没有可优化空间。唯一**未量化**且**与本仓既有痛点直接相关**的就是加载/启动 —— 它不触碰 decode 数值，却能改善 **cold start、远程 worker 调度、多 cell 实测窗口的周转时间**（每个 ABBA cell 都要重启 gateway）。

**义务 2 —— 不触碰生成路径数值的证明**（见 §5 Gate G2）：本轮唯一的代码改动是在 `native_worker::run()` 与 `replay()` 内加 `Instant::now()` / `.elapsed()` + `eprintln!`。这些语句位于 token 事件发出**之后**（Terminal `native_request_settled`）或在模型构造**之前/之间**，**不在** prefill/decode 的 `build_graph` → `eval` → `argmax` 路径上，因此**按构造不改生成数值**。该命题仍须经 G1 token 门验证，不能只凭"按构造"放行（#128 之教训）。

## 3. 假设

**H（主假设）**：native-worker 的启动成本由若干**独立 host/GPU 阶段**组成，其耗时分布**未知**（文档 / receipt 中均无分段数据）；补上分段计时后，可判断：
- (a) 是否存在 >0.5% 相对 decode 基准（34.36 ms/token，单 cell 128 tok ≈ 4.40 s）的**可优化 host 残差**（例如 safetensors mmap vs upload、tokenizer parse、`set_memory_limit` 等）；
- (b) 其中是否包含**非生成路径、按构造 token 等价**的优化机会（QUEUE #7 的硬性前提）。

**H 的 falsifier**：所有分段 host 耗时之和 < 0.5% × cell decode 总时长，且不随模型大小/ARM 开关系统性变化 ⇒ 本方向 FALSIFIED（无价值）。

## 4. 实现（冻结范围）

唯一允许的代码改动（**测量类**，非优化）：

1. **启动分段**（`crates/apxinf-serving/src/native_worker.rs` `run()`）：
   - `host_identity_ns` — banner 打印完成至 `AutoModel::load_model` 调用前（约 :525–:627 之间）；
   - `model_load_ns` — `AutoModel::load_model` 调用本身（:627–:637）；
   - `tokenizer_load_ns` — `Tokenizer::from_file` + eos 提取（:637–:650）；
   - `post_load_ns` — `synchronize` + `clear_cache` + resident 校验 + ready 帧构造（:650–:681）。
   - 一行 `eprintln!` 输出 JSON：`{"event":"native_startup_census","model":<path>,"host_identity_ns":…,"model_load_ns":…,"tokenizer_load_ns":…,"post_load_ns":…,"ready_ns":…}`，其中 `ready_ns` 为 `run()` 进入至 ready 帧发出的墙钟总时长。
2. **首次请求准备分段**（`replay()` 内，`native_request_settled` 之后追加一行）：
   - `tokenization_ns` — `prompt_tokens()`/`input.prepare()` 之前后；
   - `first_step_ns` — `prefill_start.elapsed()` 的首个 step（已有字段，补进 settle 事件即可，**不新增计时**）。
3. **tokenizer 分摊**（`native_request_settled` 已有 `detokenization_ns` / `preparation_ns`，补一行 `detok_per_token_ns = detokenization_ns / output_count` 与 `detok_per_token_pct_of_cell`（相对 cell decode 总时长）。

**显式禁止**：
- 不新增任何 `Compiled` / 图缓存 / 融合 / 量化切换；
- 不改 `model.rs` / `weights.rs` / `apxinf-mlx` / `gateway.rs` / `supervisor.rs` 任何一行；
- 不新增依赖；
- 不在 `Lfm3Model` / `build_graph` / `decode` 内打点（**那才是可能触碰数值的禁区**）；
- 不把"分段计时"与任何"优化开关"混在同一个 commit（**一个方向一个 PR**）。

## 5. Gate（冻结；PR 引用原文）

> **G1 token 一致性（硬门）**：改动后的 native-worker 与改动前的同一 binary（或同源未打点版本）在 **4 prompts × 128 token** 贪心输出上**逐 token 相同**（harness 同 #127 T1 / QUEUE #2 K3 / QUEUE #8 G1 replay 通道；`--test-threads=1`，双 Metal 锁，等不抢）。任一 prompt 任一 token 不同 ⇒ **INVALID**（不进 census 判定、不进任何优化讨论）。
> **G2 不触碰生成路径（按构造证明）**：本轮 diff 必须可证明**不含**任何 `build_graph`/`prefill`/`decode`/`argmax`/matmul 形状/归约序/权重读序/decode schedule 的改动 —— 即 `git diff --stat` 仅命中 `native_worker.rs` 一个文件，且所有 `Instant` 打点位于 token 事件之后或模型构造阶段。**违反 ⇒ INVALID**。
> **G3 空臂 / effect proof**：census 输出行的 `event=native_startup_census` 必须**真实出现**于 stderr；若 harness 要求 `APXINF_LFM3_*` 开关，则 banner 与 census 必须同源同 run。**缺行 ⇒ INVALID**。
> **G4 A/A**：census 数字本身不需要 A/A（它是单次启动的 host 计时，非 latency 窗口）；但**若本 ticket 后续要基于 census 数字开 latency 窗口**（例如"优化后"对照），则该窗口须 A/A ≤5%，且**不可引用 #121 S1 5.02%**。**本 ticket 不开 latency 窗口**（测量类），故 G4 在本 ticket 内 **N/A**；留作后续优化档的前置条件。
> **G5 五档判定输入**：census 必须覆盖 **≥10 次独立启动**（cold start，进程级，每次全新 `apxinf-native-worker`）× **≥2 个模型**（BF16 checkpoint + QUEUE #8 现场 4-bit 产物，验证分段是否随权重布局变化），取每段**中位与 p90**。样本不足 ⇒ 判定 INCONCLUSIVE。

**Gate 依据（原文出处）**：
- `crates/apxinf-model/src/profiling/mod.rs:66` `run_unprofiled` —— detokenization/terminal I/O 已从 engine TPOT 中扣除；**故"detokenize 吃掉 decode 时间"在 engine 口径下不成立**，本 ticket 只需量化其**绝对**量级（是否构成 cell 周转成本）。
- `crates/apxinf-serving/src/native_worker.rs:1284` `native_request_settled` —— 现成 `preparation_ns` / `detokenization_ns` / `cleanup_ns`，**不新增**。
- `doc/lfm3-compiled-decode-step.md` §Results（17:37 修订）—— QUEUE #1 判定 INCONCLUSIVE 的先例：**未测量的臂不得判 FALSIFIED**。本 ticket 引以为戒：在 census 跑起来之前，**不得**断言"加载残差值得优化"或"不值得"。

## 6. Decision（冻结；PR 引用原文）

> **方向**：QUEUE #7 自选 —— native-worker 加载 / 启动**分段计时 census**（纯 host 测量，不优化、不改生成路径）。
> **实现范围**：仅 `crates/apxinf-serving/src/native_worker.rs`；启动 4 段 + settle 1 行 tokenizer 分摊；**禁改 model/weights/apxinf-mlx/gateway/supervisor**。
> **先决条件**：**预登记 commit 先于任何设备运行**（本节）；随后实现 → 本地/M4 `cargo check` → G1 token 门（双锁，等不抢）→ census 数据采集（G5 样本量）→ 五档判定 → （若有残差）另起一轮写优化档 Gate。
> **判定后动作**：census 结果与判定写入本 doc §Results 并 commit；若 FALSIFIED ⇒ 记录后进 QUEUE #7 的下一个候选（tokenizer 分摊已含在本 census 中，可直接出结论）或按 RUNNER §7 转入自选下一方向；若显示 >0.5% 残差 ⇒ **不静默开优化**，先写优化档预登记。
> **五个 token**：WIN / PARITY / PARTIAL / FALSIFIED / INCONCLUSIVE（+ INVALID），附机制算术（每段 ms、占 cell decode 总时长 %、与 34.36 ms/token 地板的相对量级）。算不平写"机制未知"。

## 7. 五档判定（冻结）

| 判定 | 条件 |
|---|---|
| **WIN** | **本 ticket 不产生 WIN**（测量类，无优化对照）。WIN 留给后续优化档。 |
| **PARITY** | G1/G2/G3 通过；census 样本 ≥G5；各 host 分段中位之和 ≤0.5% × cell decode 总时长（128 tok × 34.36 ms = 4.40 s ⇒ ≤22 ms） ⇒ **无可观残差，QUEUE #7 本方向关闭** |
| **PARTIAL** | G1/G2/G3 通过；某一段 host 中位 >0.5% 但 p90 中位比 <3×（稳定可优化）⇒ 记录段名与量级，**转优化档预登记** |
| **FALSIFIED** | G1/G2/G3 通过；但所有分段之和 <0.5% **且** 与模型大小/开关无相关性（即"加载不是瓶颈"被 census 直接否证）⇒ **本方向 FALSIFIED，不进优化** |
| **INCONCLUSIVE** | G1/G2/G3 通过但样本不足（<10 cold starts）或段间耦合无法分离（机制未知） |
| **INVALID** | G1 token 不一致 / G2 diff 越界（碰了 model/weights/apxinf-mlx）/ G3 census 行缺失（空臂）任一成立 |

**注**：QUEUE #7 明文要求"先证明不改单轨迹数值" —— G1 通过即完成该证明；**G1 失败则本方向直接 INVALID，不讨论收益**。

## 8. 机制算术（预估，待 census 实测替换）

- **decode 基准**：BF16 3.4576 GB/token ÷ 100.6 GB/s = **34.36 ms/token**（RUNNER §基线）；cell = 128 tok ⇒ **4.40 s**。
- **0.5% 残差阈** = 22 ms / cell。
- **4-bit 参照**：0.973 GB ÷ 100.6 GB/s = 9.67 ms/token ⇒ cell = 1.24 s；0.5% ⇒ 6.2 ms（阈值更严苛）。
- **已知参照（非本 ticket 测量）**：lane engine load 0.22 s（round-44 sanity，`4bit` 目录，含 mmap/upload）；官方残缺文件曾报 0.165 s（round-46，**已确认为读到全零 payload 的假快**）—— 两者差异 **55 ms** 即足以说明：**分段计时是根因取证的必要条件**（round-46/47 在 G1 上反复踩空臂/vacuous-pass，若早期有 load 分段数据可提前发现"读到零 payload"）。
- **launch 数 / per-kernel GB/s**：**机制未知（本 ticket 不 census GPU 侧；`xctrace` 不可用、MLX SDK 无 per-kernel timer，round-18/24 同款诚实注记）**。本 ticket 的分段全部为 host `Instant`，无 GPU 依赖。

## 9. 测试

| 编号 | 内容 | 门 |
|---|---|---|
| **L1** | G1 token 门：打点后 binary vs 未打点 binary（或同源 diff 前版本），4 prompts × 128 token，replay 通道逐 token 比；banner `native_4bit` 字段正常、census 行正常 | 硬门，一个 token 不同 ⇒ INVALID |
| **L2** | G2 diff 越界检查：`git diff --name-only` 必须仅 `crates/apxinf-serving/src/native_worker.rs`；`git diff` 中不得出现 `build_graph\|prefill\|decode\|argmax\|matmul\|quantized_matmul\|Compiled` 等字样（除已存在的字段名） | 硬门 |
| **L3** | G3 census 行：每次 `apxinf-native-worker` 启动 stderr 必须含 `event":"native_startup_census"`；字段齐全（5 个 `_ns` + `ready_ns`），数值 >0 | 硬门 |
| **L4** | G5 样本：`≥10 cold starts × {BF16, 4bit}`，每段取中位/p90；样本不足 ⇒ INCONCLUSIVE | 判定输入 |
| **L5** | tokenizer 分摊：`detok_per_token_ns × output_count` 必须与已有 `detokenization_ns` 相等（±0.01%），且 `detok_per_token_pct_of_cell < 0.5%`（预期成立，否则转 PARTIAL） | 判定输入 |

## 10. §8 census 诚实注记

- `xctrace` 不可用（round-18 / round-24 同款）；MLX SDK 无 per-kernel timer/counter ⇒ 本 ticket **不测 GPU 侧**，所有分段为 host `Instant`，**不冒充** per-kernel GB/s。
- `StreamCounters`（`apxinf-mlx/array.rs:153`）只有 `trace_callbacks / eval_calls / uploads / downloads` —— **本 ticket 不依赖它**，也**不新增 launch 计数**（launch 数 = 机制未知，留待真正需要的优化档）。
- **不把分段耗时误读为"优化收益"**：分段大 ≠ 可优化；例如 `model_load_ns` 大可能是纯 GPU upload（受 100.6 GB/s 上限约束），属于"不可优化"。判定时须区分 **host 可优化** 与 **GPU/bandwidth-bound** —— 前者记 PARTIAL 并转优化档，后者记 PARITY/FALSIFIED。

## 11. Relation to 先例

- **#115 / #128 / QUEUE #1–#6 / QUEUE #8**：见 §1 关门清单；本 ticket **不重开**任何一项。
- **QUEUE #1 的 INCONCLUSIVE（lead correction 17:31）**：本 ticket 直接继承该教训 —— **在 census 真实跑起来之前，不写 FALSIFIED**；判定必须基于真实测量，不基于 prior 或"按构造"。
- **QUEUE #8 的 G1 INVALID（v12，step 59 分歧）**：本 ticket 的 **L1 harness 直接复用其 replay 通道**（`/v1/replay` 的 `emitted_token_ids` 是唯一可信的 token id 通道，SSE 文本通道与 lane `ids` 字段均不可靠 —— round-46 v5–v7 教训）。
- **QUEUE #2 的 K3 harness**（entry/exit 复位开关防污染）：L1 沿用同款 discipline。
- **QUEUE #7 自选义务**：本 ticket 是该义务的**度量**步骤，不是优化 —— 与 §6 Decision 的"测量后另起优化档"一致。

## 12. Receipts（执行后填写）

- 实现 commit：_（待填）_
- L1 token 门 harness / 结果：_（待填）_
- L4 census raw（≥10 × 2 模型）：_（待填）_
- 五档判定写入 §Results 的 commit：_（待填）_

---

## Gate（原文，供 PR 引用）

> **G1** 打点后的 native-worker 与打点前版本在 4 prompts × 128 token 贪心输出上逐 token 相同（harness 同 #127 T1 / QUEUE #2 K3 / QUEUE #8 G1 replay；双 Metal 锁，等不抢）；任一 token 不同 ⇒ INVALID。**G2** 本轮 diff 仅命中 `native_worker.rs`，且不含任何 `build_graph`/`prefill`/`decode`/`argmax`/matmul 形状/归约序/权重读序/decode schedule 改动；违反 ⇒ INVALID。**G3** 每次启动 stderr 必须真实出现 `event":"native_startup_census` 行；缺行 ⇒ 空臂 INVALID。**G4** 本 ticket 不开 latency 窗口，A/A 不适用；若后续优化档开窗，须 A/A ≤5% 且不可引用 #121 S1 5.02%。**G5** census 须 ≥10 cold starts × {BF16, 4bit}，取每段中位与 p90；样本不足 ⇒ INCONCLUSIVE。

## Decision（原文，供 PR 引用）

> **方向**：QUEUE #7 自选 —— native-worker 加载/启动**分段计时 census**（纯 host 测量）。**实现范围**：仅 `crates/apxinf-serving/src/native_worker.rs`（启动 4 段 + settle tokenizer 分摊），禁改 model/weights/apxinf-mlx/gateway/supervisor。**流程**：预登记（本文）→ 实现 → `cargo check` → L1 token 硬门 → L4 census 采集 → 五档判定（PARITY = 分段之和 ≤0.5% cell decode 时长；PARTIAL = 单段 >0.5% 且稳定 ⇒ 转优化档预登记；FALSIFIED = 分段之和 <0.5% 且与模型/开关无关；INCONCLUSIVE = 样本不足；INVALID = 任一硬门）。**本 ticket 不产生 WIN**（测量类）。**不静默开优化**：任何基于 census 的优化必须另起一轮写自己的 Gate+Decision。
