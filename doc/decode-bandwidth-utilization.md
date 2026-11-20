# 预登记：M1 Pro lane decode 带宽利用率 38% 的三项实测（Issue #135，Lead 11:30）

**状态**：预登记已冻结，commit 在先；测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-106，2026-11-20 12:03 CST（`date` 实取）。Lead 11:30 追加任务 =
Issue #135（原文见 §1，`gh issue view 135` 可复核）。Lead 明令：**排在 #134 之前**。

## 1. Gate（冻结原文，供 PR 引用）

```
Gate（Lead 11:30 / Issue #135，逐字）：
三件事，同一个 M1 Pro 锁窗口内做完：
- A1 真实带宽：MLX streaming copy/reduce ≥1 GB、≥5 轮
  ⇒ 用实测 GB/s 替换标称 200 GB/s；
- A2 上下文阶梯：ctx 0.5k/4k/16k/32k × {mlx-lm 参考, lane worker}，
  max_tokens=128，≥3 次中位 ⇒ 拟合常数项+斜率、算 worker 自身开销、
  验证 16k/32k 斜率是否成立；
- A3 GPU busy vs 墙钟：xctrace record --template 'Metal System Trace'
  抓 128-token decode，统计每 token GPU busy；不可用就退化为
  per-step 分解（build/eval/sample/callback）+ 字节/BW 推算。
  判定：busy≈12 ms ⇒ 算子问题（值得写融合 kernel）；
        busy≈5–6 ms ⇒ 主机侧问题（异步提交/整步图，写算子没用）。
附带：prefill 4-bit vs BF16 同窗 A/B（用户说那是推断：
      2.0 vs 0.96 ms/token，需同窗确认）。

参考对照（不是本 ticket 的判定输入）：native 4-bit（M4，#132）实测
η≈87%（11.1 ms，0.973 GB，100.6 GB/s）⇒ 38% 很可能是 lane/worker
实现问题，不是 M1 Pro 硬件上限。这点在分析里要写清。

纪律：M1 Pro 上做；双 Metal 锁 fcntl 排队；显示休眠 + caffeinate -si；
A/A ≤5%；每项先 commit 预登记（Gate:/Decision:），再动设备；
产出 doc/decode-bandwidth-utilization.md（三项方法+原始数字+判定+六场景式映射）
+ receipts + PR（base feat/lfm3-metal）。
```

必须读的账本拟合（用户给出，n=57 个 ≥100 token 输出请求，**本 ticket 复现/检验的对象**）：

- 每 token 时间 ≈ **11.66 ms + 0.61 ms / 1k ctx**；
- 常数项 11.66 ms = 权重带宽下限 4.9 ms（标称 200 GB/s 估的）+ **~6.8 ms 未知**；
- 斜率项：仅 6 层全局 attention 的 KV 随 ctx 涨，每 1k ctx KV ≈ 12 MB
  ⇒ 带宽下限 0.06 ms，实测 0.61 ms（差 10×）；拟合只覆盖 142–3844 ctx。

## 2. Decision（已冻结）

### 2.0 机器与窗口归属
- **执行机器 = 本机**：`sysctl machdep.cpu.brand_string` = **Apple M1 Pro**，
  Model MacBookPro18,3 / 16 GB（本步读数取证，与 Issue #135 "M1 Pro" 一致；
  M4 = `ssh haiyan-mac-mini`，不用于本 ticket）。开窗第一步必须把机型读数写进 receipt，
  机型不符 ⇒ 立即中止不占锁。
- **锁路径（冻结）**：沿用 RUNNER §3 的双 fcntl 文件 `/tmp/enginetailor-metal.lock`
  与 `~/.cache/enginetailor/metal-measure.lock`（**等待不抢占**；Issue #135 提到的
  `/dev/sem/*` 在本机不存在，故不采用）。本机锁不与 M4 campaign 互斥，若 M4 侧
  同时开窗互不影响。
- **窗口纪律**：`pmset displaysleepnow` + `caffeinate -si`（从不用 `-u`）；
  engine basis（lane 自报 step_ms / mlx-lm 的 `mx.eval` 计时 / prefill engine 侧计时），
  **不用客户端墙钟作主口径**（#120 SSE 全缓冲）；A/A：A2/prefill 每 cell ≥3 次
  分段中位 spread >5% ⇒ 窗口作废（RUNNER §4，不取中位凑）；A1 ≥5 轮 spread >5%
  同样作废。
- **单窗顺序**（A1 → A2 → A3 → 附带 prefill，全部在同一锁窗、同一热态/电源态）：
  A1 最先（给后面 η 提供实测 BW）；A3 的 trace 环境准备（xctrace 冷启动/权限）
  在锁外先做一次 dry-run，锁内只抓正式 decode 段，避免空臂。

### 2.1 A1 真实可达带宽
- 同一 lane venv（`~/ag0-lfm3-0f6bbbfaa348/.venvs/lfm3`，本步已验证
  `mlx.ok Device(gpu,0)`）跑 MLX streaming copy 与 reduce，数组 ≥1 GB，
  warmup 2 轮 + 计时 ≥5 轮，报 GB/s 中位与 spread；同时记
  `sysctl machdep.cpu.brand_string` + `system_profiler SPHardwareDataType` 全文进 receipt。
- 产出：**实测 BW**（后续 A2/附带 prefill 的 η 一律用它，替换标称 200 GB/s）；
  若实测 ≈ 标称 ⇒ 38% 的"带宽没用上"前提不成立，须在结论里先说明这一 falsification。

### 2.2 A2 上下文阶梯（两臂同窗）
- ctx = **0.5k / 4k / 16k / 32k** 四档，固定 prompt，每档 `max_tokens=128`、
  greedy（`temperature:0`），两臂同 prompt 逐 token（**跨引擎 token 不同不判 INVALID**，
  本项只测速度；G1-diag 已降级为诊断）：
  - 臂 A = **lane worker**：`lfm3_worker.py` 路径（`~/ag1-serving-025dfff3bd11/serving/`，
    engine root 同目录，4-bit 同用户账本场景）；
  - 臂 B = **mlx-lm 参考**：同权重，engine 侧 `mx.eval` 计时。**前置（实现档）**：
    lane venv 缺 `mlx_lm`（本步实测 `ModuleNotFoundError`）⇒ 开窗前须建独立 venv
    装 `mlx-lm`（不污染 lane venv），否则该臂无参考值 ⇒ **A2 BLOCKED（如实报，不硬凑）**。
- 每 cell ≥3 次取中位；每档产出 (ms/token, GB/s, η，η 用 A1 实测 BW)。
- 分析：对每臂作 **线性拟合 time = 常数项 + 斜率×ctx(k)**：
  - 与账本 11.66 + 0.61 拟合对读（是否复现）；
  - **worker − mlx-lm 差值 = worker 自身开销**（常数项差与斜率差分开）；
  - **16k/32k 两点验证斜率外推**：成立 ⇒ 斜率是实物理量，指向 KV 路径优化新方向；
    不成立（变平/变陡）⇒ 账本外推猜测被证伪，如实写。
- INVALID（A2 范围）：某 cell <3 次有效计时、A/A >5% 作废后仍超、两臂 ctx/prompt
  不一致、lane 账本 step_ms 缺失、mlx-lm 参考 venv 未就绪（此时整项 A2 BLOCKED）。

### 2.3 A3 GPU busy vs 墙钟（决定做算子还是主机侧）
- 主路径：锁内 `xctrace record --template 'Metal System Trace'` 抓一次
  lane worker 128-token decode（本机 `/usr/bin/xctrace` 存在；锁外先 dry-run
  验证模板/权限/是否 root 可用）；解析每 token GPU busy 时间（按 trace 中
  GPU command 区间聚合，receipt 存 `.trace` 与解析 json）。
- 退化路径（trace 不可用时逐字执行 Issue #135）：lane per-step 分解
  （build / eval / sample / callback，若 worker 无插桩则用 engine 计时 ± 客户端墙钟
  差反推 host 段）+ **字节 / 实测 A1 BW** = 理论 GPU 时间。
- **判定（冻结原文）**：
  - GPU busy ≈ 12 ms ⇒ **kernel/算子问题**，值得写融合 4-bit GEMV / 整层 fusion；
  - GPU busy ≈ 5–6 ms ⇒ **主机侧问题**（异步提交 / 整步编译成一张图），
    写算子帮不上忙。
  - 中间值 / 误差带覆盖两者 ⇒ 记 INCONCLUSIVE + 机制未知（五档口径）。
- INVALID（A3 范围）：trace 抓取进程与 decode 不同窗口/不同进程、无 GPU 区间数据、
  per-step 各段之和 ≠ engine 总计时（>5%）且无法解释。

### 2.4 附带：prefill 4-bit vs BF16 同窗 A/B
- 同锁窗两臂：4-bit worker vs BF16 D1 backbone，**prefill ms/token**（engine basis，
  ≥3 次中位，A/A ≤5%）。
- 判定：4-bit prefill ≈ **2×** BF16 ⇒ 用户的"推断"确认（反量化 GEMM 开销），
  支持"仅 GPU 上换 prefill 路径即可拿到 ANE 方案约八成收益"；
  ≈1× ⇒ 推断证伪，如实写。
- 不与 A2 的 ctx 阶梯合并：prefill 用固定短 prompt + 长 prefill 批（口径在实现档定死）。

### 2.5 不变量 / receipts / 交付
- **不改任何 `crates/` 源码、不改任何既有 gate/threshold/decision**；只新增本 ticket 的
  doc 与（实现档）测量脚本；错误 body 必读，stderr 独立 `.log`。
- **receipts 路径（冻结）**：`~/lfm3-dig-run/bw-util/`（本机，目录实现档创建）：
  `hw.txt`（机型/内存全文）、`a1-*.json`（copy/reduce 各轮原始值）、
  `a2-{arm}-ctx*.json`（≥3 次原始）、`a3.trace` + `a3-parse.json`（或退化分解 json）、
  `prefill-ab.json`、`window.log`、`locks.log`。
- 交付：本 doc 回填 §4 三项 + 附带的原始数字与判定（五档：WIN/PARTIAL/FALSIFIED/
  INCONCLUSIVE/INVALID + 机制算术 GB/s 与 η）、六场景式映射（A3 结果决定"算子 vs 主机侧"
  哪些场景存活/排除）、receipts 路径，rebase 后开 PR（base `feat/lfm3-metal`）。

## 3. 判据（冻结）

| 量 | 判定 |
| --- | --- |
| A1 实测 BW | 实测值代入 η；若与 200 GB/s 标称差 >10% ⇒ 后续 η 叙述以实测为准并在结论首页声明 |
| A2 斜率（16k/32k） | 复现 ≈0.61 ms/k ⇒ 斜率坐实，**KV 路径优化**入队为新大方向；斜率趋平 ⇒ 账本外推被证伪 |
| A3 GPU busy | ≈12 ms ⇒ kernel 方向；≈5–6 ms ⇒ 主机侧方向；中间 ⇒ INCONCLUSIVE |
| 附带 prefill | 4-bit ≈2× BF16 ⇒ 用户推断确认；≈1× ⇒ 证伪 |
| 总体 INVALID | A/A >5%（A1/A2/prefill 各自范围）、机型不符、trace 与 decode 不同窗、A2 参考臂缺失 |

## 4. 机制算术（预期，非判定）

- 账本常数 11.66 ms − 权重下限 4.9 ms（200 GB/s 标称）= **~6.8 ms 未知**；
  若 A1 实测 BW 低于 200 GB/s（M1 Pro LPDDR5 标称上界本身就不一定可达），
  权重下限会上调、未知项相应缩小 ⇒ 这是 A1 必须最先做的原因。
- 参考：native 4-bit M4 0.973 GB ÷ 11.1 ms = 87.7 GB/s，对 100.6 GB/s 实测峰值 η≈87%。
  同量级下若 M1 Pro lane 仅 38%，差距只能来自**实现**（kernel 选型 / 每 token 建图 +
  数百 kernel 提交 / Python 同步），不是硬件带宽 —— A3 的二分（算子 vs 主机侧）
  就是为了把这 62% 拆开。
- 预估锁内：A1 ~1 min + A2 4 ctx × 2 臂 × ≥3 次 × 128 tok（lane ~13 ms/tok ⇒
  ~3–5 min）+ A3 trace ~2–3 min + prefill AB ~2 min ⇒ **锁内 ≤15 min**（实现档
  可微调；超出则如实报超时原因）。

## 5. 测量结果（待回填，round-107+，窗口完成后）

> 占位：本节在测量完成后由文档档回填原始数字（A1 GB/s、A2 两臂四档中位与拟合、
> A3 GPU busy 与退化分解、prefill AB）、五档判定、receipts 路径与 harness commit 指针。
> 预登记阶段不得预填任何数字。
