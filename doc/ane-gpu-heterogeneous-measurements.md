# 预登记：ANE+GPU 异构 decode 四项物理量测量（Issue #134）

**状态**：预登记已冻结，commit 在先；测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-113，2026-11-20 13:41 CST（`date` 实取）。源 = Lead 11:12 追加任务
（`STATUS.md` Lead 新任务 / Issue #134 原文，`gh issue view 134` 可复核）+ Lead 13:38
分流指令（M4 优先做 #134 的 T1/T2，纯硬件测量，M4 完全够用；设备时间分散后每台机
各自双锁、各自 `pmset displaysleepnow` + `caffeinate -si`；不跨机合并中位）。
**当前优先级**：Lead 11:12 明令本项高于 20 题冒烟与 #135；Lead 13:38 进一步要求
**M4 空闲优先跑 T1/T2**（M1 Pro 上的 #135 lane 窗口由另一会话继续，与本 ticket 无互斥）。

## 1. Gate（冻结原文，供 PR 引用）

```
Gate（Lead 11:12 / Issue #134，逐字）：
T1 η（带宽效率）—— 决定场景 1/2 生死
- 方法：LFM3 decode，B = 1/4/8/16（用 llama.cpp 的并行/连续批处理，或 MLX 批，
  不依赖我们自己的 batch 实现）；每格 ≥3 段中位；engine 侧计时。
- 计算：η = 每步字节数 ÷ (每步时间 × 峰值 BW)，峰值 BW 用同窗口 streaming probe
  （copy/reduce）先测。
- Falsifier / 判定：η ≥ 0.85 ⇒ 场景 1 与 2 全部排除（单会话/小并发加 ANE 无意义）；
  η < 0.7 ⇒ GPU 侧还有"空等/launch"空间，应先做 kernel/graph 优化（更便宜）。
- 记录：每格 (B, ms/token, GB/s, η, A/A)。

T2 并发带宽 —— 验证"ANE 不增加带宽"
- 方法：GPU 跑流式带宽（copy/reduce 大数组）同时 ANE 跑一个读大权重的模型
  （复用 crates/apxinf-coreml 或 minicpm5 ANE MLP 路径）。
- 判定：若总带宽未超过 GPU 单独运行（超出 <5%）⇒ "ANE 不增加带宽"在该机成立
  ⇒ 所有带宽瓶颈区的异构方案排除；若显著超出 ⇒ 需重新推导。
- 记录：GPU-only GB/s、GPU+ANE 总 GB/s、ANE 单独 GB/s、三者关系。

T3 ANE↔GPU 往返同步延迟 —— 决定层内切是否不可行
- 方法：一次空 ANE eval + 一次 MTLSharedEvent 等待，测往返；再测 2×L 次模拟（L=24）。
- 判定：单次往返 × 2L 若 ≥ 10% 的单 token 时间 ⇒ 层内切（张量并行）排除；
  同时给出"流水线每 token 需多付的同步成本"。

T4 ANE 小形状矩阵乘吞吐 —— 决定大并发场景 3 是否成立
- 方法：d×d matmul，M = 1/4/8/16/32/64（d = 2048/4096），ANE vs GPU 同形状对比。
- 判定：若 ANE 在 M=16~32 的吞吐 ≤ GPU 同形状 ⇒ 场景 3（B > 2B*）无收益；
  若 ANE 显著更高 ⇒ 场景 3 保留，需进一步算 B>2B* 的流水线收益。

约束（逐字）：
- 每项先 commit 预登记（Gate:/Decision:），再动设备；双 Metal 锁排队；A/A ≤5%；
  engine basis。
- 产出：doc/ane-gpu-heterogeneous-measurements.md（四项方法+原始数字+判定）、
  receipts 路径 + 一个 PR（base feat/lfm3-metal）。
- B>1 的 decode 用 llama.cpp 并行/连续批（不要重新实现原生 batch，那个已判 NOT-WORTH）。
  若某测量在 M4 上不可行（比如 ANE 需要特定 Core ML 路径），如实报 BLOCKED + 原因，
  不要硬凑。
```

问题定义（Issue #134 推导，逐字收录，不重做）：
一步 decode（B 个会话各生成 1 token）：读取字节 = W + B·K（W 每步读一次、与 B 无关；
K 每会话一份，batch 摊不薄）；计算量 = 2·P·B；时间下界
t ≥ max((W + B·K)/BW, 2PB/F, 固定开销)。GPU 带宽→算力拐点 B* = F·b/(2·BW)；
M2 上 int4 约 9、fp16 约 36（int4 反量化占 ALU，实际更低）。ANE 与 GPU 共享同一条
~100 GB/s 总线（**待 T1/T2 实测确认**）⇒ 带宽瓶颈区加速上界 = 1/η − 1（η=0.8 时 25%，
未扣同步）。本 ticket 只测物理量、验证前提；**不是优化单**，不写 kernel、不改 crates。

## 2. Decision（已冻结）

### 2.0 机器、分支、窗口归属

- **执行机器 = M4**：`ssh haiyan-mac-mini`（用户 haiyan-mini），`sysctl
  machdep.cpu.brand_string` = **Apple M4**（Lead 13:38 已确认 M4 完全够用、M4 空闲
  优先做 T1/T2；本机 = M1 Pro，**不参与本 ticket 任何测量**）。开窗第一步把机型读数
  写进 receipt；机型不符 ⇒ 立即中止不占锁。
- **分支**：本 ticket 一律 `agent/lfm3-dig-ane`，从 `origin/feat/lfm3-metal` 切出，
  一个方向（ANE 物理量测量）一个分支一个 PR。
- **锁（M4，RUNNER §3 逐字）**：双 fcntl 文件 `/tmp/enginetailor-metal.lock` 然后
  `~/.cache/enginetailor/metal-measure.lock` —— **固定顺序 acquire、等，不抢**（LOCK_NB
  轮询，6 h deadline）。**各机各自双锁**：M4 的锁不与 M1 Pro 侧 #135 的锁互斥。
- **窗口纪律（RUNNER §3–§4 逐字）**：开窗前 `pmset displaysleepnow` + `caffeinate -si`
  （**不要 -u**）；engine basis 计时（llama-server 用 `--timings` 自报 prompt/eval；
  Core ML 用 host 侧 `Instant` 包住 `Model::predict`；streaming probe 用每轮
  `Instant` 包住提交+同步）—— **不用客户端墙钟作主口径**（#120 SSE 全缓冲）。
  A/A ≤5%：每项每格 ≥3 段分段中位的最坏 spread >5% ⇒ **窗口作废重跑**，不许取中位
  凑。
- **窗口范围与顺序**（冻结，与 Lead 13:38 "M4 先跑 #134 T1/T2" 对齐）：
  **本 ticket 拆两个锁窗**（避免单窗过长与跨项污染）：
  - **窗 W1（T1 + T2，M4）= 下一步「测量」档执行对象**：A0 同窗口 streaming probe
    （≥1 GB，≥5 轮 copy/reduce，给 η 与 T2 分母）→ T1（B=1/4/8/16 llama-server 并行
    批，≥3 段/格）→ T2（GPU-only / GPU+ANE 同窗并发，ANE 单独基线在 T2 之前先测一次）。
    单锁窗内完成，预计锁内 ≤15 min。
  - **窗 W2（T3 + T4，M4）**：T4（ANE vs GPU 小形状 matmul，先做 —— 与 T3 无依赖且
    能独立给出场景 3 判定）→ T3（ANE↔GPU MTLSharedEvent 往返）。预计锁内 ≤15 min。
    W1 结论出来后单独开窗（避免 A0 probe / T2 Core ML 热态影响 T3 往返基线）。
  - 两窗各自 `pmset displaysleepnow` + `caffeinate -si`、各自双锁、各自 receipts。
  - B>1 decode 一律走 **llama.cpp 并行/连续批**（`llama-server -np <B>`），不依赖、
    不重新实现我们自己的 batch 实现（QUEUE #6 已判 NOT-WORTH）。
  - 每格换 corpus seed（`--seed $((20261140 + grid_index))`），receipts 记录。
- **错开原则**：T1 的 streaming probe（A0）用 3.4576 GB 量级大数组（与 #135 A1 的
  1 GiB 口径不同，**两台机器、两个 ticket，数字不合并**），copy/reduce 各 ≥5 轮；
  T2 的"GPU 跑流式带宽"复用 A0 的 probe 进程/并发调用形态（同一 array 尺寸、同一
  probe 二进制）保证 T2 与 A0 可比。

### 2.1 A0 同窗口 streaming probe（T1/T2 的 BW 分母）

- 方法：复用 M4 现成 Rust probe `lfm3-pipelined-decode/scripts/m4_bw_probe.rs`
  （`apxinf_mlx` Array + `stream.eval` + `stream.synchronize`，3,457,635,448 B
  BF16 array，20 轮 copy/reduce，打印 `ms_median`/`gbps_median`/p10/p90 JSON）——
  与 #135 A1 在 M1 Pro 上的 MLX probe 是不同机器、不同 array 尺寸，**不可合并**。
- 前置（实现档）：确认该 probe 在 M4 `$RUN` 现场可构建（`SDKROOT=
  /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`、`MLX_ROOT=~/mlx-sdk`），
  或已有构建产物可直接运行；构建/运行错误体全量进 receipt（round-50–57 教训）。
- 产出：`m4`、array 尺寸、copy/reduce `gbps_median`、spread、A/A 判定 —— **同窗口
  （W1）内测得**，给 T1 的 η 与 T2 的 GPU-only 分母。
- INVALID：probe 未跑 / A/A >5% / GB/s 超过物理上限（M4 LPDDR5 ~100–150 GB/s 量级；
  harness 加 `>500 GB/s ⇒ invalid` sanity 门，与 #135 round-111 ① 同款）。

### 2.2 T1：η（B = 1/4/8/16，同窗口 W1）

- 臂：llama-server（`~/lfm3bench/engines/llama.cpp/build/bin/llama-server`，
  binary sha 前 8 必须记 receipt；与 PR #127/#137 认证基线 `d635b19f…` 对读）
  `--model ~/lfm3bench/models/LFM3-1.7B-Q4_K_M.gguf`（sha 前 8 记 receipt）、
  `-c 16384 -ngl 99 --port 8082 --parallel <B> --timings`（engine basis）。
  **每次改 B 重启 server 以令 `-np` 生效**；restart + load 的墙钟单独记（不进
  ms/token 分母，但进"每格真实成本"旁注）。
- 网格：B ∈ {1, 4, 8, 16}；每格 ≥3 段，每段 64 prompt tokens（`-n 128`、
  `--temp 0` greedy）× <B> 并发请求同时发出；段间换 corpus seed
  `--corpus-seed $((20261140 + seg))`；prompt 用固定短字符串（与 §2.3 T2 同一份
  prompt 文件，避免口径漂移）。并发请求用 bash 并发发 64-token POST（客户端只作
  触发，**计时一律取 server `--timings` 汇总行**；客户端墙钟仅作 sanity 记录）。
- 每段从 `llama-server --timings` 输出解析：`prompt eval time`（prefill，另记）、
  `eval time`（decode 总）、`eval count`（token 数）⇒
  `ms_per_token = eval_time_ms / eval_count`；GB/s = 每步字节 ÷ (ms_per_token/1000)；
  **每步字节（冻结）= 0.973 GB（LLM3-1.7B Q4_K_M 权重）+ B × KV_per_token
  （KV_per_token 从模型 metadata 实取：n_layer × 2 × n_kv_head × head_dim ×
  bytes/kv，Q4_K_M 的 KV 为 F16 ⇒ 实取值写进 receipt；实现档读 `llama.cpp`
  模型元数据或 `-dump-kv` 类信息，不猜）**。η = GB/s ÷ A0 实测峰值 BW。
- A/A：每格 ≥3 段中位 spread ≤5%（RUNNER §4），超 ⇒ 该格作废、窗口 INVALID。
- 判定（Issue #134 冻结，逐字）：**η ≥ 0.85 ⇒ 场景 1（单会话）与场景 2（B < B*）
  全部排除**（单会话/小并发加 ANE 无意义）；**η < 0.7 ⇒ GPU 侧还有"空等/launch"空间，
  应先做 kernel/graph 优化（更便宜）**；0.7 ≤ η < 0.85 ⇒ 灰区，记录机制未知。
- 附带产出：每格实测 GB/s 随 B 的曲线（验证"共享同一 ~100 GB/s 总线"前提是否在
  M4 成立——`W + B·K` 随 B 线性涨但 BW 摊薄）。

### 2.3 T2：并发带宽（GPU + ANE，同窗口 W1）

- 三个测时点（同 W1、同一 GPU thermal state）：
  1. **GPU-only**：A0 probe（3.4576 GB copy/reduce）单独跑，3 段中位 ⇒
     `gpu_only_gbps`（必须与 §2.1 A0 一致，作为 W1 内自洽 check）；
  2. **ANE-only**：一个读大权重的 Core ML 模型跑 `Model::predict` 循环（每秒
     预测次数 × 每次预测的常量权重字节 ⇒ ANE 侧 GB/s），3 段中位；
  3. **GPU+ANE**：A0 probe 与 Core ML 预测循环**同时**进行（两线程；Core ML
     `ComputeUnits::CpuAndNeuralEngine`，placement 经 `load_with_placement` 读到
     `NeuralEngine` 算子占比写进 receipt —— 这是"ANE 真的在跑"的效果证明，G3 类
     provenance），3 段中位 ⇒ `gpu_anes_gbps`（GPU probe 部分）与
     `total_gbps = gpu_anes_gbps + ane_gbps`。
- ANE 模型：用仓库现成 `crates/apxinf-coreml` `ProgramBuilder` + `matmul` 或
  `conv_fp16`（常量权重 ≥1 GB，激活 = 每次预测固定输入），`ComputeUnits
  ::CpuAndNeuralEngine`；实现档复用 `apxinf-model` 既有 ANE 路径（`metal_minicpm5::
  ane` / `metal_clef:: ane`）或自建 probe（倾向后者 —— 避免拉起整个 model 栈，
  只需"读大权重的 ANE 循环"）。实现档冻结具体形状与权重字节数。
- 判定（Issue #134 冻结，逐字）：**若 `total_gbps` 未超过 `gpu_only_gbps`（超出
  <5%）⇒ "ANE 不增加带宽"在该机成立 ⇒ 所有带宽瓶颈区的异构方案排除**（场景 1/2/4
  的带宽论据关门）；**若显著超出（≥5%）⇒ 需重新推导**（可能有独立路径/缓存），场景
  3/5 复活空间，如实写并指向 T4。
- A/A：每个测时点 ≥3 段中位 spread ≤5%（RUNNER §4）；超 ⇒ 该测时点作废、W1
  INVALID。
- receipts：`t2-gpu-only.json` / `t2-ane-only.json` / `t2-both.json` + placement
  facts 原文 + 各 stderr。

### 2.4 T3：ANE↔GPU 往返同步延迟（窗口 W2）

- 方法：一次空 ANE eval（`apxinf-coreml` `Model::predict` 空/最小形状）+ 一次
  `MTLSharedEvent` 等待（Metal `MTLSharedEvent` signal/wait，**用仓库既有 Metal
  绑定**；实现档先盘点 `apxinf-metal` / `apxinf-gpu` 侧是否有现成 shared-event
  原语 —— `grep -rn "shared_event\|SharedEvent"` 目前只命中 `crates/apxinf-serving/
  src/contracts.rs:1957` 的 transcript 测试，**原语需在实现档确认/新增在 examples 或
  测试内，不改 crates 运行时代码**），测单次往返 ms；再测 2×L 次模拟（L=24 ⇒ 48 次
  串行）中位。
- 单 token 时间参考 = T1 的 B=1 段实测 ms/token（同机器 M4、同 ticket，**跨 ticket
  （#135 M1 Pro 11.66 ms）不可直接引用**，仅作量级对照注记）。
- 判定（Issue #134 冻结，逐字）：**单次往返 × 2L 若 ≥ 10% 的单 token 时间 ⇒ 层内切
  （张量并行）排除**；同时给出"流水线每 token 需多付的同步成本"（ms 与百分比）。
- INVALID：ANE eval 与 GPU 侧未在同一窗口/同一进程、无 GPU 区间数据、往返计时
  < 3 段。

### 2.5 T4：ANE 小形状 matmul 吞吐（窗口 W2，先于 T3）

- 网格：d ∈ {2048, 4096} × M ∈ {1, 4, 8, 16, 32, 64}；两臂同形状：
  - **ANE 臂**：`apxinf-coreml` `ProgramBuilder::matmul`（x = activation
    `[M, d]`，y = 常量权重 `[d, d]`，`ComputeUnits::CpuAndNeuralEngine`，
    placement facts 记录该 matmul 的设备）；激活/权重 FP16；`Model::predict`
    每格 warmup 2 + ≥5 次，取中位 ms ⇒ MAC/s = 2·M·d·d / 中位时间。
  - **GPU 臂**：同一形状用 MLX（`mx.matmul`，`apxinf_mlx` Rust 绑定或 MLX python）
    或 Metal GEMM；实现档冻结选择（倾向 `apxinf_mlx` Rust，与 A0 probe 同绑定、
    同 stream sync 语义）；warmup 2 + ≥5 次，取中位 ms。
- A/A：每格 ≥5 次中位 spread ≤5%（ANE/GPU 各自）；超 ⇒ 该格作废。
- 判定（Issue #134 冻结，逐字）：**若 ANE 在 M=16~32 的吞吐 ≤ GPU 同形状 ⇒ 场景 3
  （B > 2B*）无收益**；**若 ANE 显著更高 ⇒ 场景 3 保留**，需进一步算 B>2B* 的流水线
  收益（本 ticket 只给数字，不算完整流水线收益，如实标注"需后续 ticket"）。
- 注意：Core ML `matmul` 是否在 ANE 上被调度 **不保证**（README 明言"Plan devices
  are anticipated placement, not proof of actual execution"）⇒ 实现档必须报
  placement facts 中该 op 的 `ComputeDeviceKind`；若核心 op 被判 GPU/CPU ⇒ ANE 臂
  数据作废、T4 BLOCKED 如实报，不硬凑。

### 2.6 不变量 / receipts / 交付

- **不改任何 `crates/` 源码、不改任何既有 gate/threshold/decision**；只新增本 ticket
  的 doc 与（实现档）测量 harness（`scripts/run-ane-measurements.sh` 或 `.py`，
  分析块独立 `.py` 不用 heredoc —— round-55 教训）；错误 body 必读，stderr 独立
  `.log`（round-50–57 教训）。llama-server / apxinf-serve 启停与双锁 harness 骨架
  沿用 `scripts/run-4bit-vs-llama.sh`（round-102）与 `scripts/run-bw-util.py`
  （round-107/109/111）的成熟模式；stale receipts 先 `mv prev/`（round-64 教训）。
- **receipts 路径（冻结）**：M4 `~/lfm3-dig-run/ane/`：`hw.txt`（`sysctl machdep.cpu
  .brand_string` + `system_profiler SPHardwareDataType` 全文）、`locks.log`、
  `window-w1.log`/`window-w2.log`、`a0-probe.json`、`t1.json`（含每格每段
  `eval_time_ms`/`eval_count`/`ms_per_token`/`gbps`/`eta`/`spread` + KV bytes
  实取）、`t2-{gpu-only,ane-only,both}.json` + placement facts、`t3.json`、
  `t4.json` + placement facts、各 server/child stderr。
- 交付：本 doc §4 回填两项窗口原始数字与判定（五档：WIN/PARTIAL/FALSIFIED/
  INCONCLUSIVE/INVALID + 机制算术 GB/s、η、MAC/s、同步 ms）、六场景映射表
  （Issue #134 表格逐行填"存活/排除/依据（T1/T2/T3/T4）"）、receipts 路径、
  harness commit 指针；一个 PR（base `feat/lfm3-metal`），正文带 Gate 原文、
  before/after 原始数字、A/A、token 一致性说明（**本 ticket 无 token 门 —— 不改
  生成路径，只测物理量**；但若实现档使用 apxinf-serve 臂须显式声明该臂不判 token）、
  receipts 路径（RUNNER §6 六要素）。
- **BLOCKED 协议（Issue #134 逐字）**：任一 T 在 M4 上不可行（ANE 需特定 Core ML
  路径/`matmul` 不被 ANE 调度/shared-event 原语缺失）⇒ **如实报 BLOCKED + 原因**，
  不硬凑、不改门（改门须 Lead 批）。

### 2.7 预计成本与 fuzzing 风险

- W1（T1+T2）锁内 ≤15 min：A0 probe ~1 min + T1（4 B 值 × 3 段 × 64 tok
  × B 并发；llama-server 单机 load 已知 ~10–15 ms/token ⇒ 16 并发段 ~0.8–2 s/段，
  加 restart/load ~10 s/格 ⇒ ~2–4 min）+ T2（3 测时点 × 3 段，probe ~20 s/段 +
  Core ML 循环 ~1 min/测时点 ⇒ ~5–8 min）。
- W2（T4+T3）锁内 ≤15 min：T4（2 d × 6 M × 2 臂 × 5 次 predict，单次 ms 级 ⇒
  ~2–5 min，Core ML compile 在锁外预编译）+ T3（往返测 ~1–2 min）。
- 风险（如实暴露，实现档先验）：(a) M4 当前双锁 FREE、无 apxinf/llama 进程（本步
  13:40 取证）；若另一 campaign 占用 ⇒ 轮询等待不抢（RUNNER §3）。(b) `-np` 并发
  请求在 gateway 侧的实际 join 需求 —— T1 直连 llama-server:8082 绕过 apxinf gateway
  （llama-server 自带 HTTP），避免 #120 SSE 全缓冲影响（客户端墙钟仍不作主口径）。
  (c) `m4_bw_probe.rs` 构建需 `apxinf-mlx` crate 与 MLX SDK —— 实现档先 `cargo build`
  验证；失败则回退到 python mlx 同款 probe（array 同尺寸、同 sync 语义）并记 BLOCKED
  原因。 (d) placement facts 是"anticipated placement"非执行证明 —— README 已明言；
  实现档用 `load_with_placement` + 精确 profile warm prediction，receipt 同时记 facts
  与实测时间，**结论限定为"该形状下 Core ML 建议 ANE 执行"**，不夸大为"已证实 ANE
  独占执行"。

## 3. 判据（冻结）

- **WIN**：W1+W2 均 OUTCOME COMPLETE、A/A ≤5%、T1 η ≥0.85 ⇒ 场景 1/2 排除 +
  T2 显示"ANE 不增带宽"（排除所有带宽瓶颈区异构） + T4 显示 ANE M=16~32 ≤ GPU
  （场景 3 无收益）+ T3 往返 × 48 ≥10% 单 token（层内切排除）⇒ **结论 = 异构 decode
  四场景（1–4）全部排除，场景 5（prefill on ANE）与 6（功耗受限）存活**（场景 6
  未测 10 min 持续负载，如实标注"未测"）。机制算术平：GB/s、η、MAC/s、同步 ms
  全部给出。
- **PARTIAL**：部分 T 有效、部分 BLOCKED（如实标注哪项、为何），剩余有效项得出部分
  场景判定。
- **FALSIFIED**：T2 显示总带宽显著超出（≥5%）⇒ "ANE 不增带宽"前提在 M4 被证伪；
  或 T4 显示 ANE M=16~32 显著更高 ⇒ 场景 3 存活 ⇒ 需后续 ticket 算流水线收益。
- **INCONCLUSIVE**：T1 η 落 0.7–0.85 灰区且其他项亦无强信号；或机制算术不平且
  无法解释（记"机制未知"）。
- **INVALID**：A/A >5% 作废后仍超、机型不符、receipts 缺失、`m4` 机型门失败、
  G3 效果证明（llama-server binary sha / placement ANE facts）缺失、probe 超物理
  sanity 上限。**跨引擎（llama vs apxinf）/跨精度 token 差异不判 INVALID**（G1-diag
  已降级为诊断；本 ticket 更是无 token 门 —— 但要在 doc §4 声明清楚）。

## 4. 机制算术（预期，非判定；预登记阶段不得预填任何数字）

- 权重字节：Q4_K_M 每步 0.973 GB（与 QUEUE #8 的 4-bit 权重字节一致）；M4 streaming
  峰值先按 ~100 GB/s 量级估（**必须用 A0 同窗口实测替换**，round-103/106 同款纪律）；
  单 B=1 段时间下限 ≈ 0.973 GB / 实测 BW ⇒ η 由此算。
- KV bytes/token：n_layer × 2 × n_kv_head × head_dim × 2 B（F16 KV）—— 从模型
  metadata 实取（实现档），不猜；L=24（LFM3 层数）用于 T3 的 2×L 次模拟。
- 同步占比：单次往返 ms × 48 ÷ 单 token ms；10% 线为 Issue #134 冻结判据。
- 场景映射依据分配：场景 1←T1；场景 2←T1+T3；场景 3←T1+T2+T4；场景 4←T2+T3；
  场景 5←T2+T4；场景 6←T1（扩展，**未测 10 min 负载，标注"未测"**）。
- 与 #135 的对照（分析时必须写清，不是判定输入）：M1 Pro lane #135 round-112 A1
  实测 181.65/172.34 GB/s（**另一台机器、另一个 ticket、另一个 array 尺寸 —— 不合并、
  不替代 M4 A0 数字**）；M4 native 4-bit #132 η≈87%（100.6 GB/s，M4，同机同口径
  对照可作 sanity）。

## 5. 测量结果（预登记阶段不得预填任何数字）

*（占位：W1/W2 实测后由「实现」档回填 —— 三项 T 的原始数字 + 判定 + receipts +
harness commit 指针；round-113 预登记冻结，此处为空是设计，不是遗漏。）*
