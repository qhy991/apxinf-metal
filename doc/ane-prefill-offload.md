# 场景 5：prefill 放 ANE —— 预登记 + 四项测量（Issue #161）

**状态**：预登记已冻结（P2 部分），commit 在先；测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-132，2026-11-22 11:37 CST（`date` 实取）。源 = Issue #161 原文（`gh issue view 161`
可复核）+ Lead 11:50 优先级指令（STATUS.md Lead 段）+ Lead P1 preliminary 评论（#161 唯一评论，
2026-11-21 15:52）。
**当前优先级**：Lead 11:50 明令 `#161 P2 → P3 → P4` 为「现在唯一存活的方向」，高于 #136 与 #135。
**本 ticket 分档**：P1 已由 Lead 在 M4 跑完（preliminary，数字见 §2.1）；本轮执行档 = **预登记 P2**
（零设备；`doc/` 写 gate + 仪器规格 + decision 冻结）。P2 测量、P3、P4 各占后续单独轮次。

## 1. Gate（冻结原文；P1/P3/P4 引 #161 原文，P2 补充冻结口径）

```
Gate（Lead / Issue #161，逐字）：
P1：ANE 在 M≥512 上 ≥ 1.5× GPU 吞吐 ⇒ P1 通过（否则场景 5 也死）。
P2：KV/状态交接 ≤ prefill 时间的 10 % ⇒ P2 通过。
P3：MLP-on-ANE ≤ 0.9 × 全 GPU ⇒ P3 通过。
P4：ANE prefill 期间 decode 退化 ≤ 5 % ⇒ P4 通过。

判定（冻结，逐字）：
- P1 与 P2 通过 才值得做 P3/P4；P3+P4 都通过 ⇒ 记录为「prefill 卸载 WIN」（按预期收益写成并发
  场景的收益，不是单 token 延迟）；任一不过 ⇒ 记录五档 + 数字，场景 5 也关掉，六场景全灭，campaign 收口。

约束（逐字）：
- 先预登记再动设备，双锁排队；只在 M4 上做。
- ANE 需要静态形状：prefill 的 M 按 bucket（512/1024/2048/4096）走，不能任意 M。
- 不准改 decode 路径数值（decode 的 token 一致性门仍适用；prefill 侧的数值要记录但不强制与 GPU
  逐位一致，但要报 prefill 输出的 KV 与 GPU 版 KV 的偏差）。
```

### 1.1 P2 口径冻结（本预登记新增；首次测量后不许改）

P2 Issue 原文只有「per-prefill 交接 ms（512/2048/8192 三个上下文档）+ 占 prefill 时间比例 + ≤10% 判定」。
本轮把仪器与口径写死，避免首次测量后临时改口径：

- **被交接的状态 = GPU 解码器在一次 prefill 后真正续用的全部 request state**，与
  `metal_lfm3::state_bytes`（`model.rs`，冻结参考测试 helper）完全同构：
  1. **11 个 attention 层的 KV**（6 full + 5 sliding）：`keys`/`values` 各
     `[1, 4, C, 128]` BF16，按实际 prefill 长度 `P` 只交接已写入的行 `[0, P)`（sliding 层同构：
     KV 缓冲是 dense C 行，decode 消费 `[0, end)` 后仅读窗口）；
  2. **13 个 conv 层的窗口**：`[1, 2048, 6]` BF16（conv width 7，保留 6 列，config conv_state_len()=6）。
  - **不交接**：rotary tables（GPU 侧由 position 自算，与交接无关）、prefill 中间激活、logits。
  - 布局按 ANE 侧 prefill 的天然输出布局（`[1, kvh, P, hd]` 与 `[1, hidden, 6]`）——
    **不做 transpose/reshape**（P2-1 的「纯拷贝」臂即此布局；IOSurface 臂测零拷贝布局不变性）。
- **状态字节算术（先算平，再测；测量必须复核与本表一致）**：
  - 每 attention 层每 token 每 request：`2 (K,V) × 4 kv_heads × 128 × 2 B = 2048 B`；
  - full（6 层）：`6 × 2048 × P`；sliding（5 层）：`5 × 2048 × 256 = 2,621,440 B`（256 窗口，与 P 无关）；
  - conv（13 层）：`13 × 2048 × 6 × 2 = 319,488 B`；
  - 总计 `S(P) = 12288·P + 2,940,928` bytes：

    | P（上下文档） | 交接字节 | MiB |
    |---:| ---: | ---: |
    | 512 | 9,232,384 | 8.80 |
    | 2048 | 28,106,752 | 26.80 |
    | 8192 | 103,604,224 | 98.80 |

  - 分母 prefill 时间：优先 **P3 同窗口的全-GPU 端到端 engine-basis prefill_ms**
    （`bench_engines.sh --engine-timings`，`ttft_basis=engine`，口径 = state reset + capacity preparation +
    prefill；与 P3 判定同源，可直接算「占 prefill 比例」）；若 P3 窗未开，用同窗口
    `metal_lfm3` engine-basis prefill_ms（同一脚本、同一 model、同一 prompt）并显式标注。
    **不用客户端墙钟作分母**（#120 SSE 全缓冲，RUNNER §3）。
- **三臂（Issue P2-1/P2-2 冻结为三臂 + 仪器校核）**：
  1. **C（copy，同进程 GPU buffer→GPU buffer）**：把状态 buffer 从「源布局」拷到「解码器布局」。
     源布局 = 交接前状态（mock 生成：attention KV 用**确定性 f32 在 host 生成、cast BF16、
     `[0,P)` 写入**，其余行强制置零——避免随机/未求值行进入 readback 时序；conv 窗口同法）；
     解码器布局 = 与 decode 消费形状逐位同构（拷贝后 `state_bytes` 形状对照）。测量 = 提交 + 同步。
  2. **H（host round-trip，**只报告，不进 P2 判定**）**：`Array::to_bytes_ready` 把状态读到 host
     `Vec<u8>`（等价于跨设备零拷贝的最坏形态：必须经过一次 PCIe/统一内存读）。机制对账用。
  3. **I（IOSurface 零拷贝，可行性探针）**：同尺寸 buffer 跨设备共享。
     **判定子项 = 布局转换在 GPU 侧的耗时**（Issue P2-2）：AE 是**借名探针**——
     ANE 输出写入共享 buffer，GPU 侧做 `[1,4,P,128] ↔ [1,P,4,128]` 的 transpose + 相同 shape 的
     reshape/copy（解码器布局），测 GPU 侧提交+同步。**ANE 真实生产写入在 P3 实现**（P3 才有真实
     KV 生产路径）；本臂的 AE 只证明「一次 ANE eval + 一次 GPU 布局变换」的可行与量级。
- **仪器与探针模式**：新增 `crates/apxinf-model/examples/ane_kv_handoff_probe.rs`
  （`--features coreml`，位置参数 mode 同 ane_probe 约定：`copy | host | iosurface`；
  `--out <json>`；`--p <len>` 默认 512/2048/8192 全跑；`--reps <n>` 默认 21 + warmup 5；
  `--cache-dir <dir>` 默认 `/tmp/apxinf-ane-kv-handoff-cache`）。带：
  - mode 内置 **状态字节核算**：每 arm 输出 `S_measured`，与 §1.1 表逐 P 对拍；不一致 ⇒ INVALID；
  - **IOSurface：零库/零新依赖，只做可行性 + GPU 侧布局变换**：
    - 前置：`/System/Library/Frameworks/IOSurface.framework` 存在检查（缺 ⇒ 记录 NOT_AVAILABLE，
      不降级成普通 buffer 冒充 IOSurface）；
    - 共享 buffer = **固定 shape `[1, 4, 8192, 128]` BF16（98.80 MB）的 IOSurface**，
      独立模块 `iosurface_probe.rs`（`#[cfg(target_os="macos")]`；inline `extern "C"` + dlopen
      符号解析，**无 `link` 属性、无 crates.io 新依赖**）。可行 = `IOSurfaceCreate` +
      属性可查（`IOSurfaceGetAllocSize` == 98,552,320 B，唯一引用计数）；布局变换 =
      GPU 侧按 GPU-stdlib `transpose_slice` 的**轴序**（`to_chw`/`to_torch`）做
      `[1, 4, C, 128] ↔ [1, C, 4, 128]` + reshape 后 `copy()` + eval；
      AE 经 `IOSurfaceGetBaseAddress` + `IOMobileFramebufferGetLayerDefaultSurface` 替代 = **不实现**
      （留给 P3；本轮不进判定）。
  - **C/H/I 三臂在 GPU 上测量**（同一 `MlxBackend`，双锁独占 ANE 期间跑，口径 = 提交+同步的
    engine 自报，不用客户端墙钟）；
  - placement：C/H 无 ANE；I 臂的 AE 替代不实现 ⇒ **G2 placement 门不适用**（G2 只对 P1/P3 的
    CoreML program 生效），但 I 臂的 GPU 侧布局变换须报 `metal_caps` 与 stream。
- **A/A 门**：每臂每 P 分段中位最坏 spread ≤5%（RUNNER §4）；>5% ⇒ 该臂该 P 窗口作废重跑。
- **进程/独占**：窗口前后 `ps` 扫描不得有其他 ANE 使用进程（apxinf-serve/clef worker/coreml 探针/
  python coremltools）；clef 默认在用 ANE（5632/12288 列），扫描输出存 receipts。
- **判定细化（三档，Issue ≤10% 判定的细化，不放宽）**：
  - `C ≤ prefill 的 10%` 且 I 可行（GPU 侧布局变换 ≤ prefill 的 10%）⇒ **P2 PASS**（可做 P3/P4）；
  - `C > 10%` 但 IOSurface 不可行/不可用 ⇒ **P2 FAIL**（即使 C 满足比例，P2-2 不可用 ⇒ P3/P4
    的预登记依赖不成立；记录五档 + 数字）；
  - `C ≤ 10%` 但 `H > prefill 的 50%`（host round-trip 明显更贵）⇒ 记录机制「零拷贝必要性」，
    不改 PASS 判定（C 已过门）。

## 2. Decision（已冻结；本轮 = 预登记，不跑任何测量）

### 2.1 已完成与未完成（按 Lead 指令，P1 已由 Lead 跑完）

- **P1 = PASS（Lead preliminary，M4，2026-11-21 15:52）**：ANE vs GPU（MLX，同形状，fp16，5 reps 中位）
  M=256/512/1024/2048 上 **2.31/3.51/4.39/6.11 vs 0.65/0.99/1.12/1.24 TMAC/s ⇒ 3.54×/3.56×/3.92×/4.92×**，
  远超 P1 的 ≥1.5× 门。原文在 #161 评论。**本轮不重测**（P1 已达门；若 Lead 要求补 M=4096 或
  attention 投影形状，另开窗）。
- **P2 = 本轮预登记**（零设备；Gate/Decision/仪器冻结；不跑三臂）。
- **P3/P4 = 未开始**：P3 依赖 §1.1 的状态交接口径 + #160 的 `ane_slice`（FP16 program 已在
  `agent/ane-fp16-port` 分支，未合入本 ticket base）；P4 依赖 P3 的 MLP-on-ANE 服务形态。

### 2.2 机器、分支、窗口归属

- **执行机器 = M4**：`ssh haiyan-mac-mini`（用户 haiyan-mini）；`/usr/sbin/sysctl -n machdep.cpu.brand_string`
  = Apple M4（round-125 已验）；本机 = M1 Pro，**不参与本 ticket 任何测量**。
- **分支**：本 ticket 一律 `agent/lfm3-dig-ane`，从 `origin/feat/lfm3-metal` 切出；**一个方向一个 PR**。
  base `feat/lfm3-metal`（P1 preliminary 数字来自 Lead 分支，不属本 ticket base 范围）。
- **锁（M4，RUNNER §3 逐字）**：双 fcntl `/tmp/enginetailor-metal.lock` →
  `~/.cache/enginetailor/metal-measure.lock`，固定顺序 acquire、等，不抢（LOCK_NB 轮询，6h deadline）。
  各机各自双锁（M1 Pro 侧 #135 不互斥）。
- **窗口纪律**：开窗前 `pmset displaysleepnow` + `caffeinate -si`（不带 -u）；engine basis 计时
  （GPU 侧 = 提交+同步 `Instant`；CoreML = host `Instant` 包 `predict`）；A/A ≤5%；process scan（§1.1）。
- **窗口归属（冻结，两窗）**：
  - **窗 W1（P2 交接，M4）= 下一步「测量」档执行对象**：`ane_kv_handoff_probe` 三臂 × 3 上下文档
    + 分母 prefill（engine-basis，同窗口或 P3 同源窗口）；analyzer 出 `ane-p2-verdict.json`；
    driver rc 判窗（0/101/102/103/104）。
  - **窗 W2（P3 端到端，M4）= P2 PASS 后**：`全 GPU` vs `MLP-on-ANE`（512/2048 prompt）engine-basis
    prefill_ms + token 对照；**未预登记 P3 细则前不许开**（P3 是另一轮的「预登记」档）。
  - **窗 W3（P4 干扰，M4）= P3 PASS 后**：双会话 decode/prefill 干扰；同上，未预登记不许开。
- **单轮预算**：每轮 ≤25 min；先 commit 再继续（Lead 老规矩）。

### 2.3 判定与五档口径（Issue 冻结 + 本 ticket 口径）

- **WIN（场景 5 记录为 prefill 卸载 WIN）**：P1 PASS + P2 PASS + P3 PASS + P4 PASS ⇒ 按并发场景
  收益口径写（不是单 token 延迟）；**默认开关仍 off**（默认开启是 owner 决定）。
- **PARTIAL**：P1/P2 PASS 但 P3 或 P4 任一不过（数字 + 机制，记录后不再重试本 ticket）。
- **FALSIFIED**：P2 不过（`C > prefill 的 10%` 或 IOSurface 不可用且 C 也不满足）⇒ 记录五档 + 数字，
  **场景 5 关掉，六场景全灭，campaign 收口**（Issue 判定逐字）。
- **INVALID**：字节核算与 §1.1 表不符 / A/A >5% / G1 独占失败 / 进程状态污染 / analyzer schema 错位。
- **INCONCLUSIVE**：连接不可达、分母 prefill 不可得、IOSurface 探测无结论且无法归因。
- **机制算术（报告强制）**：交接 GB/s = `S(P) / t_copy`；与 M4 streaming 峰值 100.6 GB/s（RUNNER §2）/
  P1 的 ANE 6.11 TMAC/s 对账；算不平写「机制未知」。

### 2.4 不碰什么（明确清单）

- **decode 全系**：`len==1` 路径、融合 conv_step、decode MLP —— 不改（token 一致性门仍适用）。
- **prefill 的 attention/conv/norm 数值**：P2 三臂是**测量仪器**，不动 `metal_lfm3` 运行时；
  P2 不产生任何 `metal_lfm3` diff。
- **KV/conv 状态的布局、容量、prepare/reset/release**：P2 只读（`state_bytes` 口径）；
  P3 才可能新增 ANE 生产路径，届时走 P3 的白名单。
- **4-bit/Q4、serving 协议、tokenizer、gate 语料/harness 语义**：不改。
- **共享 kernel / 其他家族**（`metal_clef`/`metal_qwen_image`/`apxinf-mlx` 运行时 src）：不改；
  本 ticket 只加 `examples/ane_kv_handoff_probe.rs` + `examples/iosurface_probe.rs`（独立新文件）+
  `Cargo.toml` 的 example 声明（如有）+ `doc/` 本文件 + `STATUS.md`。
- **判定工具**：`bench_engines.sh` 的 engine-basis 口径、P1 数字、Lead 冻结阈值 —— 不改。

### 2.5 已知风险（必须写进文档）

1. **分母污染**：P2 的「占 prefill 比例」分母若跨窗口取（P3 未开时用同 engine-basis 但不同窗），
   会带入热漂移 ⇒ 记录为「分母来源」字段，A/A 门对分子分母各自独立。
2. **C 臂的「解码器布局」若做成 transpose，则 P2-1 不再是纯拷贝** ⇒ §1.1 冻结：布局相同，
   transpose 只在 I 臂的 GPU 侧（Issue P2-2）做。
3. **mock 状态与真实 prefill 状态的数值差异不影响时序**，但若 `to_bytes_ready` 走到未求值/未初始化
   行会污染时序 ⇒ mock 必须先 `stream.eval` 全量、`[P, C)` 置零（§1.1 臂 1）。
4. **IOSurface 可行 ≠ ANE 可写入**：本轮只证「共享 buffer 存在 + GPU 侧布局变换可行 + 量级」；
   ANE 真实写入是 P3 实现，P3 预登记必须把 IOSurface 路径列为依赖并重跑 G2 placement。
5. **ANE 与 clef 争抢**：G1 双锁 + ps 扫描；deploy 期默认 off 是唯一缓解（继承 FP16 port 风险 #2）。
6. **P2 PASS 不保证 P3**：`C ≤ 10%` 只排除交接成为瓶颈；P3 还要过端到端 ≤0.9×（Issue 冻结）。
7. **六场景全灭风险**：P2 FAIL ⇒ 按 Issue 判定直接收口 campaign，不再开新方向（本 ticket 内不重复）。

## 3. 本 ticket 交付物与顺序（冻结）

1. **本轮（预登记档）**：`doc/ane-prefill-offload.md`（本文档，Gate/Decision/仪器冻结）+ `STATUS.md`
   记录；零设备、零测量；commit `agent/lfm3-dig-ane`；push origin（Lead 18:18 git 指示范围）。
2. **下一轮（实现档）**：`examples/ane_kv_handoff_probe.rs` + `examples/iosurface_probe.rs` +
   `Cargo.toml` example 声明（若需）；离线验证（`cargo check` 无 device 不可跑 ⇒ 用 `git diff --check`
   + `rustc` 语法级检查 + 字节核算单测，`cargo check --features coreml` 在 M4 lock-free build 阶段做）；
   commit + push。
3. **再下一轮（测量档 = W1）**：双锁 + process scan + `pmset` + probe 三臂 + 分母 + analyzer
   ⇒ `ane-p2-verdict.json`；A/A 门；五档结论。
4. **P2 PASS ⇒ P3 预登记档**（新轮）→ P3 测量窗 → P4 预登记档 → P4 测量窗 → PR
   （base `feat/lfm3-metal`，正文带 gate 原文、before/after 原始数字、A/A、token 一致性、receipts 路径）。

**本 ticket 一个 PR**（P1 preliminary 不进本 PR 的判定，但可引用；若 Lead 要求 P1 数字进 PR，
在 PR 正文引 #161 评论）。
