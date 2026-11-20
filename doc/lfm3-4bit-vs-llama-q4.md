# 预登记：同窗口三臂确认 "native 4-bit 快于 llama.cpp Q4_K_M"（Lead 11:05）

**状态**：预登记已冻结，commit 在先；测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-101，2026-11-20 11:21 CST（`date` 实取）。Lead 11:05 追加任务（STATUS 原文，见 §1）。

## 1. Gate（冻结原文，供 PR 引用）

```
Gate（Lead 11:05，逐字）：
1. 臂：apxinf-4bit（APXINF_LFM3_4BIT=1）、apxinf-bf16（unset）、
   llamacpp-Q4_K_M（~/lfm3bench/models/LFM3-1.7B-Q4_K_M.gguf，llama-server d635b19f…）。
2. 协议：ABBA 交错（如 A B C C B A A B C）、≥3 段/臂、
   S2 decode ctx1024/4096 gen128、engine basis（--engine-timings）、
   每段换 corpus seed、双 Metal 锁排队、显示休眠 + caffeinate -si。
3. A/A ≤5% 每臂每 cell；banner 证明 arms 生效
   （native_4bit=true/false；llama 侧 engine_source=llama.cpp.timings）。
4. 判定（冻结）：
   - 4-bit / llama ≤ 0.95（快 ≥5%）⇒ 写"4-bit 同窗确认快于 llama Q4"；
   - 0.95–1.00 ⇒ "略快"；>1.00 ⇒ 之前的跨窗结论被推翻，如实写。
5. 产出 doc/lfm3-4bit-vs-llama-q4.md（预登记先 commit）+ receipts，并入文档 PR。
```

## 2. Decision（已冻结）

- **臂与证明**：
  - A = `native-4bit`：`apxinf-serve` + `--worker-env APXINF_LFM3_4BIT=1`，
    model root `~/lfm3-dig-run/4bit`（M4 现场量化，G4-speed 已用），G3 banner 必须
    `native_4bit=true` + census `quantized_triples=154/154`；
  - B = `native-bf16`：同一 `apxinf-serve`，switch unset，**不传任何 `--worker-env`**
    （防空臂），model root `~/lfm3-spec/weights-local/LFM3-1.7B`，banner
    `native_4bit=false`；
  - C = `llamacpp-Q4_K_M`：`llama-server`（已构建，d635b19f…）
    `--model ~/lfm3bench/models/LFM3-1.7B-Q4_K_M.gguf`，engine basis 靠
    `bench_engines.sh` 从 `/metrics` 取 `llama.cpp.timings`，
    `engine_source=llama.cpp.timings` 必须命中（否则该臂读数无效）。
- **窗口**：单锁窗内完成三臂全部段（同一硬件/热态/电源态 ⇒ 消除跨窗漂移，这是本 ticket
  相对认证基线 12.39/12.71 的唯一增量）。顺序 **A B C C B A A B C**（ABBA 交错 ×9 段 =
  3 段/臂，每段 corpus seed 独立递增 `20261130 + seg`）。cells = S2
  ctx 1024 / 4096、gen 128（沿用 #127 / G4-speed 口径；S1 短生成 cell 不可用）。
- **分析**：复用 `scripts/run-g4speed.sh` 的分析块骨架（分段中位 → cell 中位 → A/A），
  扩为三臂：每臂 cell 值 = 该臂 3 段中位；A/A = 每臂每 cell 段间 spread，
  **任一臂任一 cell >5% ⇒ 窗口作废**（RUNNER §4，不取中位凑）。比值 = A / C
  （4bit ÷ llama Q4），按 §3 判定。BF16 (B) 只作同一窗内 sanity（预期 3.3× vs A，
  与 G4-speed PASS 一致），不进本 ticket 判定。
- **不变量**：**不改任何 `crates/` 源码、不改任何既有 gate/threshold/decision**；只新增
  本 ticket 的 doc 与（若需）harness 脚本。二进制 freshness 与 banner/census 三件套照
  `scripts/run-g1prime.sh` 执行；错误 body 必读，stderr 按段独立 `.log`。
- **INVALID 范围**：G3 空臂/banner 缺失/引擎 stale、llama 臂 `engine_source` 不符、
  A/A >5% 作废后仍超、三臂请求 ledger 计数不等。**不因跨引擎 token 差异判 INVALID**
  （本 ticket 只测速度；G1-diag 已降级为诊断，doc/lfm3-native-4bit.md §2）。

## 3. 判据（冻结）

| 4bit / llama Q4 | 结论 |
| --- | --- |
| ≤ 0.95 | **4-bit 同窗确认快于 llama Q4**（快 ≥5%） |
| 0.95 – 1.00 | **略快**（方向成立，幅度在噪声量级） |
| > 1.00 | 之前的跨窗结论**被推翻**，如实写 |

交付：本 doc §4 回填数字 + receipts 路径（三臂 raw json、request ledger、banner/census log、
分析输出），并入文档 PR。

## 4. 机制算术（预期，非判定）

- 4-bit 权重字节 0.973 GB ÷ 实测 M4 streaming 带宽 100.6 GB/s = **9.67 ms** 地板；
  G4-speed 实测 10.74 ms ⇒ η≈90%。
- llama Q4_K_M 权重量级同为 ~0.97 GB，同带宽地板相同；两者差距只可能来自
  dequant/kernel 效率与调度开销，因此 **同窗比值** 才能定位（跨窗 ±4% 漂移
  已实测，见 STATUS Lead 11:05）。
- 9 段 × (apxinf 起服务 ~数 s + 2 cells × 256 tok)；llama 起服务带加载更慢。
  预估锁内 ≤10 min（实测以 ledger 为准）。

## 5. 测量结果（round-103，2026-11-20 11:42–11:49 M4）

窗口：11:42:36–11:49:07，锁内 ~391s（符合预登记 ≤10 min）；`ANALYZE_EXIT=0`、
`OUTCOME: VALID`；双锁 11:49:24 复核 FREE、无残留进程。9 段全部 exit=0、failures=0、
无 request error；顺序 A B C C B A A B C 按冻结 doc §2 执行，每段 `--corpus-seed
20261130+seg` 独立。

**G3 效果证明（9/9 段命中）**：`apxinf-serve=7f11e798…`、`worker=42a4fe23…`（A/B 同二进制）、
`llama-server=d635b19f…`、Q4_K_M gguf `3793e1e6…`；A 臂 banner `native_4bit=true` +
census `quantized_triples=154/154`（seg 0/3/6）；B 臂 banner `native_4bit=false`、env 为空
（不传任何 worker-env，防空臂）；C 臂 `engine_source=['llama.cpp.timings']` 3/3 命中，
A/B 臂 `engine_source=['apxinf.ledger']`。

**速度数字（ms/token，engine basis，每 cell n=6 = 3 段 × repeats 2）**：

| 臂 | ctx1024（median，A/A） | ctx4096（median，A/A） |
| --- | --- | --- |
| A native-4bit | **10.9850**（1.31%） | **11.3751**（1.30%） |
| B native-bf16 | 36.4235（1.03%） | 36.7605（0.30%） |
| C llamacpp Q4_K_M | 13.7977（3.31%） | 14.1385（3.51%） |

A/A 全部 ≤5%（最坏 3.51%，无窗口作废，RUNNER §4）。

**冻结判定（§3 三档，逐字）**：A/C = **0.7961**（ctx1024）、**0.8045**（ctx4096），两 cell
均 ≤0.95 ⇒ **4-bit same-window CONFIRMED faster than llama Q4（快 ≥5%）**，实际快
20.4% / 19.6%。B/A in-window sanity = 3.32× / 3.23×（与 G4-speed PASS 3.3× 一致；
仅 sanity，不进 ticket 判定）。"被推翻"档（>1.00）未触发：之前跨窗证据（−8%～−11%）
的幅度被**低估**，同窗下差距更大（−20%）。

**机制算术**：A 臂 0.973 GB ÷ 10.99 ms = 88.5 GB/s（1024）/ 85.5 GB/s（4096），对
100.6 GB/s streaming 峰值 η≈88%/85%。C 臂 Q4_K_M 混合精度权重 >0.973 GB 且无 dequant
融合，故同量级下比值差距来自 kernel/调度效率。

**Receipts**：`M4:~/lfm3-dig-run/window-4bit/vs-llama/`（`vs-llama.log`、`raw/` 9 文件、
`logs/`、`locks.log`、`vs-llama-requests.jsonl`、`vs-llama-diag.jsonl`）。
**Harness**：`scripts/run-4bit-vs-llama.sh` + `scripts/analyze-4bit-vs-llama.py`
（commit `151187c0`）。
