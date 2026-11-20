# QUEUE #8 — 原生路径 4-bit 量化支持（Gate + Decision）

预登记日期：2026-11-19 22:07 CST（`date` 实取）。**修订 2026-11-20 01:05 CST（`date` 实取），Lead 纠偏（2026-11-20 00:16）**：G1 跨引擎逐 token 硬门降级为诊断项 `G1-diag`，新增产品门 `G1′`（质量）与 `G4-speed`（速度）；修订先行 commit，设备运行（G4-speed 窗、G1′ 评测）在本 commit 之后。本文件为 pre-registration：Gate 与 Decision 在任何设备运行前冻结；实现、测试、速度窗、质量评估分属后续轮次。

- 相关队列：QUEUE.md v2 第 8 项（Lead 决策 2026-11-19 22:01「4-bit 可以，最高价值杠杆」；Lead 修订 2026-11-19 22:05「M4 现场量化，不从 M1 拷」）。
- 权威代码基座：`agent/lfm3-dig`，tip（本 commit 之前）`9f12e4d9`（QUEUE #6 预登记），base `feat/lfm3-metal`。
- **交付状态（round-81 定稿）**：QUEUE #8 已随 PR #132 **合并**入 `feat/lfm3-metal`（merge commit `e353eda5`）；两产品门判定：**G4-speed PASS（3.26–3.31×，≤ llama Q4）+ G1′ PARTIAL（n=20，both −5.00pp / either 0.00pp，A/A DISAGREE caveat 保留）** ⇒ 按 §5 = **PARTIAL**：4-bit 保持 opt-in（`APXINF_LFM3_4BIT=1`），**BF16 默认**。Lead 09:31 已接受该 opt-in 形态为 campaign 收口状态，无待裁决项。

---

## 1. 量化格式（步骤 1 已在 M1 实测，写死为 Gate 事实）

格式探测 receipt：`~/lfm3-dig/4bit-format-probe.txt`（M1，`~/models/LFM3-1.7B-MLX-4bit/model.safetensors`，973,083,386 B，`mx.load` 全量 594 keys，round 42）。实测结论：

1. **packed 权重 = uint32**：每 uint32 装 8 个 4-bit nibble；shape 形如 `(out, in/8)`（uint32 列数 = in/8，因为 8 nibbles/word × 4 bits = 32 bits）。例：`embed_tokens.weight (128000, 256)` uint32 ⇒ 行内 256×8 = 2048 个 nibble = hidden 2048；`ff.w2.weight (2048, 960)` ⇒ 960×8 = 7680 = in 7680。
2. **group_size = 64**（跨模块由 per-group 列数反推一致：embed `scales (128000,32)`，2048/32=64；ff.w2 `scales (2048,120)`，7680/120=64；ff.w1 `scales (7680,32)`，2048/32=64；attn.q_proj `scales (2048,32)`，2048/32=64）。
3. **scales 与 biases 均存在、均为 bfloat16** ⇒ MLX `mx.quantize(mode="affine")` 形式：`v=scales`、`b=biases`。Rust 侧必须使用对应 affine dequant / affine qmm（`apxinf-mlx` 的 `q4_f16` / `qmm_bm64` 注释自述「MLX affine」，实现轮核对符号契约：`value = (q − bias) * scale`）。
4. **lm_head = tied embedding**：`grep lm_head` = 0 keys ⇒ 无独立 lm_head；`embed_tokens` 自身已被 4-bit 量化（uint32 (128000,256) + scales/biases (128000,32)）。vocab = 128000。lm_head decode 投影天然走 4-bit，收益含在总量内。
5. **保持 BF16 的部分**：所有 norm（embedding_norm / ffn_norm / operator_norm / post_*）、per-layer depthwise `conv.conv.weight (2048,7,1)`。
6. **量化的线性**：conv/attention 的 `in_proj/out_proj/q/k/v/o`、FFN `w1/w2/w3`、`gate_proj (16,256)` 全部 4-bit。
7. **字节账**：973,083,386 B ≈ 0.973 GB ⇒ 每 token 权重读 0.973 GB（vs BF16 3.4576 GB，26.4%）；M4 100.6 GB/s roofline ⇒ **9.67 ms/token 地板**（vs BF16 34.36 ms）；目标 ≥2.5× vs BF16（≤13.7 ms）且 ≤ llama.cpp Q4（12.39/12.71 ms/token）。

## 2. Gate（冻结；任一硬门触发即 INVALID，窗口数字不可解读）

| # | 门 | 判据 | 失败后果 |
| ---|----|------|----------|
| **G1-diag**（诊断项，**不再是硬门**，Lead 2026-11-20 00:16） | 同精度跨引擎 token 分叉取证 | 记录原生 4-bit 与 **Python lane 4-bit**（同 prompt、同权重文件语义、贪心）的**首处分叉位置**（已知：G1 v12 prompt 0 step 59，lane=125020 vs native=25797）；并**尝试用 FP16 scale 路径复现 lane**（lane 走私有 `packed_mode` FP16 scales，原生走 MLX 通用 `quantized_matmul` + 文件 BF16 scales；`apxinf-mlx` 已有 `q4_f16.rs`）。 | **修好 ⇒ 升级回硬门（G1 硬门恢复）**；修不好 ⇒ 记录为**已知引擎差**（机制：不同引擎舍入差，近平局 logit 翻转），**不判 INVALID、不阻断速度/质量门**。 |
| **G1′**（**产品门：质量**，新增） | solved rate 回落 | 同一批 agent 任务（MBPP/HumanEval 套，**≥6 题**），4-bit vs BF16 的 **solved rate** 与**解出耗时**：solved rate 掉 **≤3 个百分点 ⇒ 通过**；**>10 个百分点 ⇒ 停**（3–10 点之间记数字并由 Lead 裁决）。 | ≤3 点：通过；3–10 点：PARTIAL 候选（4-bit opt-in）；>10 点：停，4-bit 不可默认。 |
| **G4-speed**（**产品门：速度**，新增） | ABBA 速度 | ABBA、**≥3 段/臂**、A/A ≤5%、engine basis（`--engine-timings`；客户端计时因 #120 SSE 全缓冲不可用），三臂 `native-4bit` vs `native-bf16` vs `llamacpp-Q4_K_M`。判定：4-bit 相对 BF16 **≥2.5×** 且 **≤ llama Q4**（12.39/12.71 ms/token）⇒ **速度门通过**。 | 未达标记五档（速度 <1.5× ⇒ NEGATIVE；机制预测差 >15% ⇒ FALSIFIED；1.5–2.5× 且算不平 ⇒ INCONCLUSIVE），仍开 PR 作为"已测量实现"。 |
| G2 | 跨精度不设逐位门 | 4-bit vs BF16 贪心输出**必然**不同，用户已明确接受；本项不因 4-bit ≠ BF16 token 判 INVALID。质量以 G1′ solved rate 记录。 | 无（不是门）。 |
| G3 | 空臂 / effect proof | worker banner `native_4bit=<true\|false>` 字段与 `APXINF_LFM3_4BIT=1` 同源；速度窗内 banner 必须含该字段且 armed 臂 `=true`。banner 缺失 / 二进制早于源码 mtime（未重新构建）⇒ 空臂陷阱，独立 INVALID（#4 教训）。 | INVALID（独立于速度/质量）。 |
| G4-shape | shape 不漂移（原 G4，改名避让 G4-speed） | 4-bit 臂的 forward 输出 shape / logits shape / KV 与 conv state 形状与 BF16 臂一致；不做 padded/bucketed/wider matmul，不拆 K，不改读向（#128 教训）。实现若被迫引入任一 shape 漂移 ⇒ 回退该实现，记录机制。 | 视情形 INVALID 或 FALSIFIED（速度门若已过则 FALSIFIED 机制记录）。 |
| G5 | A/A ≤ 5% | G4-speed 窗每臂每 cell 分段中位最坏 spread ≤ 5%；S1 短生成 cell（#121 A/A 5.02%）**不可引用**，须本 ticket 自有 A/A。超 ⇒ 窗口作废重跑。 | 窗口作废（重跑）；不取中位凑数。 |

## 3. Decision（实现与推进路线；不可静默改）

### 3.1 权重来源（Lead 修订 22:05 冻结）

- **不在 M1 编 Rust MLX**：M1 的 homebrew MLX 为 0.31.x，apxinf-mlx pin 0.32.3 headers/library（`cargo check -p apxinf-mlx --features native` 报错，Lead 实测）。
- **不从 M1 拷 973 MB 到 M4**：实测 ~1.2 KB/s，不可用。
- **M4 现场量化**：M4 有 mlx 0.31.2 python venv（`~/lfm3-lane/.venvs/lfm3/bin/python`）与 BF16 checkpoint（`~/lfm3-spec/weights-local/LFM3-1.7B/`）。写 `quantize_lfm3.py`：读 BF16 safetensors → 对每个线性权重 `mx.quantize(group_size=64, bits=4, mode="affine")` → 存 HF 风格目录（`weight`(uint32) / `scales` / `biases`，key 命名 `language_model.model.<...>.{weight,scales,biases}`）+ `config.json` 写 `quantization: {group_size:64, bits:4, mode:"affine"}`。MLX 量化确定性（同参数同输入同输出），故 M4 现场产出的 4-bit 文件与 M1 的文件在数值上应一致（可用 size/samples 抽查，不强求逐字节）。
- sanity：用 lane 的 engine 加载现场产出的 4-bit 文件，跑一句生成确认合理文本（能加载、不 NaN、文本像样）。

### 3.2 原生实现（照抄既有模式，不发明）

- 参照 `crates/apxinf-model/src/metal_qwen_image/weights.rs`（已用量化权重加载的完整范例）+ `apxinf-mlx/src/{quantized,q4_f16,qmm_bm64}.rs`（`Quantized` 类型、affine dequant、BM64 retile）。
- `metal_lfm3` 增加 quantized Linear（packed uint32 + BF16 scales + BF16 biases [+ 原 BF16 bias 若有]）；`call` → `quantized_matmul` / 对应 affine qmm。
- **缺失回退**：若某权重缺 4-bit 三元组（scales/biases 缺、shape 不符、dtype 非预期），该层回退 BF16 加载路径（保持现有路径可用），并在 log / banner 说明 fallback 层数（诚实记录，不假装全部 4-bit）。
- B1 / BF16 路径（默认 `APXINF_LFM3_4BIT` unset）字节级不动。

### 3.3 开关与 banner

- 开关 `APXINF_LFM3_4BIT=1`（默认关）+ `Lfm3Model` 构造时捕获 `native_4bit: bool`；`native_worker.rs` banner 增 `native_4bit=<true|false>` 字段（与 switch 同源 env，空臂门）。导出 `native_4bit_enabled()` 供 census / 测试。
- 空臂门实证：`G3`。

### 3.4 速度窗（实现 + G1 过后，单独轮次）

- ABBA：`native-4bit` vs `native-bf16` vs `llama.cpp Q4_K_M`（12.39/12.71 ms/token 认证基线，不必重跑 llama，但若要同窗 receipt 需重跑）。
- A/A ≤ 5%（本 ticket 自有）；engine basis（`--engine-timings`，客户端计时因 #120 SSE 全缓冲不可用）。
- 目标：**≥2.5× vs BF16** 且 **≤ llama.cpp Q4**。

### 3.5 质量 = G1′ 产品门（Lead 修订：与速度门并列为交付门）

- 同一批 agent 任务（MBPP/HumanEval 套，≥6 题）比 4-bit vs BF16 solved rate 与解出耗时，记录差值与绝对值；≤3 点 ⇒ G1′ 通过；>10 点 ⇒ 停（4-bit 不默认）；Lead 批准先跑 G1′（不敏感于锁竞争，但 A/A 仍须干净）。

## 4. Cells / Order

- 速度窗 cells：S2 ctx 1024 / 4096 各 128 tok（同 #127/#4 口径），arms `native-4bit (A) / native-bf16 (B)`，`A B B A A B`；llama Q4 用既有 receipts，不重跑（若同窗重跑则单列）。
- 质量：MBPP / HumanEval 套，batch 与 prompt 规格与 #127 已有质量评测一致（引用已有 receipts 口径，不新发明）。
- S1 decode-at-8 / #121 短生成 cell 不可用。

## 5. 判据（冻结；Lead 2026-11-20 00:16 修订）

交付规则：**两门（G1′ 质量 + G4-speed 速度）都过 ⇒ 开 PR**（4-bit 默认可用 `APXINF_LFM3_4BIT=1` 打开，BF16 保持默认直到用户确认切换）；**任一不过 ⇒ 记录五档 + 数字，仍开 PR 作为"已测量实现"**。G1-diag 仅诊断，不再单独判 INVALID。

- **WIN**：G4-speed 过（≥2.5× vs BF16 且 ≤ llama Q4）+ G1′ 过（solved rate 掉 ≤3 点）+ A/A 净 + G3 空臂检查净 ⇒ 建议 4-bit 为原生默认（BF16 仍默认直至用户确认），4-bit opt-in 就绪。
- **PARTIAL**：G4-speed 过但 G1′ 掉 3–10 点 ⇒ 默认 BF16，4-bit opt-in。
- **NEGATIVE**：速度 <1.5× ⇒ 机制未知，记录（字节账若仍对平则 roofline 归因问题）。
- **FALSIFIED**：机制预测（0.973 GB ÷ 实测带宽）与实测速度差 >15% 且 A/A 干净 ⇒ 机制被否，记录算术。
- **INCONCLUSIVE**：A/A 净 + 速度落在 1.5–2.5× 且机制算术算不平，写"机制未知"。
- **INVALID**（仅限）：G3 空臂 / G5 A/A > 5% 作废（重跑后仍超）/ G4-shape 漂移。**G1-diag 分叉不再判 INVALID**。

## 6. Relation / Receipts

- Relation：QUEUE #8 是**唯一经用户批准的量化方向**，与 QUEUE #1–#5（结构重组类封盘）无关：4-bit 改变权重精度，不改执行图结构 / tiling 逻辑（量化 kernel 走 MLX 现成 qmm，BM64 retile bit-exact 对齐 Python lane），故不继承"结构改动必扰 token"的负结果结论——但 G1 跨引擎 token 门仍然前置（同精度才有资格比）。
- 已关死的门（不重开）：编译/分桶（#115）、prompt-lookup（#547）、prefix cache、合并 GEMV（#128）、自写 kernel（QUEUE #2）、prefill cache（QUEUE #3）、eval split（QUEUE #4）。
- receipts 路径（预期，实现后回填真实值）：M4 `~/lfm3-dig-run/4bit/quantize_lfm3.py` 与产出目录、`~/lfm3-dig-run/window-4bit/`、各 doc §Results、PR 号（待开）。

---

## 7. Results

### 2026-11-19 23:39 CST — G1 v12 判定：INVALID（§2 G1 / §5；速度窗未开）

**背景（v11 为何作废）**：G1 第一版（v5–v11）两侧各返回 128 × `0`，是 vacuous pass，根因为官方 HF 4-bit 副本在 M4 上 99%+ 字节全零（594 tensor 中 590 个 payload 全零），两臂读的是同一份残缺文件。round-46 判 v11 INVALID 但根因在文件、不在实现。

**本轮动作（仅换权重根，同 harness）**：`MODEL4`（native 侧）与 `LANE_DIR`（lane 侧）同时改指 **`~/lfm3-dig-run/4bit/`**——round-44 在 M4 现场 `mx.quantize(group_size=64, bits=4, mode="affine")` 产物（973,082,900 B，sha256 `96b021b3…`，594/594 tensor payload 非零，lane sanity 已跑出真实文本）。v12 harness 追加 vacuous-pass 守卫（常量序列 / 全零通道 ⇒ 直接 INVALID），旧 official-root harness 备份于 `~/lfm3-dig-run/g1/run-g1.v11-official-root.sh`。

**Effect proof（G3）三件套全部成立**：banner `decode_schedule=… native_4bit=true`（与 `APXINF_LFM3_4BIT=1` 同源）；census `quantized_triples=154/154`；binary freshness（overlay 树 `find crates -newer worker` = 空，`apxinf-serve` sha256 `c8cec3ea…`）。双 Metal 锁 fcntl 顺序等待、等不抢，全程锁内；进程清理干净，锁已释放。

**门结果：G1 硬门 FAIL ⇒ INVALID**（预登记 §2 G1 / §5：token 不一致且无法解释 ⇒ INVALID，不进速度窗）：

- replay 通道，prompt = "What is the capital of France?"，16 prompt tokens，128 decode steps。两侧前 59 个生成 token **逐 token 相同**（非退化，v12 守卫通过）。
- **first mismatch：step 59**（1-based；注意 `/v1/replay` 数组含 prefill argmax 头，runner 日志原始报告 step=60 即 array index 60 = 去头后 step 59）。两侧 context `[41176, 88050, 124900, 124895]` 相同：lane = `125020`，native = `25797`。此后 128 步中 67 处不同；lane 序列 uniq=75，native uniq=55。
- 序列 sha256（array 含头）：lane `d632913f…`，native `26f9af37…`。
- 前 16 个 id 两侧完全一致：`[597, 4695, 20589, 34, 496, 2992, 355, 278, 5205, 302, 3980, 8022, 969, 355, 267, 90139]`。

**机制算术 / speed signature（仅作 receipt，不作判定依据）**：原生臂 request log `decode_ms_per_token=11.51`（prefill 52.1 ms，128 tok），lane 自报 `step_ms=11.98`——与 0.973 GB ÷ 100.6 GB/s = **9.67 ms/token floor** 量级吻合，说明速度真实；但 G1 在此之前已失败，**不进 ABBA 速度窗**。per-kernel GB/s 与 launch 数：**机制未知（未 census）**。

**机制未知的解释（如实写，不编造）**：两臂读同一份权重文件、同一 quant 格式（lane `packed_mode=True, packed_tensors=153, packed_bytes=972,232,704`），前 59 token 相同 ⇒ 输入/权重语义一致；但通用 `quantized_matmul`（MLX affine qmm / BM64 路径）在两个 ABI 上的**归约顺序/tiling**未能逐 token 对齐（`apxinf-mlx` SDK 0.32.x vs lane venv 0.31.2，二者 libmlx ABI 不同，step_ms 亦不同：11.51 vs 11.98）。首个分歧落点、group 内归约序的精确定位：**机制未知，未测量**。按冻结规则，无法解释的跨引擎 token 差异即 INVALID，不强行归因。

**Disposition**：QUEUE #8 G1 硬门失败 ⇒ **本方向 INVALID，不进速度窗（ABBA）、不进质量评测、不开 PR**。原生 4-bit 量化 Linear 实现（commit `0bdbb05e`）保留在 `agent/lfm3-dig` 分支作 receipt，默认 `APXINF_LFM3_4BIT` unset 时路径零改动；是否清理剥离不属本轮。下一步按 RUNNER §1-5 / §7 记录后进 **QUEUE #7（自选：不触碰生成路径数值的方向）**。

---

### 2026-11-20 01:26 CST — G4-speed 判定：PASS（§2 G4-speed / §5；Lead 01:41 重算确认）

> 注：本节结果在 Lead 2026-11-20 01:41 指示「把上表写进 doc」后补记，原始窗口运行于 01:19:05–01:24:31（round-55，预登记修订 commit `f10cae5f` 之后）。分析块当时因 typo（`floaint`，见 round-58 STATUS）SyntaxError 挂掉，以下数字为 Lead 从 raw 重算，已复核。

**窗口**：ABBA `A B B A A B`，6 段 3/臂，cells `s2-c1024-g128` × 2 repeats + `s2-c4096-g128` × 2 repeats，engine basis（`--engine-timings`；`decode_ms_per_token` 由引擎自报，客户端计时因 #120 SSE 全缓冲不作主度量）。raw：`~/lfm3-dig-run/window-4bit/raw/*.json`（6 文件）。

**原始数字（Lead 重算，每 case 3 段中位）**：

| case | native-4bit | native-bf16 | 加速 | A/A（4bit / bf16） |
|---|---:|---:| ---:| ---:|
| s2-c1024-r0 | 11.13 ms/tok | 36.59 | 3.29× | 3.5% / 0.3% |
| s2-c1024-r1 | 11.06 | 36.59 | 3.31× | 1.7% / 0.5% |
| s2-c4096-r0 | 11.39 | 37.13 | 3.26× | 0.8% / 1.1% |
| s2-c4096-r1 | 11.44 | 37.26 | 3.26× | 1.7% / 2.4% |

**G3 effect proof**：armed 臂 banner `decode_schedule=… native_4bit=true` + census `quantized_triples=154/154`；BF16 臂 banner `native_4bit=false`（census 空属预期）；无空臂。

**判据（§2 G4-speed，冻结）**：4-bit vs BF16 **≥2.5×** 且 **≤ llama Q4（12.39/12.71 ms/token 认证基线）**；A/A ≤5%（§2 G5）。

**结论：G4-speed PASS**：3.26–3.31× ≥ 2.5×；11.06–11.44 ≤ 12.39/12.71，4bit/llama = 0.89–0.92（快 8–11%）；A/A 全部 ≤3.5% ≤ 5%；G3 空臂检查 PASS。

**QUEUE #8 完整五档**：G4-speed 通过，但 §5 判据要求两门都过才判 WIN；G1′ 结果见本文件 §7 后续节（n=6 ⇒ INCONCLUSIVE；n=20 扩测 ⇒ **PARTIAL 候选 −5.00pp（both）/ 0.00pp（either）** ⇒ 4-bit 保持 opt-in、暂不设默认）。

---

### 2026-11-20 01:50 CST — G1′ 质量门判定：**INCONCLUSIVE（n=6）**（§2 G1′ / §5；round-57 实测；**改判 2026-11-20 02:18 CST Lead 裁决**）

> 本节结果在 round-57 窗口（01:49:05–02:02:53，commit `7ab6ad5d` 之后）后回填。原 M4 运行 harness commit `b4ec75cb` + 修正 `7ab6ad5d`；receipts：`~/lfm3-dig-run/window-4bit/g1prime/`（verdict：`g1prime-verdict.json`，6 题 per-task 表、各臂 2 段 json、第 1 次作废运行 `round57-failed1/`）。

**评测配置**：suite = MBPP sanitized rows 0..4 + HumanEval/80（`build-g1prime-suite.py` 离线生成并冻结 sha256），ABBA 2 段/臂，A=`native-4bit`（`APXINF_LFM3_4BIT=1`，模型 `~/lfm3-dig-run/4bit/` sha `96b021b3…`）、B=`native-bf16`（开关 unset，模型 `~/lfm3-spec/weights-local/LFM3-1.7B/` sha `135922ca…`）；每臂 2 gateway × 6 题 × ≤3 retry，PV-11 code_loop 模板（gold code 不进 prompt）。窗口前双 Metal 锁 fcntl 顺序等待 + `pmset displaysleepnow` + `caffeinate -si`；锁在 02:02:53 释放，全程锁内。

**原始数字（per-segment solved count）**：

| 臂 | seg | solved | solved ids | 备注 |
| ---|---| ---|---| ---|
| A（4-bit） | 0 | 2/6（33.3%） | `mbpp_0012`, `mbpp_0017` | |
| A（4-bit） | 3 | 1/6（16.7%） | `mbpp_0017` | 与 seg0 不一致 ⇒ spread 16.67pp |
| B（BF16） | 1 | 2/6（33.3%） | `mbpp_0012`, `mbpp_0017` | |
| B（BF16） | 2 | 3/6（50.0%） | `humaneval_0080`, `mbpp_0012`, `mbpp_0017` | 与 seg1 不一致 ⇒ spread 16.67pp |

**汇总（两口径，`g1prime-verdict.json`）**：

- both 口径（每任务 2 段都解出才算解出）：4-bit **1/6（16.7%）** vs BF16 **2/6（33.3%）**，**delta = −16.67pp**；
- either 口径（任一段解出即算解出）：4-bit **2/6（33.3%）** vs BF16 **3/6（50.0%）**，**delta = −16.67pp**；
- per-task disagreements：1/6（`mbpp_0012`：4-bit 1/2 vs BF16 2/2，是唯一一题跨口径分歧；其余 5 题两口径结果相同）。

**判据（§2 G1′，冻结）**：掉 ≤3pp ⇒ PASS；3–10pp ⇒ PARTIAL 候选；**>10pp ⇒ STOP（4-bit 不可设默认）**。

**A/A caveat（如实记录，round-57 §59）**：两臂 2 段 solved 集合各自均不一致（spread 16.67pp > 5%）；按 RUNNER §4 严格口径本窗 A/A 不干净，但 G1′ 的 A/A 门在预登记中定义为「每臂 solved 集合一致」，analyzer 报告 DISAGREE 并以逐任务表为 tiebreaker，**不因 DISAGREE 自动判 INVALID**（INVALID 仅限 G3 空臂 / G5 速度窗 A/A >5% / G4-shape 漂移）。n=6、≤3 retry 下解谜率本身抖动极大（同臂同精度两段就差 1 题：A 臂 seg0 2/6 vs seg3 1/6、B 臂 seg1 2/6 vs seg2 3/6，各自 spread 16.67pp），与「两臂差值 16.67pp」**完全同量级**。

**改判（2026-11-20 02:18 CST，Lead 裁决，仅改结论标签、不改门）**：上述 A/A caveat 表明**同一臂同一精度两段之间的噪声与「两臂之差」同量级**，且 n=6、≤3 retry、480-token 上限可能截断解法 ⇒ 原判 STOP（质量不可接受已确立）**统计上站不住**，效应在噪声内。按裁决改判 **INCONCLUSIVE（样本不足）**；STOP 仅在扩到 ≥20 题且两臂 A/A 干净后若 solved 差仍 >10pp 时成立。**门（§2 G1′ 判据）保持冻结不变**：≤3pp ⇒ PASS；3–10pp ⇒ PARTIAL；>10pp ⇒ STOP。

**结论：G1′ INCONCLUSIVE（n=6）—— 质量门未测出可靠差异**：两口径 −16.67pp 是噪声内读数，不构成质量已劣化的证明；**4-bit 继续暂不设为原生默认（与 round-60 处置一致），保持 opt-in（`APXINF_LFM3_4BIT=1`）**，等 ≥20 题重测结论。速度窗的 3.26–3.31× 优势（round-58 本文件上节）已成立、不受影响（G4-speed PASS）。G1-diag（step-59 token 分叉 + FP16 scale 复现）本轮未跑，仍为诊断项。

**下一动作（Lead 02:18 裁决）**：扩到 **≥20 题**（从冻结的 MBPP/HumanEval 池选，写死 suite sha256），每臂 **≥3 段**，同一协议；20 题下 solved 差仍 >10pp 且 A/A 干净 ⇒ 真 STOP；≤3pp ⇒ PASS；3–10pp ⇒ PARTIAL（4-bit opt-in）。PR #132 保持 open，标题/正文同步为「G1′ INCONCLUSIVE (n=6; expanding to ≥20)」。该扩测已于 round-65（2026-11-20 02:47–03:54 窗）完成，结果见下节。

---

### 2026-11-20 03:55 CST — G1′ 质量门判定：**PARTIAL（n=20，候选）**（§2 G1′ / §5；round-65 实测，Lead 02:18 裁决扩测结果）

> 本节结果在 round-65 窗口（02:47:02–03:54:38，预登记修订 `f10cae5f` + suite v2 `82ebc66c` + window harness `8bf1db30` 之后）后回填。receipts：`~/lfm3-dig-run/window-4bit/g1prime/`（verdict：`g1prime-verdict.json`、6 段 result json、6 份 ledger）。

**评测配置**：suite v2 = MBPP sanitized rows 0..19 + HumanEval/80 = **20 题**，**任务集 sha256 = `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`（与 round-63 冻结值逐字一致，payload 冻结未破）**；ABBA 交错 3 段/臂（A-0/B-0/A-1/B-1/A-2/B-2），A=`native-4bit`（`APXINF_LFM3_4BIT=1`，模型 sha `96b021b3…`）、B=`native-bf16`（开关 unset，模型 sha `135922ca…`）；每臂 3 gateway × 20 题 × ≤3 retry；G3 banner/census/freshness 三件套全过（armed 臂 `native_4bit=true batch=1` + census `154/154`，BF16 臂 `native_4bit=false`，无空臂）；窗口前双 Metal 锁 fcntl 顺序等待 + `pmset displaysleepnow` + `caffeinate -si`；锁在 03:54:38 释放。

**原始数字（per-segment solved count）**：

| 臂 | seg | solved | solved ids |
| ---|---| ---|---|
| A（4-bit） | 0 | 4/20（20.0%） | `mbpp_0017`, `mbpp_0062`, `mbpp_0064`, `mbpp_0066` |
| A（4-bit） | 1 | 3/20（15.0%） | `mbpp_0017`, `mbpp_0062`, `mbpp_0066` |
| A（4-bit） | 2 | 5/20（25.0%） | `humaneval_0080`, `mbpp_0012`, `mbpp_0017`, `mbpp_0062`, `mbpp_0066` |
| B（BF16） | 0 | 6/20（30.0%） | `humaneval_0080`, `mbpp_0012`, `mbpp_0017`, `mbpp_0062`, `mbpp_0064`, `mbpp_0066` |
| B（BF16） | 1 | 5/20（25.0%） | `humaneval_0080`, `mbpp_0012`, `mbpp_0017`, `mbpp_0064`, `mbpp_0066` |
| B（BF16） | 2 | 5/20（25.0%） | `mbpp_0012`, `mbpp_0017`, `mbpp_0062`, `mbpp_0064`, `mbpp_0066` |

**汇总（两口径，`g1prime-verdict.json`）**：

- both 口径（每任务 3 段都解出才算解出）：4-bit **3/20（15.0%）** vs BF16 **4/20（20.0%）**，**delta = −5.00pp**；
- either 口径（任一段解出即算解出）：4-bit **6/20（30.0%）** vs BF16 **6/20（30.0%）**，**delta = 0.00pp**；
- per-task disagreements：3/20 —— `humaneval_0080`（4bit 1/3 vs bf16 2/3，倾向 BF16）、`mbpp_0012`（1/3 vs 3/3，倾向 BF16）、`mbpp_0062`（3/3 vs 2/3，倾向 4-bit）；`mbpp_0017`、`mbpp_0066` 双方 3/3 一致全解，14 题双方 0/3 一致失败（有效分辨题仅 6/20）。

**判据（§2 G1′，冻结）**：掉 ≤3pp ⇒ PASS；3–10pp ⇒ PARTIAL 候选；**>10pp ⇒ STOP（4-bit 不可设默认）**。

**A/A caveat（如实记录，round-65 §103）**：两臂 3 段 spread 分别 10.00pp（4-bit）/ 5.00pp（BF16），**均 >5%（RUNNER §4）** ⇒ 按本项预登记口径 analyzer 报 **DISAGREE**、以逐任务表为 tiebreaker、**不自动 INVALID**（INVALID 仅限 G3 空臂 / G5 速度窗 A/A >5% / G4-shape 漂移）。较 n=6 窗（两臂各 16.67pp）spread 已减半但仍不「干净」；**结论强度：PARTIAL 候选，非精确 −5.00pp 点估计**；n=20、≤3 retry、480-token 上限下同臂 3 段仍差 1–2 题，要更稳需 ≥40 题或更多重复段，属 Lead 裁决。

**结论：G1′ PARTIAL（n=20，候选）—— 不再支持 STOP**：both 口径 −5.00pp 落在 3–10pp 区间（PARTIAL），either 口径 0.00pp（PASS 级无差异）；扩测使 delta 从未扩 suite 时的 INCONCLUSIVE / 原 STOP −16.67pp 收敛到 −5.00pp / 0.00pp。按 §5 **PARTIAL ⇒ 默认 BF16，4-bit 保持 opt-in（`APXINF_LFM3_4BIT=1`），不设原生默认**，留 Lead 裁决是否接受 −5pp（both）质量代价换 3.26–3.31× 速度。G4-speed PASS（上上节）与 G1-diag 诊断项状态均不受影响。

**机制 receipt（engine basis，round-65 §104，非本门判定字段）**：6 段 ledger `decode_ms_per_token` median —— 4-bit 10.935/10.949/11.010 ms/tok、BF16 36.366/36.438/36.211 ms/tok ⇒ 质量窗内速度优势复现 ~3.3×，与 G4-speed 一致。

---

## Gate（原文，供 PR 引用）

> **G1-diag（诊断，非门）** 记录原生 4-bit vs Python lane 4-bit 首处分叉（已知 v12 prompt 0 step 59，lane=125020 vs native=25797）并尝试 FP16 scale 路径复现 lane（`q4_f16.rs`）；修好则升级回 G1 硬门，修不好记为已知引擎差，不判 INVALID。**G1′（质量门）** 同批 agent 任务（≥6 题）4-bit vs BF16 solved rate 掉 ≤3 点通过、>10 点停。**G4-speed（速度门）** ABBA ≥3 段/臂、A/A ≤5%、engine basis，4-bit vs BF16 ≥2.5× 且 ≤ llama Q4（12.39/12.71 ms/token）。**G3** banner `native_4bit` 字段缺失或二进制未重编译 ⇒ 空臂 INVALID。**G4-shape** 任何 shape 漂移（padded/bucketed/wider/拆 K/换读向）⇒ 实现不可用。**G5** A/A ≤ 5%（本 ticket 自有）。

## Decision（原文，供 PR 引用）

> 权重：M4 从 BF16 现场 `mx.quantize(group_size=64, bits=4, mode="affine")`（不从 M1 拷，不在 M1 编 Rust MLX）；格式：packed uint32、group_size 64、scales/biases BF16、lm_head=tied 量化 embedding、norm/depthwise conv BF16。实现：`metal_qwen_image/weights.rs` + `apxinf-mlx` quantized 模式照抄，缺失回退 BF16。窗口：ABBA，A/A ≤5%，engine basis，目标 ≥2.5× vs BF16 且 ≤ llama Q4 12.39/12.71 ms/token。默认规则：G1/G3/G5 硬门通过才判五档；WIN ⇒ 4-bit 默认，PARTIAL ⇒ opt-in；跨精度 4-bit vs BF16 不设逐位门（用户已接受）。
