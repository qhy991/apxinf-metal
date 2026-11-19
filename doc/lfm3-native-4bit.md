# QUEUE #8 — 原生路径 4-bit 量化支持（Gate + Decision）

预登记日期：2026-11-19 22:07 CST（`date` 实取）。本文件为 pre-registration：Gate 与 Decision 在任何设备运行前冻结；实现、测试、速度窗、质量评估分属后续轮次。

- 相关队列：QUEUE.md v2 第 8 项（Lead 决策 2026-11-19 22:01「4-bit 可以，最高价值杠杆」；Lead 修订 2026-11-19 22:05「M4 现场量化，不从 M1 拷」）。
- 权威代码基座：`agent/lfm3-dig`，tip（本 commit 之前）`9f12e4d9`（QUEUE #6 预登记），base `feat/lfm3-metal`。

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
|---|----|------|----------|
| G1 | **同精度跨引擎 token 硬门** | 原生 4-bit（native engine）与 **Python lane 4-bit**（M1 上 python+mlx engine，同一份 4-bit 权重文件语义、同 prompt、同采样：贪心）在 4 prompts × 128 token 上**逐 token 相同**；逐 prompt 断言，记录第一处差异位置（prompt id / step index）与两侧 token id。无法解释差异（tiling / group 顺序 / nibble 序 / affine 符号）⇒ INVALID。 | INVALID，直接按 RUNNER §6 记录结论，不进速度窗。 |
| G2 | 跨精度不设逐位门 | 4-bit vs BF16 贪心输出**必然**不同，用户已明确接受；本项不因 4-bit ≠ BF16 token 判 INVALID。质量以 solved rate 记录（见 §5）。 | 无（不是门）。 |
| G3 | 空臂 / effect proof | worker banner `native_4bit=<true\|false>` 字段与 `APXINF_LFM3_4BIT=1` 同源；速度窗内 banner 必须含该字段且 armed 臂 `=true`。banner 缺失 / 二进制早于源码 mtime（未重新构建）⇒ 空臂陷阱，独立 INVALID（#4 教训）。 | INVALID（独立于 token）。 |
| G4 | shape 不漂移 | 4-bit 臂的 forward 输出 shape / logits shape / KV 与 conv state 形状与 BF16 臂一致；不做 padded/bucketed/wider matmul，不拆 K，不改读向（#128 教训）。实现若被迫引入任一 shape 漂移 ⇒ 回退该实现，记录机制。 | 视情形 INVALID 或 FALSIFIED（token 门若已过则 FALSIFIED 机制记录）。 |
| G5 | A/A ≤ 5% | 速度窗每臂每 cell 分段中位最坏 spread ≤ 5%；S1 短生成 cell（#121 A/A 5.02%）**不可引用**，须本 ticket 自有 A/A。超 ⇒ 窗口作废重跑。 | 窗口作废（重跑）；不取中位凑数。 |
| G6 | 质量回落阈值 | 4-bit vs BF16 在同一批 agent 任务（MBPP/HumanEval 套）solved rate 差 > 10 个百分点 ⇒ 默认 BF16，4-bit 仅 opt-in（PARTIAL）。 | PARTIAL（速度达标也只 opt-in）。 |

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

### 3.5 质量（速度窗过后）

- 同一批 agent 任务（MBPP/HumanEval 套）比 4-bit vs BF16 solved rate，记录差值与绝对值；>10 点 ⇒ G6 PARTIAL，4-bit opt-in。

## 4. Cells / Order

- 速度窗 cells：S2 ctx 1024 / 4096 各 128 tok（同 #127/#4 口径），arms `native-4bit (A) / native-bf16 (B)`，`A B B A A B`；llama Q4 用既有 receipts，不重跑（若同窗重跑则单列）。
- 质量：MBPP / HumanEval 套，batch 与 prompt 规格与 #127 已有质量评测一致（引用已有 receipts 口径，不新发明）。
- S1 decode-at-8 / #121 短生成 cell 不可用。

## 5. 判据（冻结；五档 + INVALID）

- **INVALID**：G1 token 不一致且无法解释 / G3 空臂 / G5 A/A > 5% 作废（重跑后仍超）。
- **WIN**：G1 过 + A/A 净 + 速度 ≥2.5× vs BF16 且 ≤ llama Q4 ⇒ **4-bit 设为原生默认**（保留开关可回退 BF16；G6 质量 ≤10 点）。
- **PARTIAL**：速度达标（≥2.5× 且 ≤ llama Q4）但 G6 质量差 >10 点 ⇒ 默认 BF16，4-bit opt-in。
- **NEGATIVE**：速度 <1.5× ⇒ 机制未知，记录（字节账若仍对平则 roofline 归因问题）。
- **FALSIFIED**：机制预测（0.973 GB ÷ 实测带宽）与实测速度差 >15% 且 token/A/A 干净 ⇒ 机制被否，记录算术。
- **INCONCLUSIVE**：A/A 净 + token 净，但速度落在 <1.5× 与 ≥2.5× 之外（1.5–2.5×）且机制算术算不平，写"机制未知"。

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

## Gate（原文，供 PR 引用）

> **G1** 原生 4-bit 与 Python lane 4-bit（同 prompt、同权重文件语义、贪心）在 4 prompts × 128 token 上逐 token 相同；任一 prompt 首处差异即 INVALID（可解释的 tiling/group/nibble/affine 符号差异记为机制，无法解释 ⇒ INVALID）。**G3** banner `native_4bit` 字段缺失或二进制未重编译 ⇒ 空臂 INVALID。**G4** 任何 shape 漂移（padded/bucketed/wider/拆 K/换读向）⇒ 实现不可用。**G5** A/A ≤ 5%（本 ticket 自有）。**G6** solved rate 差 >10 点 ⇒ PARTIAL（默认 BF16，4-bit opt-in）。

## Decision（原文，供 PR 引用）

> 权重：M4 从 BF16 现场 `mx.quantize(group_size=64, bits=4, mode="affine")`（不从 M1 拷，不在 M1 编 Rust MLX）；格式：packed uint32、group_size 64、scales/biases BF16、lm_head=tied 量化 embedding、norm/depthwise conv BF16。实现：`metal_qwen_image/weights.rs` + `apxinf-mlx` quantized 模式照抄，缺失回退 BF16。窗口：ABBA，A/A ≤5%，engine basis，目标 ≥2.5× vs BF16 且 ≤ llama Q4 12.39/12.71 ms/token。默认规则：G1/G3/G5 硬门通过才判五档；WIN ⇒ 4-bit 默认，PARTIAL ⇒ opt-in；跨精度 4-bit vs BF16 不设逐位门（用户已接受）。
