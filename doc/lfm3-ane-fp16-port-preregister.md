# LFM3 的 MLP 走 FP16 ANE 列切分：移植预注册（零设备；端到端 TTFT 是唯一判定依据）

预注册时间：**2026-11-21 09:32 CST**。本 commit 先于任何 `metal_lfm3` 运行时改动与任何设备运行；
**§4 Gate、§5 判据/判定、§6 标定协议在首次设备运行后不许改**（结果只填 §10）。

**来源**：issue #153（owner 派单，@enginetailor 执行，@Open-Cake 同伴）。
**追加要求（owner，2026-11-21 01:27 UTC）**：这一棒的预注册必须包含端到端验证，**端到端是判定的唯一依据** ——
不是 ρ、不是 R_max；必须同时报两臂贪心输出是否逐字相同与零量化对照，并量出墙钟 `f` 与 R_max 预测对账。
本 doc 已并入该要求（§2 末、§5.2–§5.3、§7）。
**本棒**：零设备、零运行时改动 —— 交付物只有本文档，以及 `doc/lfm3-ane-mlp-preregister.md`
头部的一条日期更正与 #145 的一条更正评论。
**移植目标**：`feat/lfm3-metal` 分支上的 `metal_lfm3`（BF16-only 家族）。本文档先在 `main` 落地；
移植分支合入 `main` 后才允许动代码（§9）。
**依赖**：**无**。4-bit 不是 FP16 ANE 的前提（§0 的更正）。

---

## 0. 依赖更正（这条要写进 #145 的记录）

#145 原文：

> **Blocked by #144**（packed 4-bit）：clef 的 ANE 权重是**从 Q4 checkpoint 在 host 上量化**的，
> 所以 LFM3 的 ANE 路径也以 packed 4-bit 为前提。

**更正**：那条依赖描述的是 **clef 的 INT8 配方如何取权重**（它为了省显存从 Q4 反量化），
不是「ANE 需要 packed 4-bit」。**FP16 配方直接从 BF16 checkpoint 取权重**：host 侧
`f16(w)` 成 FP16 常量（#151 的 FP16 臂就是这么建的，见
`crates/apxinf-coreml/examples/lfm3_mlp_parity_probe.rs` 的 `fp16_arm`），实测 **ρ = 1.89 / 1.85**
（T=512 / 2048）、**relL2 中位 0.36%**，全程没有 packed 4-bit 张量。
另外 #144 本身以 **measured NEGATIVE 关闭**（未合入），它不可能成为任何东西的合入前提。

⇒ **FP16 ANE 移植的依赖是「无」**。INT8 配方若要从 BF16 直接 per-row 量化同样不需要 #144
（#151 的 INT8 臂就是从 BF16 量的）；4-bit 只在「想省显存/带宽」时才是话题，不是本方向的前提。

更正落点：本节 + `doc/lfm3-ane-mlp-preregister.md` 头部的日期更正行 + #145 的 issue 评论。

---

## 1. 事实基座（引用，非本轮测量）

| 事实 | 出处 |
|---|---|
| LFM3-1.7B：hidden 2048、intermediate 7680、24 层（13 conv / 6 full / 5 sliding-256）、16Q/4KV、head_dim 128、BF16、tied head；上下文上限 8192 | `crates/apxinf-model/src/metal_lfm3/README.md`（`feat/lfm3-metal`） |
| `metal_lfm3` 是 BF16-only、eager MLX 组合：**无 ANE、零 CoreML 引用、无 Q4 路径**；MLP 为 `silu(w1)·w3 → w2` | 同上；`model.rs::mix_and_residual` |
| `metal_clef/ane.rs` 已验证 GPU+ANE 列切分：每层一个 CoreML 程序，gate/up 1×1 conv、`0.5v(1+tanh(v/2))` SiLU、gate/up 行归一、down K-split conv、512-token tile；GPU 先算自己的列，ANE 预测，FP32 还原后相加 | `crates/apxinf-model/src/metal_clef/ane.rs`；`model.rs::mlp` |
| #151（M4 Mac mini，真实 BF16 权重 + 真实激活）FP16 ANE 整层：relL2 中位 **0.36%**、vs BF16 自身噪声底 **1.1–2.1×**、ρ(T=512/2048) = **1.89 / 1.85**、R_max ≈ **47%**（f=0.73） | `doc/lfm3-ane-int8-parity.md` §6.3/§6.8 |
| 同链 INT8：relL2 中位 **1.04%**、vs 噪声底 **2.6–3.8×**、ρ = **2.42 / 2.34**、R_max ≈ **51%** | 同上 |
| BF16 自身噪声底（同 BF16 权重，BF16 vs F32）：0.249 / 0.273 / 0.315 / 0.315% | 同上 §6.4 |
| 真实权重下 FP16 的全链 token 表现：A3 与基线只在 P2@12（BF16 并列）不同；teacher-forced top-1 **99.602%**，与基线自身天花板 99.602% 相同 | 同上 §6.5/§6.7 |
| 本 lane 的注册口径：`metal_lfm3` teacher-forced top-1 = **720/731 = 98.50%**，栏 98%，轨迹是 lane 的冻结 float32 oracle；lane 自己的 bf16 引擎 98.76% | `metal_lfm3/README.md`（Verification） |
| 缺口：#130 第 3 项，短 prompt prefill（128/512 token）比 llama.cpp 慢 **8–12%**，2048 基本持平 | #130 / #145 |
| 列切分的跨层流水不可能：下一层 RMSNorm 依赖本层 MLP 完成；只能同层 GPU∥ANE | #145 的事实基座 |
| 端到端计时口径：`scripts/bench_engines.sh --engine-timings`（engine basis，`ttft_basis=engine`），apxinf 的 `prefill_ms` = state reset + capacity preparation + prefill；客户端墙钟保留在 `*_observed` 但不决定主指标 | #112/#141 的 LFM3 engine-comparison harness |

---

## 2. 为什么选 FP16（owner 表 + 机制核算）

| | INT8 | **FP16** |
|---|---:|---:|
| relL2 vs BF16 | 1.04% | **0.36%** |
| vs BF16 噪声底 | 2.6–3.8× | **1.1–2.1×** |
| ρ（T=512 / 2048） | 2.42 / 2.34 | **1.89 / 1.85** |
| 回收上界（f=0.73） | 51.6% / 51.2% | **47.7% / 47.4%** |

- **「FP16 只损失 8% 的吞吐」要说准**：ANE 的原始速率是 −22%（2.42 → 1.89），但回收上界只从
  51.6% 掉到 47.7%（−7.6% 相对），因为 `R_max = f·ρ/(1+ρ)` 在 ρ≈2 处饱和。
- **精度**：0.36% 只有 BF16 自身算术底（0.29%）的 1.2×；INT8 的 1.04% 是 3.6×。
- **BF16→FP16 的权重 cast 在 f16 正规范围内是精确的**（BF16 的 7 位尾数是 FP16 10 位尾数的子集），
  所以 0.36% 主要来自 **ANE 引擎的 FP16 算术 + clef 的 INT8 时代归一化链**，不是「权重量化」。
  载入时必须检查权重范围（`|w| ∈ {0} ∪ [2^-14, 65504)`，越界 ⇒ 拒绝装载，不静默），并把结果记入证据。
- **对 98% 门的意义**：基线 720/731 只比 98%（717/731）高 **3 个位置**，精度余量很薄；0.36% 比 1.04%
  更有把握，但不是保证（#151 已证明 BF16 并列会翻 token）。这正是 §5.1 必须带零量化对照的原因。
- **ρ 与 R_max 只是「为什么可能」，不是「确实有效」**：它们来自局部探针 + 算术外推（f=0.73 还是 FLOP 口径）。
  判定只看 §5.2 的端到端 TTFT；ρ 只用来在 §5.3 解释偏差。**ρ 再漂亮，端到端不改善就是 FALSIFIED。**

---

## 3. 移植范围（写死边界）

### 3.1 改什么（只允许这些）

在 `feat/lfm3-metal` 的 `crates/apxinf-model/src/metal_lfm3/` 内，学 `metal_clef/ane.rs` 的结构：

1. **新增 `ane.rs`**：FP16 CoreML 程序构建 + 装载 + placement 准入 + tile 打包 + `predict` + FP32 还原。
   几何 `HIDDEN=2048`、`INTERMEDIATE=7680`、`TILE=512`、`DOWN_PART=4096`；配方见 §3.4。
2. **`weights.rs`**：从完整 BF16 `w1/w3/w2` 构建每层程序，然后只保留 GPU 份额的权重
   （`w1/w3` 的列 `[0, 7680−S)`、`w2` 的对应行；与 clef 的 `gpu_share` 同构）。
3. **`model.rs`**：`Lfm3Model` 增加 `#[cfg(feature = "coreml")] ane: Vec<AneMlp>`；
   `mix_and_residual` 的 MLP 段增加列切分分支：**GPU 份额先入队 → ANE 逐层 `predict`（host 阻塞，
   与 GPU 份额重叠）→ FP32 还原（`y × gate_scale × up_scale`）→ 与 GPU 份额相加 →
   `post_ffn_norm` → 残差**。跨层不重叠（§1）。
4. **`mod.rs`**：`#[cfg(feature = "coreml")] mod ane;`；装载时解析开关并**拒绝非法值**；
   把生效份额/程序数写进 worker banner/状态（供 G3 见证与复现）。
5. **`Cargo.toml`**（`apxinf-model`）：`coreml` feature 下启用（已有 `apxinf-coreml` 依赖）；
   `cargo check -p apxinf-model` 不带 `coreml` 必须照常编译（cfg 隔离）。
6. **`README.md`**：写清开关、内存、decode 不动、测量口径（端到端 engine basis）。
7. **测试**：开关解析/拒绝、份额到列数的取整、tile 补齐、GPU 份额切片几何的单元测试；
   程序同一性用 §5.1 的 relL2 检查（可复用/扩展 `lfm3_mlp_parity_probe` 的 FP16 臂，不改库行为）。
8. **只用于测量的仪器**（默认 off，不进判定臂）：
   - `APXINF_LFM3_MLP_ABLATE=1`：prefill 时把 MLP 段短路成同形状零张量（其余路径不变），
     用 A′ 的 prefill 时间差量出墙钟 `f`（§5.3）。**只用于计时窗口**，不得出现在 B0/B1/B2 的
     正确性/性能臂里；它改变数值，必须显式标注。

### 3.2 不碰什么（明确清单）

- **attention** 全系：full/sliding 的 SDPA（含 windowed 分解）、per-head `sigmoid` 门、两套 partial-RoPE、
  所有投影 —— 不改。
- **conv** 全系：prefill 的 `conv_layer` 与 decode 的融合短卷积 `conv_step` —— 不改（decode 融合明确不动）。
- **KV/conv 状态**：布局、容量、`prepare`/`reset`/`release_request_state` —— 不改。
- **decode（`len == 1`）整条路径**：含 decode 的 MLP —— 继续走 GPU，不接 ANE。
- **sampling/generation、tokenizer、lane ChatML prompt 协议、serving 适配层/协议** —— 不改。
- **head/embedding/final norm、数值类型与权重来源** —— 不改（唯一新增的是 host 侧 FP16 常量）。
- **4-bit/Q4**：本家族没有该路径；本棒不新增（也不依赖 #144）。
- **其他家族与共享 kernel**：`metal_clef`、`metal_qwen_image` 的 ANE 代码不改；
  `apxinf-mlx` 的 `ane_tile_*` 共享 kernel **原样复用**（列切分在模型侧用 `add` 合并，不需要改 kernel）。
  如确需改共享 kernel，必须先单独立票，不能混进本棒。
- **判定工具**：冻结语料（q01..q08、731 位置、oracle 轨迹）、gate harness 语义不改。
  harness 只允许在 `devlocal/` 里扩展出 B1/B2 两条臂与 f 窗口。

### 3.3 开关（冻结）

- 名字：`APXINF_LFM3_ANE`。
- **默认 off**：未设置或空串 ⇒ 纯 GPU，路径与今天**逐位一致**（G6），不构建/不装载任何 CoreML 程序。
- 值 = **ANE 列份额**（小数，`(0,1)`，例如 `0.7333`）。生效列数
  `S = round(share × 7680 / 64) × 64`（取整到 64 列，确定性；记录 `S` 与 `S/7680`）。
  `S < 64` 或 `S > 7680−64`、非数字、非有限数 ⇒ **装载报错拒绝**（不静默回落、不换份额）。
- ANE 拿**最后** `S` 列（`[7680−S, 7680)`），GPU 拿前 `7680−S` 列（与 clef 的「GPU 在前」约定一致）。
- 开关只影响 **prefill（`len > 1`）**；decode 与 off 时相同。
- 生效配置写入 worker banner（如 `ane=0.7333/5632/24`），否则测量无法归因（G3）。
- **判定的 B 臂取 `S=5632`**（即 `share=0.7333`，owner 指定：探针最优；端到端最优点可能不同，见 §6）。

### 3.4 ANE 程序配方（冻结：必须复刻 #151 实测的那条链）

每层一个程序，逐算子复刻 `fp16_arm`（`crates/apxinf-coreml/examples/lfm3_mlp_parity_probe.rs`）：

- gate/up：FP16 常量 1×1 conv，输出 `S` 列；
- SiLU：`0.5v(1+tanh(v/2))`（不是 `v·sigmoid(v)`）；
- gate/up 各按行最大值归一（`1/cap`、下限 `1/FACTOR_SCALE`，`FACTOR_SCALE=16`，
  `DOWN_ABS_SUM_LIMIT=8192`，`cap` 按真实 down 权重算），输出 `gate_scale`/`up_scale`；
- 乘积 `a = (silu/scale_g)·(u/scale_u)`，identity 输出 `a`（仅取证用）；
- down：K-split FP16 conv（`ceil(S/4096)` 份、等宽），求和；
- GPU 侧还原 `y × gate_scale × up_scale`（FP32）后与 GPU 份额相加；
- **不允许**在首轮设备运行后改配方（含去掉归一化链）。配方是 0.36% 的载体；改配方等于换实验，须新票。

理由：归一化链对 FP16 不是动态范围必需，但它是 #151 测到 0.36% 的那条链，且为任意输入兜住 FP16 上限；
去掉它可能更准也更快，但会改变已被 gate 的数值配方 —— 列为后续（§7）。

### 3.5 tile、份额与内存（写死）

- **只编译 512-token tile**（clef 的发货 tile，也是 #151 实测 tile）。T=128 的注册 case 用**一个补齐的
  512-tile**：1×1 conv/tanh/行归一/down 都是逐 token 运算，零填充行不影响真实行（数值惰性），
  但 ANE 侧要做 4× 算术 —— 这是 128 case 的真实部署形态，不许换成「理想小 tile」的成绩。
  128 专用 tile 程序会把每层权重再复制一份（+1.5 GB 量级），不在本棒。
- **一个**生效份额（单值），不是按长度两套。
- **全 24 层**，每层一个程序；不做层子集。
- 内存核算（估算，移植棒必须实测上报）：S=5632 时每层 ANE 权重 3×2048×5632×2 ≈ 69 MB，
  24 层 ≈ **1.66 GB**；GPU 份额 BF16 ≈ 0.61 GB；两者之和 ≈ 原始完整 BF16 MLP 的 2.27 GB（近似中性）。
  CoreML 可能另留副本，且首次编译/装载要时间 —— 计入证据（G7），不进 TTFT 判定（warm 后计时）。

---

## 4. Gate（冻结；PR 引用原文）

> **G0 预注册先行**：本 doc 合并进 `main` 的时间必须早于移植分支上任何 `metal_lfm3` 改动
> （以 commit/PR 时间戳为准）。本棒自身零设备、零运行时改动。
> **G1 双锁 + ANE 独占**：整个测量窗口持有 `/tmp/enginetailor-metal.lock` →
> `~/.cache/enginetailor/metal-measure.lock`（flock，等不抢，不杀其他 campaign）；窗口前后各做一次 `ps` 扫描，
> 不得存在其他可能用 ANE 的进程（`apxinf-serve`/`apxinf-native-worker`/clef worker/任何 coreml 探针/python coremltools）。
> **clef 的列切分默认就在用 ANE**（5632/12288 列）——两条 lane 不能同时测；扫描输出存 receipts。
> **G2 placement 取证**：每个程序用 `load_with_placement`，证据必须显示**全部 conv op preferred = NeuralEngine**；
> 任一 op 落 CPU/GPU ⇒ 该臂 **INVALID**（不是 ANE 成绩，不进判定）。
> **G3 见证/空臂**：worker banner/状态必须显示生效份额、程序数（24）、`coreml` feature；声称 ANE 而 banner 显示
> `ane=off`/程序数不符 ⇒ **INVALID**（空臂）。off 臂必须显示无 ANE 程序。
> **G4 A/A 稳定性**：端到端每 case 两窗中位相对差 **≤1.5%**，否则重跑一次，仍 >1.5% ⇒ 该 case **INCONCLUSIVE**；
> 贪心/top-1 臂重复运行必须给出**逐位相同**的 token 序列（BF16 引擎在固定 tie-breaking 下确定），否则 INCONCLUSIVE。
> **G5 不越界**：diff 白名单 = `doc/`（本 doc + #145 doc 的更正行）、`metal_lfm3/`（`ane.rs`/`model.rs`/
> `weights.rs`/`mod.rs`/`README.md`/测试）、`apxinf-model/Cargo.toml` 的 coreml feature、§3.1 的测量开关、`devlocal/` receipts。
> 其他家族、共享 kernel、serving 代码、gate 语料/harness 语义越界 ⇒ **INVALID**。
> **G6 off = no-op**：`APXINF_LFM3_ANE` 未设置时，移植分支必须与 B0 **逐位相同**（3 条 prefill prompt 的 logits
> 与 731 位置 top-1），TTFT 在 A/A 内，且不构建/装载 CoreML 程序。
> **G7 证据齐全**：engine-basis TTFT 原始日志（`ttft_basis=engine`）、份额扫描表、贪心/top-1 对比 JSON
> （含每个分歧位置的 top1−top2 margin）、placement、ps 扫描、A/A、权重范围检查、墙钟 `f` 与 R_max 对账表、
> 内存/装载时间全部落 `devlocal/lfm3-ane-fp16-port/`，缺项 ⇒ INCONCLUSIVE。
> **G8 端到端口径**：TTFT 只能来自 engine basis（`--engine-timings`，`ttft_basis=engine`）；客户端墙钟
> 只作 `*_observed` 记录，不进判定。**ρ 不是判定输入**：判定表里不得用 ρ/R_max 代替 TTFT；
> §5.3 的 f 对账只解释机制。违反 ⇒ **INVALID**。

---

## 5. 判据（先写死；跑完不许改）

### 5.1 正确性（token）：两臂逐字一致 + 零量化对照

语料/协议 = 本 lane 的注册口径：冻结真实问题集 **q01..q08**，lane ChatML + BOS，**temperature=0**（贪心），
最多 64 个新 token（EOS 停）；两臂的 **token id 序列逐位比较**（报首个分歧位置与分歧率）。
另跑 teacher-forced top-1：同语料 **731 个位置**，对照 lane 的冻结 float32 oracle 轨迹。不得改语料/oracle。

| 臂 | 构造 | 作用 |
|---|---|---|
| **A** 基线 | 移植分支、开关 off（= 当前 GPU-only MLP） | 贪心/top-1 的对照；必须复现 **720/731**，否则 INCONCLUSIVE（主机/协议漂移） |
| **B** 移植臂 | 开关 on，**S=5632**（§3.3） | 判定臂 |
| **B2** 零量化对照 | **同一条切分路径**（同份额、同 512-tile、同 gate/up 归一与 cap、同 FP32 还原/相加顺序），但 MLP 份额用**原始 BF16 权重**在 GPU 上以 F32 计算，**不做任何 f16 cast** | 隔离「路径算术」（#151 的 A4 教训） |
| **B1′** 重试臂 | 仅当 §5.4 规则③触发：份额降一档（§6 阶梯） | 一次，不再多 |
| **B1″** 交付臂 | 仅当 §6 的端到端扫描选出的推荐开启值 ≠ 5632 时：在推荐 rung 上重跑本节 | 保证交付的配置也被 gate 过 |

**冻结判据**：

1. `B` 的 teacher-forced top-1 `≥ 717/731`（98.00%）；`B2 ≥ 717/731`；`A = 720/731`。
2. **两臂贪心输出逐字相同**（8/8 prompt 的全部 token id 相同）= token 门通过。
   若不同：**用 B2 归因**（同一位置 B2 也不同 ⇒ 路径/baseline，不是 ANE；B2 相同 ⇒ FP16 不够）。
   每个分歧位置必须报 `top1−top2` margin 与是否精确并列。
3. **程序同一性**：移植臂的 ANE 份额 vs 同份额 GPU BF16 参考，在 #151 的 4 层真实激活上
   relL2 中位 **≤0.5%**（实测 0.36% + 余量）；超限 ⇒ **INVALID**（程序不是被 gate 的那条链）。
4. #151 已证明「token 相等」在 bf16 下会被并列触发（P2@12 的 margin=0）；本判据**不因此放宽**
   （门是 owner 定的逐字相同），但**归因必须走 B2**：把并列翻 token 写成「FP16 精度结论」是错的。

### 5.2 性能（注册 case：128 / 512 token 的 TTFT；唯一判定依据）

- **被测对象**：注册的 prefill case —— **128 与 512 token**（#130 第 3 项：这两个上 apxinf 比 llama.cpp 慢 8–12%）。
- **输入**：#151 用的那条冻结 prompt（`doc/model-layer-architecture.md` 前 512 token，
  token-ids sha256 `4bcda0c7…`，见 #151 receipts）的**前 128** 与**整 512** token；从空状态 prefill。
- **指标**：**engine basis TTFT** = `scripts/bench_engines.sh --protocol openai-chat --engine-timings`
  记录的 `ttft_ms`（`ttft_basis=engine`），即 apxinf 自己的 `prefill_ms`（state reset + capacity preparation +
  prefill）；**不是客户端墙钟**（客户端值只留在 `client_ttft_ms_observed`）。若移植分支还没有该 flag，
  先合入 #141 的 harness 或记录等价的 engine `prefill_ms` 口径，不得用客户端墙钟代替。
- **采样**：每 case 3 次 warmup + **21** 次测量取中位；A/B 两臂**交替成块**运行以抵消热漂移；
  同一 prompt、同 `temperature=0`、`max_tokens=8`。
- **判定阈值**：**128 与 512 两个 case 的中位 TTFT 都必须改善 ≥3%**。
  理由：「显著低于」在没有 sudo/powermetrics 的机器上只能用 A/A 离散度定义 —— 冻结 A/A ≤1.5%，
  3% = 2× 该上界，是能与漂移区分的最小值；低于 3% 视为「无改善」，不是「有一点收益」。
  8–12% 缺口用 8% 作**报告线**（是否把缺口关上），不是另一档判定。
- **2048 只报不判**（一次测量，机制参照）；**禁止**把 128/512 的结果外推成「全 range 改善」。

### 5.3 墙钟 `f` 与 R_max 对账（owner 追加；只解释机制，不改变判定）

之前没有的量：**`f` = MLP 段占 prefill 墙钟时间的比例**（0.73 是 FLOP 口径，不是墙钟）。

- **测法**：同一 engine-basis 窗口内，对同一 case 跑 **A**（完整 GPU-only）与 **A′**（`APXINF_LFM3_MLP_ABLATE=1`，
  MLP 段短路成零，其余不变）各 21 次；`f_wall = (T_A − T_A′)/T_A`。A′ 只用于计时，数值无效，不进 token 臂。
- **对账表**（每 case 一行，强制上报）：

  | case | f_wall | ρ_port（本次窗口 ANE 份额/GPU 全 MLP 速率比，若测）或探针 1.89 | R_pred = f_wall·(1 − max(1−s, s/ρ))，s=0.7333 | R_realized = (T_A − T_B)/T_A | 偏差 + 机制 |
  |---|---:|---:|---:|---:|---|

  `R_pred` 在 s=0.7333、ρ=1.89 时 = `f_wall × 0.612`；`R_max`（最佳份额）= `f_wall × ρ/(1+ρ)`。
- **要求**：偏差（预测 vs 实测）必须给出机制解释（候选：Amdahl 的墙钟 f 被小 kernel/dispatch 稀释、
  ANE 与 GPU 争抢、512-tile 在 128 case 的 4× 补齐、还原/打包/相加开销、GPU 份额不均衡、host 阻塞的 predict）。
  **对不上就写「R_max 模型错在哪」，这比结果好坏更有价值**（owner 原话）。
- **边界**：ρ/R_max 只出现在本节的机制对账里；§5.5 的判定只看 §5.2 的 TTFT 与 §5.1 的 token。

### 5.4 拒绝规则（owner 四条 + 追加）

- **① 端到端 TTFT 无改善或变差**（任一注册 case 改善 <3%）⇒ **FALSIFIED**：回退 `metal_lfm3` 的改动
  （PR 关闭/回退，不留死代码开关），写死「ANE MLP 在此 workload 上无收益」，并写清机制
  （Amdahl？ANE 争抢？列切分点不对？还原/打包开销？）。ρ 再漂亮也不翻案。
- **② token 不同且零量化对照 B2 也不同（同一位置）** ⇒ **REJECT-QUALITY**：是路径/baseline 的问题，
  不是 ANE。不许写成「FP16 不行」；先修 baseline/路径，本方向不在这一棒继续。
- **③ token 不同而 B2 相同** ⇒ **REJECT-FP16（可重试一次）**：FP16 ANE 的引擎算术不够 ⇒ 份额降一档
  （B1′，只一次），再跑 §5.1 + §5.2；仍不过 ⇒ **放弃**（保持 off，记录结论）。
- **④ top-1 <98% 但 B2 ≥98%** 按 ③ 处理；**B2 也 <98%** 按 ② 处理。
- 补充：若 G2/G3/G6/G8/程序同一性任一失败 ⇒ **INVALID**，先修实现，不进判定。

### 5.5 判定表（冻结）

前提：G0–G8 全过。

| 判定 | 条件 |
|---|---|
| **WIN** | 128/512 两个 case 的 engine-basis TTFT 中位都改善 ≥3% **且** §5.1 全过（8/8 贪心逐字相同、top-1 ≥98%、B2 ≥98%）⇒ 方向兑现；flag 保留、**默认仍 off**，owner 决定是否开默认。报告是否同时 ≥8%（缺口关上） |
| **FALSIFIED** | 任一注册 case 改善 <3%（含变差/无变化）⇒ 回退；写清机制，ρ 不构成翻案理由 |
| **REJECT-QUALITY** | token 不同且 B2 在同一位置也不同 ⇒ 路径/baseline 问题，不是 ANE |
| **REJECT-FP16** | token 不同而 B2 相同，且降一档重试后仍不同 ⇒ FP16 ANE 不够，放弃 |
| **INCONCLUSIVE** | G0/G1/G4/G7 不满足，或 A 未复现 720/731，或 A/A 超限，或 engine-basis 证据缺失 |
| **INVALID** | G2 placement 未落 ANE、G3 空臂、G5 越界、G6 off 不等价、G8 口径违规、程序同一性 relL2 >0.5% |

---

## 6. 份额：判定点与端到端扫描（冻结）

- **判定点 = S=5632（share 0.7333）**：这是 #146/#151 的探针最优，也是 owner 指定的 B 臂。
  在该点，ρ=1.89 的预测是 `R_pred = f_wall × 0.612`（不是 47.7% 的最佳份额上界）。
- **端到端扫描（必须做，解释用）**：阶梯 `{2048, 3072, 4096, 4992, 5632, 6144}` 列，
  对应 `APXINF_LFM3_ANE` 取值 `{0.267, 0.4, 0.533, 0.65, 0.733, 0.8}`（按 §3.3 的 64 列取整）。
  其中 **4992 = FP16 吞吐最优**（`s* = ρ/(1+ρ)`，ρ=1.89 ⇒ 0.654），**5632 = 探测/部署份额**。
  每个 rung 在 128/512 上按 §5.2 同一方法测 TTFT（各做 A/A）。
- **用途**：
  1. **解释**：若 5632 无改善，扫描表回答「是 split point 不对，还是根本无收益」——即 §5.4 ① 的机制之一。
  2. **推荐开启值**：若 5632 **WIN**，交付的「推荐开启值」= 扫描的 maximin rung
     `argmax_s min(gain_128(s), gain_512(s))`（可能不是 5632）；该 rung 必须另跑 **B1″**（§5.1）并通过，
     否则交付 5632 或保持 off。**交付的默认仍是 off**。
  3. **不翻案**：若 5632 不改善、而某个别的 rung 改善 ≥3%，判定仍是 **FALSIFIED（注册点）**，
     机制写「split point」；那个 rung 是**新候选**，需要新预注册，不能在本票里改判。
- 标定在 **M4**（lane 的目标硬件）上做，不在 M1 Pro；报告主机、macOS、revision。

---

## 7. 已知风险（必须写进文档）

1. **端到端可能根本不改善**（owner 追加要求的核心）：ρ=1.89 / R_max=47.7% 是局部探针 + 算术外推；
   `f=0.73` 是 FLOP 份额，不是墙钟。逐层 ANE `predict` 的 host 阻塞、tile 打包、GPU 份额不均衡、
   还原相加、24 层串行都会稀释兑现率。**本预注册不接受「ρ 漂亮 ⇒ 方向成立」**，只看 TTFT。
2. **ANE 与 clef 争抢**：clef 的发货列切分默认在用 ANE（5632/12288 列，`GPU_COLUMNS=6656`）。
   两条 lane 不能同时测；本棒的 G1 用双锁 + 进程扫描。**部署期**同样存在争抢：LFM3 打开 ANE 时若 clef 在跑，
   两者会互相抢 ANE —— 本棒不做仲裁/调度，默认 off 是唯一缓解；默认开启是 owner 的决定，不是移植的自作主张。
3. **列切分点可能不对**：5632 是 INT8 时代的吞吐最优，FP16 的吞吐最优在 4992，端到端最优还受 GPU 份额效率、
   合并开销、128 case 的 512-tile 补齐影响。判定点用 5632（owner 指定），扫描表负责说明偏差；
   若「split point」是 FALSIFIED 的机制，下一棒要用扫描赢家重新预注册。
4. **128 case 的 4× 补齐**：T=128 用一个 512-tile，ANE 做 4 倍算术；FP16 在 T=128 的 ρ **没有测过**
   （#151 只测 T=512/2048）。128 的最优份额可能更小。
5. **精度余量薄**：基线只比 98% 高 3 个位置；一个并列翻转就可能不通过「逐字相同」。B2 与 margin 报告是归因工具，
   **不是**放宽门的理由；门是 owner 定的逐字相同 + 98%。
6. **内存/装载**：24 个程序 × FP16 权重（S=5632 时 ~1.66 GB），CoreML 可能再复制；首次编译/装载耗时。
   warm 后的 TTFT 不受影响，但要上报冷/热装载时间与 RSS；16 GiB 主机要留余量。
7. **跨层不能重叠**：下一层 RMSNorm 依赖本层 MLP 完成 ⇒ 只有同层 GPU∥ANE；ANE 的 `predict` 是 host 阻塞的
   逐 tile 调用，重叠窗口窄。
8. **配方带着 INT8 时代的归一化链**：对 FP16 不是必需；去掉可能更准/更快，但会换掉 0.36% 的载体，
   属于后续实验（新预注册），不在本棒。
9. **默认 off = 没有用户可见收益**：本方向的价值要等 owner 决定默认开启（或在 serving 里显式配置）才兑现。
10. **失败归因**：BF16→FP16 权重 cast 在正规范围内精确，失败几乎只能归给 ANE 引擎 FP16 算术或路径/并列；
    唯一杠杆是份额。别把引擎算术/并列问题误写成「权重量化问题」。
11. **f 的测量会改数值**（A′ 短路 MLP）：只用时间差，绝不用它的输出做 token 判定；且必须标注。

---

## 8. 局限（先写死，免得事后找补）

- 本文档只写判据，**零设备**；所有数字引用 #146/#149/#151，不是本轮测量。
- 注册语料是 q01..q08（8 条贪心 + 731 位置 teacher-forced），不是通用质量保证，也不覆盖长上下文/采样解码。
- TTFT 只有 128/512 两个 case、来自同一条冻结文档 prompt；不是 workload 分布，不外推（2048 只报不判）。
- engine basis 的 `prefill_ms` 包含 state reset + capacity preparation + prefill（#112 口径）；A/B 都含，
  常数项在比值里抵消，但它不是纯矩阵时间。
- 无 sudo/powermetrics：独占证据 = 双锁 + 进程扫描 + A/A；热漂移靠交替运行与 1.5% 的 A/A 栏兜住。
- 只在 M4 上判；其他 Apple Silicon 不在本棒。
- `metal_lfm3` 的 decode 不动，所以本判据只覆盖 prefill 的影响；decode 的 ANE 是另一个方向。
- f 的消融是时间仪器：A′ 的输出无效，f_wall 只用于对账；R_pred 仍依赖 ρ 的点估计，偏差本身就是结论的一部分。
- 若实现被迫改动 §3.1 白名单之外的文件，本预注册失效，需重新预注册。

---

## 9. 交接纪律（写死）

- **本棒（#153）**：只写文档；零设备；diff = 本 doc + `doc/lfm3-ane-mlp-preregister.md` 的一条日期更正。
  不创建 `ane.rs`、不改 `metal_lfm3`、不跑任何 ANE/GPU 测量。
- **下一棒（移植）**：只有在本文档合并进 `main` 之后，才允许动 `metal_lfm3`。顺序：
  ①（本 doc 已合并）②实现 §3 ③（G0 满足后）跑 §5.2 的 A/B（S=5632）与 §5.1 的 A/B/B2 ④跑 §5.3 的 f 窗口
  与 §6 的端到端扫描 ⑤填 §10 并给出 §5.5 判定 ⑥按判定交付（WIN 保留 flag 且默认 off；FALSIFIED/REJECT 回退）。
- **判据不许改**：§4/§5/§6 在首次设备运行后冻结。要改判据，先开新 issue、重新预注册，
  不能在原 PR 里偷偷改口径。§10 只填结果，不回头改判据。

---

## 10. Results（运行后填写；本节以上全部冻结）

**判定：FALSIFIED（§5.4 ①/§5.5）。** 2026-11-28 于 M4（10 核 GPU，16 GiB，
macOS 27.0.1 build 26A434 / Darwin 27.0.0）测量，两把 Metal 锁全程持有，
engine basis（server request log 的 `prefill_ms`）。完整证据与复现命令见
`devlocal/ane-prefill-16k/RESULTS.md`。

### 10.1 注册点（S=5632，share 0.7333）

`results/ab-0.7333.json`，每臂 21 样本、3 预热，A1→B1→A2→B2：

| case | A1 | A2 | B1 | B2 | R_realized | A/A |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 221.8 ms | 221.3 | 346.5 | 325.3 | -51.4% | 0.2% |
| 512 | 598.4 ms | 599.0 | 651.5 | 645.8 | -8.4% | 0.1% |

两个注册 case 都变差（128 +47–56%，512 +8–9%），远低于 3% 改善门；8 个贪心
token 四臂逐字相同。

### 10.2 墙钟 f_wall（§5.3）

| case | A 完整 GPU MLP | A′ 消融 | f_wall |
|---:|---:|---:|---:|
| 128 | 224.7 ms | 101.4 ms | 54.9% |
| 512 | 634.3 ms | 244.1 ms | 61.5% |

**仪器缺陷（已修，`df13fc7c`）**：`ablate` 分支原来排在 `share == 0` 之后，
不设 `APXINF_LFM3_ANE` 时 A′ 走的是完整 GPU MLP。第一次窗口
（`results/ablate.json`：128 220.6 vs 224.8、512 598.8 vs 634.5）因此是 A/A
零结果。修正后 A′ 打印 `lfm3_mlp_ablate=on`，两臂 prompt 均为 156/540 token。

### 10.3 份额阶梯（§6）

| share | case | A1 | B1 | B2 | R_realized | A/A |
|---:|---:|---:|---:|---:|---:|---:|
| 0.267 | 128 | 224.2 | 324.8 | 317.9 | -43.3% | 1.3% |
| 0.267 | 512 | 626.4 | 691.3 | 718.0 | -12.5% | 0.6% |
| 0.4 | 128 | 219.8 | 330.8 | 313.0 | -45.5% | 1.4% |
| 0.4 | 512 | 625.5 | 661.4 | 668.9 | -8.0% | 3.1% |
| 0.533 | 128 | 221.1 | 321.7 | 324.5 | -46.1% | 0.2% |
| 0.533 | 512 | 616.2 | 676.0 | 705.2 | -12.1% | 0.3% |
| 0.65 | 128 | 228.5 | 315.7 | 313.1 | -37.6% | 0.1% |
| 0.65 | 512 | 622.8 | 729.1 | 868.3 | -28.2% | 3.4% |
| 0.733 | 128 | 221.8 | 346.5 | 325.3 | -51.4% | 0.2% |
| 0.733 | 512 | 598.4 | 651.5 | 645.8 | -8.4% | 0.1% |

每一档在两个 case 上都变差，所以 FALSIFIED 不是切分点的选择问题。0.65 的 512
行被一次 868 ms 离群拉低（该臂 spread 58%）；符号不受影响。

0.8 档（S=6144）记 **INVALID**，不是数据点：ANE 臂在 Core ML 计划构造阶段失败
（`cannot inspect the Core ML compute plan: Failed to construct compute plan,
internal failure`），worker 退出。两次尝试：第一次 B1 装载成功、B2 在 48 个请求
后崩溃；重试在 B1 就崩。这是 6144 列的 Core ML 限制，而且 0.8 是对 WIN 最不利
的一档。

### 10.4 对账（§5.3 表）

| case | f_wall | R_pred = f_wall·0.612 | R_realized | 偏差 |
|---:|---:|---:|---:|---:|
| 128 | 54.9% | +33.6% | -51.4% | -85.0 pt |
| 512 | 61.5% | +37.6% | -8.4% | -46.0 pt |

每档把切分后的 MLP 成本反推为 `A = T_B - (T_A - F)` 再除以 `F`：

| share | A/F（128） | A/F（512） |
|---:|---:|---:|
| 0.267 | 1.79 | 1.20 |
| 0.4 | 1.85 | 1.10 |
| 0.533 | 1.84 | 1.20 |
| 0.65 | 1.69 | 1.46 |
| 0.733 | 1.94 | 1.14 |

`A/F` 对份额是平的（0.65 两档最噪：61% / 58% 臂内离散，512 有一个 868 ms
离群）。真实的 1.89× ANE 优势会让它随份额下降；它没有，所以 ANE 的边际每列
速率与 GPU 相当，不是快 1.89×。切分相对完整 GPU MLP 多付一个固定项：512 上
0.10–0.20 `F`（40–78 ms），128 上 0.70–0.94 `F`（86–114 ms）。形状就是 24 次
逐层 host 阻塞 `predict` + FP32 还原 + 相加；它与搬多少列无关，128 上显得大
是因为 ANE 仍跑 512 行 tile，而 GPU 半边只有四分之一。噪声不足以把 `c` 和
`ρ_served` 分开，也不需要：不存在让切分打赢完整 GPU MLP 的份额。**R_max
模型错在把 #151 孤立 predict 的 ρ 当成发货程序的属性**：发货程序每层是
1×1 FP16 gate/up 卷积 + SiLU + 逐行归一化 + K 拆分 down 卷积 + FP32 还原，
T=128 还要 512 行补齐。

### 10.5 未跑的注册门

§5.1 的 731 位置 teacher-forced top-1 没有跑：TTFT 门以 50 个百分点失败，
质量门只能否决 WIN，不能把 -50% 翻成 WIN。8 个贪心 token 的逐字一致
（§5.1 的另一半）全臂通过。B2 零量化对照也没跑：§5.1 只用它在 token 分歧时
归因，而这里没有分歧。两者都记为偏离，不改变判定。注意 A/B 脚本里的
`B2-ane` 是 `B1-ane` 的重复臂，不是预注册的 B2 对照。

### 10.6 交付

flag 保持 off，不打开 PR；`feat/lfm3-ane-fp16`（含 `df13fc7c` 仪器修复）
留在 qhy991 fork 作记录。M4 上 prefill 的 MLP 段仍占 55–62% 墙钟且在 GPU
上以 ~3.15 TFLOP/s 跑（FP32 峰值的 68%，与既有 GEMM roofline 一致），ANE
不是替代路径。
