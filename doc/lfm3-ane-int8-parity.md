# LFM3 的 INT8 精度代价：真实权重 + 真实激活下 ANE INT8 vs GPU BF16 的 MLP 输出误差（只测量，不移植）

预登记时间：**2026-11-20 23:20 CST**（commit 先于任何 ANE 设备运行；本 doc 的 §3 Gate、§4 Decision、§5 五档判定在测量后**不许改**）。
测量主机：**M4 Mac mini（Mac16,10，16 GiB），macOS 27.0.1**，`main` @ `61c82285`。

**来源**：issue #150（owner 派单，@enginetailor 执行）；owner 2026-11-20 14:53 的补充实测与对
本棒要求的更新（[comment](https://github.com/qhy991/apxinf-metal/issues/150#issuecomment-7042627924)）。
**承接**：#145 / #146 / #148 / #149（ANE MLP 吞吐 WIN，ρ = 2.88 / 2.60 / 2.46 on M4，**合成权重**）。

**人工约束（owner 原文）**：
1. 不移植、不改运行时：只加探针或扩展现有两个 example。
2. 预登记先于设备（写死 relerr 判据）。
3. 必须报 BF16 自比噪声底。
4. owner 更新后的判据：`relL2` 不是判据，**贪心 token 一致性才是**；并要在 **clef 的精确 CoreML
   路径**上复现他的 numpy 近似（中位 1.72%）；若 token 不一致，写明替代方案（FP16 ANE / 缩小
   share）并给出 **FP16 的吞吐预算**。

---

## 0. 事实基座（引用，非本轮测量）

| 事实 | 出处 |
|---|---|
| clef MLP 的 ANE 程序：gate/up 为 per-row 对称 INT8 1×1 conv；SiLU 用 `0.5v(1+tanh(v/2))`；gate/up 各自按行最大值归一后相乘；down 为 K-split INT8 conv；512-token tile；`factor_cap` 限幅 | `crates/apxinf-model/src/metal_clef/ane.rs` |
| ANE 侧 `y` 必须乘 `gate_scale × up_scale` 才还原成 MLP 输出（FP32 合并） | `crates/apxinf-mlx/native/ane_tile_merge.metal` |
| LFM3-1.7B：hidden 2048、intermediate 7680、24 层（13 conv / 6 full / 5 sliding）、BF16 | 本地 checkpoint `config.json` |
| owner 的 numpy 近似：全部 24 层真实 `w1/w3/w2`、真实 embedding 行、INT8 per-row 对称，**relL2 中位 1.72% / p90 2.03% / 最大 2.11%**；f32 数值底 0.000165% | issue #150 comment 7042627924 |
| ANE INT8 吞吐（合成权重）：M4 上 ρ = 2.88 / 2.60 / 2.46（T=128/512/2048，S=7680） | #146 / #148 / #149 |

**真实权重**：`/Users/haiyan-mini/Agent4Kernel/lfm3-bf16`（BF16 safetensors，3.2 GB，
`model.layers.{i}.feed_forward.w{1,3}: [7680,2048]`、`w2: [2048,7680]`）。
**真实激活**：同一 checkpoint 的 MLX BF16 引擎（外部 lane 运行时，只读使用，见 §7）在一条
固定真实 prompt 上 prefill 512 token 后、各目标层的 `ffn_norm` 输出 `[512, 2048]`。

---

## 1. 假设与 falsifier

**H1（量级复现）**：在 clef 的精确 CoreML 路径上（INT8 per-row + tanh-SiLU + gate/up 归一 +
down K-split + 512 tile），真实权重 + 真实激活的 MLP 输出误差落在 owner numpy 近似的量级
（relL2 1–2.5%），不是 BF16 噪声底量级。

**H2（下游可用性）**：该量级的扰动在 **部署列份额**（S=5632/7680）上不改变贪心输出：
同一 prompt、4 条真实问题、64 个贪心 token，BF16 与 INT8 的 token 序列完全相同。

**falsifier（owner 更新版）**：若部署份额的贪心 token 出现不一致 ⇒ INT8 ANE 这条路**不能按
`ρ ≈ 2.5` 兑现**，必须给出 FP16 ANE / 更小 share 的替代预算，并停工。

---

## 2. 测什么（冻结配置）

### 2.1 ANE 臂（`apxinf-coreml` 探针）

逐层（`L = {0, 4, 11, 22}`，覆盖 conv / sliding-attention / conv / full-attention 四种上下文）
构建 clef 的 `program()` 的逐算子复刻，形状 `H=2048`、`T=512`（clef 的 TILE），列份额
`S ∈ {7680, 5632}`：

- `S=7680`：整个 MLP 的中间维在 ANE 上（最坏情形；`parts=2`，`width=3840`）。
- `S=5632`：部署份额（ρ≈2.5 时的 `s* = ρ/(1+ρ)` 附近；`parts=2`，`width=2816`）。

权重按 clef 的 `int8_per_row` 量化：`scale = f16(max|w| / 127)`，`q = rint(w/scale)` clamp 到
`[-127,127]`；down 按 clef 的 `DOWN_PART=4096` 切分；`factor_cap` 按真实权重算（`FACTOR_SCALE=16`
上限、`DOWN_ABS_SUM_LIMIT=8192`）。输出 `y`、`gate_scale`、`up_scale`，另加一个仅探针使用的
`a`（归一后的 SwiGLU 乘积）用于"silu·mul 之后"的误差。

**FP16 对照臂**（`S=7680`）：同一链条，权重用 FP16 常量的 1×1 conv（clef 的 neuron-split FP16
写法），用于回答"若 INT8 不行，FP16 ANE 的精度是否够"。

### 2.2 GPU 臂（`apxinf-mlx` 探针）

同一层、同一真实权重、同一真实激活，MLX 的部署链 `gate/up matmul → silu·mul → down matmul`：
- **BF16** 权重与激活（#121 认证基线的引擎口径），输出 `y_bf16` 与 `silu(g)*u` 中间量；
- **F32** 同一批权重（数值底对照），输出 `y_f32` / `h_f32`。

### 2.3 误差指标（每层 × 每臂）

- `relerr`（相对 L2）：`||y_ane - y_gpu||_2 / ||y_gpu||_2`。
- `max|Δ|`、`RMS(Δ)`、`cos(y_ane, y_gpu)`。
- **silu·mul 之后**：`h_ane = a * gate_scale * up_scale` 对 `h_gpu = silu(g)*u` 的 relL2。
- BF16 自比噪声底（三项，全部报）：
  - N1：BF16 链跑两遍的差（确定性）；
  - N2：BF16 链 vs F32 链（同一批 BF16 权重）——"BF16 引擎口径"的舍入底；
  - N3：F32 链 vs FP64 链（numpy）——数值底（对账 owner 的 0.000165%）。
- INT8 量化底（N4）：F32 链用 INT8 per-row 反量化权重 vs F32 链原权重——对账 owner 的 1.72%。
- 引擎链误差（N5）：ANE INT8 输出 vs INT8 反量化 F32 链——归因给 tanh-SiLU/归一/FP16 引擎算术。

### 2.4 吞吐（只服务 FP16 预算）

同一探针在同一窗口内报 ANE 每层 512-token 的 `predict` 中位 ms（warmup 3 + 20 次），
`T=2048` 以 4×512 顺序 predict 表示（#146 的部署形态）；GPU BF16 的 ms 由既有
`lfm3_mlp_gpu_probe` 在同窗口复跑 `S=7680` 的 T=512/2048 得到。
`ρ = GPU BF16 ms / ANE ms`（同 T、同 S 的无量纲时间比，单位安全；等价于
`ρ = 2 × ANE TMAC/s ÷ GPU BF16 TFLOP/s`）。**不要写 `tmacs ÷ tflops`**：探针的 `tmacs`
按 MAC 计数（一次乘加记 1），`tflops` 按 FLOP 计数（一次乘加记 2 = 乘 + 加），两者不可直接
相除，否则 ρ 会系统性偏小一半。`R_max = f·ρ/(1+ρ)`，`f=0.73`（#146 冻结值）。

---

## 3. Gate（冻结）

> **G0 预登记先行**：本节与 §4/§5 的 commit 先于任何 ANE 设备运行。
> **G1 双锁 + ANE 独占**：测量窗口持有 `/tmp/enginetailor-metal.lock` → `~/.cache/enginetailor/metal-measure.lock`（POSIX flock，等不抢）；窗口前后各做一次 `ps` 扫描，不得存在其他可能用 ANE 的进程。本机无 sudo、powermetrics 不可用。
> **G2 placement 取证**：每个 ANE cell 的 `load_with_placement` 必须显示全部 conv op preferred = NeuralEngine；否则该 cell INVALID。
> **G3 A/A 稳定性**：完整精度矩阵跑两遍；每层 relL2 两窗相对差 ≤5% 为有效，>5% 重跑一次仍超 ⇒ 该层 INCONCLUSIVE。
> **G4 不移植（按构造）**：diff 只允许 `doc/`、两个 crate 的 `examples/` 新探针、`Cargo.toml` 的 `[[example]]` 注册、`devlocal/` receipts。不得触碰 `metal_lfm3` 或任何库代码。
> **G5 证据齐全**：原始 JSON / 原始输出二进制 / placement / A/A / ps 扫描 / token 序列全部落 `devlocal/lfm3-ane-int8-parity/`。
> **G6 仿真保真（token 判据的前提）**：token 一致性用"逐行 INT8 反量化 + clef 归一链"的数值仿真实现在 BF16 引擎内；仿真输出与同层 ANE 实测输出的 relL2 必须 ≤ 0.25 ×（该层 ANE-vs-GPU relL2），否则 token 判据 INCONCLUSIVE（不能拿仿真代替引擎）。

---

## 4. 贪心 token 一致性（冻结）

- **模型**：同一 BF16 引擎（外部 lane 运行时，只读引用，不改其代码；脚本在 `devlocal/`）。
- **prompt**：4 条真实问题（下文 P1–P4），按 lane 的 ChatML 模板 + BOS 124894；贪心；最多 64 个新 token；遇 EOS 124900 停。**token 序列逐位比较**，报首个分歧位置与分歧率。
- **四臂**（同一进程、同一 tokenizer、同一权重来源）：
  - **A0**：BF16（基线）；
  - **A1**：全部 24 层 MLP 的 `w1/w3/w2` 全部 7680 列做 per-row INT8 量化并反量化（clef 形状），MLP 在 **F32** 里按 clef 链（tanh-SiLU + gate/up 归一 + 反归一）计算，结果 cast 回 BF16 后进残差；
  - **A2**：**部署份额** —— 每层列 `[2048, 7680)`（5632 列）走 A1 的 INT8 链，`[0, 2048)` 保持 BF16；两半按列相加；
  - **A3**：FP16 权重（`f32→f16→f32`）全列，F32 链，结果 cast 回 BF16（FP16 ANE 的权重精度上界）。
- 冻结判据：**A2 与 A0 的 4×64 token 完全一致** = 部署份额可用；A1 全一致 = 全份额也可用；A3 用于替代方案。

---

## 5. 五档判定（冻结）

前提：G0–G6 全过。否则落到 INCONCLUSIVE / INVALID。

| 判定 | 条件 |
|---|---|
| **PASS** | A2 与 A0 完全一致 **且** A1 与 A0 完全一致 ⇒ INT8 ANE 在 1.7% 量级下仍保 token，方向继续 |
| **PASS-SHARE** | A2 完全一致、A1 有分歧 ⇒ INT8 只在 ≤5632/7680 列份额下可用；写清份额上限，不需要 FP16 |
| **FAIL** | A2 出现分歧 ⇒ 部署份额不可用；给 FP16 ANE 的吞吐预算（§2.4）与 A3 的 token 结果，并停（不移植） |
| **INCONCLUSIVE** | 锁/独占/placement/A-A/G6 任一不满足，或关键 cell 缺失 |
| **INVALID** | G4 越界（碰了库代码 / metal_lfm3），或探针没有测到所声称的链条（空臂） |

**与 #150 原文判据的关系（明确写出，不悄悄改）**：原文"relerr 在 BF16 噪声底 2 倍以内 ⇒ 可接受；
显著更大 ⇒ 量化 ANE 走不通并停"在 owner 14:53 的更新里被显式取代——**relL2 不作为判据，token
一致性才是**。本棒仍完整报 relL2 与噪声底（作为机制证据），但五档判定只由 §4/§5 的 token 结果驱动；
若 owner 仍想按原文判据执行，本表的 relL2 列可直接查出结论（预期显著大于噪声底）。

---

## 5a. 补充诊断（post-hoc；不改变 §5 冻结判定）

冻结矩阵的首轮结果（§6）里出现了一个必须先证伪的现象：**部署份额的贪心 token 出现分歧**。
在把它解释成"INT8 精度代价"之前，先排除三个更平凡的解释。本节是看到首轮 token 结果后
追加的，**明确是 post-hoc**，没有、也不会改动 §4/§5 的冻结判据与判定：

- **M0 基线自一致**：BF16 基线在独立进程里重跑 3 次，token 序列逐位比较。若 BF16 自己都
  不稳定，token 相等就不能当判据。
- **D1 零量化对照（a4）**：原 BF16 权重不变，只把 MLP 换成本棒同一个 F32 链（无 INT8、
  无 FP16 舍入）。a4 与 a0 的分歧 = 纯数值路径差异。
- **D2 同路径换权重（a5）**：把 INT8 反量化权重转回 BF16，MLP 用引擎原生的 BF16 数学
  （`silu(g)*u` 后 down），与 a0 唯一的差别是权重值本身（含 INT8 量化 + 回 BF16 的舍入）。
- **D3 teacher-forced top-1 一致率（只报告，不判定）**：固定 a0 的 prompt+生成序列，逐位置
  比较各臂的 argmax 与 a0 的下一 token，报 4 prompt × 生成位置的一致率。98%（lane 的
  registration-gate 口径）作为"与 gate 同量级"的参照线；**这不是冻结判据**，只是给
  owner 决定"要不要把判据从 token 相等改成 top-1 一致率"的输入。

D3 的指标在此 commit 时冻结（先于运行）。

## 6. Results（测量后填写；本节以上全部冻结）

**判定（冻结，§5）：FAIL** —— 部署份额（A2）的贪心 token 在 P1/P2 出现分歧。
**但 §5a 的 post-hoc 诊断表明：这个 FAIL 由 BF16 的数值并列触发，不是 INT8 精度造成的**
（零量化对照在同一位置同样分歧；部署份额对照与零量化对照逐 token 相同）。证据与机制见 6.5–6.7。

### 6.1 环境与输入

- 主机：M4 Mac mini（`Mac16,10`，16 GiB），macOS 27.0.1；`main` @ `61c82285` + 本票分支。
- 真实权重：`/Users/haiyan-mini/Agent4Kernel/lfm3-bf16`（BF16 safetensors；
  checkpoint digest `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`）。
- 层 `{0, 4, 11, 22}`（conv / sliding-attention / conv / full-attention）。
- 真实激活：MLX BF16 引擎在固定 prompt（`doc/model-layer-architecture.md` 前 512 token，
  token-ids sha256 `4bcda0c7…`）上 prefill 后各层的 `ffn_norm` 输出 `[512, 2048]`；
  抓取脚本与 lane runtime 的 sha256 在 `devlocal/lfm3-ane-int8-parity/inputs/meta.json`。
- ANE 程序：clef `program()` 的逐算子复刻（INT8 per-row、tanh-SiLU、gate/up 归一、down K-split、
  512 tile），`S=7680`（整层）与 `S=5632`（部署份额，列 `[2048,7680)`）；FP16 对照臂同链。
  真实 LFM3 的 `factor_cap` 落在 **4 或 8**（不是合成权重时的 16）——这是真实 down 权重的行绝对值
  和决定的，已按 clef 公式照算。

### 6.2 Gate 对账（冻结项）

| Gate | 结果 | 证据 |
|---|---|---|
| G0 预登记先行 | PASS | 预登记 commit `30255a96`（23:20 前），补充诊断 `52aadbc2`（D3 运行前） |
| G1 双锁 + 独占 | PASS | 两个窗口全程持锁；`ps-before/window/after.txt` 三份均为空（无其他 ANE 用户） |
| G2 placement | PASS | `ane-A4` 的 12 个程序全部 conv op preferred = ANE（`ane/L*/ane-*.json`） |
| G3 A/A | PASS | A4 vs B4：所有输出文件 **逐位相同**（relL2 = 0.0）；每 cell 计时差 ≤ 2.97% |
| G4 不移植 | PASS | `git diff --stat`：doc + 两个 example + 两个 `Cargo.toml` 的 `[[example]]` 注册；无库代码/`metal_lfm3` 改动 |
| G5 证据齐全 | PASS | `devlocal/lfm3-ane-int8-parity/`：inputs、A4/B4、GPU B/B2、token A0–A5、tf-*、analysis.json |
| G6 仿真保真 | PASS | 仿真 vs ANE relL2 最大 0.1456% = ANE-vs-BF16 误差的 **15.4%**（栏 25%） |

**M4 的操作性发现（已验证）**：引擎的 `cannot inspect the Core ML compute plan: …
internal failure` 在第二次窗口复现于 layer 22；清空 `/tmp/lfm3-mlp-parity-cache`
（编译产物缓存）后，12 个程序全部装载成功。两个干净窗口均按"每层一个探针进程"运行。
失败的窗口 `ane-A2/`、`ane-A3/` 保留在 receipts 里。

### 6.3 精度：ANE INT8 / FP16 vs GPU BF16（真实权重 + 真实激活）

单位 %（relL2，除非说明）；`S=7680` 是整层走上 ANE，`S=5632` 是部署份额。

| layer | INT8 S=7680 | INT8 post-SwiGLU | INT8 量化底 (N4) | 引擎链残差 (N5) | INT8 S=5632 合并估计 | FP16 S=7680 | FP16 post-SwiGLU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.658 | 0.887 | 0.593 | 0.101 | 0.573 | 0.282 | 0.432 |
| 4 | 0.896 | 1.053 | 0.809 | 0.136 | 0.771 | 0.330 | 0.423 |
| 11 | 1.184 | 1.402 | 1.157 | 0.146 | 0.964 | 0.384 | 0.487 |
| 22 | 1.200 | 0.680 | 1.152 | 0.135 | 1.105 | 0.656 | 0.378 |
| **中位** | **1.040** | 0.970 | 0.981 | 0.136 | **0.867** | **0.357** | 0.428 |

- `max|Δ|`（S=7680）0.015–0.145，`RMS(y)` 0.053–0.603，`cos` 0.99993–0.99998。
- **INT8 的误差几乎全部来自权重量化**（N4 0.59–1.16%），引擎链（tanh-SiLU + 归一 + FP16 算术）
  只再加 **0.10–0.15%**；"silu·mul 之后"的相对误差与输出同量级（0.68–1.40%）。
- 份额：ANE 只做 5632 列时，份额本身 relL2 0.68–1.47%，按份额范数折算到整层输出的合并估计
  0.57–1.10%（中位 0.87%）。
- 与 owner numpy 的对账：owner 全 24 层中位 1.72%（embedding 行输入）；本棒 4 层中位 1.04%
  （量化底 0.98%），**同量级、不矛盾**；本棒不是 24 层全扫。

### 6.4 噪声底（必须先报的量）

| 底 | 值 | 说明 |
|---|---:|---|
| N1 BF16 重跑 | **0.0** | 3 次独立进程逐位相同（固定 tie-breaking 后；见 6.6） |
| N2 BF16 vs F32（同 BF16 权重） | 0.249 / 0.273 / 0.315 / 0.315% | "BF16 引擎口径"的舍入底 |
| N3 F32 vs FP64 | 0.000136–0.000157% | 对账 owner 的 0.000165%，一致 |

**INT8 / N2 = 2.64 / 3.29 / 3.76 / 3.81 倍**（4 层全部 > 2×）；
**FP16 / N2 = 1.13 / 1.21 / 1.22 / 2.08 倍**。

若按 issue #150 原文的 "relerr 在 BF16 噪声底 2 倍以内" 判据，INT8 在 4 层全部不通过；
owner 14:53 的更新已显式把判据换成 token 一致性，故这里只作为机制证据报告。

### 6.5 贪心 token 一致性（冻结判据）：FAIL

4 条真实 prompt、ChatML + BOS、贪心、最多 64 token（P1 遇 EOS 在 59）；`diff@k` 是首个分歧位置。

| prompt | A1 全列 INT8 | A2 **部署份额** INT8 | A3 FP16 | A4 零量化对照 | A5 仅换权重 |
|---|---|---|---|---|---|
| P1（59 tok） | diff@29 | **diff@29** | same | diff@29 | same |
| P2（64 tok） | diff@12 | **diff@12** | diff@12 | diff@12 | diff@12 |
| P3（64 tok） | same | same | same | same | same |
| P4（64 tok） | diff@26 | **same** | same | same | diff@26 |

| 五档（§5） | 条件 | 本棒 |
|---|---|---|
| PASS | A2、A1、A3 全一致 | ✗ |
| PASS-SHARE | A2 一致、A1 分歧 | ✗ |
| **FAIL** | **A2 分歧** | **命中** |
| INCONCLUSIVE | Gate 不全 | ✗ |
| INVALID | 越界/空臂 | ✗ |

按冻结规则：部署份额不可用；A3（FP16）也在 P2 分歧 ⇒ 按同一判据 FP16 同样"不可用"。

### 6.6 Post-hoc 诊断：FAIL 的机制是数值并列，不是 INT8 精度

§5a 的四个诊断（详表见 receipts）：

1. **M0 基线自一致**：固定 tie-breaking 后，BF16 基线 3 次独立运行逐位相同（`a0`、`a0-r4`、`a0-r5`）。
   最初两次重复在 P4 不同，根因是探针把 `argmax` 换成 `argsort` 后对**精确并列**的 tie-breaking
   改变，不是模型不稳定。
2. **分歧位置的 logit margin**：P1@29 与 P2@12 的 BF16 top1−top2 = **0.125**，P4@26 = **0.0（精确并列）**；
   每个 prompt 有 8–20% 的 decode 步 margin < 1.0。
3. **A4（零量化对照，同一条 F32 链）**：在 **P1@29、P2@12 与 A2 完全相同的两个位置分歧**；
   A2 与 A4 在全部 4×~64 个 token 上**逐位相同**。也就是说，部署份额的 INT8 量化没有在
   零量化对照之外多翻任何 token。
4. **A5（只换权重、保持引擎原生 BF16 数学）**：P2@12、P4@26 分歧，其中 P4@26 是精确并列；
   A1（全列 INT8）相对 A4 也只在 P4@26（精确并列）多一个分歧。

结论：**"token 相等"在这台机器上分辨不了 INT8 的质量**——任何把 MLP 换成 ANE 数值路径的实现
都会在 0.0–0.125 的并列/近并列处翻 token。冻结判定的 FAIL 是真实的（按规则就是 FAIL），但把它
读成"INT8 精度不可用"会与证据相反。

### 6.7 Teacher-forced top-1 一致率（D3，只报告，不判定）

固定 a0 的 prompt+生成序列，各臂逐位置 top-1 与 a0 的下一 token 比较，251 个位置：

| 臂 | 一致率 | 与 a0 自身口径的差 |
|---|---:|---:|
| A0 BF16（自身 prefill vs decode） | 99.602% | 基线自身（1 个位置是 prefill/decode 数值差） |
| A1 全列 INT8 | 99.203% | −1 位置 |
| A2 部署份额 INT8 | **99.602%** | **0** |
| A3 FP16 | **99.602%** | **0** |
| A4 零量化对照 | 100.000% | +1（把基线自身的 1 个数值差修掉） |
| A5 仅换权重 | 99.602% | 0 |

lane 的 registration-gate 口径是 98%；全部臂 ≥99.2%，比 gate 高 1.2 个百分点以上。
**注意**：这是 post-hoc 指标，没有冻结权威；且基线自身只有 99.602%（prefill 与逐步 decode
的数值路径不同），所以"100%"不是这条指标的天花板。

### 6.8 FP16 ANE 的吞吐预算（owner 更新版第 3 条）

同一窗口内 ANE 与 GPU BF16 的 512-token tile 实测（4 层中位；GPU BF16 S=7680 T=512/2048）。
下表 ρ 一律是 `GPU ms / ANE ms`（同 T 的时间比，§2.4），**不是** `tmacs ÷ tflops`（差 ×2）；
ρ 与 R_max 的数值、判定均不变：

| 臂 | T=512 ms | T=2048（4×512）ms | ρ(T=512) | ρ(T=2048) | R_max（f=0.73） |
|---|---:|---:|---:|---:|---:|
| ANE INT8 | 5.75 | 23.2 | **2.42**（2.37–2.44） | **2.34**（2.27–2.38） | 51.6% / 51.2% |
| **ANE FP16** | 7.36 | 29.4 | **1.89**（1.88–1.89） | **1.85**（1.81–1.86） | **47.7% / 47.4%** |

GPU BF16 同窗口 A/A：T=512 0.01%、T=2048 0.06%（T=128 首 cell 6.73% 是 #149 记录的冷启动）。
**结论：FP16 ANE 的吞吐仍然远超 12% 缺口（R_max ≈ 47%）——若改用 FP16，瓶颈不在吞吐，在精度门。**
INT8 的 ρ 也复现了 #146/#149 的 M4 量级（2.4）。

### 6.9 结论（给 owner；"用哪个 gate"需要人决定）

1. **冻结判定 FAIL**：部署份额 INT8 的贪心 token 不一致（P1@29、P2@12）。严格按 #150 的更新版
   判据，INT8 ANE **不能按 ρ≈2.5 兑现**，FP16 也同样过不了"token 相等"这道门。
2. **但这个 FAIL 不是精度信号**：零量化对照（A4）在同两个位置分歧，A2 与 A4 逐位相同；
   所有"分歧"都落在 logit margin 0.0–0.125 的并列/近并列上。**"token 相等"判据在本模型上
   不可用**，它测到的是 bf16 舍入的并列，不是 INT8 的误差。
3. **有意义的精度量级**（clef 精确路径、真实权重、真实激活）：INT8 整层 relL2 中位 **1.04%**
   （0.66–1.20%），部署份额折算 **0.87%**；其中量化贡献 0.98%，引擎链只加 0.14%。
   与 owner 的 1.72% 同量级。FP16 为 **0.36%**（接近 BF16 自身的 0.29% 底）。
4. **替代方案的预算已经给出**：FP16 ANE 的 R_max ≈ 47%（T=512/2048），远够 12%；不构成吞吐问题。
5. **需要 owner 选一个真正的 gate**。本棒能给的证据是：teacher-forced top-1（lane 的 98% 口径）
   下 A1/A2/A3/A5 全部 99.2–99.6%，A2/A3/A5 与基线自身的 99.602% 天花板相同；若 owner 接受
   这个 gate，下一步应是**先预登记 24 层 + 真实问题集的 top-1 gate**，再谈移植；若坚持 strict token
   相等，则 INT8 与 FP16 都应判死，ANE 方向按精度关闭（而不是按吞吐关闭）。


## 7. 外部工具（只读，不改）

- lane 运行时（`lfm3-lane`）位于 `/Users/haiyan-mini/lfm3-lane`，用于两件只读的事：
  抓真实 `ffn_norm` 激活、跑贪心 token。它的 revision 与脚本 sha256 落 receipts。
- 本票的 maintained 交付物只有：本 doc + 两个探针 example（+ `Cargo.toml` 的 example 注册）。

## 8. 局限（先写死）

1. 只测 4 层（{0,4,11,22}）× 3 个 ANE 臂，不是全部 24 层；层选择覆盖四种上下文，但层间分布未全扫。
2. 单条 512-token prompt 的激活；不同 prompt 的激活幅值会变，per-row 量化误差与激活无关（权重侧），但 FP16 引擎算术与归一链的误差与激活有关。
3. token 一致性是 **64 token 的贪心**（4 prompt），不是长程/多采样；长上下文与采样解码不在本棒。
4. ANE 路径是 clef 配方的逐算子复刻（同一 CoreML 图），但 LFM3 的部署图若改变 tile/份额/融合，数字需重测。
5. 仿真不是引擎：G6 只保证仿真误差 ≤ 0.25× 被测误差，不保证 token 序列与真 ANE 逐位相同（真 ANE 的 token 级验证需移植后做）。
6. 冻结的 token 判据在 bf16 下不可判别（6.6）：分歧位置全是 0.0–0.125 的并列/近并列，零量化对照同样分歧。这不是为 FAIL 找借口——按 §5 规则它就是 FAIL；但把它读成 INT8 的质量结论会与证据相反。
7. 探针为取 SwiGLU 中间量多输出一个 identity `a`（不改变计算图）；吞吐计时用的是同一程序。
