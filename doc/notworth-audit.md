# NOT-WORTH / 关闭结论复盘审计（2026-11-06）

对 Clef（Mario）线与 D3-lite 线此前的 NOT-WORTH 与关闭结论逐条复盘，判定哪些是**证据充分的
硬结论**、哪些是**被环境卡住的 INCONCLUSIVE 被误写成 NOT-WORTH**、哪些是**只做过一次变体就
外推到整个方向**的欠探索结论；并给出重开清单与 D3 的第一批优化方向。

- 审计日：2026-11-06；测量机：Apple M1 Pro 16 GB，macOS 27.0.1，MLX 0.32.3（`applegpu_g13s`）
- Clef 记录来源：[PR #5](https://github.com/qhy991/apxinf-metal/pull/5) 与 `showcase/clef-vision`
  分支的 `docs/mario-opt/`（`HANDOVER.md` / `FUSION-MATMUL.md` / `ANE-COLUMN-BALANCE.md` /
  `EAGER-EVAL.md` / `PREP-IMAGE-CACHE.md` / `OPT-CANDIDATES.md` / `TIMELINE-OPERATOR.md`），
  以及更早的外部归档 `~/Documents/Infinity/Agent4Kernel/clef-mario-opt-20261103/devlocal/mario-opt/`
  （`FUSION-RESULTS.md` / `MEGAKERNEL-PLAN.md` / `WAVEFRONT-PLAN.md` / `ANE-DEEPER-PLAN.md` /
  `DECISIONS.md` / `REPORT.md`）
- D3 记录来源：`doc/d3-lite-metal.md`（P1a）、`doc/d3-lite-metal-vision.md`（P1b）、issue #11
- 第二意见：`claude-opus.sh`（sorux-cc/claude-opus-5-5），结论见 §7；最终判定仍以本文件为准

## 1. 判据与证据分级

**证据分级**（每个结论标注其一）：

| 级别 | 含义 | 可否直接用于关闭方向 |
| --- | --- | --- |
| **MEASURED** | 同机、同二进制、配对、过 Swap 门的实测 | 可以 |
| **EXTRAPOLATED** | 由已测量配置推到未测配置（如「A2 只会中性偏正」、列配平模型外推） | 只能降级为待验证 |
| **MODEL-ONLY** | 工作量模型、几何下限、成本估算，零实测 | 不能单独关闭方向 |

**四个审计维度**：证据强度 / 试过的实现变体数 / 是否受环境影响（swap、TCC `open()` 挂死、
缺硬件 instrument、单 take 噪声）/ 是否存在未试的关键变体。

**纪律**（判定 KEEP 用）：3 轮反序、双锁单窗、每臂 `vm_stat` Swapouts（非 0 不进 KEEP）、
parity 通过、median ≥ 2% 且 bootstrap CI95 下界 > 0。否则 NOT-WORTH；**没有任何干净轮次时
必须判 INCONCLUSIVE，不得用污染数据判 NOT-WORTH**。

## 2. Clef 线逐条审计

### 2.0 总表

| # | 方向 | 证据分级 | 实测变体数 | 环境影响 | 未试的关键变体 | 裁决 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | fusion 组合（elementwise） | MEASURED（单候选用 #1/#2 干净 3 轮；组合 3 轮摆动大） | 2 单独 + 1 组合 | 组合轮 −2.6/−0.4/+7.1% 摆动 → swap/热噪声主导；组合 CI 未含轮间方差 | #3（q_norm+mRoPE，仅估算 ~0.1% 即 HOLD，**从未实现**）；**FUSION-PLAN 原案的 kernel 级 conv+norm 融合（预测 2–4%）被弱化成 slice+reshape 版，从未实现原案** | 单候选 **UPHOLD**；组合 **INCONCLUSIVE**；kernel 级原案 **REOPEN-LOW（预期 0.5–1.5%）** |
| 2 | matmul 级融合（A1 k/v、A2 z/b/a） | MEASURED（latency）+ MEASURED（A/A 漂移） | **1**（A1+A2 合并实现） | 干净（6/6 臂次 Swapouts +0） | A1 单独（latency 与 parity 均未测）；GPU 档 A2（仅结构性取证）；fused-kernel / 分组 scale 其它拼法 | **UPHOLD**（latency/CI/parity 三重失败 + 不删 FLOPs 的天花板论证） |
| 3 | ANE 列再平衡（6656→7680） | MODEL-ONLY（配平）+ MEASURED-smoke + **INCONCLUSIVE**（campaign） | 1 个新点（7680） | **严重**：3 轮 5/6 臂次 Swapouts≠0 → latency 不可测；auto 基线自身 vs MLX 0.0404 > 0.035 | 列数连续扫描（7168/6144 等）；ANE tile 尺寸；列×tile 联合空间 | **INCONCLUSIVE**（被误写成 NOT-WORTH）；parity 相对 auto 基线退化可作弱否定 |
| 4 | deeper-ANE（C=5632）/ wavefront K=2 | deeper: MEASURED（T1）+ MODEL-ONLY（T2，实现未测）；wavefront: **MODEL-ONLY + 半程诊断门控** | deeper **1 档实测**（5632；4608 实现了白名单但 D1 明确跳过、零测量）；wavefront **1 个 K** | wavefront 判据用的是 42 请求 trace，未过 Swap 门复核 | deeper: 4608 档（已实现、从未测）与列×tile 空间；wavefront: K∈{1,1.5,3,4}；tile/列空间；「async_eval 提到 predict 之前」单独消融；Q10b 反序复测（门控条件 (i)，**未做**） | deeper-ANE T1 **UPHOLD**、T2 **未闭环**；wavefront **INCONCLUSIVE-lean**（门控两个条件均未满足，却直接按「已关闭」写入清单） |
| 5 | eager eval（per-layer `async_eval`） | MEASURED（3 轮全干净、bit 相同、parity 0.0328 不变） | **1** | 无（reference 档干净） | chunk 粒度、head 粒度、独立线程派发；**两个候选归因未裁定**（记录里明说） | **UPHOLD**（对该变体）；「增量求值方向关闭」**过强** |
| 6 | prep 缓存 | MEASURED 硬上界 | 1（足够） | 无 | 无（上界 1.11% < 2% 不依赖实现） | **UPHOLD** |
| 7 | megakernel | **MODEL-ONLY，从未实现** | **0** | 无实测可言 | 全 GPU 持久 kernel（周级）；**ANE 专用线程+异步下发（同一次评估里被定为「下 campaign 首选」，从未实现）**；wavefront 诊断门控（见 #4） | **UPHOLD 仅对「全 GPU megakernel 本轮不做」**；「megakernel 线关闭」**不成立**——首选候选零证据 |
| 8 | mx.compile / vis014 / vision Metal batch | mx.compile: MEASURED(host only)；vis014: MODEL-ONLY；batch: MEASURED | 各 1 | xctrace 注入曾被 kperf/dyld 卡死（后经 TCC 暂存解决）；Core AI 永久禁用 | mx.compile 的**设备侧**效果（host 建图 0.7ms 只覆盖 host 轴）；batch 与其它 fusion 叠加 | mx.compile **UPHOLD（host 轴）/ 设备轴未测**；vis014 **UPHOLD（几何不可能，但零实测）**；batch **UPHOLD（1.1–1.8% 实测 <2%）** |

### 2.1 fusion 组合（#1）

**实测**：单候选 #2（q/k norm+scale → 1 kernel）median 2080.7→2068.0 ms（+0.61%），配对
+9.68 ms，CI95 [5.57, 14.14]，33/42 更快，`to_bytes()` 42/42 bit 相同；#1（GDN conv 切分
3×slice+reshape → 1 kernel）2069.9→2065.2 ms（+0.22%），+5.67 ms，CI [−4.90, +7.21] 跨 0，
24/42。两臂 Swapouts +0/+0。组合 flag（D22 / Q28）−0.372%，CI [−0.74, −0.20]，3 轮
−2.6/−0.4/+7.1%。

**审计**：
- 单候选的关闭是**硬证据**：bit 相同 + CI 分辨率 ~0.2% + 点估计 0.61%/0.22% < 1%（当时的
  门槛）→ 即使把门槛降到现在的 2% 也不会翻案。
- 但 #3（attention q_norm + mRoPE 的 3 gathers → 2 kernels）只有**源码级估算 ~0.1%** 就
  HOLD 从未实现；「kernel 级融合」这个类别只覆盖了两个极小的 elementwise 融合。
- 组合值 −0.372% 的 CI 虽然不含 0，但**逐轮 −2.6/−0.4/+7.1% 的摆动幅度是净效应的 20 倍**
  ——这是环境噪声主导的读数，把它当作「融合有害」的证据不成立；它只能说明「在该环境下测不出
  正收益」。
- **更深一层（第二意见 §2/§5.1）**：Clef campaign 的 CI 是把逐案配对增益跨轮池化后做
  bootstrap，**没有把轮间方差计入**。当轮间摆动与本例同量级时，这个 CI 是反保守的——所以
  连「CI 显著为负」本身也不可信。裁决应写成 **INCONCLUSIVE（该环境下测不出收益）**，
  而不是「融合使请求变慢」。本清单对轮间摆动 ≥ 点估计量级的 campaign 一律按此处理；
  本 leg 的 D3 campaign 也按此口径补报逐轮区间（见 `doc/d3-lite-metal-gdn.md` §4）。
- **计划与实现不符（本项最实质的欠探索证据）**：`FUSION-PLAN` 排序第 1 的目标是**kernel
  级**融合——「融合 conv kernel 直接写 q/k/v head 布局并对 q/k norm+scale，省 ~8 kernel ×
  32 层 = 256 launch」，预测 **~40–80 ms（2–4%）**；三项合计预测 **75–155 ms
  （3.9–8.1%）**。但实际落地的 Fusion #1 只是「3× `slice+reshape` → 1 kernel」（纯视图/
  切片合并，不含 conv、不含 norm），实测 **+0.22%**；Fusion #2（norm+scale）+0.61%；
  **计划里的 kernel 级 conv+norm 融合从未实现，其 2–4% 预测从未被验证**。回答任务里
  「fusion 组合的变体空间是否试尽」的准确说法是：**没试尽，而且实测的不是计划里的那个
  变体**。同时这给成本模型上了量级课：计划合计预测 3.9–8.1%，弱化版实测合计 ≤0.83%
  → 预测高估 ~5–10×。

**裁决**：两个单候选 **UPHOLD**（MEASURED、bit 相同、< 门槛）；组合 flag **INCONCLUSIVE**
（轮间方差未进推断，见上）；**kernel 级 conv+norm 融合（FUSION-PLAN 原案）REOPEN-LOW**：
现实预期 ~0.5–1.5%（reference 档，按实测/预测比折算），auto 档受 overlap 饱和约束。
若实施，与 §3 #2 的 reference 档 compile 同窗对照——更少图节点既省 host 建图，也缩小
compiled graph。#3（q_norm+mRoPE）维持不建议。

### 2.2 matmul 级融合（#2）

**实测**（唯一变体：A1 attention k/v 拼接 + A2 GDN z/b/a 合并，reference 档，3 轮全干净）：
median **+0.109%**，CI95 [−0.100, +0.303]，126 对更快占比 53.2%；A/A vs 本仓 base
42/42 决策、max|Δp| **0.0252**（shape 改变 GEMM tiling → 非 bit 一致）；vs 冻结 MLX
42/42、max|Δp| **0.04896 > 0.035 → parity 门失败**。before/after 时间线：window
5038.3→5000.3 ms（−0.75%），GPU busy 39.45→39.19%，overlap 91.36→90.58%，真气泡
5.79→6.43%。

**欠探索点（确认存在）**：
- **A1 单独从未测过 latency，也从未测过 parity**；「A2 是 3→1 个 qmm 只会中性偏正，所以
  A1 单独的上限 ≈ 组合值」是 EXTRAPOLATED。该推理对 latency 方向成立（A1 上限 ≈ +0.1%，
  无论如何 <2%），但对 parity 不成立——A1 的 k/v 都是 64 的倍数，其漂移特征与 A2 不同，
  单独跑有可能过 parity。
- GPU 档 A2（补零到 4224 行的 split GEMM）**只由一次 before/after 录制结构性取证，未经
  latency campaign**。
- 「fused-kernel（把 GEMM 与 epilogue 合成一个 Metal kernel）」与「分组 scale 的其它拼法」
  都没有实现过。

**为何仍维持关闭**：该方向**不删、不移 FLOPs**。§5.1 的天花板分析（layer host 建图 9.2%、
真气泡 5.8%、两引擎并集 94.2%）意味着任何「少几次 dispatch/decode 往返」的融合可动的
最多 ~9% 的 host 固定开销，而实测只拿到 +0.109%（CI 含 0），说明其中大部分不是 dispatch
瓶颈。EP 级 fused GEMM 若不删计算，同样受此天花板约束。**UPHOLD，但理由应写成「天花板
<2%」而不是「变体已试尽」。**

### 2.3 ANE 列再平衡（#3）——最典型的一处误判

**实测与模型**：
- smoke（每臂 4 次单案，ANE trace）：6656 臂 request 2991/3150 ms，7680 臂 4617（冷启动
  离群）/3323 ms；ANE predict/层均值 21.6/22.6 → 20.1/20.0 ms（−7~−9%），但**请求反而
  +8%**（warm 对比）。
- 配平模型：266-token chunk 输入补到 GPU 320 行（64 倍数），ANE 恒跑 512 行 tile。
  6656：GPU ∝ 320×6656 = 2.13M vs ANE ∝ 512×5632 = 2.88M → 0.74；7680：2.46M vs
  2.36M → 1.04（刚过平衡点，每层 wall 改由 GPU 决定）。
- campaign：3 轮 **5/6 臂次 Swapouts≠0**（仅 r2-cand 一次 +0）→ KEEP 池为空，
  **latency 不可测**（runner 输出 `keep: null, insufficient clean rounds`）。
- parity：A/A vs 本仓 base 42/42、0.0255；**base 臂 vs 冻结 MLX 0.0404 > 0.035（auto
  基线自身不过门）**；cand vs MLX 0.0483。

**审计**：
1. latency 结论按纪律应是 **INCONCLUSIVE**，不是 NOT-WORTH。记录本身写的是「latency 不可
   测」，但 PR #5 的实验表与 `HANDOVER.md` 把它归入「已关闭（都有实测证据）」，**分级丢失**。
2. 「默认 6656 就在配平点」是 MODEL-ONLY + 单点 smoke 的组合；模型的单位成本假设、GPU
   补行假设、ANE tile 恒定假设都未独立验证。不过**两侧都有负读数**（更深的 5632/4608 实测
   −2.48%，更浅的 7680 smoke warm +8%），经验上把最优点夹在了中间，所以「7680 不是改进」
   这个窄结论可接受，「6656 是最优点」不可接受。
3. auto 档还有一条独立的门：基线自身 vs MLX 0.0404 已超 0.035 → **auto 上不存在能过的
   vs-MLX KEEP 门**；这应改写为「auto 的 parity 判据需改为 A/A vs auto 基线」（记录里已
   提出，未落地）。

**裁决**：**INCONCLUSIVE（环境阻塞）**。REOPEN 条件：先解决 auto 档 42 案 campaign 的结构性
swap 污染（或改用更小的判定套件 + 单锁窗重试），并把 parity 判据改为 A/A vs auto 基线，
然后扫 {6144, 7168}（6656 两侧各一点）+ ANE tile 尺寸一档。预期上限 ≤2%（配平模型 +
两侧已夹逼），**优先级低**——它是「加速/配平」类，不是「删工作」类。

### 2.4 deeper-ANE / wavefront（#4）

**deeper-ANE**：T1（C=5632）实测 **−2.48%**（更慢）且 parity max|Δp| 0.0361 > 0.035，
MEASURED 双重失败。**T2（C=4608）实现了白名单却从未测量**——DECISIONS D1 明确「Q11
(T2) 跳过；T1 同向恶化风险高，不测」，`ANE-COLUMN-BALANCE.md` §2 的 allow-list
`[6656, 5632, 4608, 7680]` 中 4608 是**实现项不是测量项**。裁决：**T1 UPHOLD**；
**T2 MODEL-ONLY 外推，未闭环**（且外推方向是「更深更差」，与 T1 一致，故不影响
「朝 ANE 加压不行」这条窄结论）。

**wavefront K=2**：
- 证据：W1 ANE trace（42 请求 × 32 层，n=1344）gap median **1.3 ms**（87% ≤ 5 ms，
  3.4% ≥ 30 ms），wait median 17.0 ms，predict median 17.4 ms，`sum_wait ≈ sum_predict
  ≈ 560 ms/请求` → 「重叠已兑现」。
- **该方向在 MEGAKERNEL-PLAN 里是「诊断门控，本轮不做」**，门控条件：
  (i) Q10b 反序复测显示 deeper-ANE 显著更快（推翻 GPU-bound 前提）；或 (ii) 新轨迹里
  gap ≥ 30 ms 尾部占比 > 10%。**两者都未满足**：(i) 的反序复测从未做；(ii) 实测 3.4% <
  10%（但这条读数本身用的是未过 Swap 门的 trace，且 K/tile 只有一组）。
- 欠探索：K 只有 2；tile/列空间未扫；wavefront 计划里的「`async_eval` 提到 predict 之前」
  这个**单独的调度改动**从未消融；「ANE 专用线程」（#7）是它的 ANE 侧半件，也从未做。
  另（第二意见）：旧证据只报了 gap 的**中位数与尾部占比，未算 gap 总和占请求的比例**——
  中位数 1.3 ms 与「3.4% ≥ 30 ms」不能推出总 gap 可忽略；重评 wavefront 前先补这个数。

**裁决**：deeper-ANE UPHOLD；wavefront **INCONCLUSIVE-lean**——在现有诊断下不值得立刻做
（gap 尾部小），但不能以「已关闭、有实测证据」结案。若 #7 的 ANE 专用线程落地，wavefront
自然获得其双缓冲前提，届时应重新评估。

### 2.5 eager eval（#5）

**实测**（唯一变体：每层 `h.add` 后 `async_eval`，末层与 head 各一次）：3 轮全干净（6/6
臂次 Swapouts +0），A/A vs base **bit 相同**（max|Δp| 0.0），parity vs 冻结 MLX 与基线
完全相同（0.0328 PASS），median **+0.115%**，CI95 [−0.038, +0.264] 含 0。

**机制诊断（记录里的缺口）**：chrome 显示 eager=1 时 layer span 238.4→2226.6 ms（+1988），
logits_sync 2506.4→141.8 ms（−2365）——GPU 工作确实被提前了，但**每层 span 正好吸收了
该层的 GPU 执行时长**，host 建图与 GPU 执行仍然串行。记录**明确拒绝裁定**两种归因
（`async_eval` 调用本身阻塞 vs 下一层对 `h` 的消费 op 在等），并指出跨流被 `Array` 单流
约束（`inputs` 检查 `same(stream)`）排除。

**裁决**：对该变体 **UPHOLD**（干净、bit 相同、点估计 ≪ 2%）。未试变体（chunk 粒度、head
粒度、独立线程派发、两个归因的裁定实验）**不改变当前裁决**——因为裁定实验的产出是「为什么
拿不到重叠」，而不是一条新的收益路径；跨流不可行意味着 host/GPU 重叠在该结构下基本无解。
**优先级：低**，除非 MLX 构建或 `Array` 流约束发生变化。

### 2.6 prep 缓存（#6）

**裁决**：**UPHOLD**。硬上界 12.7 ms ÷ 1140 ms 命中窗 = **1.11% < 2%**，且命中时
bit-preserving（A/A 0）。上界不依赖实现变体，属 MEASURED 硬边界；该结论同时修掉了一个
「两臂都跑 cache-off」的 runner bug。无未试变体。（补记：该方向还跑过一轮 3×2 repeat
campaign，median −0.174%，CI [−0.359, +0.214] 含 0，与硬上界一致。）

### 2.7 megakernel（#7）——关闭清单里最大的窟窿

**事实**：从未实现任何 megakernel 变体，全部结论来自 MEGAKERNEL-PLAN 的评估：

| 方案 | 当时判决 | 实际执行 |
| --- | --- | --- |
| A. 全 GPU 持久 megakernel（32 层调度收进一个 kernel） | 不做：上界 2–3%、周级、parity 风险大 | 未实现（MODEL-ONLY 上界） |
| B. wavefront / 双缓冲（K=2 对角调度） | 诊断门控 | 见 §2.4，门控未走完 |
| **C. ANE 专用线程 + 异步下发 + 双缓冲** | **「下 campaign 首选候选」**：直接攻击 ~560 ms/请求的串行 ANE 等待；只改调度 → parity 友好；工程量数天；env 门控可回退 | **从未实现，零证据** |

**裁决**：
- 「全 GPU megakernel 本轮不做」UPHOLD（工程/风险比差，上界贴门槛）。
- **「megakernel 方向关闭」不成立**。方案 C 是当时评估里唯一被点名为首选、且机制上只
  「移动工作到线程、不改数值」的候选；把它和方案 A 一起写进「已关闭（都有实测证据）」是
  分级错误——方案 C 连外推都算不上，是**未探索**。
- 现实天花板：ANE predict 合计 ~560 ms/请求（auto 稳态窗 ~1429 ms）。若其中相当部分已
  被 GPU 执行掩盖（W1 trace 的 `sum_wait ≈ sum_predict` 与 gap 1.3 ms 暗示重叠良好），
  可动部分可能只有尾部；若请求线程在 predict 期间无 GPU 派发（当前 issue order 是
  wait_tiles → async_eval → predict → merge），则上限可接近数百 ms（**10%+ 量级**）。
  实测前无法收敛——这正是需要实现一次的原因。

**REOPEN 优先级：Clef 线第 1**。前置条件见 §3。

### 2.8 mx.compile / vis014 / vision Metal batch（#8）

- **mx.compile vision**：host 图构建实测 **0.7 ms（0.05%）** → 「compile 省 host 建图」
  这条路径 UPHOLD。但 compile 的另一半收益是**设备侧**（图融合、更少 kernel 间隙），0.7 ms
  的 host 测量完全不覆盖；vision Metal batch 的 1.1–1.8% 是设备侧 fusion 的一个点，说明
  设备轴**部分**探索过。裁决：host 轴 UPHOLD；设备轴 EXTRAPOLATED 未闭环，但单独重开的
  预期 <2%，优先级低。
- **vis014 vision row split**：结论来自几何模型——单图 432–600 rows < 1024-row floor，拆
  到两个引擎后每份都低于 floor → 零收益。**MODEL-ONLY，未找到任何实测**。裁决：UPHOLD
  （几何下限论证对「GPU/ANE 行拆分」这一具体设计是结构性的），但应在清单里标注「模型论证、
  零实测」，不得写「有实测证据」。若将来 vision 走 ANE 且单请求多图（3 图 1512 rows > floor），
  该结论需要按新几何重评。
- **vision Metal batch**：实测独立 **1.1–1.8% < 2%**，判 NOT-WORTH。裁决：UPHOLD（MEASURED、
  低于门槛）。备注：这是 Clef 上所有「加速类」候选里点估计最大的一个；它「仅作 vision-ANE
  配套」的定位意味着一旦 vision-ANE 落地应作为子项复测。

## 3. 重开清单（按「预期上限 × 通过门槛概率」排序）

| 排名 | 方向 | 线 | 缺的变体 / 前置 | 机制 | 预期上限 | 通过门槛概率 | 环境前置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **ANE 专用线程 + 异步下发 + 双缓冲**（MEGAKERNEL 方案 C） | Clef | 完整实现（从未做） | 把 ~560 ms/请求的串行 ANE 等待移出请求线程；只改调度 → parity 友好 | 若 predict 未被掩盖：**10%+**；若已掩盖：仅 gap 尾部 ~2–3% | 中（parity 风险低，但上限两极化） | **先解决 auto 档 42 案结构性 Swapouts**，否则无 CLEAN 窗可判 KEEP |
| 2 | **Clef reference 档整骨干 `mx.compile`（prepared compiled callable）** | Clef | 整个 backbone 从未 compile（branch 只对小 `g_beta` 用 `Compiled`） | reference 档每请求重建 ~230 ms 的 lazy 图且 GPU 全程空闲；按 shape 准备一次 compile 后，稳态请求跳过整段 host 建图 → 它与 GPU 执行串行的那部分被删掉 | **~8–9%**（230 ms / ~2500 ms；即 timeline 的 layer host 9.2%） | 中高（reference 档历来干净；parity 需过门：compile 会改设备侧融合顺序） | 无（reference 档双锁单窗可测） |
| 3 | **D3 GDN 内核移植（剩余变体）** | D3 | `gdn_block=fused_prep`、`gemm_dtype=f16`、图路径 `eight_lanes`（建议配 fp32 状态累加） | fused prep 删 stock 链 ~10 个 op 并少 2 次中间 BF16 舍入；f16 攻 gemm 67.8%@η0.93；图路径 eight-lanes 需先补回 parity 头寸 | fused prep 文本增量 **2–4%**（并可能改善 image parity）；f16 **0–7%**（roofline）；图路径 = 文本路径同款增益 | 中（fused prep/f16 均未实现；图路径 parity 已证紧） | 无 |
| 4 | **D3 末层行剪枝**（layer 23 只跑 last row 的 q/out/mlp） | D3 | 实现 + parity 门 | **删工作**：readout 只吃 last position；末层 MLP/out_proj/q_proj 只算 1 行（K/V 仍全行） | **~4%**（Graft：~1/24） | 中（shape 变 → 漂移；但删的是整行 FLOPs） | 无 |
| 5 | **同图视觉前缀复用**（D3 带图请求跳过 vision tower） | D3 | 实现；**traffic preflight** | **删工作**：同图重复问时复用 merged features，跳过 12-block tower | 同图工作负载上 **25–30%**（vision 增量 53–75 ms / 175–218 ms）；唯一图流量 = 0 | 高（命中即与 miss 路径 bit 相同，parity 零风险） | **P4 Mario demo 流量形态未量化**；按「先量化流量再决定」应在 P4 做 preflight |
| 6 | wavefront K 扫描 / `async_eval`-before-`predict` 消融 | Clef | K∈{1.5,3,4}；单点调度消融；**先补 gap 总和占比**（旧证据只有中位数与尾部占比） | 缩短 gap ≥30 ms 尾部（当前 3.4%） | ≤5% | 低（gap 尾部小；auto 不可测） | 同 #1 |
| 7 | ANE 列：T2（C=4608）+ 无 swap 配置下复测 7680 + tile 一档 | Clef | 4608 实现未测；7680 只有污染轮 | 配平模型两侧夹逼内的再配平；7680 的 +8% smoke 读数需要一个干净窗复核 | ≤2%（模型 + 两侧已夹逼）；若复测翻正则重估 | 低–中（第二意见把 7680 列为「最该重开」，与本表的低预期上限存在分歧，见 §7） | 同 #1；且需改 parity 判据为 A/A vs auto 基线 |
| — | matmul A1 单独 / epilogue-fused GEMM / 分组 scale | Clef | 各一个实现 | 少 dispatch，不删 FLOPs | ≤0.3%（受 §2.2 天花板约束） | 低（parity 风险仍在） | 不建议重开 |
| — | fusion #3（q_norm+mRoPE）/ #1+#2+#3 合一 kernel | Clef | 实现（仅估算 ~0.1%） | 少 3 个 gather | ~0.1% | 高但 <2% | 不建议单独重开；若做，按轮间方差评估 |

**Clef 线真正的阻塞不是「候选池干了」，而是「auto 档不可测」**：42 案 campaign 结构性
Swapouts 污染（16 GB 上的换页，与候选无关）+ auto 基线自身 vs-MLX 0.0404 超 0.035 门。
`EAGER-EVAL.md` §3.3 的筛选表已经把五条候选全部因这两个门跳过——**那张表筛选掉的是
「不可测」，不是「不值得」**。下一根治动作（与具体优化无关的基础设施方向）：
1. 给 campaign runner 加「干净窗探测」（短 probe + Swapouts 判定），只在 clean 时启动；
2. 把 auto 的 parity 判据正式改为 A/A vs auto 基线（at-risk 决策与参考构建 bit 一致）；
3. 降低 auto 档常驻内存（42 案逐案流式、缩小 allocator cache 上限）以消除结构性换页；
4. **根因级（本 leg 新证）：给 main 的 `apxinf-mlx` 恢复 `set_cache_limit` FFI，并在
   `metal_d3_lite::load` 按 1 GiB 设置**。showcase 分支的 `bridge.cpp`/`array.rs`/`ffi.rs` 已有
   该 API 且 `metal_clef/mod.rs:166` 以 `CACHE_LIMIT = 1 GiB` 调用，EngineTailor 亦设
   `CACHE_LIMIT=1 GiB`；**main 在重构中丢失了这个 API，`metal_d3_lite` 因而完全不限制 MLX
   allocator cache**。本 leg `sample` 实测：48 案 gate 进程 footprint 涨到 **11.2 GB**
   （P1a 也记录过 gate 轮 +4528 Swapouts）→ 单进程跑完 48 案必然换页。这是 D3 判定窗
   「自我污染」的主要内因，也是 Clef/D3 一系列 INCONCLUSIVE 的共同上游。前置提醒：设置上限
   会改变中间量的 cache 命中与重算，现有 KEEP 数字是在无上限状态下测的，落地前须重跑 campaign。

## 4. D3-lite 线：尚未优化状态的审计与第一批方向

### 4.1 状态审计

P1a/P1b 交付的是**未优化基线**，记录里明确把以下 PROMOTED 点列为「P3 再做」：

| 引擎选项 | reference 的 PROMOTED 值 | native **本 leg 前**的值 | 攻击的成本项（Graft 头寸） |
| --- | --- | --- | --- |
| `conv1d` | `apxinf`（FP32 tap order，单 launch，自带零历史） | stock `Array::conv1d` + 拼接 S−1 行零历史 | conv1d **2.9%** |
| `gdn_core` | `apxinf_eight_lanes` | `apxinf_four_values` | gdn **6.9% @ η0.26** |
| `gdn_block` | `fused_prep`（conv+SiLU+split+l2norm 一个 kernel，中间保持 FP32 → **少 2 次 BF16 舍入**） | separate（约 10 个 op） | gdn + conv + 部分 elementwise |
| `gemm_dtype` | `f16`（F16 权重副本） | BF16 as stored | gemm **67.8% @ η0.93（近 roofline）** |
| `rope` | `apxinf` | `apxinf` ✓ | — |
| `compile` | `model` | `model` ✓ | — |

**关键交叉验证（同机、同 8 timing 案、不同 harness）**：EngineTailor stock 臂 suite 合计
**3529 ms**，promoted 臂 **2808 ms（+25.8%，paired p 2.4e-4，且过 48 案 gate）**；native
合计 **3693 ms** ≈ stock 臂。即 native 目前**没有拿到任何 Graft 已 promote 的加速**。
（注意：跨 harness 的绝对值只作动机，权威判定必须用本仓配对 campaign。）

**parity 头寸**（必须正视）：text gate worst 0.02918 / 门 0.03766（余 0.0085）；image gate
worst 0.0554 / 门 0.06（余 **0.0046**）。任何改 GEMM shape 或 reduction 顺序的变体都必须
逐档过门；好消息是 fused 类（causal conv / fused prep）把中间步骤保持 FP32、每输出只舍入
一次，**相对 FP32 oracle 的漂移预期是减小而不是增大**（ET promoted worst 0.0251 vs stock
0.0232，仍在 2× 预算内）。

### 4.2 第一批方向

#### D3-A（本 leg 实施）：移植 ApxInf causal conv 内核 + eight-lane GDN 递归

**机制预期（测量前写下）**：
- causal conv：stock 路径先拼接 `[1, S−1, C]` 零历史再走 `conv1d`（隐式 im2col 级的额外
  搬运），换成 `apxinf_causal_depthwise_conv`（一线程一输出、FP32 tap order、自带零历史、
  单 launch）→ 删掉零历史拼接的物化与 conv1d 的通用路径开销；每输出舍入一次。
- eight-lanes：把每个 SIMD group 的 4 个 value 行换成 8 个、每 lane 持两行 `Dk/8` 列、
  `simd_shuffle_xor` 归约并预取下一步操作数 → 攻击 Graft 标注 η=0.26 的 gdn 项。
- 两者都是**调度/内核选择**改动，不改权重与算子语义。
- 逐档实现 3 个 arm：base / +conv / +conv+eight_lanes；再视 parity 与收益决定是否追加
  fused prep（Graft 内核，需从 EngineTailor vendor）与 f16 权重副本（独立的 gemm 档）。

**测量方案**：8 timing 案配对（同二进制、两 env gate），3 轮反序、双锁单窗、每臂
Swapouts；48 案 text gate + 8 案 image gate parity（决策 + max|Δp|）；KEEP 门槛 ≥2% +
CI 下界 >0 + parity 通过；before/after 算子时间线（同 signpost recorder，xctrace Metal +
Points of Interest，PNG）。

**预期量级**：gdn 6.9% @ η0.26 项若提速 2–3× → 3.5–4.6%；conv1d 2.9% 若近乎消除 →
~2.5%；叠加后 **~5–7%**。**保守预期 3–6%**；falsifier = 任一档 median <2% 或 CI 含 0 或
parity 越门。

**parity 风险**：中（shape/reduction 顺序改变 → 非 bit 一致，必须过门）；但 fused 类
倾向改善 vs oracle 漂移。

**本 leg 结果**（完整证据与 timeline 见 `doc/d3-lite-metal-gdn.md`）：
- `conv` 单独：**+0.5345 %**（CI [0.3724, 0.9608]，24 对中 83.3 % 更快）→ 低于 2 % 门，
  按 §6 #7 记为**组件级「可叠加」**；且与基线 **bit 相同**（48 案 + 8 案 max|Δp| = 0）。
- `conv` + `eight_lanes`（两个 callable 同用 eight-lanes，出厂前的 bundled 读数）：
  **+4.2714 %**（CI [4.1555, 4.8408]，24/24 对更快，3/3 轮 clean）；text parity **改善**
  （worst 0.02435 < 基线 0.02918，预算 0.03766）；**image parity 失败**
  （0.06390 > 0.06；8/8 决策一致、回执 byte 稳定）。
- 解法：`eight_lanes` 只用于文本 callable，带图 callable 保持 `four_values` → 带图回执与
  P1b **bit 相同**，image gate 通过；文本增益不变。这属于「按 parity 头寸放置变更」，不是
  放宽预算；证伪清单见 §7.3。默认值与单旋钮归因见 D3 报告 §7。
- **before/after 时间线（双臂 Swapouts +0 的干净窗）**：window 1126.4 → 1050.9 ms
  （−75.5，单 take −6.7%），全部来自 `forward` 阶段（−75.4 ms）；真气泡恒为 0、CB 恒 62
  → 机制是「删工作」而非「填气泡/合并 dispatch」；与 campaign 的 +4.27%（CI [4.16, 4.84]）
  同号同机制，幅度差 2.4 pp 属单 take 分辨率（D3 报告 §6）。

#### D3-B：末层行剪枝（删工作）

- **机制**：readout 只吃 last position；末层（layer 23，full attention）之后只有 last row
  被消费 → 该层的 q_proj、out_proj、MLP、post-norm、residual 都只算 1 行；K/V 仍需全行
  （非因果 attention 的 last query 看全 prompt），可把 Q 切成 last row 后用 `sdpa` 的
  单 query 形态。
- **测量**：gate 8 timing 案配对 + 48/8 parity；before/after 时间线。
- **预期**：~4%（Graft ~1/24）；纯删 FLOPs。
- **parity 风险**：中（GEMM shape 变 → 漂移）；falsifier = parity 越门或 <2%。

**本 leg 结果**（完整证据与时间线见 `doc/d3-lite-metal-last-row.md`）：
- 实现 `APXINF_D3_LAST_ROW=full`（现为默认）/`off`/`attn`：末层（layer 23，full attention）
  只算 readout 行 —— 单行 q/gate 投影、单 query 行 SDPA（K/V 仍全 prompt）、单行 o_proj 与
  MLP；K/V 与 0..22 层不变。契约由代码复核（末层之后只有 row-local final norm + last-position
  readout；GDN 层循环不可剪），并用单测钉住位置切片 = `L−1`（不是 0）。
- parity：text **48/48**（worst max|Δp| 0.023195 vs 预算 0.037659；mean KL 2.02e-4 vs
  3.98e-4），image **8/8**（worst 0.055251 vs 预算 0.06，本方向预声明的严格门 0.0575 也过；
  头寸 0.004749，比基线 0.004615 略**宽**）；与 `off` 的最坏逐案漂移 text 0.004889 /
  image 0.001547，无决策翻转；`off` 与改动前回执 **bit 相同**（0.0 / 0.0）。
- latency：3 轮反序配对（2/3 干净轮；round 0 的 `full` 臂 Swapouts +4071 被排除），
  **+4.374 %**，CI95 **[4.124, 4.511]**，16/16 对更快；逐轮 +4.17 / +4.50 %；逐案
  2.31–4.97 %（随 prompt 长度上升）。
- before/after 时间线（双臂 Swapouts +0）：window 1066.1 → 1019.2 ms（−46.9，−4.40 %），
  降幅全部落在 `forward`；GPU busy 99.56 → 98.86 %，真气泡恒 0，CB 62 → 59 → 机制是
  **删 GPU 工作**，不是调度变化。
- 裁决 **KEEP**，默认值翻为 `full`；图像路径未单独计时（预期小），只做了 parity。

#### D3-C：GDN fused prep 移植（vendor Graft 内核）

- **机制**：一个 kernel 完成 conv+SiLU+split+l2norm，中间保持 FP32、每输出一次 BF16
  舍入 → 删掉 stock 链的约 10 个 op 与 2 次中间舍入，**同时**向 FP32 oracle 靠拢。
- **测量**：在 D3-A 的 conv gate 通过后叠加（`gdn_block=fused_prep` 要求 `conv1d=apxinf`）。
- **预期**：与 D3-A 叠加的增量 2–4%；**parity 预期改善**（ET promoted worst 0.0251 vs
  stock 0.0232）。
- **前置**：从 EngineTailor `native/gdn_fused_prep.metal` vendor（MIT，需在 NOTICE 中登记）。

#### D3-D：同图视觉前缀复用（删工作，parity 零风险）

- **机制**：同图重复请求时复用 merged features（命中 = 同一 device array → 编译后的 text
  forward bit 相同），跳过 12-block vision tower + position 插值 + merger。
- **测量**：先做 **traffic preflight**（只读计数同图复用率）；P4 Mario demo 流量形态明确
  前不实现。命中窗上预期 **25–30%**（Mario 裁图 vision 增量 53–75 ms / 175–218 ms）；
  no-hit 控制必须中性（−0.1% 内）。
- **parity**：命中与 miss 路径 bit 相同 → A/A 0 漂移；vs 参考 parity 不变。

#### D3-E：F16 权重副本（gemm 档，最后做）

- **机制**：`gemm_dtype=f16` 的 F16 权重副本让投影走 F16 GEMM；Graft 标注 gemm 67.8%、
  η0.93 → **roofline 只留 ≤7%**，且是唯一可能显著影响 parity 的档（权重值本身变了）。
- **测量**：单独 gate，parity 门预期最难通过（image 余量仅 0.0046）。
- **预期**：0–7%；**falsifier 很可能是 parity**——若失败，说明 native 的 BF16 权重是
  parity 余量的主要来源，该档永久关闭。

### 4.3 D3 交付规则的具体化

D3 目前是**单引擎（GPU-only）**，没有 ANE → 交付规则里的「两引擎 overlap%」需按归因边界
推广：第二轴用 host os_signpost 区间（REQUEST / VISION / FORWARD / SYNC）代替 Clef 的
ANE 窗，四分类写作 `both / gpu-only / host-only / none`，并在 PR 中写明「host 轴，不是
硬件引擎轴」。GPU busy% 仍取 xctrace `metal-gpu-intervals`（M1 Pro compute channel，无
kernel 名）。

## 5. 证据分级总结

**硬证据（MEASURED，可直接引用）**：
- Tier-1 前缀缓存 +8.72%（parity 42/42，max|Δp| 0.0153，Swapouts +0）
- Tier-2 同图行前缀缓存 +45.327%，CI95 [42.799, 47.640]（reference 档；auto 档未测）
- prep 缓存硬上界 1.11%；matmul 融合 +0.109%（CI 含 0）+ parity 0.04896 越门；eager
  eval +0.115%（CI 含 0，bit 相同）；deeper-ANE −2.48% + parity 0.0361 越门；vision
  Metal batch 1.1–1.8%；单 elementwise fusion #1/#2 +0.22%/+0.61%（bit 相同）
- D3 基线：文本 84.5/205.4/453.2/1116.4 ms，48/48 gate（worst 0.02918）；带图 8/8
  （worst 0.0554）、median 216.1 ms、Swapouts 全 0
- EngineTailor 同机：stock 3529 ms → promoted 2808 ms（+25.8%，p 2.4e-4，gate 过）
- 本 leg（D3-A，详见 `doc/d3-lite-metal-gdn.md`）：causal conv 单独 **+0.51%**（CI
  [0.18, 0.85]，bit 相同，<2% 门 → 组件级）；bundled `keep` **+4.27%**（CI [4.16, 4.84]，
  逐轮 4.20–4.86%，24/24 对更快，3/3 轮 clean，全部门通过）；text parity 改善
  （0.02435 vs 0.02918，预算 0.03766）；image parity = P1b 回执 **bit 相同**；单旋钮 `text8`
  latency **INCONCLUSIVE**（demo server 占 ~10 GB，0/3 干净轮）
- 本 leg（D3-B，详见 `doc/d3-lite-metal-last-row.md`）：末层行剪枝 **+4.374 %**（CI95
  [4.124, 4.511]，16/16 对更快，2/3 干净轮，第 3 轮因单臂 Swapouts +4071 被排除）；text
  48/48 worst 0.023195、image 8/8 worst 0.055251（预算 0.06 / 严格门 0.0575，头寸未减少），
  `off` 与改动前回执 bit 相同；时间线 window 1066.1 → 1019.2 ms（−4.40 %），气泡恒 0、
  CB 62 → 59

**外推（EXTRAPOLATED，降级为待验证）**：
- 「A2 只会中性偏正 → A1 单独上限 ≈ 组合值」（latency 方向成立，parity 方向不成立）
- 「默认 6656 就在配平点」（配平模型 + 两侧单点夹逼）
- 「wavefront 重叠已兑现」（单 trace、未过 Swap 门、K=2 单点）

**模型估算（MODEL-ONLY，不能单独关闭方向）**：
- 全 GPU megakernel 上界 2–3%（**从未实现**）
- vis014 行拆分零收益（几何 floor，零实测）
- fusion #3 ~0.1%（从未实现）
- mx.compile 的设备侧效果（host 0.7 ms 不覆盖）
- ANE 专用线程的天花板（从未实现，两极化 2–3% vs 10%+）

**被环境改写的结论（应为 INCONCLUSIVE）**：
- ANE 列再平衡 7680（3 轮 5/6 臂次 Swapouts≠0）
- Tier-2 auto 档、prefix cache auto 档（auto 42 案结构性换页；auto 基线 vs-MLX 0.0404 超门）
- auto 尾部 sync 气泡 113 ms 的归属（记录明说未定因，且 auto 上无能过的 KEEP 门）

## 6. 方法论修正（写给下一根棒）

1. **NOT-WORTH ≠ NOT-MEASURABLE**。实验表必须保留 `inconclusive` 独立列；`keep: null,
   insufficient clean rounds` 不得折叠进 NOT-WORTH。
2. **变体数是裁决的一部分**。「方向关闭」至少需要：(a) 该方向的主导变体实测且失败，或
   (b) 一个与实现无关的硬上界（如 prep 缓存）。单变体实测只能关闭该变体。
3. **诊断门控必须走完**。写下的门控条件（如 wavefront 的 (i)/(ii)）要么执行，要么把结论
   写成「未启动，条件未满足」，不能写成「已关闭」。
4. **「加速类」候选先问天花板**：不删/不移 FLOPs 的候选，天花板 = host 固定开销占比
   （Clef ~9%，真气泡 ~6%）；先算再测（#3 的先例）。**「删工作」类没有这个限制**。
5. **引擎第二轴缺失时要改口径**：D3 GPU-only 的 before/after 时间线用 host 轴推广，写明
   归因边界，不假装有双引擎 overlap。
6. **轮间方差必须进推断**。跨轮池化的 case-level bootstrap **不含轮间方差**；逐轮摆动接近
   点估计量级时该 CI 反保守。至少补报逐轮区间，理想做法是按轮聚类重采样（第二意见 §5.1）。
7. **低于门槛但显著为正的组件标「可叠加」**，不要写成负结果：elementwise #2（+0.61%，
   CI 含 0 之上）与 D3 causal conv（+0.53%，CI [0.37, 0.96]）都属此类——同时保留
   「单独不过 KEEP 门」的记录，避免把它当成已 KEEP 的方向。

## 7. Opus 第二意见（`claude-opus.sh`，只读）

按「每棒最多 2 次」授权实际调用 2 次：

| 次 | 输入 | 参数 | 结果 |
| --- | --- | --- | --- |
| 1 | `opus-brief.md` 全量材料 | 默认 effort（settings 为 `claude-opus-5-5` xhigh） | **失败**：连接建立后上游 ~35 min 无响应，最终 `API Error 524`（网关 `127.0.0.1:3425` 的 120 s Proxy Read Timeout） |
| 2 | `audit-second-opinion.md` 精简 5 问 | `--effort medium`，硬超时 200 s | **成功**：48.3 s，`devlocal/notworth-audit/opus-opinion-2.md`（1997 B） |

复现教训：xhigh + 全量 brief 会超出网关 120 s 读超时；medium + 精简清单可用。第二意见全文
保留在 `devlocal/`，不进本仓。

### 7.1 采纳（已写入本文件）

- 「候选池已干」主要是 **NOT-MEASURABLE 被写成 NOT-WORTH**；站得住的硬 NOT-WORTH 只有
  5 条：elementwise #1、A1+A2、eager eval、prep 缓存、vision Metal batch。与本审计一致，
  措辞并入 §1/§5。
- **wavefront 改判 BLOCKED/INCONCLUSIVE**：用未满足的门控条件关闭是循环论证；且旧证据
  只有 gap 中位数与尾部占比，**未算 gap 总和占比**（§2.4、§3 #6）。
- **7680 降为 NOT-MEASURABLE**，并把「无 swap 配置下复测 7680」列入重开项（§3 #7）。
- **新增重开项：Clef reference 档整骨干 `mx.compile`**。核对 `origin/showcase/clef-vision`
  源码确认：branch 只对小 `g_beta` 用 `Compiled`，整个 backbone 从未 compile；reference 档
  每请求重建 ~230 ms lazy 图且 GPU 全程空闲 → prepared callable 可删掉这段与 GPU 串行的
  host 建图，上限 ~8–9%（§3 #2）。
- 「带图请求 decode 走哪个 callable」的疑问对 D3 **不适用**：每请求单次 forward、无
  decode 阶段；带图与文本是两个独立 compiled callable，split 只动后者（D3 报告 §5/§9）。
- 「D3 +4.37% 只有 8 个 timing 案」的幅度外推限制写入 D3 报告（方向 24/24 对更快可信）。

### 7.2 部分采纳 / 驳回

- **「配对 CI 不含轮间方差 → 组合 flag 结论无效」**：采纳为方法论修正（§2.1、§6 #6），
  并要求本 leg 补报逐轮区间。但不据此翻转 elementwise #1/#2 单候选结论——它们逐轮摆动
  远小于点估计且 bit 相同。
- **「D3 采了单独 +0.53% 的 conv，与 2% 门双标」**：部分采纳。已补测**单旋钮候选** `text8`
  （stock conv + 文本 eight-lanes + 图路径 four_values），但该 campaign **0/3 轮干净**
  （`keep: null, insufficient clean rounds` → INCONCLUSIVE）：测量窗内一台 Mario demo 的
  `apxinf-decide-server` 占用 ~10 GB（6.5 GB 常驻 + 3.4 GB 压缩），每臂 Swapouts +745 k~936 k，
  未杀他人 demo。因此**双标争议保留**：默认值翻的是 bundled `keep` 配置（自身 3/3 轮干净、
  CI [4.16, 4.84]），其 conv 组件另有干净 CI [0.18, 0.85]；`text8` 的 latency 留作下根棒在干净窗
  的第 1 个测量项——若 ≥2%，默认应收窄为 text8（见 D3 报告 §4.1/§7）。
- **「7680 最该重开」**：**分歧，保留本表低排名**。机制上界（配平模型 + 两侧夹逼）
  ≤2%，且需 auto 档（环境前置未解决），上限×概率低于 #2/#3；已在 §3 #7 显式标注分歧。
- **「A1+A2 的 0.0252 未解释来源」**：不采纳——`FUSION-MATMUL.md` §2 已用设备单测归因到
  GEMM tiling 改变（含 rms_norm 的 strided slice 重排）并给出逐项 max|Δ|。
- **「megakernel 2–3% 与 host 9% + 气泡 5.8% 未对账」**：部分成立，并入 §2.7：host 固定
  开销是「加速类」候选的共同上界，megakernel 的 2–3% 估计应显式引用它。
- **「eager eval 的 chrome 测量可能被隐式同步扭曲」**：与 §2.5 记录的两种未裁定归因一致；
  仍判该变体 NOT-WORTH（bit 相同、+0.115%），裁定实验列低优先。

### 7.3 对 D3「按 callable 拆布局」的裁决

第二意见判「**有条件正当**」（两个 callable 本就是独立图；带图回执与 P1b bit 相同；失败
集中在 P1b 已标注最不稳定的同一件事例），并给出证伪条件：(a) 对同案施加其它等精度扰动若
都不越 0.06，则问题在 eight-lanes 本身而非事例噪声；(b) 扩大图像集后失败扩散或误差随
图像 token / 序列长度单调增长 → 递归累积误差；(c) fp64 基准下 eight-lanes 更远；(d) 更长
文本同向漂移。**本 leg 采纳为下一根棒的证伪清单**：vision tower 数值条件改善前图路径保持
four_values；任一条件成立则收窄或撤销 split，而不是放宽预算。

## 8. 产物与复现

- 本文件为维护产物；Clef 负结果原文在 `showcase/clef-vision:docs/mario-opt/`，本仓未复制。
- 审计用的证据快照与 opus brief/回复在 `devlocal/notworth-audit/`（gitignored）。
- D3 本 leg 的实现、campaign 与时间线证据见 `doc/d3-lite-metal-gdn.md`（随实施提交）。
