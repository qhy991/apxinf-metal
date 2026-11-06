# auto 档 Tier-1 前缀缓存 — 严格重测（2026-11-06）

D39 曾判 auto 档 Tier-1 前缀缓存 **NOT-WORTH**（+1.68～2.15%，幅度 <2% KEEP 门；D30 的
+3.273% KEEP 因 runner 统计 bug 撤销）。但该轮所有 arm-process Swapouts ≠ 0（auto 42 案
在 16 GB 上结构性换页），按纪律应为 **INCONCLUSIVE**——复盘审计（main 的
[`doc/notworth-audit.md`](../../doc/notworth-audit.md) §5，PR #21）已把它列入「被环境改写
的结论」。2026-11-06 14:45 起窗口干净（Swapouts 3 s 探测 = 0、磁盘 65 GiB），
按严格口径重测一次。

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1，MLX 0.32.3（`applegpu_g13s`）
- 代码：本 worktree @ `origin/showcase/clef-vision` tip（`agent/notworth-audit-ee3c4e7da150`）
- 工作负载：42 案 `suite/separate/separate-proto14.json`，variant = **auto**（GPU+ANE）
- 时间线图：_（待回填）_

## 1. 协议（测量前声明）

- 3 轮；偶数轮 base→cache，奇数轮 **反序** cache→base；每轮一个双锁单窗
  （`~/.cache/enginetailor/metal-measure.lock` → `/tmp/enginetailor-metal.lock`）
- 每臂 `--warmup`（1 个全量 pass）+ `--measured 1`；每臂 job 内独立进程
- 唯一变化的 env：`APXINF_CLEF_PREFIX_CACHE` base=0 / cache=1。Tier-2 与它同步
  （`off→(0,0)`、`on→(1,1)`），在 42 张唯一图上 Tier-2 恒不命中 → **有效差异 = Tier-1**
- 每臂 `vm_stat` 前后：arm-process Swapouts ≠ 0 → 该轮不进 KEEP 池；另报 bench 侧
  `APXINF_CLEF_ANE_TRACE` 逐层 gap/wait/predict
- KEEP = ≥2 个干净轮 且 median ≥2% 且 bootstrap CI95 下界 >0 且 parity 通过

**parity 两个口径（同时计算，不预设取哪个）**：

| 口径 | 定义 | 背景 |
| --- | --- | --- |
| **A（字面）** | cache 臂与 base 臂各自 vs 冻结 MLX：42/42 决策 + max\|Δp\| ≤ 0.035 | 现行字面规则；auto 基线自身历史读数 0.0404 > 0.035 → **结构性不可过** |
| **B（audit/D39）** | 每轮 cache 臂 vs **自身 auto 基线**：42/42 决策 + max\|Δp\| ≤ 0.035；只池化干净轮 | `ANE-COLUMN-BALANCE.md` §3 与 D39 实际采用；audit §5 推荐 |

两口径给出不同 KEEP 结论时，那是**协议决策**，不是测量结果——如实上交，不在数据出来后选口径。

## 2. 结果

**clean rounds = []（0/3）→ runner 判 `keep: null, "insufficient clean rounds"`（INCONCLUSIVE）。**

arm-process `vm_stat`（每轮一个双锁单窗；r0 正序 base→cache，r1 反序 cache→base，r2 正序）：

| 轮 | 臂序 | Swapouts | Swapins | wall | bench 逐决策 Swapouts | ANE 层组数 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 0 | base → cache | +81,180 | +55,107 | 188.9 s | 脏 | 2,688 |
| 0 |  | +791,138 | +642,882 | 212.4 s | 脏 | 3,392 |
| 1 | cache → base | +18,480 | +21,327 | 177.9 s | 脏 | 3,392 |
| 1 |  | +16,044 | +25,530 | 152.9 s | 脏 | 2,688 |
| 2 | base → cache | **+0** | +2,999 | 155.8 s | **clean** | 2,688 |
| 2 |  | +50,928 | +24,146 | 183.2 s | 脏 | 3,392 |

读数：

- **cache-ON 臂三轮全脏**（+791,138 / +18,480 / +50,928）；cache-OFF 臂 r2 干净（+0），
  r0/r1 脏。→ 换页更像**由 cache-ON 路径的内存形态驱动**，而非纯案量（否则 r2-base 不会
  干净）。但 r1 两臂相近（18k/16k），归因仍需 §2.1 的 N=12 探针收敛。
- cache-ON 臂的 ANE 层组数 3,392 = 106×32，cache-OFF 为 84×32=2,688 → cache-ON 下 84 个
  请求中有 **22 个走了 chunk0 fill + chunk1 双趟**（64 层组），其余 62 个为命中单趟。
- 池化延迟：0 个干净轮 → `gains_vs_base.cache.ci = {n: 0}`，**无 KEEP 可用的 latency 数**。
  污染轮的配对中位（仅作区间参考）逐轮 **−0.569 / +0.280 / +1.859 %**，池化 median
  **+0.402 %**，CI [−1.617, +2.204]，51.6 % 更快 —— 与 D39 的 +1.68~2.15 % 相差一个门
  量级，且轮间摆动 (2.4 pp) 远大于点估计，正是「污染轮不能定案」的活样本。

parity（r0 回执；两口径同时算，均不取巧）：

| 口径 | 臂 | 结果 |
| --- | --- | --- |
| **A 字面** vs 冻结 MLX | base（cache off） | 42/42 决策，max\|Δp\| **0.044935** > 0.035 → **fail** |
| **A 字面** vs 冻结 MLX | cache（cache on） | 42/42 决策，max\|Δp\| **0.040417** > 0.035 → **fail** |
| **B A/A** cache vs 自身 auto 基线 | r0 | 42/42 决策，max\|Δp\| **0.031589** ≤ 0.035 ✓，bit_equal false，worst `f0065-enemy/enemy/true` |

两个口径都拒绝 KEEP：A 对 auto 结构性不可过（连 cache-OFF 的 0.0449 也超门；注意
**cache-ON 反而比 cache-OFF 更贴近 MLX**，0.0404 < 0.0449，因此「为 parity 关掉 cache」
在 auto 上会更糟）；B 的逐轮 A/A 本身干净（决策与数值预算都过），但**没有干净轮可池化
latency**。

## 2.1 N=12 诊断探针（同协议 2 轮）

套件：42 案套件的前 12 案（`f0000/f0009/f0012/f0013` 各 3 问）；其余协议与 §1 完全相同。

| 轮 | 臂序 | base（cache-OFF） | cache（cache-ON） |
| --- | --- | ---: | ---: |
| 0 | base → cache | +508（wall 60.7 s） | **+0**（75.1 s） |
| 1 | cache → base（反序） | **+0**（60.1 s） | **+0**（74.5 s） |

**clean rounds = [1]（1/2）**（runner 仍判 `insufficient clean rounds`，因为要 ≥2；探针只作
归因）。

**结论：换页是案量驱动的。** N=42 的 arm-process Swapouts 是 16k–791k 页（16k×16 KB =
256 MB ~ 12 GB 级），N=12 塌到 **0–508 页（≤ 8 MB）**，低三个数量级；cache-ON 臂在 N=12
两轮全干净（与 N=42 三轮全脏相反），唯一非零读数是 r0-base 的 508 页。因此：

- auto 档不是「结构上永远脏」，而是「42 案的驻留把 16 GB 压穿」；
- 但 exact-zero 门在 N=12 仍会被单次 8 MB 的 blip 卡掉（r0）→ 想在 auto 档拿 KEEP，必须
  **预声明**一个小套件并多跑几轮凑 ≥2 干净轮；
- **本 leg 不据此重判**（把判定套件从 42 换到 12 是协议变更，属请求者决策；且在看到哪个
  套件干净之后再换 = 套件挑选）。42 案 verdict 维持 INCONCLUSIVE（§4）。

## 3. 时间线

**不录制 before/after 算子时间线。** 本轮无可 KEEP 的实测优化（0 干净轮 → 连 latency 数都
没有），属 negative/inconclusive candidate；按本线先例
[`ANE-COLUMN-BALANCE.md`](ANE-COLUMN-BALANCE.md) §1「negative candidate 不做 before/after
录制」处理。时间线是给**已测得的优化**做机制归属用的，给 INCONCLUSIVE 录一张会把单 take
噪声误读成结构差异（参见 `FUSION-MATMUL.md` §4 的单 take 噪声告诫）。若下根棒在**预声明
的新套件**上拿到干净 KEEP，则必须补时间线。

## 4. 裁决

**INCONCLUSIVE**（不是 NOT-WORTH，也不是 KEEP）。

- 环境：0/3 干净轮；cache-ON 臂三轮全脏、cache-OFF 臂 r2 才干净（§2）。starting window
  本身干净（2 s Swapouts 探测 = 0）→ 污染来自测量过程，不是起始状态。
- latency：无干净轮 → 无 KEEP 可用数；污染轮区间 −0.569/+0.280/+1.859 %（池化 +0.402 %，
  CI [−1.617, +2.204]，见 `analysis.json`）。
- parity：口径 A 对 auto 结构性不可过（两臂都 >0.035；cache-OFF 0.0449、cache-ON 0.0404）；
  口径 B 的逐轮 A/A 过门（42/42、max|Δp| 0.03159 ≤ 0.035，三轮读数逐位相同）但无干净轮
  可池化 → 两口径都拒绝 KEEP，`disagreement = false`，**本 leg 不存在需要人裁的口径分歧**。
- 对记录的修正：**D39 的 NOT-WORTH 应改读为 INCONCLUSIVE（环境）**——它的 +1.68~2.15%
  来自全脏轮，幅度本身不可信；本轮在干净起始窗下仍拿不到干净轮，故既不能 KEEP 也不能
  hard NOT-WORTH。代码默认值不变（Tier-1 在 reference 档有独立 KEEP，unset = ON）。
- N=12 探针：1/2 干净轮，swap 从 16k–791k 页塌到 0–508 页 → 换页案量驱动（§2.1）；据此
  的协议建议（预声明小套件）留给请求者，本 leg 不套用。
- **协议开口（留给请求者）**：auto 档是否允许改用**预声明**的小判定套件（如 N=12）？允许则
  auto 候选池（Tier-1 前缀缓存、ANE 列 7680、auto 尾部 sync 等）才可能拿到 KEEP；不允许则
  auto 档在本机只能出 INCONCLUSIVE，应把优化火力全部放在 reference 档与 D3 线（后者本来
  就与 auto 无关）。

## 5. 产物与复现

- 脚本（devlocal）：`devlocal/notworth-audit/clef-campaign/`
  （`run_fusion_campaign.py` / `run_column_campaign.py` + `record_before_after.py` stub +
  `analyze_auto_prefix.py`）。注意 runner 的结果根目录硬编码为
  `devlocal/matmul-fusion/results/<run-root>`，本记录的 receipt 收纳在
  `devlocal/notworth-audit/results/auto-prefix-strict/`。
- 关键修复：复制来的 runner 对 `APXINF_CLEF_PREFIX_CACHE` 存在 **dict 重复键覆盖**
  （`env_key: arm_value` 写在固定 `"APXINF_CLEF_PREFIX_CACHE": "1"` 之前 → 两臂都会跑
  cache-on，与当年 prep-cache 首轮作废同款）。本记录已改为 arm 值最后写入，并在
  `run_meta.json` 落盘 `gate_env` 自证两臂 env 只差 prefix 两键。

```bash
export MLX_ROOT="$HOME/apxinf-runtime/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
cargo build --release -p apxinf --bin apxinf-latency-worker --features mlx,coreml
cd devlocal/notworth-audit/clef-campaign
python3 run_column_campaign.py --run --env-key APXINF_CLEF_PREFIX_CACHE \
  --arms base=0 --arms cache=1 --rounds 3 --warmup --measured 1 \
  --variant auto --suite 42 --run-root auto-prefix-strict
python3 analyze_auto_prefix.py
```

ANE cache（binary-bound，全新目录）：`/tmp/clef-ane-cache-notworth-audit`。首臂（r0-base）负责
编译 ~32 个 program，其 +81,180 含编译期换页；即便豁免该臂，r0-cache 的 +791,138 仍使 r0
不干净，故 0/3 干净轮的判定不受此影响。
