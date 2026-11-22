# GPU/ANE MLP 列拆分再配平（shallower ANE：6656 → 7680）

任务 3（自主选向）的候选记录。授权：PR #5 评论
`c2797f63ba9ca92530d33c441b474f68`——"由你基于 timeline 证据自主选择、
自主实施"，同一套 3 轮反序 / 双锁单窗 / Swapouts 门 / 42 案 parity /
before-after 时间线规则。

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1，MLX arch `applegpu_g13s`
- 涉及 gate：`APXINF_CLEF_ANE_COLUMNS`（GPU 的 intermediate 列份额；
  ANE 跑 `[columns, 12288)`，层输出 = 两半 down 的**和**）
- **不做 before/after 录制**：negative candidate（smokes + ANE_TRACE +
  campaign + §4 bubble analysis 已足以定案），且 auto tier 的 swap 污染
  使单窗对比本身不可靠。§4 的气泡剖面来自既有 base 录制重算。

## 1. 选向与预期收益机制

时间线（[TIMELINE-OPERATOR.md](TIMELINE-OPERATOR.md) + 本 ticket 的
before 录制）给出的结构：

- ANE 调用窗是长板：take-2 里 `ane_predict` 46.3 % + `ane_wait` 38.3 %
  的窗口占比；layer host 图构建只有 9.2 %；
- **engine-only 段很大**：take-2 约 1.2 s、before 录制约 2.8 s 的 ANE
  窗口内**没有**本进程的 GPU 命令缓冲 → GPU 有余量；
- 已关闭的 **deeper-ANE**（GPU 列 5632/4608，即把更多列给 ANE）是
  **−2.48 % + parity 门失败** → 朝 ANE 加压不行。

**机制主张（测量前写下）**：不删 FLOPs，把列从 ANE 挪到空闲 GPU——
`ane_predict`（最大单项）变短，多出来的 GPU share 尽量藏进剩下的 ANE
窗内；INT8 面积变小，parity 大概率改善。若预期收益 <2 % 就按新规写清
"为什么仍值得试"，否则跳过。

## 2. 实现

`crates/apxinf-model/src/metal_clef/ane.rs`：

- `ANE_DEEPER_COLUMNS` → `ANE_QUALIFIED_COLUMNS`，列表
  `[6656, 5632, 4608, 7680]`（前三个是既往 campaign 的深度，含已关闭的
  deeper 两个；7680 是本次候选），error 文案同步；
- `GPU_COLUMNS = 6656` 默认**不动**（KEEP 才翻默认值）；
- 其余零改动：`ane_split_geometry(7680)` = ANE 4608、1 part、part width
  4608（64- 与 128-倍数 ✓），`gpu_share_with_columns` 与
  `program_with_columns` 都按既有约束走。

测量用 worker 是带这段临时改动的 release 构建（14:55）。

## 3. 预检（smoke，4 次单案）

新程序几何会在 load 时量化+编译，campaign 之前必须先确认它能加载、
能跑——这一步同时给出了方向的初步读数。auto tier，每臂一个 Metal 锁窗
口，`APXINF_CLEF_ANE_TRACE=1`（每层一行 `gap/wait/predict_ms`，写入
`results/smoke-column-*/worker-stderr.log`；env 查询与 thread-local 读取
开销可忽略，stdout/stderr 是文件不会堵）：

| 臂 | 列 | request ms | predict_ms/层 mean | median | max |
| --- | --- | --- | --- | --- | --- |
| base 1 | 6656 | 2991.3 | 21.56 | 20.84 | 32.55 |
| base 2 | 6656 | 3149.8 | 22.64 | 23.21 | 30.62 |
| cand 1 | 7680 | 4617.2 | 20.05 | 22.41 | 26.39 |
| cand 2 | 7680 | 3323.1 | 19.96 | 21.61 | 27.53 |

读数：ANE 阶段均值 −7～−9 %（新几何编译并跑通 ✓），但**请求反而更慢**
（warm 那次 +8 %；cand 1 的 4617 是新程序首次使用的冷启动离群）。

**为什么**（按层工作量模型）：GPU share 对 266-token chunk 的输入补到
320 行（64 倍数），ANE 恒跑 512 行 tile；单位成本相近时

- base：GPU ∝ 320×6656 = 2.13 M，ANE ∝ 512×5632 = 2.88 M → 比值
  **0.74**，已经接近平衡；
- 7680：GPU ∝ 2.46 M，ANE ∝ 2.36 M → 比值 **1.04**，刚过平衡点。

过平衡点后每层 wall 由 GPU share 决定，多算的 GPU 比 ANE 少算的更贵 →
净变慢。再叠加 ANE predict 的**每次调用 ~20 ms 固定延迟**（两档 median
几乎相同、只有 base 的尾部更高），可动的计算部分本来就小。这与已关闭的
deeper-ANE 负结果合起来说明：**默认 6656 就在配平点附近，两侧都更差。**

3 轮**全部**被 Swapouts 污染（5/6 臂次 Swapouts ≠ 0，只有 r2-cand 一次
+0）→ KEEP 池为空，latency **不可测**（verdict = `keep: null,
"insufficient clean rounds"`）。逐轮墙钟：r0 base/cand 103.5/89.2 s、
r1 cand/base 87.8/101.0 s、r2 base/cand 102.1/92.2 s——**不可跨污染臂
比较**；这是 auto tier 在 16 GB 机器上的结构性换页（与 HANDOVER 对
Tier-2 auto 的结论同款，不是本候选独有）。

parity（与 swap 无关，是有效证据）：

- **A/A vs 本仓 base**（r0）：42/42 决策相同，max|Δp| = **0.025513**
  （`f0012-enemy/enemy/true`）→ 与 §3 一致：候选非 bit 一致；
- **base 臂本身** vs 冻结 MLX：42/42 决策，max|Δp| = **0.040417 > 0.035
  → auto 基线就不通过**（ANE INT8 误差把参考组合的 ~0.0328 顶了上去）
  → **auto tier 在"vs MLX ≤ 0.035"这条门下无解**；auto 候选的 parity 门
  应改为 A/A vs auto 基线（HANDOVER 本就是用"at-risk 决策与参考构建
  bit 一致"给 auto 定性的），reference tier 才有可过的 vs-MLX 门；
- **cand vs MLX**：max|Δp| = **0.048346 > 0.035 → 失败**（比 auto 基线
  再差 0.0079，与 §3 的配平分析一致：多算的 GPU 份额不划算）。

## 4. 气泡分析（clue b：300 ms 真气泡在哪）

用扩展后的 `timeline_metrics.py` 在既有 base 录制上重算（不需要新录制），
true bubble = **291.7 ms（5.79 %）**，按时间十分位：

| 0–10 % | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 69.8 | 5.6 | 17.2 | 6.3 | 2.2 | 3.0 | 4.4 | 18.5 | 10.1 | **154.7** |

- **开头 68.8 ms**（区间 0.0–68.8）：preprocess → encode → vision host
  串行，在第一个 `ane_wait` 之前，两引擎都没事干；
- **结尾**：`head` host 建图只有 0.4 ms，但 `sync`（`to_f32_vec` 的
  stream 同步 + 读回）**161.7 ms**，其中 **4876.1–4989.6 的 113.5 ms
  两引擎全空**，之后是 9 个 ~0.5–4 ms 的小命令缓冲、彼此隔 ~10 ms。

两种读法都与数据相容：(a) softmax+读回链的 MLX 建图/提交在 host 上串行；
(b) 我们的小 CB 排在桌面其他进程的 GPU 工作后面（`metal-gpu-intervals`
只含本进程，看不到别人的占用）。代码里已有一条"不做 eager eval，重算
~0.1–1 ms"的旧注释，与 113 ms 的实测**冲突**——在用证据判定归属之前，
不按旧注释动手，也不凭这条时间线就改尾部调度。

在 steady-state（chunk0 命中、window ~1429 ms）的 auto 请求上，这 113 ms
≈ **7.9 %**、开头 69 ms ≈ **4.8 %**，所以 clue (b) 仍是下一阶段最有价值
的方向——冷 fill 窗口把两段都摊薄了，看起来才只有 5.8 %。

## 5. 结论与下一步

**NOT-WORTH**（不进主线）。依据：latency 门不可测（swap 污染）、parity
相对 auto 基线退化且 vs MLX 失败、§3 预检 warm +8 %。allow-list 的 7680
已回退——`ane.rs` 与 9aa0a8f0 **逐字节一致**，本方向的维护产物只有本文；
ANE_TRACE 预检与 campaign 原始记录留在 `devlocal/matmul-fusion/`。

clue (c)（decode M=1 kernel 选择）**不适用**：`decide` 每请求只跑一次
prefill 式 forward，没有自回归生成，不存在 decode 热路径。

## 6. 归因边界

与 [TIMELINE-OPERATOR.md](TIMELINE-OPERATOR.md) §5、
[FUSION-MATMUL.md](FUSION-MATMUL.md) §6 相同：GPU = 硬件 command buffer
级（无 kernel 名）、算子标签 = host os_signpost、ANE = 软件轴 wall-clock
（Core AI instrument 本机禁用）；overlap/bubble 都是 host 窗口求交。

## 7. 产物与复现

维护产物：本文（negative candidate，无 timeline PNG；gate 的 7680 增项
已回退，`ane.rs` 与 9aa0a8f0 逐字节一致）。

中间产物（`devlocal/matmul-fusion/`，gitignored）：
`results/column-7680/summary.json`（campaign）、
`results/smoke-column-{6656,7680}/`（预检 + ANE trace）、
`results/fusion-{before,after}.trace`、`results/timeline-*/`（录制）、
`logs/`、`scripts/`。

```bash
export MLX_ROOT="$HOME/apxinf-runtime/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
cargo build --release --features mlx,coreml --bin apxinf-latency-worker
python3 devlocal/matmul-fusion/scripts/smoke_column_depth.py --columns 7680 --run
python3 devlocal/matmul-fusion/scripts/run_column_campaign.py --run \
  --arms base=6656 --arms cand=7680 --rounds 3 --measured 1 --variant auto \
  --run-root column-7680
```
（7680 的 allow-list 增项是测量期临时代码；campaign 结束后已回退，所以
重跑预检前需要先把 7680 加回 `ANE_DEEPER_COLUMNS`——见 §2 的说明。）
