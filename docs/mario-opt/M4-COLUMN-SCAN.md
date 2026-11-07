# M4 mini：Clef GPU/ANE 列拆分扫描（第一棒）

Issue #41 第一棒：在 M4 mini 上扫描 Clef GPU/ANE 列拆分，找最优点。授权来自
Issue #41 评论 `6703453938`（本 Issue 只由 M4 mini 实例执行；基线数据与二进制
直接用，不重做）。

- 测量机：Apple M4（Mac mini，16 GB，MLX arch `applegpu_g16g`），macOS 27.0.1
  (26A434)，Xcode 27
- 基线构建：`showcase/clef-vision` @ `882448a1`（owner 的
  `~/work-clef-showcase/target/release/apxinf-decide-server`，sha256
  `033cd909…b480fe`）
- 本棒构建：分支 `agent/clef-column-scan-928c4a3c1597` @ `7d5df90c`
  （`882448a1` + 诊断门 + 扫描深度），sha256 `14289a92…c0ce`
- 请求语料：owner 的 `~/test-decide-loop.py`（`test-enemy.png`，320×400，1 个
  `noul` 问句，state 固定）；同图同问
- 纪律：每配置独立进程；2 warmup + 20 measured；3 个独立进程（轮次反序），
  同配置取轮中位数的中位数；EngineTailor 双测量锁；记录 `vm_stat` Swapouts、
  swap 使用量与 loadavg
- 硬件计数：Core AI instrument，`xctrace record --attach <pid>`，15 s 窗口；
  GPU 按 server pid 过滤；ANE 硬件区间（无 pid 列）在锁内静默窗口记录，并保留
  程序标签直方图以识别外来 ANE 活动；overlap = ANE 区间 ∩ GPU 区间

## 0. 结论

1. **M4 的 `auto` 不是列拆分，而是神经元拆分**：`select()` 在 `applegpu_g16g`
   返回 `GpuAneNeurons`（5632 INT8 + 1248 FP16 神经元在 ANE），列拆分只在
   `applegpu_g13s`（M1 Pro）自动启用。owner 的 ~630 ms 基线是神经元拆分；
   `APXINF_CLEF_ANE_COLUMNS` 在 M4 `auto` 下 inert。
2. **列拆分在 M4 上不可竞争**：本批最好列拆分点是 **GPU 列 6144**
   （ANE 跑 `[6144, 12288)`），warm median **571.4 ms**；同机同语料的
   `auto`（神经元拆分）**437.3 ms**，列拆分最好点慢 **30.7 %**；GPU-only
   741.6 ms。
3. **列曲线**：6656 → 781 ms，6144 → **571 ms**，5632 → 593 ms，5120 → 612 ms，
   4608 → 644 ms，4096 → 672 ms，3584 → 676 ms，3072 → 690 ms。拐点在
   C≈6144：再往下把更多列交给 ANE，延迟回升；M1 Pro 默认的 6656 在 M4 上
   明显更慢（三个独立进程一致）。
4. **引擎工作量**（按请求，trace）：auto（神经元拆分）GPU 364 ms、ANE 75 ms，
   引擎下界 364 ms，实测 437 ms；C=6144（列拆分最好点）GPU 513 ms、ANE
   311 ms，引擎下界 513 ms，实测 571 ms。列拆分把 6144 列交给 ANE 的每请求
   ANE 成本（311 ms）是神经元拆分（75 ms）的 **4.2 倍**，GPU 工作量也更高
   （513 vs 364 ms）——列拆分族的引擎下界（513 ms）已高于 auto 实测
   437 ms，所以即使完美重叠、零 host 边，列拆分也追不上 M4 的神经元拆分。
5. **决策一致性**：全部 8 个列深度 + GPU-only 的 `enemy.noul` 概率在
   0.01566–0.01653 之间，max|Δp| vs auto = **0.0006**，决策全部一致
   （`p < 0.5` = 无敌）。
6. **建议**：M4 继续用 `auto`（神经元拆分）。最佳列拆分点
   `APXINF_CLEF_COMPOSITION=gpu-ane-columns` + `APXINF_CLEF_ANE_COLUMNS=6144`
   只作为 env 门控实验配置，不翻默认；M4 的下一步应扫神经元拆分的
   INT8/FP16 份额，而不是列拆分。

## 1. 旋钮确认

- 列拆分旋钮是 `APXINF_CLEF_ANE_COLUMNS`，语义为 **GPU 保留的 intermediate
  列数** `C`：GPU 计算 `[0, C)`，Neural Engine 计算 `[C, 12288)`。未设置时是
  qualified 默认值 `6656`（`GPU_COLUMNS`）。
- M4 的 auto 选 `GpuAneNeurons`（见结论 1），所以列拆分 knob 在 M4 auto 下
  inert；owner 的 `APXINF_CLEF_ANE_COLUMNS` 假设只对 M1 Pro 的 auto 成立。
- 本棒新增 `APXINF_CLEF_COMPOSITION` 诊断门（`reference` / `gpu` /
  `gpu-ane-columns` / `gpu-ane-neurons`）：未设置保持 `auto`；显式
  `LoadOptions::model_variant` 优先；强制组合绕过 architecture 资格门，因此
  只用于测量，默认值不变。
- 列深度从 6656/5632/4608 扩到 6656/6144/5632/5120/4608/4096/3584/3072。
  几何：down 权重按至多 4096 宽分 part，part 宽必须是 64 的倍数；3584 让 ANE
  拿 8704 列，分成 3×2944，尾部 128 列零填充（零行/零列，不参与求和，数值上
  严格无贡献）。6656 仍是唯一 qualified 默认值。
- 「只 GPU」 = `APXINF_CLEF_COMPOSITION=gpu`（GPU composition，无 ANE）。
  「只 ANE」不可行：attention/GDN/投影/head/vision 仍由 GPU/MLX 执行；把 MLP
  的 12288 列全部给 ANE 是另一个组合，未在本棒验证，也仍不是「只 ANE」。

## 2. 方法

- 每次 run：新进程加载 checkpoint（Core ML 程序编译可缓存）→ warmup 2 →
  同一请求连续 20 次；server 自报的 `latency_ms`（覆盖图片解码到概率可用）是
  测量量，另记 wall time。
- 每个配置 3 个独立进程，轮次顺序为正向、反向、正向；统计量取 3 个进程内
  中位数的中位数，并给出 pooled p90 与范围。
- Core AI trace：server 完成 warmup 后 `xctrace --attach`，15 s 窗口内连续发
  请求；导出 `ane-hw-intervals` 与 `metal-gpu-intervals` 表。GPU 只保留
  `process.pid == server pid` 的 command buffer 区间；ANE 表无 pid，锁内静默
  窗口内全部区间视为本进程（标签直方图见产物）。
- 决策一致性：同图同问，比较每个配置的 `enemy.noul` 概率与 auto 基线、与默认
  列拆分 `c6656` 的差 `Δp`；`p > 0.5` 视为同一决策。
- 锁：`/tmp/enginetailor-metal.lock` 与
  `~/.cache/enginetailor/metal-measure.lock`；同机另有 EngineTailor 测量
  campaign 时串行执行。本批等待过 1097 s 与 1359 s 两个窗口。

## 3. 扫描结果

| arm | rounds | warm median ms | pooled p90 ms | noul p | Δp vs auto | GPU busy % | ANE busy % | GPU∩ANE % | GPU-ANE idle % | Swapouts (load) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto (neuron) | 1 | **437.3** | 440.2 | 0.0159 | 0.0000 | 93.9 | 19.3 | 18.7 | 75.2 | 235764 |
| gpu (only GPU) | 3 | 741.6 | 764.7 | 0.0163 | +0.0004 | 95.7 | 0.0 | 0.0 | 95.7 | 0,0,0 |
| c6656 (default) | 2 | 781.0 | 828.2 | 0.0158 | −0.0001 | 75.2 | 38.5 | 28.1 | 47.1 | 11892,32712 |
| **c6144** | 3 | **571.4** | 828.7 | 0.0160 | +0.0001 | 68.7 | 41.6 | 25.2 | 43.4 | 8284,2412,20664 |
| c5632 | 3 | 592.5 | 843.3 | 0.0158 | −0.0001 | 74.4 | 39.1 | 27.1 | 47.3 | 3752,12356,43964 |
| c5120 | 1 | 611.8 | 614.9 | 0.0165 | +0.0006 | 66.7 | 40.9 | 20.4 | 46.2 | 33260 |
| c4608 | 1 | 644.1 | 968.4 | 0.0159 | 0.0000 | 62.2 | 43.3 | 17.8 | 44.4 | 27068 |
| c4096 | 1 | 671.9 | 679.8 | 0.0157 | −0.0002 | n/a‡ | n/a‡ | n/a‡ | n/a‡ | 33360 |
| c3584 | 1 | 675.9 | 680.3 | 0.0160 | +0.0001 | 63.3 | 43.7 | 17.1 | 46.2 | 20664 |
| c3072 | 1 | 690.3 | 710.5 | 0.0160 | +0.0001 | 53.6 | 42.1 | 8.6 | 45.0 | 38420 |

- owner 的 ~630 ms 基线是 24.67 s 窗口（含模型加载与首个冷请求）的平均；
  本报告的 437 ms 是同一请求语料、同一进程排除前 2 个 warmup 后的 steady-state
  median，两者不矛盾（我们的 warm 均值也明显低于 630）。
- auto 只有 1 个干净进程：另一次 auto 运行（15:27）在中断/并发状态下得到
  438.1 ms，两个独立进程一致，故 437 ms 可信；但其 request-window Swapouts
  = 133352，swap 污染无法消除（见 §8）。
- c6144 的 round 1 是离群（818.6 ms，load swap 2412），round 0/2 为
  571.5/571.2 ms；中位数取 571.4 ms。
- c5120 及更深的点只有 1 个进程：每个配置第一次编译后的首载入成功，后续
  重载入被 Core ML placement inspection 拒绝（`Clef runtime placement is
  unknown`），见 §8。每点仍有 20 个 warm 请求，但不满足 3 进程纪律。
- auto 的占用来自本批自己的 auto trace（10.46 s 窗口、27 请求；第三次尝试
  成功）。owner 基线 trace（`/tmp/coreai-clef.trace`，pid 92369）的 ANE 表被
  另一个 Qwen3.5 ANE 实验污染（`mlp_`/`qkv-` 标签 270+ 区间；Clef 的 `tmp*`
  区间只有 18 个），其 16.4 % 不能作为 Clef 值；该 trace 的 GPU 侧（pid 过滤）
  复算为 84.0 %，与本批 auto trace 的 93.9 %（attach-after-warmup、不含加载）
  量级一致。
- ‡ c4096 的 trace 在载入阶段失败（`server exited during warmup 0`，内存压力）；
  其余列点的 trace 成功。
- 6656 的 781 ms 由三个独立进程复现（smoke 782.7、r0 775.5、r1 789.8），
  不是噪声；它比 6144 慢 27 %，说明 M1 Pro 的默认几何在 M4 上不是曲线上的
  平滑点（2816 宽 part vs 3072 宽 part 的 ANE 效率差异是候选解释，未单独验证）。

## 4. 曲线

![column scan](m4-column-scan.png)

- 上：warm median（3 个进程的中位数；误差条为进程中位数的 min–max），
  虚线是 auto 与 GPU-only。
- 下：GPU/ANE 硬件占用与 overlap；虚线是 auto/GPU-only 的占用。
- 曲线在 **6144** 取最小值；6144 → 5632 → 5120 → … → 3072 单调回升。把
  ANE 份额从 6144 继续加大（C 更小）不再降低延迟：ANE 工作量继续上升，而
  GPU 的非 MLP 工作（attention/GDN/head/vision）已经不能再被抵消。

## 5. 最优点与理论下界

- 最优点（列拆分族）：**C=6144**，571.4 ms；相对默认 6656 快 26.9 %，相对
  GPU-only 快 23.0 %，但比 M4 auto（神经元拆分）慢 30.7 %。
- 引擎工作量（trace，按请求）：auto GPU 364 ms、ANE 75 ms；C=6144 GPU
  513 ms、ANE 311 ms；C=6656 GPU 551 ms、ANE 282 ms；C=5632 GPU 502 ms、
  ANE 264 ms；C=3072 GPU 520 ms、ANE 409 ms。
- GPU-only trace：GPU busy 95.7 %（11.51/12.03 s），ANE 0 %，延迟 741.6 ms。
- 列拆分族的理论下界：完美重叠、零 host 串行边时
  `max(GPU_work, ANE_work)` ≈ **513 ms**（C=6144）；实测 571 ms 距下界
  ~58 ms，是 host 串行边（preprocess/encode/head/sync）与重叠损失。
  auto 的同一算法：`max(364, 75)` = 364 ms，实测 437 ms（差 73 ms）——
  两个组成的 host 边量级相同，差距来自引擎工作量本身。
- 关键结论：**列拆分的引擎下界（512 ms）已经高于 auto 的实测 437 ms**，
  所以靠继续配平列拆分不可能追上神经元拆分。auto 的每请求 ANE 工作量在
  owner trace 中不可单独归因（见 §7），但 auto 的实测延迟 437 ms 已经低于
  列拆分族的引擎下界 512 ms，足以证明列拆分追不上。
- 「只 ANE」：不可行（见 §1）；「只 GPU」= 741.6 ms，是列拆分族的另一端。

## 6. 决策一致性

- 8 个列深度 + GPU-only 的 `enemy.noul` 概率：0.0156637（c4096）—
  0.0165296（c5120）；auto 0.0159064。
- max|Δp| vs auto = **0.000623**（c5120）；vs 默认 c6656 = **0.000745**
  （c5120）。所有 arm 决策相同（`p < 0.5`，无敌）。
- 本请求（owner 的 test-enemy 图）决策余量很大，数值差异不构成决策风险；
  42 案 parity/质量门仍是把列拆分变成产品路径的前提，本棒不做。

## 7. 硬件占用 before/after

![occupancy before/after](m4-occupancy-before-after.png)

![timeline before/after](m4-timeline-before-after.png)

- 本批成功记录了 8 条硬件 trace（每个 15 s）：gpu、c6656、c6144、c5632、c5120、
  c4608、c3584、c3072（c4096 的 trace 载入失败）。GPU 区间按 server pid 过滤；
  ANE 用锁内静默窗口，区间数 387–547，≈32×请求数（每请求 32 层 ANE predict），
  支持归属。
- 随 C 减小（更多列给 ANE）：GPU busy 95.7 %（gpu）→ 75.2（6656）→ 68.7
  （6144）→ 74.4（5632，噪声）→ 66.7（5120）→ 62.2（4608）→ 63.3（3584）→
  53.6（3072）；ANE busy 0 → 38.5 → 41.6 → 39.1 → 40.9 → 43.3 → 43.7 → 42.1；
  overlap 0 → 28.1 → 25.2 → 27.1 → 20.4 → 17.8 → 17.1 → **8.6**。GPU 始终是
  更忙的引擎，但 ANE 份额继续加大时延迟回升（ANE 每请求工作量上升，GPU 的
  非 MLP 工作无法再被抵消）；C=3072 的 overlap 掉到 8.6 %，说明再往下 ANE
  已经从「与 GPU 重叠」变成「串行长板」。
- 最佳点 c6144：GPU 68.7 %、ANE 41.6 %、overlap 25.2 %（占 ANE 的 60.7 %）、
  GPU busy 而 ANE idle 43.4 %。
- before/after（本批自己的两条 trace，同请求语料、同 attach-after-warmup 方法）：
  auto GPU 93.9 % / ANE 19.3 % / overlap 18.7 % / GPU busy ANE idle 75.2 %
  （10.46 s、27 请求）→ c6144 GPU 68.7 % / ANE 41.6 % / overlap 25.2 % /
  GPU busy ANE idle 43.4 %（11.9 s、16 请求）。列拆分确实把负载从 GPU 挪到
  ANE（GPU busy −25 个百分点，ANE +22 个百分点），但代价是每请求 ANE 工作量
  从 75 ms 涨到 311 ms，净延迟反而变慢。两张图分别给占用柱状与 2 s 时间线。
- auto trace 的前两次尝试在 16 GB 内存压力下于载入阶段失败，第三次（清空
  cache、单进程）成功；其 request-window Swapouts = 468616（trace 开销），
  硬件区间计数 27×32 与请求数吻合，GPU/ANE 占用不受 swap 影响。

## 8. 风险与归因边界

- **同图重复语料**：复用 owner 基线的同一张 `test-enemy.png`；Tier-1/Tier-2
  image-prefix cache 命中后 vision tower 不在热路径。绝对毫秒不可直接外推到
  42 案混合图；列拆分比较本身作用在 backbone MLP，仍有效。
- **M4 列拆分未 qualification**：只在 M1 Pro `applegpu_g13s` qualified。本棒
  的「最优点」是测量结论，不是默认可用路径；启用需另行过 parity/质量/内存门。
- **Core ML placement inspection 不稳定**：c5120 及更深的配置首次编译后
  首载入成功，后续重载入反复报 `invalid operator or storage input: Clef
  runtime placement is unknown`（placement retries 未救回）；auto（神经元）
  也出现一次 `cannot construct compute plan`。因此深列点只有单进程数据，
  且这些配置在 demo 重启/重载场景下的可靠性存疑。根因未完全定位；内存压力
  （16 GB、swap 已用 2.7–3.1 GB）与 ANE 权重 residency 是首要候选。
- **swap 污染**：auto 延迟 run 的 request 窗口 Swapouts = 133352（trace run
  468616，含 instruments 开销）；列点多为 0–10k，c5632 r0 request delta = 0
  （最干净）。auto 的 437 ms 在 swap 下测得，若换成干净窗口可能略有变化；
  这不影响「auto 明显快于列拆分最好点」的结论（差距 134 ms）。
- **ANE 表无 pid 列**：归属依赖锁内静默窗口；本批 ANE 区间数 488–547，
  ≈32×请求数，与「每请求 32 个 per-layer ANE predict」吻合，支持归属。
  标签直方图在本批只显示通用 `Neural Engine`（程序名未暴露）。
- 16 GB 主机内存紧张：Safari/WindowServer/其它 agent 常驻；本批测得的 loadavg
  ~1.8–2.0，swap 使用 2.7–3.1 GB。绝对延迟对桌面负载敏感，只用于同批相对比较。

## 9. 复现

```bash
export MLX_ROOT="$HOME/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --features mlx,coreml --bin apxinf-decide-server

# 每个配置一个独立进程；C 是 GPU 列数
APXINF_CLEF_COMPOSITION=gpu-ane-columns APXINF_CLEF_ANE_COLUMNS=<C> \
  apxinf-decide-server --model ~/Agent4Kernel/models/clef-flash-4bit-b5adbe2 \
  --model-variant auto
# 只 GPU
APXINF_CLEF_COMPOSITION=gpu apxinf-decide-server --model ... --model-variant auto
```

本棒的请求循环、扫描、trace 与绘图脚本在
`devlocal/clef-column-scan/scripts/`（gitignored 的中间产物；本文件所在的
docs 只保留报告与图）。原始 run JSON、trace 与导出表在
`devlocal/clef-column-scan/{results,traces,logs}/`。

## 10. 交接（Direction / Evidence / Next candidates / Cost）

- **Direction**：M4 的列拆分不是杠杆；M4 `auto`（神经元拆分）以 437 ms 明显
  领先列拆分最好点 571 ms。列拆分最好点 C=6144 只作为 env 门控实验配置保留，
  默认不动。
- **Evidence**：8 个列深度的 warm 曲线（3 进程 × 20 请求；深点为 1 进程）、
  Core AI 硬件占用（auto GPU 93.9 %/ANE 19.3 % vs c6144 68.7 %/41.6 %）、
  决策 max|Δp|=0.0006、GPU-only 741.6 ms、引擎下界：auto 364 ms vs 列拆分
  513 ms。
- **Next candidates**：
  1. **扫 M4 神经元拆分**（`NEURON_INT8` / `NEURON_FP16` 份额，env 门控）——
     这是 M4 auto 的实际杠杆；预期先把每请求 ANE 工作量压到比列拆分低，
     再谈配平。
  2. **解释 6656 的异常**：用 6144/6656 两个几何做 ANE 侧 program 尺寸对比
     （2816 vs 3072 宽 part），确认是否为 ANE 通道对齐效应；若成立，列拆分
     的默认点在 M4 应该重新选。
  3. **placement inspection 稳定性**：深列点的重载入失败需要定位（ANE
     residency / 内存压力），否则任何列拆分配置都不能进 demo。
  4. vision tower / thread 流水线（M1 Pro 的 PR #18 方向）优先级低于 1–3。
- **Cost**：1 棒 ≈ 1.5 h 墙钟（含 ~38 min 锁等待、~20 min 一次中断后的重测），
  8 个 Core ML 编译/删除循环，~3 GB 峰值 cache；Opus 0 次。

Opus: 0 次。
