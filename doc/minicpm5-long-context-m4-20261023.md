# MiniCPM5-2B 长上下文实测，M4，2026-10-23—2026-10-24

普通 BF16 generation 上限为 65792 total tokens，最高档实测为 65536 个输入 token 加 128 个输出 token。本次配对观测中，各档总时间中位数均高于 MLX；具体瓶颈归因仍不完整。上限包含 prompt 和预留 output，不代表允许 65792+128。未改变 BF16 数值边界，也未宣称跨实现 greedy 输出一致。

## 可运行范围与实现

Checkpoint 声明 131072 positions，本实现按 16 GiB 主机预算将普通 BF16 generation 限为 65792 total tokens，支持 65536 输入加最多 256 个预留输出。准备阶段按请求以 256-token 对齐分配 KV 和 RoPE。长 decode 使用有效前缀的 256-token bucket 与绝对位置 mask。容量不超过 4096 时保留原注意力形状。

不超过 4096 tokens 的 prompt 保留原 prefill 形状，更长 prompt 使用最多 512-token 的分块，每块只计算最后一行的词表投影。跨 chunk 的 mask 包含已有 KV 和当前 chunk 的因果前缀。旧 KV 句柄在求值前释放，成功后发布新状态；reset 后可以调整容量。

DSpark 仍单独限制为 4096 total tokens、1–256 output tokens，越界会明确报 DSpark 错误。这次普通 decode 的扩容不扩大 DSpark 的验证范围。

KV 预算是 `2 × 42 × 2 × 128 × 2 = 42 KiB/token`。65536 tokens 的 live KV 为 2.625 GiB；额外 256-token bucket 增加 10.5 MiB，以便实际 64K 输入档预留输出。65792 容量下，RoPE 两张表为 32.125 MiB。权重、allocator cache 和临时张量另计。

**64K 的交付配置要求 `APXINF_MLX_CACHE_LIMIT_BYTES=2147483648`。** 初次默认 cache 策略的 65408+128 请求在 prefill 中达到 12,888,709,912 bytes 物理占用，触发 12 GiB guard 后立即停止。随后诊断进程在 2 GiB cache 下完成，峰值 12,128,098,784 bytes。最终配对表使用正式 worker 和相同 cache 配置重新验证。该选项只设置进程级可回收 cache，不是 active tensors 或物理内存的硬上限。嵌入模型的应用需要自行管理 MLX allocator policy。

## 固定身份与计时方法

实现提交为 `d87e1a2a39dc15058dbd7b1f017e4daeb5e89dc7`，基于 `6ec73d133894b8de9f9f8a84330b5aaeafe1a19d`。分支为 `perf/long-context`。Checkpoint revision 为 `f97400052a43d642bbc6e9975e2397e3ae6a6b52`。

配对测量二进制为 `/Users/haiyan-mini/Agent4Kernel/apxinf-wt/long-context/target/release/apxinf-latency-worker`，SHA-256 为 `6b7029077d1f0c8681317a223c546a8204ed97f07b52b4691b7049287e44b3e6`。报告提交可能晚于实现提交，worker 的 ready identity 仍绑定上述实现。

构建使用 `cargo build --release --features mlx --bin apxinf-latency-worker`，`CARGO_BUILD_JOBS=4`，`CARGO_TARGET_DIR` 指向该 worktree 的 `target`，`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk`。编译持有 `/tmp/enginetailor-metal.lock`，完成即释放。

动态库沿用 MLX 0.32.3。`DYLD_LIBRARY_PATH` 为 `/Users/haiyan-mini/Agent4Kernel/apxinf-wt/mlx-0.32.3/devlocal/mlx-0.32.3/mlx-0.32.3-venv/lib/python3.12/site-packages/mlx/lib`。MLX 对照使用已安装的 MLX 0.32.3、MLX-LM 0.32.0，既有 checkpoint 和 tokenizer，无下载。两边均使用 2 GiB allocator cache。对照通过任务目录内的小 wrapper 在加载前设置 cache，baseline 仓库实现未修改。

ApxInf 使用默认 `bf16-compiled`，显式设置 `APXINF_MLX_CACHE_LIMIT_BYTES=2147483648`。MiniCPM5 的 worker 还必须设置 `APXINF_MODEL_NAME=minicpm5`，因为其 checkpoint 的 `model_type=llama` 不足以自动区分该 family。

机器为 Apple M4、16 GiB unified memory。使用 `mlx-model-baselines latency` 驱动，documents 工作负载，每档五个 block，顺序交替 MLX/ApxInf 与 ApxInf/MLX。每个 arm 独立进程，先一遍 warmup，再一遍 measured request；greedy、固定生成 128 tokens、thinking 关闭。计时包含请求预处理至同步 host output，不含模型加载和输出保存。

测量期间，基线仓库由其他会话从 `4c334dc` 更新至 `54488a4`。Qwen3 前三档 driver 使用前者，之后的 driver 使用后者。差异仅为 guard 补记首尾 swap/disk 观察、latency 模块说明文字，以及未在本任务显式 latency plan 中执行的 context-sweep 环境解析。模型 adapter、计时 benchmark、worker 源码和资源阈值未变。每个 receipt 保留实际来源哈希，核查见 `results/baseline-source-audit.json`；本任务没有修改基线仓库。

固定长度 MLX adapter 实际调用 `generate_step`。该版本在读取当前 token 前异步提交下一步，最后一次 yield 后仍有一步 lookahead，由 `wired_limit` 退出时同步完成。ApxInf 使用 prefill 加 127 次后续 forward。二者输出长度相同，内部提交步数不同；本任务保持现有 baseline，未删去其额外工作。TTFT 因此也受各自提交与同步顺序影响。

每个设备进程同时持有 `/tmp/enginetailor-metal.lock` 和 `~/.cache/enginetailor/metal-measure.lock`。锁以 arm 为粒度，忙时排队。每次 lease 上限 1190 秒，物理进程组上限 12 GiB，磁盘至少 8 GiB。最终扫描以最早基线运行的 5453.19 MiB swap 为固定基线，增长超过 4096 MiB 就停止，避免逐轮重置基线掩盖累计增长。

驱动发现以下 block 有 swapout，不能将全部样本视为无内存压力的比较。污染标记为 {"1024": [0, 2, 3, 4], "4096": [2, 3], "16384": [0, 1, 3, 4], "32768": [0, 1, 2, 3, 4], "65536": [0, 1, 2, 3, 4]}。表中保留全部样本，并列出每档无 swapout 的配对数量；少于五组的子集只作描述。 Pageout 非零的 block 为 {"1024": [0, 1, 2, 3, 4], "4096": [0, 1, 2, 3, 4], "16384": [0, 1, 2, 3, 4], "32768": [0, 1, 2, 3, 4], "65536": [0, 1, 2, 3, 4]}；逐 arm 计数保留在汇总 JSON。

五档累计排队 22.0 分钟，单 arm 最长排队 13.3 分钟。同一配对内两侧之间的最大间隔为 13.3 分钟。排队不计入表内推理时间，但长间隔意味着交替顺序不能消除所有整机状态漂移；结论限于此次观测配置。

## 配对结果

表内为中位数 [最小值, 最大值]。总时间单位为秒。ratio 为同 block 的 ApxInf/MLX 总时间比，中位数并非两个总体中位数的商；大于 1 表示 ApxInf 更慢。所有已统计 warmup 与 measured 请求均成功，输入 token 数正确，输出均为 128。

| 输入 tokens | MLX 总时间 s | ApxInf 总时间 s | 配对总时间比 | 无 swapout 配对数 |
| --- | --- | --- | --- | --- |
| 1024 | 8.26 [7.71, 8.89] | 8.61 [8.19, 8.74] | 1.0495 [0.9693, 1.0666] | 1 |
| 4096 | 13.19 [12.90, 13.46] | 14.72 [14.59, 15.39] | 1.1306 [1.1087, 1.1486] | 3 |
| 16384 | 43.16 [41.85, 43.27] | 47.62 [45.92, 48.14] | 1.1007 [1.0973, 1.1154] | 1 |
| 32768 | 106.47 [105.82, 111.99] | 120.80 [117.83, 126.02] | 1.1131 [1.0786, 1.1909] | 0 |
| 65536 | 344.85 [333.54, 346.93] | 379.75 [375.99, 381.99] | 1.1019 [1.0853, 1.1273] | 0 |

1024 输入档排除 swapout 标记后的 1 组，配对总时间比中位数为 0.9918；4096 输入档排除 swapout 标记后的 3 组，配对总时间比中位数为 1.1430；16384 输入档排除 swapout 标记后的 1 组，配对总时间比中位数为 1.1154。这是样本子集的描述，不补足五组无 swapout 测量。

Prefill 口径为 `prompt_tokens / TTFT`，因此也包含首 token 的请求开销。Decode 口径为 `127 / (total_time - TTFT)`。单位均为 token/s；不是只计某个 kernel 的吞吐率。

| 输入 tokens | MLX prefill | ApxInf prefill | MLX decode | ApxInf decode |
| --- | --- | --- | --- | --- |
| 1024 | 752.1 [699.7, 798.9] | 750.8 [737.2, 780.0] | 18.43 [17.11, 19.76] | 17.52 [17.25, 18.47] |
| 4096 | 683.0 [672.2, 696.9] | 575.0 [559.4, 578.7] | 17.67 [17.24, 18.13] | 16.73 [15.75, 16.94] |
| 16384 | 463.0 [460.3, 476.2] | 424.3 [422.0, 440.1] | 16.56 [16.34, 17.07] | 14.11 [13.63, 14.60] |
| 32768 | 334.0 [318.6, 335.8] | 298.2 [284.5, 306.3] | 15.17 [13.66, 15.48] | 11.70 [11.40, 11.73] |
| 65536 | 196.0 [194.9, 203.4] | 179.9 [178.8, 181.8] | 12.05 [11.17, 12.09] | 8.24 [8.21, 8.25] |

下表的物理峰值取五个进程的最大值，含该进程 warmup。Swap 是整机相对固定基线的累计增长，不能归因于单个 arm。

| 输入 tokens | MLX 物理峰值 GiB | ApxInf 物理峰值 GiB | ApxInf 期间累计 swap 增长 MiB | 双边最大 lease 秒 |
| --- | --- | --- | --- | --- |
| 1024 | 5.403 | 7.238 | 938.81 | 30.6 |
| 4096 | 5.790 | 7.215 | 893.93 | 43.7 |
| 16384 | 6.635 | 7.802 | 960.50 | 108.6 |
| 32768 | 7.816 | 8.922 | 1620.93 | 259.7 |
| 65536 | 9.927 | 10.546 | 2883.06 | 778.6 |

## 数值与边界验证

旧 worker 为未修改的 baseline-interface `6ec73d1`。最终统一交付 worker 的 text-fixed 8 题逐 token 比较全部相等，每题 128 tokens，共 1024/1024。实际短题 prompt 范围为 30–62 tokens；此结果不声称穷举所有短输入。

扩容/RoPE/分块与 attention bucket 两个实现阶段分别保存了 8 题旧/新 token 对比，全部 1024/1024 相等。最终 cache 选项启用后的复验也相等。

CPU 过滤测试合计 24 项通过，其中 `metal_qwen3` 的 16 项包含 Qwen3-TTS 测试，不能全部计作 Qwen3 家族新增测试；MiniCPM5 为 8 项。它们覆盖容量对齐、增长、空输入/加法溢出/上限拒绝、有效长度 mask 和 RoPE 前缀保持。另有 4 项 Qwen3 设备生命周期测试和 1 项 MiniCPM5 真 checkpoint 长分块/bucket 边界测试通过。边界测试记录准备后的 trace count，跨 bucket、reset、扩缩容均不在执行期新增 Rust tracing。

下表比较长扫描的完整 greedy token 序列。首差异位置从 1 开始。每个 block 均直接读取双方 token IDs；同侧 distinct output 数只统计五次 measured request。JSON 另保留从 0 开始的索引、首差异 token ID 和输出摘要。跨实现差异如实保留，没有用文本相似度替代 token 一致性。

| 输入 tokens | 完全相等的配对 | 首差异生成 token | MLX 不同序列数 | ApxInf 不同序列数 |
| --- | --- | --- | --- | --- |
| 1024 | 0/5 | 3 | 1 | 1 |
| 4096 | 5/5 | 无 | 1 | 1 |
| 16384 | 0/5 | 27 | 1 | 1 |
| 32768 | 0/5 | 76 | 1 | 1 |
| 65536 | 0/5 | 69 | 1 | 1 |

## Decode 分项诊断及其限制

原生 Metal System Trace 的两次较长捕获在 finalize 期间触发资源 guard。后续短窗口捕获完成，但 named shader intervals、PC samples、shader registrations 均为空。单步 GPU capture 成功，`gpudebug` 在初始化阶段无输出，受控停止。没有可用的原生逐 kernel 成本或真实 host wait 分解，不能将 Encoding span 或 GPU 空隙冒充同步耗时。

可用替代诊断在 MLX 0.32.3 上使用同 checkpoint BF16 GEMV 权重、实际层数和注意力 geometry，KV 宽度 16640、有效位置 16384。各类操作独立批量执行，2 次 warmup、5 次计时并同步。下表为毫秒。

| 独立组件 | 中位数 ms | 范围 ms |
| --- | --- | --- |
| 所有层及输出 head GEMV | 50.387291 | 48.789291–52.092500 |
| 所有层 SDPA | 11.233834 | 10.302125–11.499584 |
| 空闲 synchronize | 0.015792 | 0.015208–0.017459 |
| 已完成标量 host read | 0.015333 | 0.014709–0.017791 |

这些组件不是完整 compiled decode 的可加总分项。独立输入、调度和同步改变执行方式；空闲 synchronize 仅是调用开销下界，不是模型运行时的等待时间。Qwen 16K 短 trace 的 target GPU compute spans 合并后为 1884.639 ms，窗口 1971.669 ms，约 95.6%；这也不是 shader 利用率或每 token 耗时。

分项诊断中 GEMV 成本高于该 geometry 的 SDPA，未据此改动 BF16 projection、norm 或 RoPE 的舍入边界。长 attention 只改变视图范围和 mask，有效 KV 和权重精度不变。实际端到端结果以上述正式配对为准。

## 证据与交付

维护的实现说明见 [family README](../crates/apxinf-model/src/metal_minicpm5/README.md)。核心选择遵循 Model the Domain，分别表示配置上限、请求容量和有效 attention 长度；Laziness Protocol 促使实现复用既有 MLX SDPA 和编译接口，没有引入新通用 runtime 层。

完整本地证据位于 `devlocal/long-context/`，按仓库规则忽略，不随 Git 提交。关键记录为 `results/paired-summary.json`、`results/paired-minicpm5-*/comparison.json`、各 arm 的 `run.json`/`guard.json` 和 `measured/.../output.json`、`results/minicpm5-handoff-short-parity.json`、`results/tests-long-context-01/` 和 `results/tests-mini-boundary-01/`。原始 GPU capture 用完即删，保留 JSON 归一化结果、源码/原始文件摘要和删除清单。

重算汇总可在此 worktree 运行 `python3 devlocal/long-context/scripts/summarize_pairs.py`。它校验全部请求成功、固定 token 数、两侧 cache 设置、worker commit、物理内存/swap/磁盘/lease guard，再逐序列比较 token。测量计划、驱动 wrapper 和 guard 脚本也保留在同一 feature 目录。

最终将总上限从 65536 增至 65792 的修正，另通过 8 项 MiniCPM5 CPU 测试。它覆盖实际 65536+128 输入、65536+256 的最大接纳、65536+257 的拒绝，以及对应 mask/RoPE 的末端。正式 64K 配对使用 ctx-sweep 原有的 65536-token suite，没有修改该 suite。
