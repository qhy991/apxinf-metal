# Qwen3-0.6B 长上下文实测，M4，2026-10-23

普通 BF16 generation 上限为 40960 total tokens，最高档实测为 40832 个输入 token 加 128 个输出 token。本次配对观测中，各档总时间中位数均高于 MLX；具体瓶颈归因仍不完整。上限包含 prompt 和预留 output，不代表允许 40960+128。未改变 BF16 数值边界，也未宣称跨实现 greedy 输出一致。

## 可运行范围与实现

Checkpoint 上限为 40960。准备阶段按实际请求分配 KV 和 RoPE，容量按 256 对齐并受 checkpoint 上限约束。新请求 reset 后可增大或缩小容量。长 decode 的 K/V 视图只覆盖有效前缀向上取整的 256-token bucket，mask 排除未写入位置。容量不超过 2048 时保留原注意力形状。准备阶段预热所有可调用的一行 bucket，容量变化时清除旧的准备标记。

Prefill 保留最多 512-token 的分块顺序，最后一行计算词表输出。长 generation 不准备完整 `[prompt,vocabulary]` 输出。KV 更新前消费旧状态句柄，使底层可复用缓冲区；计算成功后才发布新状态。失败使当前请求失效。

16 GiB 主机上的 KV 预算是 `2 × 28 × 8 × 128 × 2 = 112 KiB/token`。40960 tokens 的 live KV 为 4.375 GiB，RoPE 两张表为 20 MiB，共享不可变 reset buffer 为 80 MiB。权重、allocator cache 和临时张量另计。内存上限由物理占用 guard 实测，不能只用 KV 字节数保证。

## 固定身份与计时方法

实现提交为 `fc77d6a4aa40a8b777aeaea23691d7bdee41a3e5`，基于 `6ec73d133894b8de9f9f8a84330b5aaeafe1a19d`。分支为 `perf/long-context`。Checkpoint revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。

配对测量二进制为 `/Users/haiyan-mini/Agent4Kernel/apxinf-wt/long-context/devlocal/long-context/bin/qwen3-fc77d6a-worker`，SHA-256 为 `7fdf00256c84893d6e2b1080f606a991fcf1d1f9175e48b9795fb363ecb2173f`。报告提交可能晚于实现提交，worker 的 ready identity 仍绑定上述实现。

构建使用 `cargo build --release --features mlx --bin apxinf-latency-worker`，`CARGO_BUILD_JOBS=4`，`CARGO_TARGET_DIR` 指向该 worktree 的 `target`，`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk`。编译持有 `/tmp/enginetailor-metal.lock`，完成即释放。

动态库沿用 MLX 0.32.3。`DYLD_LIBRARY_PATH` 为 `/Users/haiyan-mini/Agent4Kernel/apxinf-wt/mlx-0.32.3/devlocal/mlx-0.32.3/mlx-0.32.3-venv/lib/python3.12/site-packages/mlx/lib`。MLX 对照使用已安装的 MLX 0.32.3、MLX-LM 0.32.0，既有 checkpoint 和 tokenizer，无下载。两边均使用 2 GiB allocator cache。对照通过任务目录内的小 wrapper 在加载前设置 cache，baseline 仓库实现未修改。

ApxInf 使用默认 `bf16-compiled`，显式设置 `APXINF_MLX_CACHE_LIMIT_BYTES=2147483648`。MiniCPM5 的 worker 还必须设置 `APXINF_MODEL_NAME=minicpm5`，因为其 checkpoint 的 `model_type=llama` 不足以自动区分该 family。

机器为 Apple M4、16 GiB unified memory。使用 `mlx-model-baselines latency` 驱动，documents 工作负载，每档五个 block，顺序交替 MLX/ApxInf 与 ApxInf/MLX。每个 arm 独立进程，先一遍 warmup，再一遍 measured request；greedy、固定生成 128 tokens、thinking 关闭。计时包含请求预处理至同步 host output，不含模型加载和输出保存。

测量期间，基线仓库由其他会话从 `4c334dc` 更新至 `54488a4`。Qwen3 前三档 driver 使用前者，之后的 driver 使用后者。差异仅为 guard 补记首尾 swap/disk 观察、latency 模块说明文字，以及未在本任务显式 latency plan 中执行的 context-sweep 环境解析。模型 adapter、计时 benchmark、worker 源码和资源阈值未变。每个 receipt 保留实际来源哈希，核查见 `results/baseline-source-audit.json`；本任务没有修改基线仓库。

固定长度 MLX adapter 实际调用 `generate_step`。该版本在读取当前 token 前异步提交下一步，最后一次 yield 后仍有一步 lookahead，由 `wired_limit` 退出时同步完成。ApxInf 使用 prefill 加 127 次后续 forward。二者输出长度相同，内部提交步数不同；本任务保持现有 baseline，未删去其额外工作。TTFT 因此也受各自提交与同步顺序影响。

每个设备进程同时持有 `/tmp/enginetailor-metal.lock` 和 `~/.cache/enginetailor/metal-measure.lock`。锁以 arm 为粒度，忙时排队。每次 lease 上限 1190 秒，物理进程组上限 12 GiB，磁盘至少 8 GiB。最终扫描以最早基线运行的 5453.19 MiB swap 为固定基线，增长超过 4096 MiB 就停止，避免逐轮重置基线掩盖累计增长。

驱动发现以下 block 有 swapout，不能将全部样本视为无内存压力的比较。污染标记为 {"40832": [0]}。表中保留全部样本，并列出每档无 swapout 的配对数量；少于五组的子集只作描述。 Pageout 非零的 block 为 {"16384": [3], "32768": [0, 1, 2, 3, 4], "40832": [0, 1, 2, 3, 4]}；逐 arm 计数保留在汇总 JSON。

五档累计排队 123.8 分钟，单 arm 最长排队 32.3 分钟。同一配对内两侧之间的最大间隔为 20.4 分钟。排队不计入表内推理时间，但长间隔意味着交替顺序不能消除所有整机状态漂移；结论限于此次观测配置。

## 配对结果

表内为中位数 [最小值, 最大值]。总时间单位为秒。ratio 为同 block 的 ApxInf/MLX 总时间比，中位数并非两个总体中位数的商；大于 1 表示 ApxInf 更慢。所有已统计 warmup 与 measured 请求均成功，输入 token 数正确，输出均为 128。

| 输入 tokens | MLX 总时间 s | ApxInf 总时间 s | 配对总时间比 | 无 swapout 配对数 |
| --- | --- | --- | --- | --- |
| 1024 | 2.25 [2.24, 2.26] | 2.52 [2.51, 2.55] | 1.1223 [1.1148, 1.1310] | 5 |
| 4096 | 4.20 [4.16, 4.21] | 4.63 [4.60, 4.66] | 1.1043 [1.0999, 1.1105] | 5 |
| 16384 | 19.19 [18.52, 19.37] | 20.03 [19.45, 20.13] | 1.0396 [1.0307, 1.0509] | 5 |
| 32768 | 57.04 [56.93, 57.14] | 58.88 [58.72, 59.31] | 1.0315 [1.0305, 1.0394] | 5 |
| 40832 | 80.01 [79.78, 83.47] | 82.91 [82.88, 85.78] | 1.0363 [1.0276, 1.0393] | 4 |

40832 输入档排除 swapout 标记后的 4 组，配对总时间比中位数为 1.0370。这是样本子集的描述，不补足五组无 swapout 测量。

Prefill 口径为 `prompt_tokens / TTFT`，因此也包含首 token 的请求开销。Decode 口径为 `127 / (total_time - TTFT)`。单位均为 token/s；不是只计某个 kernel 的吞吐率。

| 输入 tokens | MLX prefill | ApxInf prefill | MLX decode | ApxInf decode |
| --- | --- | --- | --- | --- |
| 1024 | 3005.7 [2962.5, 3070.9] | 2775.8 [2764.3, 2821.7] | 66.48 [66.44, 67.02] | 59.01 [58.32, 59.18] |
| 4096 | 2320.4 [2283.6, 2347.6] | 2175.5 [2149.3, 2198.6] | 52.59 [51.74, 53.00] | 46.21 [45.90, 46.82] |
| 16384 | 1103.1 [1090.6, 1161.2] | 1082.5 [1070.5, 1119.1] | 29.24 [28.82, 29.51] | 26.19 [25.92, 26.43] |
| 32768 | 652.5 [651.2, 653.7] | 638.7 [637.6, 640.4] | 18.65 [18.48, 18.68] | 16.81 [16.05, 16.86] |
| 40832 | 566.4 [541.4, 567.7] | 549.8 [531.1, 550.2] | 16.04 [15.75, 16.19] | 14.62 [14.28, 14.70] |

下表的物理峰值取五个进程的最大值，含该进程 warmup。Swap 是整机相对固定基线的累计增长，不能归因于单个 arm。

| 输入 tokens | MLX 物理峰值 GiB | ApxInf 物理峰值 GiB | ApxInf 期间累计 swap 增长 MiB | 双边最大 lease 秒 |
| --- | --- | --- | --- | --- |
| 1024 | 2.113 | 2.937 | 726.37 | 7.8 |
| 4096 | 2.650 | 3.215 | 534.31 | 12.3 |
| 16384 | 4.982 | 4.364 | 1219.31 | 43.4 |
| 32768 | 7.402 | 6.211 | 491.25 | 124.3 |
| 40832 | 8.218 | 7.124 | 410.87 | 174.8 |

## 数值与边界验证

旧 worker 为未修改的 baseline-interface `6ec73d1`。最终统一交付 worker 的 text-fixed 8 题逐 token 比较全部相等，每题 128 tokens，共 1024/1024。实际短题 prompt 范围为 30–60 tokens；此结果不声称穷举所有短输入。

扩容/RoPE、旧 KV 生命周期、attention bucket 三个实现阶段均分别保存了 8 题旧/新 token 对比，全部 1024/1024 相等。最终 cache 选项启用后的复验也相等。

CPU 过滤测试合计 24 项通过，其中 `metal_qwen3` 的 16 项包含 Qwen3-TTS 测试，不能全部计作 Qwen3 家族新增测试；MiniCPM5 为 8 项。它们覆盖容量对齐、增长、空输入/加法溢出/上限拒绝、有效长度 mask 和 RoPE 前缀保持。另有 4 项 Qwen3 设备生命周期测试和 1 项 MiniCPM5 真 checkpoint 长分块/bucket 边界测试通过。边界测试记录准备后的 trace count，跨 bucket、reset、扩缩容均不在执行期新增 Rust tracing。

下表比较长扫描的完整 greedy token 序列。首差异位置从 1 开始。每个 block 均直接读取双方 token IDs；同侧 distinct output 数只统计五次 measured request。JSON 另保留从 0 开始的索引、首差异 token ID 和输出摘要。跨实现差异如实保留，没有用文本相似度替代 token 一致性。

| 输入 tokens | 完全相等的配对 | 首差异生成 token | MLX 不同序列数 | ApxInf 不同序列数 |
| --- | --- | --- | --- | --- |
| 1024 | 0/5 | 67 | 1 | 1 |
| 4096 | 0/5 | 33 | 1 | 1 |
| 16384 | 0/5 | 32 | 1 | 1 |
| 32768 | 5/5 | 无 | 1 | 1 |
| 40832 | 5/5 | 无 | 1 | 1 |

## Decode 分项诊断及其限制

原生 Metal System Trace 的两次较长捕获在 finalize 期间触发资源 guard。后续短窗口捕获完成，但 named shader intervals、PC samples、shader registrations 均为空。单步 GPU capture 成功，`gpudebug` 在初始化阶段无输出，受控停止。没有可用的原生逐 kernel 成本或真实 host wait 分解，不能将 Encoding span 或 GPU 空隙冒充同步耗时。

可用替代诊断在 MLX 0.32.3 上使用同 checkpoint BF16 GEMV 权重、实际层数和注意力 geometry，KV 宽度 16640、有效位置 16384。各类操作独立批量执行，2 次 warmup、5 次计时并同步。下表为毫秒。

| 独立组件 | 中位数 ms | 范围 ms |
| --- | --- | --- |
| 所有层及输出 head GEMV | 13.370375 | 12.835833–13.444375 |
| 所有层 SDPA | 21.572625 | 20.847000–22.369083 |
| 空闲 synchronize | 0.014375 | 0.013666–0.015583 |
| 已完成标量 host read | 0.014042 | 0.013084–0.017542 |

这些组件不是完整 compiled decode 的可加总分项。独立输入、调度和同步改变执行方式；空闲 synchronize 仅是调用开销下界，不是模型运行时的等待时间。Qwen 16K 短 trace 的 target GPU compute spans 合并后为 1884.639 ms，窗口 1971.669 ms，约 95.6%；这也不是 shader 利用率或每 token 耗时。

分项诊断支持减少长 attention 扫描范围。原始 KV 句柄寿命也阻止了缓冲复用，因此一并调整所有权。剩余总时间差距不能据此归因于某个具体 kernel。CPU 源码核查发现 ApxInf 在模型输出和 greedy sampler 之间多一个同步求值边界，MLX 使用异步提交和 lookahead；合并边界需要保留状态发布和失败语义，未把它作为未经验证的晚期修改。更大 prefill chunk、Q/K norm-RoPE fusion、移除 decode mask 都没有本次可接受的端到端收益证据。

## 证据与交付

维护的实现说明见 [family README](../crates/apxinf-model/src/metal_qwen3/README.md)。核心选择遵循 Model the Domain，分别表示配置上限、请求容量和有效 attention 长度；Laziness Protocol 促使实现复用既有 MLX SDPA 和编译接口，没有引入新通用 runtime 层。

完整本地证据位于 `devlocal/long-context/`，按仓库规则忽略，不随 Git 提交。关键记录为 `results/paired-summary.json`、`results/paired-qwen3-*/comparison.json`、各 arm 的 `run.json`/`guard.json` 和 `measured/.../output.json`、`results/qwen3-handoff-short-parity.json`、`results/tests-long-context-01/` 和 `results/tests-mini-boundary-01/`。原始 GPU capture 用完即删，保留 JSON 归一化结果、源码/原始文件摘要和删除清单。

重算汇总可在此 worktree 运行 `python3 devlocal/long-context/scripts/summarize_pairs.py`。它校验全部请求成功、固定 token 数、两侧 cache 设置、worker commit、物理内存/swap/磁盘/lease guard，再逐序列比较 token。测量计划、驱动 wrapper 和 guard 脚本也保留在同一 feature 目录。

统一交付 worker 位于 `/Users/haiyan-mini/Agent4Kernel/apxinf-wt/long-context/target/release/apxinf-latency-worker`，提交 `d87e1a2a39dc15058dbd7b1f017e4daeb5e89dc7`，SHA-256 `6b7029077d1f0c8681317a223c546a8204ed97f07b52b4691b7049287e44b3e6`。它在上述性能构建之后只修正 MiniCPM5 的 64K 输入预留空间，Qwen3、MLX bridge 和 worker 源码未改变。Qwen3 的五档配对始终使用原始 SHA 固定的构建；统一交付构建另做了 text-fixed 8 题逐 token 复验。保留两份构建身份以避免把旧构建计时冒充新构建实测。
