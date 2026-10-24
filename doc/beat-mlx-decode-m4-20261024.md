# Qwen3 与 MiniCPM5 decode 优化，M4，2026-10-24

本任务以 `b2d58ea0656732af69922dc95f75b46cdae0b484` 为参考实现，工作分支为 `perf/beat-mlx-decode`。目标为 Qwen3-0.6B 与 MiniCPM5-2B 各档配对总时间比不超过 MLX 的 1.00。当前尚未完成测量或优化，不作达标声明。

## 实验约定

按 profiling、greedy 流水线、attention 长度、prefill 分块、剩余热点的顺序执行。每个改动分别验证两模型 text-fixed 的 8 题、每题 128 token，与参考实现逐 token 比较。数值差异单列并运行已有数值检查。最终 documents 扫描使用每档 5 block，固定输出 128 token，另有 text-fixed 短输入档。

每次编译和设备运行同时持有两把 Metal 锁，单次最多 1190 秒。物理进程组上限 12 GiB，swap 相对本任务初始值最多增长 4 GiB，磁盘至少剩余 8 GiB。构建使用 4 jobs、MacOSX15.4.sdk 和独立 target。动态库固定 MLX 0.32.3。原始捕获解析后删除，只保留 JSON。

中间证据和待办位于 `devlocal/beat-mlx-decode/`。既有外部 checkpoint、baseline 仓库和长上下文结果原地只读复用。

## 中文进度

- 2026-10-24，已从指定 b2d58ea 建立独立分支和 worktree，记录资源初值并建立同时持有双锁的运行器，尚未执行设备测量。
- 2026-10-24，完成生成路径只读复核，参考版和只加计时的诊断版均构建成功。Qwen3 的 512-token 分块也适用于短输入，MiniCPM5 仅在超过 4096 时分块。
- 2026-10-24，任务初始 swap 为 7483.06 MiB，之后全机增长超过 4096 MiB，guard 在下一次 observer 编译启动前拒绝运行，已释放双锁；此前两次构建 guard 均通过且未观测到 swap 增长。
- 2026-10-24，swap 回落至固定阈值内后恢复设备运行；两模型 b2d58ea text-fixed 参考输出均已保存，每模型 8 题、1024 token，teacher-forced 基线也已完成。

## 数值参考

冻结的 b2d58ea worker 已完成 text-fixed 8 题，每模型 8×128=1024 个 greedy token。后续每个独立修改与这份 JSON 逐 token 比较。已有 teacher-forced 检查同样在 b2d58ea 上复跑：Qwen3 总体 220/222，英文 112/113，中文 108/109；MiniCPM5 总体 430/433，英文 262/262，中文 168/171。两者通过现有 0.98 门槛，后续数值变动还须对照这些基线计数。

## 源码复核

共享循环在 `crates/apxinf-model/src/llm_trait.rs` 先同步 forward，再读取 sampler 结果。MLX backend 已在 GPU 上计算 argmax，`to_u32_scalar()` 随后同步整条 stream。两 family 的 forward 也各自 eval 并同步。bridge 已提供 `async_eval()` 和仅等待指定数组的 `to_bytes_ready()`。

MLX-LM 0.32.0 的 `generate_step` 在读取当前 token 前提交下一步。它默认使用 2048-token prefill 块，块间仍同步 KV。长 attention 的现有 ApxInf 编译路径以 256 对齐宽度和 mask 调用 SDPA。

这些是源码事实，尚不等于各项成本或可获得的加速。旧长上下文报告没有可用的逐 kernel 计时，新诊断将单独记录 host span 和 GPU command buffer span。Apple 定义的 `GPUStartTime`/`GPUEndTime` 是 command buffer 的 GPU 起止时间，须在完成回调后读取；它们不提供逐 kernel 时间。见 [Apple GPUStartTime 文档](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)。

## 改动前的诊断实测

以下是带 host span 和 Metal command-buffer observer 的单次测量，每臂先 warmup 一次，固定 128 输出。它们用于定位，不作为最终性能门禁。Qwen3 上限是 40832 输入加 128 输出，无法提供 64K。

单位 ms/token；GPU 是 command buffer 区间并集，包含 buffer 内间隙，不能称为逐 kernel 时间。eval 包括提交、执行等待，不能单独称为 CPU 等待。sample 包括设备 argmax 和回传。各列存在重叠，不应相加。

| 模型/输入/实现 | GPU CB | forward | eval/等待 | forward 外围 host | sample | command submit CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicpm5-1024-apxinf | 55.696 | 57.798 | 56.313 | 1.499 | 0.714 | 0.859 |
| minicpm5-1024-mlx | 53.425 | — | — | — | 0.004 | 0.813 |
| minicpm5-16384-apxinf | 66.491 | 68.844 | 67.393 | 1.419 | 0.679 | 0.749 |
| minicpm5-16384-mlx | 62.103 | — | — | — | 0.004 | 0.835 |
| minicpm5-65536-apxinf | 115.393 | 117.765 | 116.241 | 1.526 | 0.688 | 1.101 |
| qwen3-1024-apxinf | 16.160 | 16.967 | 16.439 | 0.457 | 0.432 | 0.192 |
| qwen3-1024-mlx | 15.141 | — | — | — | 0.003 | 0.168 |
| qwen3-16384-apxinf | 36.223 | 38.190 | 37.087 | 1.112 | 0.616 | 0.477 |
| qwen3-16384-mlx | 35.189 | — | — | — | 0.004 | 0.547 |
| qwen3-40832-apxinf | 69.024 | 72.053 | 70.033 | 1.689 | 0.677 | 0.832 |
| qwen3-40832-mlx | 68.976 | — | — | — | 0.004 | 0.973 |

推断：流水线可以覆盖一部分 host 图构建和采样等待，但 MiniCPM5 的长上下文 GPU 时间差需要另外处理。Qwen3 40K 两边 GPU CB 区间相近，端到端差不能直接归因于某个 attention kernel。

M4 查询结果不支持 dispatch-boundary counter sampling，只支持 stage boundary。当前记录了实际 kernel 名称和 dispatch 数量；原始 trace 无法给出可靠逐 kernel 时间。将每 command buffer 的 MLX op 上限改为 1 的补充实验仍有多 kernel，属于改变调度后的诊断，未把其 buffer 时间冒充单 kernel 时间。

| 模型/输入/实现 | prefill eval 次数 | eval 合计 ms |
| --- | ---: | ---: |
| minicpm5-1024-apxinf | 1 | 1392.253 |
| minicpm5-1024-mlx | 1 | 1216.152 |
| minicpm5-16384-apxinf | 32 | 38089.535 |
| minicpm5-16384-mlx | 8 | 35119.666 |
| minicpm5-65536-apxinf | 128 | 344268.783 |
| qwen3-1024-apxinf | 3 | 355.289 |
| qwen3-1024-mlx | 1 | 306.624 |
| qwen3-16384-apxinf | 33 | 14463.365 |
| qwen3-16384-mlx | 8 | 14377.983 |
| qwen3-40832-apxinf | 81 | 78764.701 |
| qwen3-40832-mlx | 20 | 76940.652 |

每块 eval 合计包括有效模型计算。块间同步造成多少额外时间，目前不能从这个合计中分离；后续 512/1024/2048 同条件实验决定。

- 2026-10-24，完成 Qwen3 三档和 MiniCPM5 ApxInf 三档诊断；MiniCPM5 MLX 64K 对照仍在运行。流水线按独立数值变更单元实现，attention/chunk 尚未改动。

- 2026-10-24，MiniCPM5 MLX 64K 诊断在第二次请求期间被 swap guard 终止：进程峰值 10.60 GB，任务累计 swap 增长 4205.63 MiB，超过 4096 MiB 门槛。该臂不作为已完成对照。流水线首次构建通过；随后 CPU 测试、补充 kernel 诊断和短上下文配对因全机 swap 超限而拒绝启动，未重设初始基准。

- 2026-10-24，流水线构建成功，4 项 CPU pipeline 测试及 12 项原有生成接口测试通过。Qwen3 text-fixed 8×128 token 与 b2d58ea 全一致；单臂几何汇总延时 1934.77 ms，对照冻结参考 2128.92 ms（非配对诊断，尚非最终性能结论）。MiniCPM5 一致性检查排队中。

- 2026-10-24，MiniCPM5 流水线 text-fixed 8×128 token 也与 b2d58ea 全一致，单臂汇总 6700.14 ms 对参考 7428.95 ms。两模型各减少约 9.1%/9.8% 总耗时，仍需正式配对确认。

## Attention 隔离诊断

在实际模型的 heads/KV heads/layers 几何下，构造固定种子的 BF16 Q/K/V，批量执行各层 SDPA，先 3 次 warmup 再取 7 次中位数。输入视图共用相同完整 KV；这是合成输入的隔离算子测量，不是模型耗时可相加分解，也不是数值质量门禁。进程均通过双锁及资源 guard。

| 模型/输入 | 256 bucket ms | 1024 bucket ms | 精确长度无 mask ms | 精确对 bucket 最大绝对差（第一层） |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 1K | 1.859 | 2.020 | 2.220 | 0 |
| Qwen3 16K | 22.655 | 22.395 | 22.151 | 0 |
| Qwen3 40K | 65.095 | 62.996 | 63.493 | 0 |
| MiniCPM5 1K | 2.079 | 1.895 | 1.780 | 0 |
| MiniCPM5 16K | 10.932 | 11.191 | 8.841 | 1.53e-5 |
| MiniCPM5 64K | 64.016 | 60.058 | 31.570 | 7.63e-6 |

这些请求只预留 128 输出，capacity 以 256 对齐，所以 256 与 1024 bucket 最后都被相同 capacity 截断，两列实际视图宽度相同，其差异是测量波动。MiniCPM5 64K 的无 mask 路径有明显收益，同时 BF16 输出已经变化，不能把它作为无数值影响的调度修改接受。下一步要测完整模型及编译成本，并跑 teacher-forced 门禁。

- 2026-10-24，提交前审查将内部清理参数改成显式 `DecodeFinish`，移除 5 行内部叙述注释；重跑 16 项 CPU/接口测试和两模型各 1024-token 对照均通过。pipeline-v2 的 1K documents 单 block 配对总时间比为 Qwen3 0.9931、MiniCPM5 1.0111；后者有 paging 污染，均仅作阶段诊断，最终仍按 5 blocks 验收。

- 2026-10-24，MLX 64K 的两个单请求重试仍在约 115/125 秒时触发同一 swap 门禁，低水位起跑未解决。停止原样重试，采用“检验共同前提”原则，下一次运行移除 observer 和 host instrumentation，隔离诊断工具是否影响内存。流水线额外通过 MLX greedy 无穷值/NaN/tie 语义测试。
