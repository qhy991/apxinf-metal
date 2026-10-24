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

- 2026-10-24，流水线作为独立变更提交到本地 `5306ec3`；两个模型各 1024 token 精确一致，16 项 CPU/接口测试及 greedy 特殊值测试通过。精确 attention 继续作为未接受的实验，采用有界请求宽度预热并单独计入冷准备成本。

- 2026-10-24，无 observer/host instrumentation 的 MLX 64K 对照仍于 240.2 秒触发 swap guard，进程峰值 10.06 GB、累计 swap 增长 4108.25 MiB。诊断包装不是失败的必要条件；不能用该运行补齐 64K 性能结论。全机其他进程的物理占用明显，继续保留原始 swap 基准与 12 GB 门槛。

Attention kernel 名称实测补充：MiniCPM5 16K/64K 的 bucket+mask 使用 `sdpa_vector_2pass_1_bfloat16_t_128_128_boolmask_qnt_nc_nosinks_64`，精确无 mask 切到 `sdpa_vector_2pass_1_gqa_8_bfloat16_t_128_128_nomask_qnt_nc_nosinks_64`；两者都另有 pass-2。Qwen3 对应两路均为普通 two-pass，仅 mask 专项不同。推断 MiniCPM5 的主要收益来自 GQA kernel 选择，不只是少扫最多 255 个位置；已留存逐 case/粒度名称 JSON。

- 2026-10-24，512 MiB allocator cache 的 MLX 64K 控制运行仍在 202.8 秒触发 swap guard，进程峰值 8.24 GB、累计 swap 增长 4241.88 MiB。cache 降低不足以解决当前全机内存约束；64K 配对继续留空。精确 attention 实验已编译通过，开始数值对照。

- 2026-10-24，attention-v1（实验开关开启）在两模型 text-fixed 各 1024 token 上与 b2d58ea 完全一致。MiniCPM5 冷请求汇总约 21.92 秒，流水线基线约 6.81 秒，逐精确宽度预热带来的成本必须继续拆分，不能仅凭稳定 decode 收益推广。

额外的 MiniCPM5 几何合成 attention 使用 CPU float64 稳定 softmax 参考，长度 1025/16385/65537，Q/K 尺度 0.2/1/4。bucket 与 exact 的相对 L2 误差均约 0.14%–0.25%；exact 对 bucket 的误差有增有减，长序列两者最大差随输入尺度可达 0.0078125。此证据只衡量所测合成算子的舍入误差，不能替代模型 teacher 门禁。

- 2026-10-24，attention-v1 的现有 teacher-forced 门禁通过且计数不退化：Qwen3 220/222（en112/113、zh108/109），MiniCPM5 430/433（en262/262、zh168/171），均与冻结基线相同。该短轨迹检查仍不覆盖 64K 的 GQA 舍入变化。

- 2026-10-24，加入锁外资源恢复后，MiniCPM5 16K 三臂单 block 完成且无 swap-out 污染。总时间 MLX41.42s、pipeline45.04s、exact44.27s，比例由1.0873降至1.0689，仍未达标；此为单 block 阶段诊断。exact 冷 warmup64.85s对 pipeline46.08s，须继续解决准备成本。

16K 同 block 的实测 decode 为 pipeline15.34、exact16.88、MLX17.58 token/s；prefill 约445.75/445.87/479.15 token/s。pipeline 与 exact 的128-token 输出从下标59（第60个）开始分歧，共68个位置不同。该差异已单独保存，不能标为长文本精确一致；现有短 teacher 门禁通过不等于证明长文本轨迹不变。

- 2026-10-24，attention-v2 采用首尾 profile 实际执行、中间 profile 仅验证计算图复用。两模型 text-fixed 各1024 token仍全一致；teacher计数保持220/222与430/433。Mini冷短请求汇总降至7.081s，16题teacher准备合计由53.344s降至5.898s（基线0.747s）。2项CPU准备语义测试通过，真实Mini4609-token prompt覆盖单行尾块、同capacity输出预算2→3→0、越界拒绝/Invalidated及执行不重trace。中间长度的设备JIT/分配仍计入首次请求，未宣称全Metal预热。

MLX 64K 另有已完成首个 warmup 的局部诊断（同一进程后来在第二个请求触发 guard，不能作为完成的配对性能数据）。该首请求 GPU command-buffer 区间并集为82.91ms/token，command submit CPU0.860ms/token，scalar read中位0.003625ms；32次prefill eval合计320.65s。它支持长decode GPU差距这一方向，但热身状态与完整运行失败的限制必须保留。

- 2026-10-24，attention-v2 的16K单block配对总时间比为 exact/pipeline 1.0005，decode由15.27增至17.01 token/s；冷请求44.86s对43.09s。追加的基线轨迹teacher对照为125/128，未通过该诊断设置的98%阈值；它衡量对b2d轨迹的一致性，不是外部质量teacher，原有两模型数值门禁仍无计数退化。64K对照继续运行。

- 2026-10-24，64K精确attention的基线轨迹teacher检查完成，127/128（99.22%）通过该诊断98%阈值，物理峰值10.51GiB、累计swap增长1053.50MiB，均在门禁内。16K/64K逐步差异另存JSON，不能称长输出逐token一致；64K MLX配对性能仍缺。

- 2026-10-24，4K精确attention单block总时间比 exact/pipeline 为0.9982，收益不足以抵消短请求准备风险。先将此路径以默认关闭的实验模式独立提交，后续默认选择等待prefill和最终矩阵；未将16K基线轨迹97.66%描述为通过。接口注释审查无删除项，澄清端点/内部profile预热范围；模型边界检查通过。

Prefill补充隔离诊断：MiniCPM5几何、42层独立随机BF16输入，在16K/64K有效KV、512/1024/2048 query rows下，causal SDPA相对显式bool mask的GPU调用墙钟中位比例为16384/512 0.9694, 16384/1024 0.9664, 16384/2048 0.9248, 65536/512 0.9701, 65536/1024 0.9650, 65536/2048 0.9422。所测输出逐元素一致，但这不是完整模型或质量门禁。mask构造在计时外，峰值同时保留两种路径对照，不能据此直接推断模型内存节省。将其列入分块实验之后的候选。

Qwen3 bucket准备成本补测（独立新模型、每档单次，不含完整prefill/decode）：256→1024使1K/16K/40K准备时间由28.51/311.29/1418.17ms变为27.82/96.03/394.92ms。40K graph trace数170→50、eval数360→120，准备后MLX活跃内存约减少10.41MiB。这些固定128输出档的最终decode视图相同，未宣称稳定decode加速；实验代码恢复为256，二进制和精确diff留存。Mini同档256/1024粒度的准备宽度集合也相同，只有exact需要逐宽度图profile。

- 2026-10-24，prefill-v1三种分块共享一个冻结二进制。2048开关下，两模型各1024-token短文本仍与b2d全一致，2项CPU计划测试通过，现有teacher保持Qwen220/222、Mini430/433。真实Mini4097 prompt的单行尾块、同capacity输出2→3→0、越界拒绝及无新trace检查通过。独立源码审查未发现新增正确性/兼容性问题。

16K单block分块比较（均128输出、warmup1）：Qwen512/1024/2048的TTFT为14.016/14.198/14.065s，物理峰值4.69/4.84/5.11GiB，128输出全一致。Mini相应TTFT为36.485/36.675/37.110s，物理峰值8.16/8.28/8.36GiB；1024与512输出全一致，2048在下标91首次分歧，共24/128处不同。两模型当前均保留512默认；40K/64K尚在测量。

其他热点隔离诊断：Qwen BF16激活、FP32 norm权重下，原有compiled RMSNorm单步调用墙钟约106–159μs，MLX fast RMSNorm约83–122μs；2048行大输入有6/2097152个BF16元素不同（最大绝对差0.0078125），仍需完整模型验证。greedy的第二次max归约改为gather在Mini词表上反而变慢，暂不采用；这些孤立墙钟数据不能直接相加预测模型收益。

- 2026-10-24，Qwen40K单block分块比较完成。512/1024/2048的TTFT为74.355/73.410/72.288s，总时间83.790/82.823/81.752s，物理峰值7.59/7.80/8.33GiB，128输出全一致。2048为长档候选，16K无相同收益，默认值尚未切换；Mini64K继续运行。

Mini unweighted fast RMSNorm隔离探针保留先归一化到BF16、再乘BF16权重的顺序，所测输入全一致。512/1024/2048/4096行的调用墙钟由393/723/1267/2380μs降至207/335/664/874μs，单行127→132μs未受益。只考虑替换prefill局部norm，decode的packed residual norm保持原实现。这仍不是完整模型收益或数值门禁。

- 2026-10-24，Qwen fast RMSNorm候选的短文本门禁出现分歧，history-zh/code-zh/code-en/retrieve-en分别有20/16/116/55个位置不同，首处分歧下标108/27/11/65。共207/1024处，不能称为逐token等价优化；现有teacher和完整模型性能检查继续，暂不接受此候选。

- 2026-10-24，Mini64K三种分块全部完成。512/1024/2048的TTFT为336.153/330.468/332.888s，总时间346.901/341.343/344.049s，物理峰值10.31/10.24/10.79GiB，累计swap增长923/1431/1905MiB，均在门禁内。1024与512的128输出全一致；2048从下标86开始分歧，共41处。该block有paging污染，不能据1.6%的差距确定默认赢家，默认仍为512。

- 2026-10-24，prefill实验的计划、准备与执行使用同一分块范围，512/1024/2048由构造时私有开关选择。短文本两模型1024 token全一致，现有teacher不退化，2项CPU计划测试及实际4097-token尾块/重准备检查通过。此次仅提交可复现实验的分块选择，默认值不变，长BF16差异及单block限制保留。

- 2026-10-24，Qwen fast RMSNorm的现有teacher为221/222（en113/113、zh108/109），高于基线220/222；Mini保持430/433，tiny public/compiled轨迹检查通过。Qwen1K单block候选/原路径总时间比0.9957，差距很小；16K/40K继续测量，尚不作收益结论。
