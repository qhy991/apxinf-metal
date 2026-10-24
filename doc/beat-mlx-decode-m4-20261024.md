# Qwen3 与 MiniCPM5 decode 优化，M4，2026-10-24

本任务以 `b2d58ea0656732af69922dc95f75b46cdae0b484` 为参考实现，工作分支为 `perf/beat-mlx-decode`。目标为 Qwen3-0.6B 与 MiniCPM5-2B 各档配对总时间比不超过 MLX 的 1.00。组合版本 `5b3b6d2` 通过完整 64K 冷请求的严格内存门禁及数值/状态验证，但该版本矩阵已完成的五档中四档仍慢于 MLX。后续 Qwen 精确长度和多行表旋转已分别作为默认关闭的优化本地提交为 `88c6c7a`、`18c6f22`；Mini 最后一层 prefill 缩行已通过独立门禁并作为默认关闭改动保留，多行表旋转因完整请求收益不足撤回。统一候选的完整矩阵尚未完成，不作全档达标声明。

| 已完成的组合版本五 block 档位 | b2d/MLX 总时间比中位 | 5b3b6d2/MLX 总时间比中位 | 结论 |
| --- | ---: | ---: | --- |
| Qwen3 1K | 1.135101 | 1.006257 | 未达标 |
| Qwen3 4K | 1.119805 | 1.005949 | 未达标 |
| MiniCPM5 1K | 1.066054 | 0.993079 | 中位数达标，4/5 块更快 |
| MiniCPM5 4K | 1.139739 | 1.015020 | 未达标 |
| MiniCPM5 16K | 1.098915 | 1.010775 | 未达标 |

以上三臂均用 128 MiB allocator cache、128 输出，五块均无分页污染。该版本其余七档未收齐。Qwen 16K 的旧中断运行只有三块，不计入表中。后续候选结果另列，不覆盖这些失败记录。Qwen 精确长度候选的 16K/40K 五块总时间比中位为 0.994341/0.989877；多行表旋转候选的 4K 为 0.992206。这些属于不同阶段，不能拼接成统一版本的最终矩阵。

## 实验约定

按 profiling、greedy 流水线、attention 长度、prefill 分块、剩余热点的顺序执行。每个改动分别验证两模型 text-fixed 的 8 题、每题 128 token，与参考实现逐 token 比较。数值差异单列并运行已有数值检查。最终 documents 扫描使用每档 5 block，固定输出 128 token，另有 text-fixed 短输入档。

每次编译和设备运行同时持有两把 Metal 锁，单次最多 1190 秒。后续物理进程组上限按字面 12 GB，即 12,000,000,000 bytes 执行。早期 12 GiB 设置保留在各原始 guard JSON，超过 12 GB 的旧实验不作合规证据。swap 相对本任务初始值最多增长 4 GiB，磁盘至少剩余 8 GiB。构建使用 4 jobs、MacOSX15.4.sdk 和独立 target。动态库固定 MLX 0.32.3。原始捕获解析后删除，只保留 JSON。

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

当前 CommandLineTools 环境的 `xcrun --find xctrace` 返回72，未安装可用的 Instruments CLI。因此本次完整模型的GPU时间仍以command-buffer区间为粒度，逐算子探针另列；没有用估算填充逐kernel实测。

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

Prefill补充隔离诊断：MiniCPM5几何、42层独立随机BF16输入，在16K/64K有效KV、512/1024/2048 query rows下，causal SDPA相对显式bool mask的GPU调用墙钟中位比例为16384/512 0.9694, 16384/1024 0.9664, 16384/2048 0.9248, 65536/512 0.9701, 65536/1024 0.9650, 65536/2048 0.9422。探针仅对每组第一层输出作逐元素比较，结果一致；其余41层只计时，没有作数值比较。这不是完整模型或质量门禁。mask构造在计时外，峰值同时保留两种路径对照，不能据此直接推断模型内存节省。将其列入分块实验之后的候选。

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

- 2026-10-24，分块选择以`1bf1af5`提交到本地分支，未推送。Qwen RMSNorm的16K单block总时间比为1.0011，无明确收益；40K尚在运行。Mini causal prefill开始独立构建/数值验证，Public与DSpark保留原mask路径。

- 2026-10-24，Qwen1K的5-block四臂比较完成且全部无paging污染。b2d58ea、流水线/ordered norm、fast norm相对MLX的总时间比中位数为1.1251、1.0914、1.0875，decode分别59.03、61.42、61.36 token/s，MLX66.99。fast norm未改善decode，16K单block也无收益；40K单block比例1.0162且被paging污染。该候选不进入当前源码，数值差异和冻结二进制保留。短文本5-block补充仍在完成。

- 2026-10-24，Mini causal prefill候选在text-fixed1024 token上与b2d58ea完全一致，现有teacher保持430/433且各语言计数不退化。该阶段Qwen仍带此前norm实验，复现207处短token差异及221/222 teacher；不是causal改动导致的Qwen差异。随后已恢复Qwen ordered norm，独立验证Mini unweighted prefill norm。

- 2026-10-24，Qwen现有Q/K norm-RoPE融合核的28层几何探针完成。批量调用墙钟中位0.948→0.530ms，全部86016个Q/K元素一致。此为合成算子证据，不是模型速度或质量门禁；下一步仅评估其在BF16 compiled decode中的复用。

- 2026-10-24，Mini causal prefill的16K单block对照无paging污染，TTFT37.094→36.358s，总时间44.659→44.035s，比例0.9860，128输出一致。1K/4K比例0.9669/0.8754，但两档均有paging污染，不据此宣称稳定收益。结合独立算子和数值证据，接受该局部实现，最终收益仍由5-block矩阵决定。

- 2026-10-24，Mini unweighted prefill norm候选的两模型各1024-token短文本全一致，teacher保持Qwen220/222、Mini430/433且各语言不退化，独立源码审查无问题。自动attention策略的3项CPU边界/override测试通过；实际模型同capacity切换及失败恢复仍待运行。

- 2026-10-24，causal prefill独立提交为`c1b4722`。Mini norm的16K单block无paging污染，TTFT36.557→36.161s，总时间44.298→44.000s，128输出一致。收益约0.7%，尚待最终配对确认。其1K/4K单block比例0.9871/1.0498均有paging污染，不能忽略反向结果。

- 2026-10-24，加入causal与unweighted norm后，Mini16K的512/1024/2048分块TTFT为37.057/35.701/35.406s，总时间44.994/43.560/43.174s。该block有paging污染；1024输出一致，2048仍在下标91首次分歧，共24/128处。64K重测继续，默认分块尚未切换。

- 2026-10-24，补充源码复核确认Mini非末prefill块仍同步求值随后丢弃的logits，MLX中间块只求值KV。开始单独核对cache-only的状态边界并量化尾部开销；尚未改变该执行路径。Qwen投影合并只作隔离实验，完整性能未宣称受益。

- 2026-10-24，causal+norm候选的64K分块重测完成，512/1024/2048的TTFT为337.231/324.262/339.937s，总时间351.871/341.545/363.178s；物理峰值10.47/10.37/10.33GiB，累计swap增长2520/2882/2757MiB，全部在上限内但均有paging污染。1024输出全一致，2048仍有41/128处分歧。该次decode降至8.67/7.35/5.46token/s，不能作为分块导致decode变化的证据。请求结束实测active约7.90GB、allocator cache约2.14GB；对照源码MLX会在块间clear_cache，下一步单独检查暂存内存释放的影响。

- 2026-10-24，Mini unweighted norm独立提交为`4c19950`，提交中的模型源码与已过门禁的mini-norm-v1哈希完全一致。自动attention的真实模型五次同capacity模式/预算切换、重复prepare不重trace、越界拒绝、非法capacity及非零offset变更后的恢复全部通过；显式exact的output2→3→0和一行尾块也通过。Q/K fusion两模型各1024短token全一致，teacher仍为220/222与430/433，完整模型配对开始。

补充隔离结果：Qwen Q/K/V和gate/up分组合并的28层单行调用墙钟比约0.9944/0.9948，512/2048行也仅约0.9865–1.0039，所测输出逐元素全一致。没有足够收益证据支持新增权重布局与接口，暂不集成。Mini合成末层加最终行head在16K有效KV、512/1024/2048行时，求值全输出为40.78/74.46/143.02ms，只求值KV为1.09/1.95/2.52ms；64K为109.54/214.97/414.83ms对2.36/2.92/3.68ms，KV全一致。这是单末层随机输入的调用墙钟，不能相加充当完整模型收益；支持进一步做非末块cache-only实验。

- 2026-10-24，自动attention策略独立提交为`fc35ffe`。Q/K融合五个干净1K block全部慢于原路径，原路径/融合相对MLX总时间比中位数1.0911/1.1041；融合相对原路径中位1.0119，已撤回源码和能力说明，保留冻结二进制及数值证据。Qwen短文本ordered路径五block相对MLX中位1.0490，目标尚未达到。

- 2026-10-24，提高command buffer上限160/640在Qwen1K三个干净block的总时间比为1.0717/1.1206，均更慢；Mini中位1.0053/1.0087，且仅一个干净block，没有稳定收益，保留默认。Qwen4K分块2048相对512在三个干净block均较快，中位0.9914；1K无明确分块收益。

- 2026-10-24，Mini cache-only修改前的实际8192/8193 prompt、512/1024/2048分块、4097单行尾块零预算和Public对照共八组摘要完成，保存完整logits字节SHA及后续decode token。非法prompt与过时position拒绝不破坏状态，越宽执行失效及reset重准备恢复均通过。准备按独立设计裁决只裁剪Compiled非末块的求值根。

- 2026-10-24，Mini cache-only两模型各1024短token全一致，teacher保持220/222、430/433，各语言无退化；八组实际长输入的所有logits SHA、后续decode token及状态断言与修改前一致。独立源码及注释审查无问题，未新增注释或抑制。16K/64K同1024分块性能对照开始。

当前流水线1K再profile（真实完整请求，含observer开销，非最终配对成绩）：

|模型/路径|graph ms/token|async提交含等待 ms/token|token读取 ms/token|GPU CB union ms/token|host commit ms/token|CB/token|
|---|---:|---:|---:|---:|---:|---:|
|qwen3/ordered|0.3217|16.0732|0.0018|16.6023|0.1875|55.0079|
|qwen3/qk|0.3965|15.8191|0.0019|16.4826|0.1687|36.0000|
|qwen3/mlx|1.4782|13.2012|0.0032|14.8438|0.1712|26.9055|
|minicpm5/ordered|1.5237|48.5336|0.0029|50.1390|0.6057|84.3071|
|minicpm5/mlx|2.1730|47.2527|0.0040|50.1950|0.5669|84.2677|

实测token读取已降至微秒级，主要主机阻塞在async_eval内部；不能将其视为纯命令提交CPU成本。各列区间重叠，不可相加，GPU CB union不是逐kernel时间。Q/K融合将CB/token从55降至36，但五block完整性能仍不受益，已拒绝。此次observer能记录真实kernel名称；Qwen原路径出现F32 cast/reduce，MLX使用rmsbfloat16。日志交错按完整嵌套JSON无损重组，全部256次token读取和254次graph/submit保留，原始SHA及重组次数保存后删除raw。

MLX 0.32.3源码确认RMSNorm按输入和权重共同类型提升，FP32权重会将BF16输入提升为FP32。这解释此前weighted-fast实验与MLX native BF16路径仍有差别，属于源码事实；是否影响模型性能需另测。[pinned fast.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/fast.cpp#L75)。首轮合成dtype探针中BF16权重路径约四分之一元素有舍入差异，不能当作数值等价替换。

- 2026-10-24，Mini cache-only的16K同1024分块配对无paging污染，TTFT35.765→34.821s，总时间43.673→42.480s，比例0.9727，128输出一致。完整模型支持接受此局部裁剪；64K及最终五block仍在后续验证，不能把单block当作全档达标。模块边界和diff检查通过。

- 2026-10-24，cache-only以`170d67b`独立本地提交。下一阶段只在Compiled长prefill块求值完成后清allocator cache，含末块；源码独立审查确认live KV仍有owner，失败会在position发布前失效。构建完成，数值和长摘要检查排队，尚未声称内存或速度收益。

MLX async_eval在同一调用中执行图遍历与Metal编码，活跃任务超限时会等待GPU队列释放，支持将上述submit span解释为包含设备背压的墙钟，而非纯CPU开销。[pinned transforms.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/transforms.cpp#L260)。当前两模型的KV更新均为logits依赖，单checked-token根在源码上覆盖完整状态更新；去掉额外KV根只可能改变遍历/生命周期，性能与donation效果仍待独立实验。
## 后续进度补记

- 2026-10-24，MiniCPM5 cache-only 的 64K 单 block 配对完成。总时间 341.741 → 336.129 秒，比值 0.983577；TTFT 330.741 → 325.132 秒，decode 11.5448 → 11.5488 tok/s；128 token 全同。两臂物理峰值分别 10.15 / 11.29 GiB，固定基线 swap 增长均为 0，但驱动将该 block 标为污染。该数值只作观察，不计为清洁性能证据。JSON 位于 `devlocal/beat-mlx-decode/results/mini-cache-v1-minicpm5-65536-pair/metrics-summary.json`。
- 2026-10-24，Qwen BF16 native RMSNorm 的独立设计比较完成，选择加载时准备单一精度权重并复用现有 norm 接口。尚未修改产品实现；现有 tiny、逐 token 差异披露和分语言 teacher 门禁保持不变。
- 2026-10-24，Mini clear 候选的两模型 text-fixed 各 1024 token 全同，teacher Qwen 220/222、Mini 430/433，各语言均未退化。长输入 digest 与性能仍在运行。Mini weighted native norm 的独立算子探针在 1/512/1024/2048/4096 行全部逐元素相同，完整模型尚未验证。
- 2026-10-24，Mini clear 八组长输入的完整 logits 字节摘要、后续 decode token 和状态断言全部与修改前一致；16K/64K 配对性能运行中。token-only 提交候选四项 CPU 测试通过，覆盖 EOS 多算丢弃、budget、position、cancel 与错误失效；完整数值门禁和性能尚未结束。
- 2026-10-24，Mini clear 在 16K 的干净单 block 总时间由 41.866 增至 42.153 秒，比值 1.006873，没有提速证据。64K 尚在运行，候选仍未接受；较低缓存占用本身不代表完整模型更快。
- 2026-10-24，token-only 提交通过两模型各 1024 短 token 一致性和分语言 teacher 门禁，进入五 block 性能比较。Qwen BF16 native norm 实验单独应用并通过独立源码审查，原有 tiny 数值测试、真实 113 权重检查和完整模型门禁排队；尚未接受该算术修改。
- 2026-10-24，Qwen BF16 native norm 通过原有 tiny 数值断言，未改容差。独立临时程序直接编译当前 weights.rs，实际加载三个 variant，各 113 个 norm 数组的形状、准备精度和 checkpoint 数值全部匹配；完整短文本、teacher 和五 block 性能继续验证。
- 2026-10-24，拒绝 Mini clear 性能候选，待当前构建冻结后撤回。64K 单 block 总时间 338.886 → 336.668 秒，比值 0.993457，但有 paging 污染；TTFT 327.718 → 325.767 秒，decode 11.3718 → 11.6504 tok/s，128 输出一致。物理峰值 12.170 / 12.179 GB，均低于 12 GiB guard，未降低峰值。请求结束 allocator cache 确实由约 2.14 降至 0.27 GB，然而 16K 干净配对略慢，现有数据不足以接受为完整模型性能优化。保留 JSON 证据，不保留三行产品逻辑。
- 2026-10-24，Qwen native BF16 norm 的 text-fixed 出现 222/1024 处差异。history-zh 20 处、首次 108；code-zh 15 处、首次 111；code-en 116 处、首次 11；retrieve-en 71 处、首次 54。各位置、双方 token 和完整数组保存于候选 parity JSON，属于已披露的舍入变化。原有四项设备生命周期测试全部通过，分语言 teacher 门禁尚待完成，候选未接受。
- 2026-10-24，Qwen native BF16 norm 的 teacher 达 221/222，英文 113/113、中文 108/109，各语言不退化；Mini 短文本 1024 token 全同、teacher 保持 430/433。开始五 block 完整模型性能比较。Mini clear 三行逻辑及能力说明已实际撤回；Mini weighted norm 以撤回后的冻结 worker 为配对对照，独立验证开始。

- 2026-10-24，Qwen native BF16 norm 的五个干净 1K block 完成。对 MLX 的总时间比中位数从 ordered 1.041775 降至 native 1.012977，native 全部五块更快于 ordered，但全部仍慢于 MLX。总时间中位 2290.634 → 2222.678 ms，TTFT 375.213 → 349.554 ms，decode 66.304 → 67.883 tok/s；完整比值按 block 配对计算。此前披露的 222/1024 短文本差异保持，teacher 为 221/222；不能称为数值等价或全档达标。

- 2026-10-24，Mini weighted prefill norm 的两模型短文本和外部 teacher 门禁通过。Mini 1024 token 全同、teacher 430/433；Qwen 保持 native norm 的 222 处已披露差异及 221/222 teacher。八组长输入的完整 logits 摘要、后续 decode 和状态断言全部与基线相同。16K 同 1024 分块的干净单 block 总时间 41.766 → 41.223 秒，比值 0.986981，TTFT 34.630 → 34.135 秒；64K 与最终五 block 尚未完成。

- 2026-10-24，Qwen native norm 的 16K 同 512 分块配对完成且无 paging 污染，总时间 19.215 → 18.375 秒，比值 0.956309。此为单 block 局部收益，40K 与最终五 block 尚在后续验证。组合候选的 1K 新 profile 测得 Qwen GPU CB union 14.507 ms/token、Mini 48.367 ms/token，均是带 observer 的诊断；未将其与早期不同状态的 profile 比值当作性能门禁。

- 2026-10-24，Mini weighted norm 的 64K 单 block 为反向结果。总时间 324.782 → 326.122 秒，比值 1.004127；TTFT 314.899 → 316.174 秒，decode 12.851 → 12.766 tok/s。128 输出全同，两臂固定基线 swap 增长为 0，但该 block 被标记为 paging 污染，不作加速或退化的确定结论。候选物理峰值 11.849 GB，对照 12.132 GB，均在当前 12 GiB guard 内；16K 的干净单 block 收益不能替代最终长档五 block。

- 2026-10-24，纯 BF16 table RoPE 隔离探针完成。独立与合并 Q/K 两候选各检查 1,032,192 个 BF16 元素，覆盖 normal、大值、正负零和 0/1024/16384/40959 位置，全部位模式一致。28 层独立调用在位置 1024 的 30 次交错采样墙钟中位为原 compiled 0.489896 ms、独立 kernel 0.336063 ms、合并 Q/K 0.297625 ms。该数值含构图、提交和完成等待，不是单 kernel 时间，也不构成完整模型收益；开始比较安全封装和集成边界。首个调度因命令行参数误用在参数解析阶段失败，原错误 JSON 保留，第二次独立运行通过。

- 2026-10-24，token-only 提交的 Qwen/Mini 各五个 1K block 均无 paging 污染。Qwen 对 MLX 比值中位数 1.086641 → 1.039506；Mini 1.019054 → 1.003824，仍未达成全部不慢于 MLX。两模型短文本各 1024 token 完全一致，外部 teacher 不退化。新组合诊断在 Qwen bucket、Mini bucket/exact 各 300 输出中检查 299 次，每次追加 eval 全部保留 KV 的直接 compute dispatch 增量为零，计数器跨步严格增加，避免恒零假通过。此探针含逐步同步，只证明已覆盖的直接计算依赖，不作计时证据。
- 2026-10-24，实际 retained-prefix 检查全部通过。上述三条路径分别在发布 0、3、257 token 后丢弃额外一步，保留位置为 1024、1026、1280；不 reset 继续五步，异步与同步的完整 logits 字节摘要、token、position 及拒绝错误 position 后的状态均相同。接口清理后的四项 CPU EOS/budget/position/cancel/error 测试再次通过。接受 token-only 提交，删除无用 KV-root 回调，并在 DeviceDecode 保留 logits 必须依赖全部新状态的契约。
- 2026-10-24，Qwen native norm 的 40K 同 512 分块干净单 block 总时间 82.918 → 82.430 秒，比值 0.994113，TTFT 74.799 → 73.776 秒；decode 则由 15.643 降至 14.675 tok/s。此反向 decode 结果不删除，不以总时间的小幅收益掩盖。最终长档仍须五 block MLX 配对。

- 2026-10-24，接受 Qwen compiled 的 prepared BF16 权重及 native weighted RMSNorm。依据为各语言 teacher 不退化、原 tiny/lifecycle 断言和全部 339 个准备权重检查通过，以及五个干净 1K block 全部快于 ordered 对照。222/1024 短文本差异为已披露的舍入变化，不声称逐 token 等价；40K decode 的反向单 block 保留，最终全档目标尚未完成。

- 2026-10-24，接受 Mini compiled prefill 的 weighted native RMSNorm。新增五个干净 1K block 对照中，weighted/unweighted 配对比值中位为 0.996139，三块更快、两块略慢；相对 MLX 中位为 0.995757，范围 0.977647–1.003247。TTFT 中位 1251.856 → 1240.704 ms，decode 20.521 → 20.552 tok/s。数值及八组长状态检查均通过，16K 的干净单 block 比值 0.986981；64K 污染反向结果保留。该局部接受不替代最终全档五 block。

- 2026-10-24，分块策略选择进入组合验证。Qwen compiled 默认 2048，Mini compiled 长于 4096 时默认 1024；其他 variant 保留 512，环境覆盖及同一准备/执行计划保持。Qwen native norm 的新三块对照在 4K 总时间比 2048/512 为 0.990962、0.992121、0.987214，128 token 全同；1K 中位比 0.995207，但同算术的 1024/2048 单块出现 11.7% 离群值，不把短档微小差别当稳定收益。长档沿用先前分块与峰值证据，默认策略仍须短文本、teacher 和最终全矩阵验证。

- 2026-10-24，资源门禁纠正。早期脚本把物理上限写成 12 GiB（12,884,901,888 bytes），宽于用户字面 12 GB。后续构建、数值检查和最终性能计划收紧为 12,000,000,000 bytes；既有 guard JSON 不改写，超过 12 GB 的旧实验仅作诊断，不作字面 12 GB 合规证据。固定 swap 基线仍为 7483.06 MiB，增量上限及磁盘门禁不放宽。将试测两臂一致的 512 MiB allocator cache（仍低于 2 GiB 上限），检查 64K 是否可避免 swap-out 污染，再决定最终配对计划。

- 2026-10-24，默认分块组合的两模型短文本各 1024 token 与前一 native/weighted 候选全同；相对 b2d 仍仅有 Qwen 已披露的 222 处差异。teacher 保持 Qwen 221/222、Mini 430/433，各语言不退化。两项原有 prefill-plan CPU 测试通过，独立源码及注释审查无问题。保存选择为当前候选，完整长档性能与资源资格仍待最终矩阵，不宣称 Mini 1024 已证明稳定最快。提交前仅撤回无关 rustfmt 布局变化，两版本格式化输出相同。

- 2026-10-24，RoPE 候选的真实 Rust/Metal 三项设备测试通过，覆盖 D8/D128 位模式、strided/sliced 输入、不同表格半部、正负零、变化输入的 compiled replay、保留/延迟输出和非法形状。原有四项 Qwen 模型设备测试亦通过。两模型各 1024 短 token 与分块策略冻结对照全同，teacher 和五 block 性能尚在运行；独立源码及注释审查无问题。
- 2026-10-24，补正根依赖探针的边界描述。原 1024+300 请求容量仅 1536，低于两模型的小容量阈值，虽然位置越过 1280，attention 实际使用固定宽度，不能称为活动 bucket 切换覆盖。旧 JSON 保留，另加 4096+300 的依赖诊断，以及 4095 prompt、发布 0/3/259 后继续五步的状态检查，覆盖大容量下 4096/4352 边界；新检查仍在等待设备锁。
- 2026-10-24，RoPE 的完整外部 teacher 门禁通过，Qwen 221/222、Mini 430/433，各语言保持；两模型短文本相对前一候选均全同。五 block 性能仍待收齐。严格 12 GB、512 MiB allocator cache 的 64K 预检中，MLX 完成且物理峰值 9.053 GB、无 swap-out；b2d 在模型加载 ready 后、首请求完成前触发 12.112 GB 峰值，8.3 秒即被 guard 停止。该失败保留，不计性能结果，改用三臂相同 256 MiB cache 继续资源诊断。
- 2026-10-24，新增 copy dispatch 几何观测。Qwen 4K bucket 的 299 次 decode 编码中，显式 KV 根路径出现 5962 次 `vn_copybfloat16bfloat16`，grid 为 1179648；token-only 检查路径该几何计数为零。Mini bucket 对应为 5378 次、grid 294912，token-only 同样为零。MLX 0.32.3 的 `copy_v` 每线程处理 4 个 BF16，两个 grid 分别与各模型完整 KV 数组元素数吻合，属于结合源码的归因推断，未直接观测 buffer 身份。其余小 copy 和动态单行更新仍存在。两条诊断的同步方式不同，不将 dispatch 数量差换算成完整性能收益；证据存于 `decode-copy-shapes-v2.json`。
- 2026-10-24，启动冻结组合 `qwen-rope-v1` 的完整 12 档五 block 矩阵，三臂为 MLX、b2d 和候选，同用 256 MiB allocator cache、12,000,000,000-byte 物理上限、固定 swap 基线和双锁。RoPE 独立选择仍待五 block 收齐，完整矩阵属于候选验证，不代表已接受或全档达标。256 MiB 的 b2d 64K 资源预检已越过先前早期失败点，完整资格尚待完成。
- 2026-10-24，256 MiB 的 b2d 64K 两次请求完成，物理峰值 11.842 GB，但期间有 14768 次系统 swap-out，不能作干净性能证据。候选在 7.610 秒触发 12.123 GB guard；其加载 ready 为 5.144 秒。ready 仅表示模型加载完成，不能据此断言失败就在 prepare 内部，先前进度中的阶段归因修正为推断。零保留 allocator cache 的候选检查排队，同时复核初始 KV 与预热输出的存活关系。`Compiled::prepare` 的第一次输出在分号处已释放，不能从两次 eval 推断同时存活三份 KV。
- 2026-10-24，零保留 allocator cache 的候选 64K 检查仍失败，12.584 秒时物理峰值 12.067 GB，固定基线 swap 增长为 0。仅缩小闲置 cache 不足以满足该请求的严格门禁，开始增加阶段内存观测，尚未改变模型分配逻辑。Mini bucket/exact 与 Qwen bucket 的新 4096+300 依赖检查均完成，每条路径 299 次追加 KV 求值没有新增直接 compute dispatch，全部 300 个 token 与显式根对照相同。Mini exact 显式根的 observer 文本有交叉写入，仅保存带 SHA 的未解析 JSON transcript，不用于时间或 dispatch 结论；其独立 token 记录完好。
- 2026-10-24，补齐 256 MiB cache 下的 MLX 64K 完整 profile，warmup 和测量各一次均完成，物理峰值 8.792 GB。测量总时间 328.379 秒、TTFT 318.288 秒；decode GPU CB 区间并集 79.215 ms/token，host graph 中位 2.716 ms、async 提交含背压 75.846 ms、token 读取 0.004167 ms、host commit 1.058 ms/token，prefill 32 次 eval 合计 317.902 秒。以上区间重叠且含 observer 开销，不能相加或作为最终速度门禁。host 与 Metal 事件改为独立写入并在解析时合并，125390 个事件无交错修复；同 cache 的 b2d 诊断仍在运行。
- 2026-10-24，终止未成组的首次最终矩阵启动。只完成 Qwen 1K block 0 的 MLX 臂，baseline 仍在等锁且未创建设备进程；两个本任务调度进程均已停止，原始 driver receipt 与单臂 JSON 保留，另写取消原因。候选 64K 内存资格尚未通过，不把这次启动当作最终成绩；阶段诊断结束后以最终候选重跑，避免长诊断穿插配对臂。
- 2026-10-24，接受纯 BF16 table RoPE。五个干净 1K block 的 paired/composed 总时间比分别为 0.992101、0.995771、0.985394、0.995030、0.989903，中位 0.992101，五块均改善。相对 MLX 中位 0.999906，范围 0.995535–1.004385，三块更快，仍处于持平附近。总时间中位 2236.038 → 2216.573 ms，decode 67.040 → 67.974 tok/s；各块 documents 128 输出与前一候选全同。原有设备测试、两模型短文本相对一致性和分语言 teacher 均通过，Qwen 相对 b2d 的 222 处历史舍入差异保持披露。完整矩阵尚未完成。
- 2026-10-24，新边界状态检查收齐。Qwen bucket、Mini bucket/exact 各在 4095 prompt 后发布 0/3/259 token，保留位置 4095/4097/4353；9 组同步/异步续写五步的完整 logits 摘要、token、position 及错误位置拒绝均相同。依赖诊断三条路径各 299 次追加 KV 求值无新增直接 dispatch，300 输出全同；bucket 宽度 4352→4608 是源码几何推导，未直接遥测宽度。
- 2026-10-24，同 256 MiB cache 的 b2d 64K profile 完成，物理峰值 11.980 GB。测量总时间 364.242 秒、TTFT 349.279 秒；forward 中位 116.688 ms、其中 eval 114.706 ms、sample/read 0.741 ms；GPU CB 并集 113.589 ms/token、host commit 1.156 ms/token，prefill 128 次 eval 合计 348.892 秒。对应 MLX 为 32 次 eval、317.902 秒及 GPU CB 并集 79.215 ms/token。该观察支持长 attention 与 prefill 是主要差距，但含 observer、区间重叠，不能充作最终无插桩比值。
- 2026-10-24，阶段内存探针复现候选 64K 越限。零 allocator cache 下，prepare 入口 active 5.034 GB，初始 KV 求值后 7.897 GB；第一次 85 输出 decode 端点的 eval 尚未返回时，guard 在 7.322 秒采到 12.068 GB 并终止，尚未进入 prefill。首个 runner 漏设 Mini family 名称，已归档为加载前失败并另开正确运行。新进程的零数组隔离实验确认 84 个独立 backing 占 2,829,582,336 active bytes，共享单个 backing 为 33,685,504 bytes；这只证明初始存储差异，完整模型峰值和后续非零 KV replay 仍需验证。下一候选为共享不可变初始 KV。
- 2026-10-24，补充真正不同宽度的 attention 隔离实验。共同 backing 改为按 1024 对齐并受模型上限截断，1K/16K 的有效长度为 1025/16385，256 bucket 宽 1280/16640，1024 bucket 宽 2048/17408。Mini 的 256/1024/exact 七次中位为 1K 1.846/2.045/1.810 ms、16K 11.181/11.751/8.506 ms；Qwen 为 1.871/1.847/1.808 ms、22.023/21.850/20.637 ms。最高档两种 bucket 仍相同，Mini 为 56.923/57.854/29.217 ms，Qwen 为 55.662/56.943/53.480 ms。此实验不改变产品 capacity 策略，各几何按顺序测量，微小差异不作稳定赢家依据。若产品也扩大 backing，1K/16K 的额外 768 位置按形状推算分别增加 Qwen 88.080 MB、Mini 33.030 MB KV；这是计算值，非实测物理峰值。完整 JSON 位于 `attention-width-v2-{qwen3,minicpm5}/attention.json`。
- 2026-10-24，共享初始 KV 的完整 64K 冷请求通过严格门禁，物理峰值 9,896,417,472 bytes，固定基线 swap 增长为 0，单次租约 346.866 秒。零 cache 下，初始 KV 后 active 5.101 GB，两个预热端点峰值 7.932 GB，完整 prefill 阶段 allocator 峰值 8.158 GB；实际请求 active KV 仍为约 2.830 GB，没有缩减执行中的 KV 容量。128 token 与修改前 weighted、1024 分块的两次记录均全同。冷请求总时间 335.786 秒含准备及阶段标记，不能和不同状态的热测直接计算加速比。独立源码/注释审查无问题；两模型短文本、teacher、长 logits/state 与生命周期门禁继续。旧 DSpark draft 路径已不存在，后续使用现有 target checkpoint 做前后 target 验证，完整 drafter 测试未复跑。
- 2026-10-24，共享初始 KV 的两模型 text-fixed 各 1024 token 与 RoPE 候选全同，Qwen 相对 b2d 仍只有已披露的 222 处差异；teacher 保持 221/222 与 430/433，各语言未退化。八组长输入 full-logits SHA、后续 decode、非法位置及 reset/reprepare 对照全部通过，含 Public 4097-token 请求。原有两项 Mini checkpoint 生命周期测试通过，覆盖容量增减、非零 KV 续写、不同真实 prompt、保留输出和释放。首个检查调度在确认 draft 资产缺失后停止，已改用普通生命周期续跑；attention 模式切换与独立 DSpark target 回放尚未结束。

- 2026-10-24，共享初始 KV 的剩余门禁收齐。5 组同容量 attention 模式转换及失败容量恢复、非零 offset 改请求恢复、单行 prefill 尾块均通过。DSpark target 使用现有官方 target 权重，改动前后各跑 8 组 API/输入序列，共 40 次 logits/taps 观测与 4 个保留输出，全部 JSON 相同；两臂物理峰值 7.634 / 7.541 GB。未下载缺失 drafter，完整推测接受循环未复跑。冻结 worker SHA 为 `02241f9ec8d73ce8584dd91f641032594d1e4fd89faa0c00363d8e8d7e76568e`，全部记录源码与当前一致，接受为 64K 准备阶段的资源修复，尚无最终全档性能结论。
- 2026-10-24，最终矩阵改用独立 `mini-shared-zero-v1` 标签，MLX、b2d、候选三臂统一 128 MiB allocator cache，均低于 2 GiB 上限。此前 b2d 同 256 MiB 的峰值接近 12 GB，缩小共同 cache 是为资源留余量，不预设速度收益。每档五 block、每臂独立进程内 warmup 一次和测量一次，轮换臂顺序，固定 128 输出；12 GB、1190 秒、固定 swap 基线和双锁不变。
- 2026-10-24，最终矩阵首档 Qwen3 1K 完成，五块无 swap-out。b2d/MLX 总时间比中位 1.135101，候选为 1.006257，范围 0.982869–1.008073，尚未达到 ≤1.00；decode 中位 58.780→66.959 tok/s，MLX 66.991，TTFT 368.754→345.207 ms，MLX 336.379 ms。当前不把先前 512 MiB 的近持平结果替代本次 128 MiB 五块结果；其他档继续测量。
- 2026-10-24，最终矩阵 Qwen3 4K 五块完成且无 swap-out，b2d/MLX 中位 1.119805，候选 1.005949，范围 1.002237–1.020917，五块仍慢于 MLX。候选 TTFT 1699.712 ms、decode 52.864 tok/s，MLX 为 1677.390 ms、53.166 tok/s。调度器仅停止追加后续档，当前 16K 子进程继续完成五块；随后补充当前二进制的 prefill 阶段 profile，再决定下一项修改，不用重测挑掉这两档失败结果。
- 2026-10-24，16K 调度状态更正。仅终止调度器原本意在保留当前子进程，但终端会话回收随后也终止了其后代。该档只完成三块，比值为 1.003501、0.998964、1.000827，不计最终五块。已确认无遗留设备进程，原 driver/guard JSON 不改写，原始日志归档为 JSON，另存中断原因。后续须重跑完整 16K 档，不能把三块补称为五块。
- 2026-10-24，当前候选的 1K/4K 分阶段 profile 完成，同用 128 MiB cache，插桩前后各 128 输出一致，所有临时源码完整恢复。1K 首块 GPU CB 并集为 ApxInf 307.969、MLX 298.267 ms；4K 首块为 689.464、655.018 ms，第二块为 957.903、955.467 ms。ApxInf 首块有 56 次全容量 BF16 vector copy，末个单 token 步没有此类 copy，首采样提交仅 0.311/0.274 ms。两侧首块 RMSNorm、GEMM 和 causal attention kernel 名称/数量相同，RoPE 和 copy 路径不同；以上含 observer 开销且区间可重叠，不能作为独立 kernel 时间。下一实验只在 Bf16Compiled 的初始多行块让 attention 消费新 K/V，完整 KV 更新和显式求值仍保留；这改变布局和依赖，不会自行删除那 56 次 copy，收益与逐位一致性待测。

- 2026-10-24，拒绝并完整撤回初始 fresh-KV attention 候选。两档各五个干净 block，1K 候选/前版总时间比中位 1.005120，范围 1.004034–1.008277，五块全慢；4K 中位 1.000043，范围 0.990681–1.036746，没有稳定收益。两模型短文本各 1024 token 与前版全同，外部 teacher 保持 221/222、430/433，原有四项 Qwen tiny 设备/生命周期测试通过。因性能拒绝，不再执行该候选的完整逐层 bit 证明，也不把短文本一致等同于全状态证明；全部失败结果保留。下一步回到已测 attention 宽度差异，评估 Qwen 精确长度的完整模型效果。
- 2026-10-24，MiniCPM5 1K 最终五块完成，无 swap-out、固定基线 swap 增长为 0。b2d/MLX 总时间比中位 1.066054，候选为 0.993079，范围 0.984604–1.001164，四块更快，达到此档中位数目标。decode 中位为 b2d 18.626、候选 20.146、MLX 20.021 tok/s；prefill 为 778.189、804.185、801.272 tok/s。候选物理峰值 7.840 GB。三臂统一 128 MiB cache、每次固定 128 输出，完整矩阵现完成 3/12 档，其中两档 Qwen 仍未达标。
- 2026-10-24，MiniCPM5 4K 最终五块完成，无分页污染，但候选/MLX 总时间比中位 1.015020，范围 1.010823–1.018260，五块仍慢；b2d/MLX 为 1.139739。TTFT 中位为候选 5788.616、MLX 5696.961 ms，decode 为 19.020、19.290 tok/s，三臂各 128 输出逐 token 相同。候选物理峰值 7.698 GB。当前全档验收仍不通过，继续分离 attention、prefill 和共同 cache 配置的影响，不以 1K 的通过替代此档失败。
- 2026-10-24，Qwen 共同 cache 配置对照完成，两档各五块、每块四臂，无分页污染。2 GiB/128 MiB 的 ApxInf 总时间比中位为 1K 0.998305、4K 0.999264，均小于本轮波动，未证明稳定提速。两侧同 2 GiB 时相对 MLX 中位为 0.986770、1.005762，但 1K 的 MLX 大 cache/小 cache 中位为 1.008166，不能借较慢控制臂宣布产品达标。保留原 128 MiB 最终计划，不更换已失败结果。Mini 4K 实际 backing4352、有效长度4097的独立插桩 SDPA 诊断中，42 层调用合计 bucket/exact 七次中位为 3.177/2.758 ms，全部合成 BF16 输出相同；这是局部诊断，完整五块 exact 对照仍在运行。

- 2026-10-24，Mini 4K 精确长度五块对照完成。exact/bucket 总时间比中位 0.992172，五块均更快，三臂全部 128 输出相同且无分页污染；但 exact/MLX 中位仍为 1.005901，只有一块更快，尚未达标。bucket/exact 的 decode 中位为 18.926/19.253 tok/s，MLX 为 19.161，TTFT 为 5834.495/5833.435 ms，MLX 为 5720.534 ms。尚未改变默认 16K 起用 exact 的策略，继续测 prefill 最后一层的真实剩余开销。
- 2026-10-24，Qwen 精确长度候选完成源码复核与首轮数值门禁。候选仅通过私有环境变量启用，保留完整 bucket 回退与至多 128 个请求内精确 profile。两模型各 1024 短文本 token 与上一接受版全同，Qwen 对 b2d 仍为已披露的 222 处差异；外部 teacher 保持 221/222、430/433。独立审查要求补充同容量、非零 position prepare 后立即续写的状态证明；设备测试、真实同步/异步 retained-prefix 与四臂性能对照继续，不据编译成功接受该候选。
- 2026-10-24，Qwen 精确长度的 1K/4K 四臂五块全部完成且无分页污染。exact/前版中位为 0.995165/0.996991，exact/同二进制 bucket 为 0.998252/0.994500；exact/MLX 仍为 1.000433/1.004363，两档均未达到目标。首个冷请求 TTFT 中位为 exact397.511/1812.749 ms、bucket380.401/1792.811 ms，重复请求为 exact349.203/1708.419 ms。冷请求包含准备和首次执行，不能把差值直接称为单独编译时间。四组实际 retained-prefix 同步/异步对照通过，含 P4095 保留 0/3/120 token 和 P2050 保留 3 token，后续五步完整 logits SHA 全同；非法位置拒绝和 Ready 状态也一致。补充同容量 prepare 后立即续写及 P3/N0、N1 后，七项 Qwen 设备/生命周期测试重跑全部通过。
- 2026-10-24，完成真实 Mini 4096-token 最后一层重放诊断，物理峰值 7.604 GB、固定基线 swap 增长 0。原始整层、保留 full Q 后缩为最后一个 query、保留 full attention 后仅缩 O/MLP 的七次整段中位分别为 136.516、20.532、41.361 ms；这是局部 host 墙钟加同步求值，非端到端或逐 kernel 时间。两候选完整 K/V 各 1,114,112 个 BF16 元素均逐位相同。last-query 首次在 attention 出现差异，最终 hidden 有 653 个元素不同；last-output 的 Q、attention、O、post-norm 相同，首次在 SwiGLU 的一个元素不同，最终 hidden 一个元素、logits10440个元素不同，最大 logits 绝对差 0.0625。两者首 token 相同，但诊断仍执行原始模型后续路径，不能据正常 128 输出一致宣布候选数值通过。
- 2026-10-24，选择范围较小的 Mini last-output 路径做下一实验，只在 Compiled 最终多行 generation-prefill 且私有开关为 1 时缩行。保留完整 attention 和 K/V，普通 forward、Public、DSpark 以及单行尾块继续原路径。独立源码与注释审查无阻断项，新增固定 token 续写的完整 logits、普通多行输出、Public、三种 chunk 和热路径 Rust trace 验证。数值及性能队列等待 Qwen16K/40K 两档测量完成，避免设备负载交错。
- 2026-10-24，Qwen 精确长度候选的 16K 五块最终规格配对完成。b2d/MLX 总时间比中位 1.057579，候选为 0.994341，范围 0.993338–1.041091，四块更快，五块均无分页污染。候选/b2d/MLX 的 prefill 中位为 1128.151/1080.758/1121.674 tok/s，decode 为 29.853/26.278/29.641 tok/s；候选物理峰值 4.752 GB，固定基线 swap 增长 0。候选与 b2d 的 128 输出在五块中均逐 token 相同，两者相对 MLX 均有 94 处差异，首处下标 31。第一块相对时间较慢的 1.041091 保留在统计中。40K 继续运行，短档失败尚未解决。
- 2026-10-24，Qwen 精确长度候选的 40K 五块最终规格配对完成。b2d/MLX 总时间比中位 1.023863，候选为 0.989877，范围 0.986112–0.993106，五块均更快且无分页污染。候选/b2d/MLX 的 prefill 中位为 550.609/526.750/544.652 tok/s，decode 为 16.052/14.375/15.914 tok/s；候选物理峰值 7,663,405,936 bytes，固定基线 swap 增长 0。三臂全部五块的 128 输出相同。单独的精确 profile 准备耗时与重复替换内存仍待检查，尚未改变默认策略；Mini last-output 数值队列已启动。
- 2026-10-24，Mini last-output 的两模型短文本数值门禁通过，各 1024 token 与 qwen-exact-v1 全同。Mini 相对 b2d 无差异，Qwen 仍为此前披露的 222 处差异；外部 teacher 仍为 Qwen221/222、Mini430/433，各语言计数未下降。诊断已观察到的 BF16 logits 差异仍保留，短文本 token 一致不能取消该记录。十组真实状态检查继续，尚未据此接受默认启用。
- 2026-10-24，Mini last-output 十组真实状态检查通过，覆盖短整段、8192/8193 的 512/1024/2048 分块、8195 的多行尾块及 Public。每组固定五个输入 token 后完整 logits SHA 与关闭候选相同，普通三行 forward 相同；Public、单行尾块及8195多行尾块首步 logits 也相同。8192 的三种 chunk 首步 logits SHA 不同，保留为已披露数值变化；十组首 token 均相同。全部 20 次 prefill/续写观测的热路径 Rust trace 为零，无需增加 norm 预热分支。物理峰值 7,760,206,040 bytes、swap 增长 0，租约 293.684 秒。同步/异步丢弃 lookahead 和 DSpark target 补充检查随后继续。
- 2026-10-24，Mini 1K 的四臂五块对照完成，无分页污染。last-output/前版总时间比中位0.999631，只有三块更快，不能据此认定稳定总收益；last-output加exact/前版为0.995021，加exact/仅last-output为0.995095，均三块更快。组合候选/MLX为0.993970，范围0.993256–0.995077，五块均更快。前版/last-output/组合/MLX 的 TTFT 中位为1272.533/1256.598/1251.270/1292.444 ms，decode为20.244/20.193/20.256/20.215 tok/s。三条 ApxInf 路径的所有128输出相同，相对MLX均为97处差异，未新增差异。组合峰值7.832 GB、swap增长0；4K对照继续，默认策略未改。
- 2026-10-24，Mini 4K 四臂五块完成，无分页污染。前版、last-output、last-output加exact 相对 MLX 的总时间比中位分别为1.019504、1.010124、1.014939，组合仅一块更快，仍未达标。last-output/前版为0.993825，组合/前版为0.995522。组合与 MLX 的 TTFT 中位为5847.916/5827.747 ms、decode 为19.077/19.290 tok/s，不能把剩余差距全部归为 prefill。四臂所有五块的128输出相同；组合物理峰值7,718,688,816 bytes，swap增长0。默认开关保持关闭，下一步对当前版本做16K分项诊断。
- 2026-10-24，Qwen 精确 profile 的准备与替换检查完成。P42/1024/16384/40832 各16轮，相同请求复用均无 Rust trace；在首个新形状之后反复替换请求，exact 准备中位为14.316/15.035/23.039/40.850 ms，bucket 为0.034/0.028/0.024/0.041 ms。各档16轮的 allocator active bytes 完全恒定，两进程物理峰值分别2.638/2.707 GB，swap增长0。此探针不执行真实 decode，只证明所测准备阶段的复用与 active 分配稳定，不能据此声称所有 native 编译缓存或进程内存不增长。
- 2026-10-24，真实 Qwen prefill RoPE 探针完成，半维表达式在84个实际层/块观测、共429,748,224个 BF16 元素上逐位一致。1K 原表达式/半维表达式的各层耗时中位之和为25.864/19.696 ms；4K两块为41.327/33.619和40.060/32.052 ms。这是保留输入并额外同步的局部诊断，不是完整模型节省时间。native RoPE 在1K/4K分别有22,417,702/102,485,260处不同，最大绝对差0.25/4.0，不纳入本次候选。插桩前、中、后三次正常生成各128输出一致，物理峰值不超过3.474 GB。选择半维表达式做独立 opt-in 实验，仍需实际特殊值、短文本、teacher、长状态及完整性能门禁。
- 2026-10-24，Mini last-output 剩余状态门禁完成。四组同步/异步 retained-prefix 对照的后续五步完整 logits SHA、位置拒绝和 Ready 状态相同。原有长分块/桶边界测试通过；保留输出测试首次因 runner 漏设已有 replay 文件路径在加载前失败，修正环境后单独重跑通过，保留失败记录。DSpark target 前后各8组序列、40次 logits/taps 和4个保留输出全部 JSON 相同，未下载缺失 drafter，也未复跑完整推测接受循环。
- 2026-10-24，Qwen text-fixed 四臂五块完成，无分页污染。b2d、同候选 bucket、exact 相对 MLX 的总时间比中位为1.131876、0.999729、1.014421；exact/bucket 为1.017102，五块均变慢，因此短文本保留 bucket。bucket/exact/MLX 的 TTFT 几何均值再取块中位为54.863/81.959/47.588 ms，decode 为73.571/73.326/73.205 tok/s。bucket仅三块快于MLX，属于接近持平；不能把该档通过替代1K/4K documents的失败。精确模式准备开销与此方向一致，但这些独立观测不能证明其占据全部差值。
- 2026-10-24，当前 Mini 16K profile 的六个进程完成，各自插桩前/中/后的 warmup 和测量均保持128输出一致，全部无分页污染。ApxInf/MLX 的 decode GPU CB 并集为56.633/56.958 ms/token，host graph为1.544/2.765 ms、async提交含背压54.553/54.051 ms、scalar读取0.0758/0.0041 ms；这些区间重叠，不能相加。prefill对应2048-token区间合计35.017/34.609秒，ApxInf执行16个1024块，MLX执行7个2048块、一个2047块和最终单token。首区间ApxInf出现168个split-K主kernel及168个累加kernel，MLX对应区间未出现该名称；次数不等于耗时贡献。两臂物理峰值7.948/6.650 GB，源码完整恢复。ApxInf无插桩前/后TTFT为32.877/35.001秒，存在明显波动，因此本轮只作诊断，不能宣布端到端速度胜出。完整证据位于`proposals/mini-current-prefill-profile-v1/runs/`。
- 2026-10-24，Qwen 半维 prefill 表达式已作为默认关闭的私有实验实现。四份独立设计与交叉审查收敛到现有shape-specific RoPE callable，在构造时捕获开关，预热和执行共用同一表达式选择；仅BF16 compiled多行生效。开关开/关的八项设备/状态测试均通过，原七项未改；新增D8/D128检查覆盖Q/K头数、单行/多行、转置输入、不同表格左右半部、实际正负零、分别改变x/cos/sin及保留输出，全部BF16字节相同且复用无热trace。两模型短文本和teacher门禁运行中，长状态和五块完整worker配对随后执行。
- 2026-10-24，Qwen 半维候选两模型各1024个短文本输出与前一候选全同，b2d差异仍为Qwen222处、Mini0处；分语言teacher保持221/222和430/433。六组开/关retained-prefix对照全部通过，覆盖P4095发布0/3/120、P2050、P16384和P40832发布3，首步及后续五步完整logits SHA、位置和状态相同。独立审查指出最初合成测试的乘积恰好可用BF16表示，已补充实际GPU舍入反例：D8/D128分别舍入的输出为0x3b00，延迟到FP32乘加后再舍入为0x3b02；九项设备/状态测试全通过。冻结worker的产品源码逐字节不变，仅新增测试不同，已另存身份差异证明，独立复核待收齐。
- 2026-10-24，Qwen 半维候选1K四臂五块完成，无分页污染。b2d、同二进制原表达式、半维表达式相对MLX的总时间比中位为1.129723、0.994500、0.997114。半维/原表达式为0.997671，仅三块更快，尚不能确认稳定总收益。原表达式/半维/MLX的TTFT中位350.964/346.607/338.204 ms，decode68.061/68.029/67.215 tok/s。本轮原表达式也比先前轮次更快，保留先前1K失败记录，不能把这次绝对比值改善全部归功于半维改动。4K配对继续，默认未变。
- 2026-10-24，拒绝并完整撤回Qwen半维prefill候选。4K五块干净结果中，半维/原表达式TTFT比五块均大于1，中位1.003559，TTFT中位1708.641/1703.404 ms。总时间比虽为0.994072且三块更快，但decode波动明显，原表达式最后一块decode仅47.558 tok/s，不能据此把差值归为prefill优化；半维/MLX仍为1.002402，目标未达。全部原始结果保留，Qwen源码恢复到精确长度候选的原SHA42c。已测native RoPE的局部时间更低但BF16结果不同，下一实验单独验证其数值门禁，不接受半维候选的局部微测作为完整收益。
- 2026-10-25，当前 Mini 16K 的 512/1024/2048 分块单组试跑完成，全部无分页污染。TTFT 分别34.915/34.969/34.806秒，MLX32.494秒；总时间42.137/42.176/42.036秒，MLX39.789秒。物理峰值7.791/7.944/8.049 GB，固定基线swap增长0，均使用同一冻结worker、last-output开启、Auto注意力和128 MiB缓存。512与1024的128输出相同，2048有24处差异，首处下标91。单组不能选择最快分块，当前观测也未显示分块大小足以填平差距，继续保留默认1024；完整JSON位于`results/mini-current-chunks-v1-minicpm5-16384-pilot/`。
- 2026-10-25，Qwen native 多行 RoPE 候选经四份设计、独立交叉评审和源码审查后完成数值试验，现已完整撤回。原七项及新增四项设备测试在开/关两种模式下均通过；新增检查覆盖D8/D128、不同base、转置布局、偏移0/7/2048/38912、实际非零位置KV、保留输出和失败恢复。Qwen短文本相对前版有243处新差异，相对b2d为158处；Mini1024输出不变。Qwen teacher总计仍221/222，但英文113/113降至112/113、中文108/109升至109/109，未通过固定的分语言不退化门禁，不能用整体计数掩盖英文下降。Mini保持430/433。自动队列在数值门禁处停止，未运行性能配对、长状态或MixedW8补充检查。源码恢复QSHA42c，完整补丁与结果留在忽略目录，下一方向评估保持BF16表格和逐步舍入的多行kernel。
- 2026-10-25，Mini16K 的已接受共享零KV版本完成最终规格五block。b2d/MLX总时间比中位1.098915，候选1.010775，范围1.002586–1.062747，五块仍慢且全部无swap-out。候选/b2d/MLX 的decode中位17.624/14.795/17.590 tok/s，prefill468.326/437.779/472.455 tok/s；候选TTFT34.984秒，MLX34.678秒。候选物理峰值7,951,686,256 bytes，固定基线swap增长0。五块候选相对b2d各68处token差异、首处59，与此前exact数值变化一致；相对MLX各100处、首处26。第一块MLX较快的32.461秒TTFT保留，未剔除。该档仍未达目标，继续prefill逐算子诊断。
- 2026-10-25，Mini真实4K RoPE重放完成，前后冻结worker及三种诊断模式的正常128输出全同，全部无分页污染，峰值7.736 GB。共84次实际Q/K旋转、396,361,728个BF16元素，native有108,127,581处不同，最大绝对差0.0625、RMS0.002632。独立同步重放的各层耗时中位之和为table75.559 ms、native35.581 ms，不代表完整TTFT收益。观测到table的168次copy、84次negate和84次表格表达式dispatch，native为84次RoPE dispatch；无单kernel时间。正常执行仍使用table，不能据128输出控制认定native数值通过。16K逐块覆盖继续。
- 2026-10-25，Mini真实16K RoPE重放也完成。672个层/块观测覆盖1329次实际旋转、1,553,989,632个BF16元素；native有480,550,899处差异，最大绝对差0.125、RMS0.003001，全部有限。各层独立同步耗时中位之和为table504.239 ms、native321.735 ms；table观测到2658次copy、1329次negate和1329次表格表达式dispatch，native为1329次RoPE。全部五个进程正常table路径的128输出一致且无分页污染，物理峰值8,343,558,768 bytes、swap增长0。编译源已逐字节恢复，探针全部结束并释放双锁；原始成功日志已解析归档为JSON。此结果只给出局部优化依据，未资格化native产品路径。
- 2026-10-25，Qwen多行BF16表格kernel的静态独立代码与注释审查无问题，但运行门禁尚未通过。首次后端命令遗漏native特性，实际0项测试，明确作废；修正后几何检查1项通过、设备测试3/4通过，剩余项在输入正负零覆盖断言处失败，未形成kernel数值结论。模型原七项全通过，新三项中两项失败；源码确认构造阶段已prepare(1,0)，新夹具随后替换cold seams却保留prepared_lengths，导致状态和trace断言不一致。修正仅限夹具及驱动计数校验，保留原输出/KV/热路径检查，数值和性能资格门禁继续关闭。
- 2026-10-25，测试夹具修正后，几何1项、后端设备4项以及开关两种模式下各10项Qwen模型测试全部通过，实际用例数由驱动强制核验。原七项测试正文未变，生产代码未因夹具修复而改动。正负零改为周期2以使步长3视图保留两种符号，所有逐位oracle断言保留；新seams清除旧准备长度后重新走prepare(1,0)，热路径trace、KV、保留输出和失效恢复断言均通过。开始两模型各1024输出和固定分语言teacher检查，尚无性能接受结论。
- 2026-10-25，Qwen多行表格kernel短文本两模型各1024输出相对前版全同，分语言teacher保持Qwen221/222（英文113/113、中文108/109）和Mini430/433。实际1K/4K规范documents输入的55/110次Q/K旋转全部比较，共85,899,264/343,848,960个BF16元素，零位差、全部有限；五种进程的正常128输出与前后冻结控制全同且无swap-out。原table的110/220次copy、55/110次negate和55/110次表格计算分别合为55/110次custom dispatch。独立重放各层中位之和为{'compiled_custom_table': 18.912461, 'compiled_table': 28.222707}和{'compiled_custom_table': 39.503292, 'compiled_table': 73.158627} ms，仅作局部诊断，不推算TTFT。编译源2151项全部恢复并与冻结worker一致；后续继续长状态和完整五block配对。
- 2026-10-25，Qwen多行kernel六组开/关retained-prefix状态对照全部完成，覆盖P4095发布0/3/120、P2050、P16384、P40832发布3，完整logits、位置及保留状态全同。40K原路径物理峰值7.845 GB，未触及12 GB限制。开始1K/4K四臂五block完整请求选择，MLX、b2d、同二进制table和custom均用128 MiB缓存；性能结论待收齐。
- 2026-10-25，保留Qwen精确长度decode为默认关闭的本地opt-in。已完成逐token/分语言teacher及准备、保留KV、失效恢复门禁；16K/40K五干净block候选/MLX总时间比中位0.994341/0.989877，短文本exact/bucket中位1.017102，故保持默认bucket。零/一输出预算的实际final-prefill和热路径断言已补齐并通过，现有源审查中的对应测试覆盖缺口关闭。此提交不代表最终全档矩阵完成。
- 2026-10-25，保留Qwen多行BF16表格kernel为默认关闭的本地opt-in。1K/4K四臂各五block全部无分页污染，所有warmup和measured请求的table/custom128输出相同。1K b2d/table/custom相对MLX总时间比中位1.107202/0.979696/0.986769，custom/table1.000235，未证明1K完整收益。4K对应1.114441/1.005508/0.992206；custom/table总时间0.993926、四块更快，TTFT0.987195、五块均更低，TTFT中位1713.396→1687.372 ms。候选4K最后一块/MLX为1.054461，保留异常慢块，不据中位数宣称每块都赢。候选/MLX4K decode53.723/53.249tok/s、prefill2427.443/2413.674tok/s。设备、真实BF16位、两模型短输出、固定分语言teacher、六组长状态及两轮独立源审查均通过；长性能矩阵仍待完成。

- 2026-10-25 02:12。Qwen 有界精确长度与多行 BF16 表旋转分别本地提交为 `88c6c7a`、`18c6f22`，均保留可选回退。Mini 复用方案已完成独立设计评审，采用现有三数组 compiled callable 捕获后端算子，并用实际 Q/K 投影准备；实现和 Mini 独立数值、状态、完整请求门禁尚未完成。

- 2026-10-25 02:20。Mini 多行表旋转候选 `8195a089` 的两个纯选择测试、真实首层投影逐位/无热回调测试，以及开关两侧各两项原有长块边界和输出保留/释放测试全部通过；独立代码评审无待修问题。候选仍为默认关闭，完整短文本、教师、全层回放和性能门禁未据此豁免。

- 2026-10-25 02:28。Mini 多行表旋转两模型各1024短输出与上一合格版本全同；Qwen相对b2d仍为历史222处、Mini0处。固定分语言teacher保持Qwen221/222、Mini430/433。六组开关状态对照全部通过，覆盖P4095发布0/3/120、P8193、P8195、P16384发布3，首步和后续五步完整logits、位置及保留状态全同；12个进程均通过严格资源门禁。全层算子与DSpark目标检查继续，尚未性能验收。

- 2026-10-25 02:36。Mini全层表旋转回放4K/16K分别比较396,361,728/1,553,989,632个BF16元素，零位差；五个进程正常128输出全同且无分页污染。原168/2658次copy、84/1329次取负、84/1329次表格计算变为84/1329次custom dispatch。三次预热、七次交替独立计时的各层中位之和为77.145→37.628ms及502.669→339.642ms，不推算TTFT收益；选中旋转的投影准备单次另记16.527/7.536ms，非全模型准备成本。物理峰值7.770/8.327GB。DSpark目标开关两侧8组序列、40组logits/taps与4份模型释放后保留输出全同，未验证缺少drafter的完整投机链路。已开始4K/16K四臂五block完整请求选择。

- 2026-10-25，Mini多行表格kernel的4K四臂五block完成，全部无分页污染，十组warmup/measured完整128输出与原表路径相同。custom/table总时间比中位1.003371、TTFT1.000632，两者均仅两块更快；custom/MLX1.008441，五块仍慢。原表/MLX1.012563、b2d/MLX1.142669。独立kernel加速未形成4K完整请求收益，不能据微测接受；16K五block仍在继续，最终保留决定待其完成。

- 2026-10-25，复核已存Mini16K实际profile及当前源码，whole-step相关函数逐字节未变。测量提交窗口覆盖126个完整step-equivalent，每层Q/K共有2次表格表达式、2次negate、4次半部copy；KV的dynamic-copy单列，不计入RoPE。现有paired BF16 kernel支持Q16/K2/D128，但实际输入的隐式contiguous成本和整步舍入仍待验证。四份只读设计及独立交叉审查选择在既有compiled step内捕获kernel的默认关闭实验；此为下一假设，尚未修改产品源码或宣称收益。

- 2026-10-25，Mini多行表旋转16K五block完成，全部无分页污染，十组完整128输出与原表相同。custom/table热请求总时间比中位0.997920、TTFT0.997052，四块更快；首个请求对应1.002172/1.001555，仅两块更快。custom/MLX1.006328，五块仍慢，原表/MLX1.003618、b2d/MLX1.092798。custom/原表/MLX的decode中位17.494/17.516/17.542tok/s，TTFT34.849/34.923/34.697秒；物理峰值7.946GB、swap增长0。结合4K无稳定收益，拒绝此候选并完整恢复Mini源码9f33；数值通过和局部kernel加速均不替代完整请求收益。

- 2026-10-25，保留Mini末层last-output缩行为默认关闭的独立改动。4K五个干净block的TTFT比中位0.979675，五块均改善；总时间比0.993825，仅三块更快，不作稳定总时间或全档达标声明。两模型各1024短输出与前版全同、分语言teacher不退化，十组固定续写、四组同步/异步保留前缀、原生命周期及DSpark目标检查通过。4K诊断和8192首logits的BF16差异保留披露。README的当前资源约定同步为严格12GB和配对128MiB cache，早期12GiB/2GiB诊断不作为该约定证据。
