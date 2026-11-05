# 模型专用二进制：2026-10-10 首轮实验

本轮实现了 Qwen3.5 原生模型族的编译裁剪。构建时间减少，但没有获得稳定的推理提速证据。
它仍是模型族级原型，尚未固定到单个 checkpoint，也未排除所有无关算子和 Metal 模块。

## 实现和边界

根包与 `apxinf-model` 提供 `model-llama`、`model-qwen35`、`model-qwen3vl`、`model-pi05`。
默认启用全部模型，保持原有默认能力。专用构建通过 Cargo feature 在编译前排除其他模型模块、导出和注册入口。
Rust `.d` 依赖文件确认专用构建没有编译 Llama、Qwen3-VL、PI05 源文件。
这比只检查链接后的文件大小更直接。

两组使用同一源码快照、锁文件、release 配置、算法、权重和精度。
本轮没有新增依赖，也没有使用外部实现代码。
所测路径为原生 Rust CLI，CPU/Accelerate F32 主体加 decode 阶段已有的 Metal W8 head/MLP。
它不是全 Metal 模型。该 native 推理路径不需要 Python。

专用 CLI 仍保留通用参数、loader、tokenizer、core、图像处理依赖、MLX provider 入口和实验路径。
外部 MLX provider 的可用范围不受这组 native 模型 feature 限制。
两组都编译了 10 个 Metal 原生桥接模块，并保留运行时 Metal 源码编译与 pipeline 创建。
因此，本轮不能称为最终最小运行时或完全 AOT 的 Metal 交付。

## 构建结果

机器为 Apple M4、10 GPU 核、16 GiB 统一内存。
工具链为 rustc 1.98.0 / LLVM 22.1.8，固定 SDKROOT 为 MacOSX15.4.sdk。
首次工具链探测因旧链接器无法读取默认 27.0 SDK 而失败，记录单独保留，没有计入结果。

采用完整、专用、专用、完整的顺序，每次使用新的 Cargo target 目录，4 个编译任务。
依赖下载缓存已存在，文件系统缓存未清空。表中均为两次结果的中位数。

| 指标 | 完整模型集合 | Qwen3.5 专用集合 | 观察 |
|---|---:|---:|---|
| 干净构建 | 46.681 s | 39.819 s | 耗时减少 14.7% |
| CLI 注释修改后重建 | 3.128 s | 2.813 s | 两次样本，波动范围重叠 |
| 无修改重建 | 0.193 s | 0.145 s | 时间很短，波动明显 |
| 执行文件 | 8,926,976 B | 8,735,440 B | 减少 191,536 B，约 2.1% |
| native 模型族 | 4 | 1 | 编译依赖文件确认 |
| Metal 原生桥接库 | 10 | 10 | 尚未裁剪 |

release 默认不启用 Rust 的 incremental 编译。
这里的“重建”指保留依赖构建缓存后，修改 CLI 源文件再执行相同命令。
未给专用组额外启用 LTO、strip 或不同优化级别。

## 生成结果

使用现有 Qwen3.5-0.8B 本地权重。历史记录标注 revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。
本轮固定实际本地资产哈希，没有重新联网验证该 revision。
batch 和并发均为 1，context 2048，最多生成 64 tokens，按 EOS 自然停止。

两个构建各启动两次常驻测量进程。每个进程、每个问题先预热两次，再测三次。
另用实际交付 CLI 对每个问题执行每组两次独立进程生成。
全部 60 次常驻生成和 12 次 CLI 生成，在对应问题上都得到完全一致的 token IDs。
模型配置、输入 IDs、运行设置和实际 Metal 路径也通过一致性检查。

| 问题 | 输出 tokens，含 EOS | 完整请求中位数：完整 → 专用 | decode tokens/s：完整 → 专用 |
|---|---:|---:|---:|
| 法国首都 | 7 | 336.47 → 355.12 ms | 30.13 → 28.76 |
| 7 × 8 | 3 | 245.15 → 210.79 ms | 27.50 → 29.58 |
| 冰为什么浮在水上 | 26 | 982.42 → 1110.24 ms | 29.84 → 28.05 |

常驻请求时间包含 reset 到生成返回，不包含分词、文本解码和输出序列化。
这组常驻数据来自相同 API 的独立测量驱动，实际 CLI 的完整进程耗时单独保存在结果中。
模型构造中位数为完整组 3.055 s、专用组 4.577 s。
首个问题的完整 CLI 进程中位数为 3.953 s、5.347 s，专用组也更慢。

测量前后 swap 使用量从 6092.50 MiB 增加到 6982.75 MiB，并出现 swap-in、swap-out 和压缩活动。
没有热警告记录，但这不证明主机安静或没有瞬时争用。
上述数据不能把延迟变化可靠地归因于模型裁剪，也不能宣布稳定提速或确定的算法退化。
尤其是只有两次独立进程、输出很短，不足以评估长上下文或持续长生成。

`summary.json` 的 `accepted: true` 只表示身份、样本完整性和本语料的输出等价检查通过。
`formal_performance_accepted` 为 `false`。本轮没有执行完整模型质量验收或既有冻结 campaign 的重新资格认证。

## 产物和重建

实验目录：`.apxinf/experiments/model-specialization-20261010/`。

- `artifacts/2-candidate/apxinf`：已测试的专用 CLI。
- `artifacts/1-baseline/apxinf`：对应完整 CLI。
- `summary.json`：结果和接受范围。
- `builds.json`、`runtime.json`、`resident-*.json`：原始命令、测量和输出。
- `inputs.json`、`contract.json` 的归档副本：固定输入及其哈希。
- `*-model-dependencies.d`：模型源码排除证据。
- `delivery.json`：系统动态依赖、外部资产大小和源码归档身份。
- `source-snapshot.tar.gz`：实际构建的源码快照。

执行文件为约 8.33 MiB，所需模型资产另有 1,759,828,853 B，约 1.64 GiB。
模型资产未嵌入执行文件，不能把执行文件大小当作完整部署包大小。
动态依赖来自 macOS 系统，包含 Metal、Accelerate、Foundation 等。

从当前仓库重建专用 CLI：

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
cargo build -p apxinf --release --locked --offline --no-default-features \
  --features model-qwen35,accelerate,metal-w8
```

对照完整构建时，移除 `--no-default-features`，features 改为 `accelerate,metal-w8`。
不要用 `--workspace` 做本对照：其他成员可能通过 Cargo feature 合并重新启用全部模型。
重复实验应使用新 snapshot 和新 output 目录，依次执行 `compare.py` 的 snapshot、prepare、build、runtime、summarize。
build 与 runtime 需持有现有 Metal 测量锁。prepare 后不得修改已绑定输入。

## 下一步顺序

1. 拆开通用 CLI 与专用入口，按功能选择图像、MLX provider、模型注册和诊断入口。
2. 拆开 Qwen3.5 的普通 head/MLP 路径与实验状态，再同步裁剪 Rust 模块、FFI 和 Metal 桥。
3. 固定 checkpoint 配置、精度和支持的 shape 范围，使无关算子与分支可以在编译前排除。
4. 单独评估离线 Metal library、权重预打包和加载策略；pipeline 创建成本仍需实测。
5. 以单用户真实请求评估 GPU 驻留、融合、同步与状态读写。改变 kernel 或量化的收益应单独归因。

构建裁剪适合降低构建和交付成本。单用户生成极限仍需要针对实际执行路径的性能实验。
EngineTailor 可负责计划绑定、独立测量和经验反馈，专用 native 产物由 ApxInf 的构建与运行时实现提供。
