# Qwen3.5 专用二进制实测

本轮已经交付可运行的 `apxinf-qwen35-08b`。它通过 ApxInf 原生 API 执行固定的本地 Qwen3.5-0.8B，支持单次文本、单次 JSON 和常驻 JSONL。用户运行时不需要安装 Python、Rust 或 ApxInf 开发环境。

**结论：专用构建减少了编译工作和文件体积；本轮尚不能证明稳定的生成加速。**

最终证据使用 `model-specialization-v2-20261010-r3`。前两次准备记录和 r2 的 EOS 校验失败均保留，未计入最终结果。

| 构建选择 | 干净编译，两次中位数 | 两次范围 | 入口注释改动重建 | 文件大小，十进制 MB |
|---|---:|---:|---:|---:|
| 通用 ApxInf CLI | 43.67 s | 42.80–44.54 s | 3.031 s | 8.944 |
| 专用入口，保留完整组件 | 40.85 s | 38.95–42.76 s | 0.946 s | 7.536 |
| 专用入口，裁剪组件 | 33.23 s | 30.41–36.05 s | 0.796 s | 7.267 |

相对通用 CLI，专用构建耗时减少 **23.9%**，文件体积减少 **18.7%**。这个比较同时包含入口简化和组件裁剪。严格保持同一入口时，组件裁剪使本次编译耗时减少 **18.7%**，文件体积减少 **3.57%**。入口注释改动不代表任意模型或算子改动的重建成本。无改动重建约 0.08–0.19 秒，不作为加速依据。

六次构建按 full/reference/minimal/minimal/reference/full 顺序执行，各用独立 target 目录、相同 release 设置、四个编译任务、同一 SDK 和锁定依赖。依赖下载缓存已经存在，操作系统文件缓存没有控制。这是本机两次观测的结果，不能保证每台机器都取得相同百分比。

编译证据同时检查了 Cargo features、Rust 依赖文件、原生对象和符号。模型 crate 参与编译的 Rust 文件从 35 个降到 10 个；保留 Qwen3.5，排除 Llama、Qwen3-VL、PI0.5、注册表、VLA 和 debug 模块。Metal 桥接对象从 10 个降到 2 个，只保留 head 与 MLP，MatVec 和实验模块不参与最小构建。GGUF、通用 CLI 的 clap、tokenizer 的进度条与 C++ 训练代码也已排除。

这仍不是所有代码的全局最小集合。公共 core 的算子边界尚未继续拆分，第三方 tokenizer 仍含上游没有 feature 开关的 Rust 代码。Qwen3.5 的维度与组网仍通过现有配置和执行实现处理。Metal shader 仍是内嵌源码，在模型构造时由 Metal 编译，尚未改为预编译 GPU 库。

| 实际生成案例 | 输出 token 数，含 EOS | reference 解码中位数 | minimal 解码中位数 |
|---|---:|---:|---:|
| 中文短回答 | 7 | 29.70 token/s | 29.88 token/s |
| 英文一句话 | 26 | 29.49 token/s | 31.68 token/s |
| 算术短回答 | 3 | 29.86 token/s | 31.19 token/s |
| 英文长回答 | 212 | 29.59 token/s | 31.56 token/s |

两个专用构建按 reference/minimal/minimal/reference 顺序各运行两个独立进程。每个进程完成两轮预热和三轮测量，每轮四个案例，共 80 个请求，其中 48 个参与性能统计。每个案例每种构建有六个测量值，但它们只来自两个独立进程。解码吞吐使用首 token 后的 token 数除以解码时间，长回答对应 211 次 decode。

所有驻留请求的输入 token、输出 token、最终文字、终止原因和 head/24 层 MLP 执行路径一致。四次实际通用 CLI 对照的输出 token 也完全相同；其现有协议只输出 prompt 长度，因此不能宣称它的完整输入 token 身份已经直接比对。另用实际专用二进制验证了一次单次 JSON 和一次中文流式输出，文本均为“巴黎是法国的首都。”。

长回答的观测差异为 **+6.68%**，但两个范围重叠：reference 为 27.25–32.41 token/s，minimal 为 29.81–33.04 token/s。四个进程运行区间内，全系统发生了 5,333 页 swap-in，共 83.33 MiB；该数字不能归属为测试进程自身的换页量。热状态命令未记录警告，也不能证明频率稳定。因此报告保留 `formal_performance_accepted=false`，不把这一差异归因于裁剪。

最小构建的启动阶段两次为 5.94 和 6.80 秒，包含资产校验、tokenizer 加载和模型构造；reference 对应 6.04 和 6.41 秒。最小构建的完整资产校验中位数约 3.58 秒，模型构造约 2.58 秒，启动没有显示收益。对个人连续使用，JSONL 常驻模式可以摊薄这项成本。各请求仍独立重置，不共享聊天历史或前缀缓存。

实际配置是 Mac mini / Apple M4 / 16 GB / macOS 27.0.1 / arm64，编译使用 MacOSX15.4 SDK。精度保持 CPU/Accelerate F32 主体、Metal W8 head，以及全部 24 层 decode MLP；没有借此实验更换量化或生成算法。模型固定为本地 revision `2fc06364715b967f1860aea9cf38778875588b17`，五个必要资产都校验 SHA-256。

模型配置中的 EOS 为 248044，而现有聊天 CLI 使用 tokenizer 的 248046。早期专用入口错误地要求两者一致，真实模型检查在生成前阻止了运行。最终实现沿用现有 CLI 的 tokenizer EOS，并新增回归测试。最终入口 17 项 CPU 测试、无 registry 模型 23 项 CPU 测试通过；另有 12 项 tokenizer CPU 测试记录。五项模型 Metal 单元测试没有在 CPU 测试命令中运行，实际 GPU 路径由上述端到端请求验证。

二进制大小为 **7,267,376 bytes**，不包含权重。五个外部资产合计 **1,759,777,953 bytes**；专用程序不需要分片索引文件。运行还依赖 macOS 内置的 Metal、Foundation、Accelerate 和系统动态库。它是可直接运行的原生程序，不能把 7.27 MB 解释为包含模型及所有系统依赖的完整包。

可以直接运行本轮冻结产物：

```sh
/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/experiments/model-specialization-v2-20261010-r3/artifacts/3-minimal/apxinf-qwen35-08b \
  --model /Users/haiyan-mini/Agent4Kernel/.apxinf-models/Qwen3.5-0.8B-2fc063647-staged \
  --prompt '法国的首都是哪里？请只用一句话回答。' --max-tokens 64
```

连续使用时把 `--prompt ...` 换为 `--jsonl`，逐行输入 `{"prompt":"What is 7 times 8?","max_tokens":64}`。资产内容应在程序运行期间保持不变。

继续专用化时，优先做固定文本权重与 W8 数据的离线预打包，减少启动时加载和量化准备；再把固定模型的执行计划与公共算子依赖拆得更细。若目标是单用户 token/s，下一项工作应单独测量 CPU/GPU 同步、数据驻留和 dispatch 开销，然后验证对应的融合或 GPU 路径优化。每项都需要独立保持精度和正确性比较，不能把算子替换收益算到编译裁剪上。

EngineTailor 可以负责离线选择、实验约束与产物证据；ApxInf 提供可复用的原生执行实现。最终用户只需要专用可执行文件和固定资产，不必安装优化工具。完全更换 ApxInf 执行实现则属于另一项运行时比较。

证据和使用说明：

- [最终汇总](/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/experiments/model-specialization-v2-20261010-r3/summary.json)
- [构建原始记录](/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/experiments/model-specialization-v2-20261010-r3/builds.json)
- [实际进程与请求记录](/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/experiments/model-specialization-v2-20261010-r3/runtime.json)
- [源码快照](/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/experiments/model-specialization-v2-20261010-r3/source-snapshot.tar.gz)
- [专用入口和构建说明](/Users/haiyan-mini/Agent4Kernel/ApxInf/crates/apxinf-qwen35/README.md)
- [已安装的 model-specialized-binary skill](/Users/haiyan-mini/.codex/skills/model-specialized-binary/SKILL.md)
