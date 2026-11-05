# 音频 serving 社区调研与当前能力

日期：2026-10-25。本文记录源码审计和社区设计事实，不包含实现指令。
设计选择、拟议接口和阶段验收见[音频 serving 方案](audio-serving-proposal-20261025.md)。
本轮没有安装依赖、下载模型、运行推理或测量性能。
社区的接口说明、模型列表和性能宣传不代表 ApxInf 已获得相同能力。

## ApxInf 当前实际边界

审计对象是当前主工作区，Git HEAD 为 `df5c55a06140107b24964d9e4abbefd9fa2a4733`。
该工作区包含未提交的 serving、原生模型和 Metal 工作，HEAD 本身不能唯一标识这些文件。
本轮文件摘要保存在忽略目录 `devlocal/audio-serving-20261025/`。
之前用于 PR 的 `serving-upstream` 工作树目前不存在，因此没有将当前主工作区等同于远端 PR。

| 范围 | 查到的实现 | 对音频的实际含义 |
| --- | --- | --- |
| 原生模型注册 | Llama、Qwen3-VL、Qwen3.5，以及 CUDA 条件下的 Pi0.5 | 未发现 TTS、ASR 模型注册 |
| 原生语言模型接口 | token 输入、可选图像、词表 logits 输出 | 不是 waveform、音频特征或声码器接口 |
| 新 HTTP 服务 | Messages、Chat Completions、计数、模型列表、健康、就绪、指标 | 没有 `/v1/audio/*` 或音频上传接口 |
| MLX worker | 固定 MLX-LM 的串行文本生成 | 没有音频解码、ASR encoder 或 TTS vocoder |
| worker 协议 | 1 MiB 上限的 UTF-8 JSONL、token 序列和文本增量 | 没有二进制音频帧、采样位置或转写修订语义 |
| OpenPI WebSocket | observation 输入、actions 输出的 policy 通道 | 不属于音频服务，不能证明具备音频流控 |
| 内存治理 | 单 worker、固定 sequence reservation、压力检查 | 尚未实现通用多模型物理内存仲裁 |

对应源码入口：

- [模型注册](../../crates/apxinf-model/src/builtin.rs)与[模型接口](../../crates/apxinf-model/src/llm_trait.rs)。
- [公共路由](../../crates/apxinf-serving/src/gateway.rs)、[监管与请求](../../crates/apxinf-serving/src/supervisor.rs)。
- [Rust 协议校验](../../crates/apxinf-serving/src/contracts.rs)与[Python 文本 worker](../../python/apxinf/apxinf/serving/text_worker.py)。
- [现有 policy WebSocket](../../python/apxinf/apxinf/serving/websocket.py)。
- [实施计划](implementation-plan.md)中的 coordinator、device arbiter 等独立模块仍属于拟议路径。

## 调研版本

下表固定本轮阅读对象。commit 的存在和时间通过 GitHub 元数据读取，不能视为运行验证。
后续部署仍需为选定依赖、模型和前处理资源另行固定版本。

| 项目 | 本轮固定 commit | 阅读范围 |
| --- | --- | --- |
| oMLX | `78fa08bc1aef7290c93c83bb70811547763ebf33` | audio routes、STT、TTS、executor、依赖 |
| MLX Audio | `f976eab8b47a3cada5aeb3b52a78451ee06117bc` | server、模型公开入口、音频结果、实时接口说明、依赖 |
| Speaches | `993994f7984bf3fe9655b267448328cf66fccb42` | 文件转写、SSE、语音合成、model manager、实时文档 |
| vLLM | `fc0cfcdc05b1717cadb17a0e24a6b1f611aec456` | 文件 ASR 与 realtime 转写说明 |
| vLLM-Omni | `dca119877bd420ee0abc117e637eeac7aa28e4dd` | speech API、音频流、增量文本输入、多阶段边界 |
| SGLang-Omni | `85d309fbca53d651272df9a55600bdf5c8867d19` | 多阶段架构、ASR/TTS 范围、Apple Silicon 限制 |
| sherpa-onnx | `532f1b5a8a207bf53b116b747207c1506696c843` | 在线 ASR server、端点检测、分批与连接队列 |
| whisper.cpp | `231638a0717b7822784297e5980965ffcec05810` | 原生服务器、Metal 与 Core ML 路径 |
| faster-whisper | `9e472ae9de8bb374552f531f752654554b76b2d6` | 惰性 segment generator、VAD、时间戳、CPU/CUDA 示例 |

## 社区实现的不同层次

### oMLX 将音频放在独立 engine 中

oMLX 提供 `/v1/audio/transcriptions`、`/v1/audio/speech` 和自身的 audio processing 接口。
STT、TTS 有独立 engine，沿用模型解析和 engine pool。
当前 TTS 路径优先使用模型原生音频流，不支持时还能按文本分段合成。
这两条路径的上下文连续性和首音频时间含义不同。
TTS 文件开头仍有旧的“不支持 streaming”描述，但文件中的原生流方法和路由已经实现流式路径。
本轮结论依据实际方法和调用关系，而非该旧注释。

TTS 和 STT 的相关调用使用共享 MLX executor。
该 executor 的 `max_workers=1` 说明共享它的调用串行执行，不能推导所有 LLM engine 也只有一个执行线程。
来源：[路由](https://github.com/jundot/omlx/blob/78fa08bc1aef7290c93c83bb70811547763ebf33/omlx/api/audio_routes.py)、[TTS engine](https://github.com/jundot/omlx/blob/78fa08bc1aef7290c93c83bb70811547763ebf33/omlx/engine/tts.py)、[STT engine](https://github.com/jundot/omlx/blob/78fa08bc1aef7290c93c83bb70811547763ebf33/omlx/engine/stt.py)、[executor](https://github.com/jundot/omlx/blob/78fa08bc1aef7290c93c83bb70811547763ebf33/omlx/engine_core.py)。

### MLX Audio 提供模型生态和公开生成入口

MLX Audio 的服务层提供文件转写、语音合成和两种实时转写接口。
文件转写的流式结果可以使用 NDJSON，因此不能把所有社区 `stream=true` 都当成 SSE。
原生实时转写依赖模型的 streaming session 能力；另一条路径可以先通过 VAD 分出片段再识别。
两条路径的输入延迟和模型状态语义不同。
其 TTS 结果包含音频、采样率及片段信息，部分模型另有流式 chunk 标记。
来源：[接口说明](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/docs/guides/web-ui-api-server.md)、[服务实现](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/mlx_audio/server.py)、[TTS 结果](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/mlx_audio/tts/models/base.py)。

Kokoro 文档和实现提供预设音色及逐段生成；它的普通 generator 不能单凭存在 `yield` 就证明段内原生低延迟流式。
Qwen3-TTS 有 Base、CustomVoice、VoiceDesign，不同变体使用不同生成入口。
Base 依赖参考音频的用途不能等同于具备固定音色的 CustomVoice。
这些模型是候选，不是本机批准的部署配置。
来源：[Kokoro](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/docs/models/tts/kokoro.md)、[Qwen3-TTS](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/docs/models/tts/qwen3-tts.md)。

### Speaches 强调音频 API、模型驻留和组合

Speaches 将 faster-whisper、Kokoro、Piper 等能力暴露为音频接口。
文件转写可以逐段发送 SSE；其 model manager 用使用计数和空闲时间管理卸载。
实时接口进一步组合 ASR、LLM 和 TTS，也提供只转写模式。
其文档记录了模型参数和会话更新方面的兼容差异，不能把“兼容”理解为所有字段行为一致。
来源：[项目说明](https://github.com/speaches-ai/speaches/blob/993994f7984bf3fe9655b267448328cf66fccb42/README.md)、[转写路由](https://github.com/speaches-ai/speaches/blob/993994f7984bf3fe9655b267448328cf66fccb42/src/speaches/routers/stt.py)、[驻留管理](https://github.com/speaches-ai/speaches/blob/993994f7984bf3fe9655b267448328cf66fccb42/src/speaches/executors/shared/base_model_manager.py)、[实时说明](https://speaches.ai/usage/realtime-api/)。

### vLLM 与 Omni 项目区分模型和执行阶段

vLLM 的文件 ASR 与 WebSocket realtime 转写是不同入口。
所查 realtime 文档规定明确的 PCM 格式与事件流，不代表任意 ASR 模型都具有在线状态。
来源：[固定版本 ASR 文档](https://github.com/vllm-project/vllm/blob/fc0cfcdc05b1717cadb17a0e24a6b1f611aec456/docs/serving/online_serving/speech_to_text.md)。

vLLM-Omni 的 speech API 区分完整响应、原始音频字节流、SSE 和增量文本 WebSocket。
增量文本输入仍可能先缓冲，或把文本分段后分别发起合成。
当前文档明确说明部分文本提交模式不延续前段的模型或声学状态。
所以“边输入文本边收到声音”和“同一次合成延续声学状态”是两项不同能力。
来源：[固定版本 speech API](https://github.com/vllm-project/vllm-omni/blob/dca119877bd420ee0abc117e637eeac7aa28e4dd/docs/serving/speech_api.md)。

SGLang-Omni 将前处理、encoder、生成器、talker、decoder、vocoder 和聚合组织成不同阶段。
所查 Apple Silicon 路径仍标为实验性，Qwen3-ASR 使用 Torch MPS，而不是由此证明其 CUDA 调度可直接用于 MLX。
来源：[项目架构](https://github.com/sgl-project/sglang-omni/blob/85d309fbca53d651272df9a55600bdf5c8867d19/README.md)、[Qwen3-ASR 的 Apple 路径](https://github.com/sgl-project/sglang-omni/blob/85d309fbca53d651272df9a55600bdf5c8867d19/docs/cookbook/qwen3_asr.md)。

### sherpa-onnx、whisper.cpp 和 faster-whisper 各有适用边界

sherpa-onnx 分开在线和离线识别，并在在线服务器中提供端点检测、等待分批和连接队列限制。
其模型接口与消息格式可以作为在线 ASR 的研究依据，不能当成 OpenAI realtime 协议的同义实现。
来源：[在线服务器](https://github.com/k2-fsa/sherpa-onnx/blob/532f1b5a8a207bf53b116b747207c1506696c843/python-api-examples/streaming_server.py)、[在线 WebSocket 文档](https://k2-fsa.github.io/sherpa/onnx/websocket/online-websocket.html)。

whisper.cpp 提供原生推理和文件上传服务器，另有 Metal 与 Core ML encoder 路径。
Core ML encoder 的存在不等于整条 ASR 管线都在 ANE 上执行。
来源：[项目说明](https://github.com/ggml-org/whisper.cpp/blob/231638a0717b7822784297e5980965ffcec05810/README.md)、[HTTP 服务器](https://github.com/ggml-org/whisper.cpp/blob/231638a0717b7822784297e5980965ffcec05810/examples/server/README.md)。

faster-whisper 的转写结果是惰性 generator，调用函数并不代表实际识别已经完成。
其公开说明提供 CPU、CUDA、VAD 和词时间戳用法；本轮没有验证 Metal 执行能力。
来源：[固定版本说明](https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/README.md)。

## 依赖和运行设置差异

ApxInf 当前文本 worker 固定 Python 3.14.3、MLX 0.32.1、MLX-LM 0.31.3。
本轮 MLX Audio 的 `pyproject.toml` 要求 `mlx==0.32.4`。
oMLX 要求 Python `>=3.11,<3.14`，并固定另一份 MLX Audio revision。
这些记录说明不能直接把社区完整 server 放入当前文本环境。
它们不证明独立音频环境已经通过依赖解析或本机兼容验证。
来源：[ApxInf 固定值](../../python/apxinf/apxinf/serving/text_worker.py)、[MLX Audio 依赖](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/pyproject.toml)、[oMLX 依赖](https://github.com/jundot/omlx/blob/78fa08bc1aef7290c93c83bb70811547763ebf33/pyproject.toml)。

MLX Audio 的 STT utility 提供会临时修改 wired limit 的 context manager。
这只是该辅助函数的事实，不能推出每个模型的公开 `generate` 路径都必然调用它。
使用何种模型、入口和参数时发生哪些全局设置变化，仍需逐个调用路径核对。
来源：[STT utility](https://github.com/Blaizzy/mlx-audio/blob/f976eab8b47a3cada5aeb3b52a78451ee06117bc/mlx_audio/stt/utils.py)。

## 兼容 API 的边界

OpenAI 的文件转写与 realtime 转写也是不同工作流。
speech 接口区分 `response_format` 和 `stream_format`；文件格式与传输形式不是同一个参数。
社区 `stream=true`、`stream_format`、NDJSON 和 SSE 的具体行为并不完全相同。
任何 ApxInf 兼容声明都需要指定参数子集和客户端测试。
来源：[文件转写](https://developers.openai.com/api/docs/guides/speech-to-text)、[speech 接口](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create)。

## 证据限制

上述判断来自当前文件和固定社区源码，未运行这些社区 server。
没有本机首音频时间、实时率、识别质量、合成质量、取消延迟或内存上界的实测。
库函数有 generator、callback 或 WebSocket 接口，均不能单独证明计算可以及时中断。
新 ApxInf 代码、测试和 fixtures 仍须独立手写，不能复制或改写参考项目实现。
