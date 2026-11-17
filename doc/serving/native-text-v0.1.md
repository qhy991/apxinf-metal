# 原生文本家族（`--provider native`）

这个 profile 用 `apxinf-worker/2.8`，把 provider 从"每个家族一份"改成"参数化"：
同一个 worker 二进制、同一套 identity manifest 与校验，家族名/变体/prompt 协议由启动参数给出。
MiniCPM5 的 `native-minicpm5*` profile 仍用各自的协议，行为与默认值不变。

## 启动

```
apxinf-serve --provider native --model <snapshot 目录> --model-id <id> \
  --native-worker <apxinf-native-worker> --metal-library <libmlx.dylib> \
  [--model-name metal_lfm3] [--model-variant <变体>] [--request-log <jsonl>]
```

`--model-name` 省略时由 checkpoint 自身元数据选择家族（`model_type: lfm3` 解析为 `metal_lfm3`）。
`--model-variant` 省略时用该家族的默认变体。
`--model-name`/`--model-variant` 只允许与 `--provider native` 同时使用。
`--ane on` 目前被拒绝：本 profile 尚未实现 ANE prefill（见 `devlocal/lfm3-port/reports/ANE-PLAN.md`）。

## 运行时身份

`runtime.kind` 是 `native_text`（MiniCPM5 profile 仍是 `native_minicpm5`）。
`adapter_revision` 是 `apxinf-native-text-v1`。
执行身份同时绑定 `provider`（registry 名）、`precision`、`variant` 和 `prompt_protocol`，
所以另一个家族或另一个精度产出的 manifest 无法通过本 profile 的校验。
校验逻辑与其他 native profile 相同，没有跳过项：
worker 上报的 `runtime`、`adapter_revision`、`execution` 必须与启动配置推算值相等。
manifest 逐个摘要模型目录中的**常规文件**；
目录里出现子目录会被拒绝，因此该目录必须只放快照文件。

## Prompt 协议

`prompt_protocol=lane-chatml`：prompt 由**家族自己渲染**（`metal_lfm3::lane_chat_prompt`），
assistant 前缀停在 `<|im_start|>assistant\n`，让模型自己生成首 token。
checkpoint 的 `chat_template.jinja` 在此路径上**从不使用**：
即使 `enable_thinking=false`，它也会把 `<think>` 与结束 token 注入 assistant 前缀，
那是另一套从未冻结过语料与 gate 的协议。
需要 HF thinking-prefill 协议的调用方必须建立独立实验（自己的参考捕获、冻结答案与 gate）。

## `replay` 命令（强制解码回放）

命令：`{"kind":"replay","prompt_token_ids":[...],"target_token_ids":[...],"steps"?:N}`。
worker 先 prefill，然后逐步把 `target_token_ids[k]` 作为输入喂入（不采样），记录每步 argmax 与耗时；
每次 replay 前后都释放请求状态，因此同一 worker 上连续 replay 不会串味。
响应 `replay_result` 含 `prefill_ms`（完整 TTFT：状态重置 + 容量准备 + prefill）、
`decode_ms_total`、`decode_ms_per_token{per_step_ms,mean_ms,median_ms,p10_ms,p90_ms}`、
`match_rate`/`matches`/`comparisons`、`prefill_argmax`、`emitted_token_ids`、
`config_echo{quant,ane,prompt_protocol,eos,provider,variant}`、`thermal`、`memory`。

`match_rate = matches / comparisons`：比较点是 prefill 行 argmax 对 `target[0]`，
外加每个被喂入步的 argmax 对 `target[k+1]`。
因此**轨迹必须包含第一个生成 token**（录制返回的 `[prefill_argmax] + emitted_token_ids`）；
省略它会把整条轨迹错位一格，match_rate 掉到约 0.7。
不传 `target_token_ids` 时为纯自回归计时（`comparisons=0`）。

对应 HTTP 路由：`POST /v1/replay`（不属于 OpenAI schema），body 字段同上，
另外可带 `"quant"`/`"ane"` 声明，服务会与实际加载配置**校验**，不一致返回 400，
避免把某个进程的计时标成另一个配置。

## 每请求计时日志

`--request-log PATH` 追加 JSONL，每个落定请求一行，`kind` 为 `chat` 或 `replay`：
`ts, request_id, kind, model, prefill_ms, decode_ms, decode_ms_per_token, ttft_ms,
prompt_tokens, completion_tokens, quant, ane, thermal`（replay 另带 `match_rate`，chat 另带
`status`/`cause`）。

`thermal` 是 `{source, thermal_state, load_avg_1m, load_avg_5m, load_avg_15m}`。
Apple Silicon 上非特权进程拿不到 thermal level（`pmset -g therm` 无记录，`powermetrics` 需 root），
所以该字段如实报 `thermal_state: "unavailable"` 并给出 load average，
而不是编造一个档位。worker 用绝对路径 `/usr/sbin/sysctl` 读取，因为它以清空的环境变量启动。

## 能力与限制

- 单 worker 单请求串行；不声明 sampling、图像、session、prefix cache。
- 发布方只报它真正准备好的 scope：`metal_lfm3` 目前是 `bf16-eager`，不声称任何 compiled/capture scope。
- 长上下文受家族自身的 `MAX_CONTEXT` 约束（LFM3 为 8192）。
