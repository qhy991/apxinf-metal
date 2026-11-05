# Serving 审查修复与 CPU 验证

日期：2026-10-11。

本轮修复进程退出、工具参数解析，以及阻塞提交测试的时序问题。
验证没有加载模型或执行 GPU 推理。

## 进程退出

CLI 在模型加载前注册 SIGINT 和 SIGTERM。
新增 `start_with_shutdown(Config, shutdown)`，在取消加载时显式停止并回收 worker。
只有确认回收成功，接口才返回 `Ok(None)`；启动错误和回收错误仍返回失败。
原有 `start(Config)` 保持可用，运行后的停服仍使用 `Service::shutdown` 和 `Service::wait_stopped`。
轮换期间加载 replacement 也能响应停服。

修复前的 CPU 探针中，向加载阶段的 CLI 发送 SIGINT 或 SIGTERM 后，worker 仍存活且父进程变为系统进程。
修复后的测试检查 CLI 返回成功时，worker PID 已不存在。

复查还发现，未完成的 HTTP 请求体能无限阻塞 CLI 退出，即使 worker 已经回收。
现在 CLI 先等待活动工作结算和 worker 回收，再给剩余 HTTP 连接 20 秒完成。
超时会记录诊断并结束 CLI；这 20 秒不包括活动请求的执行和结算时间。
测试保留一个未完成请求体，同时让已执行的计数请求在退出信号后继续运行 2 秒并返回 HTTP 200。
它再检查 worker 回收后的 HTTP 宽限期，防止实现误用覆盖整个退出流程的短超时。

## 工具 JSON

解析器现在记录 JSON 字符串、反斜线转义和已扫描位置。
字符串中的 `</tool_call>` 保持为参数内容，不再截断工具调用。
除未完成的固定长度结束标记外，解析器不重复扫描已经检查的内容。
XML 工具语法和现有参数验证规则保持原有行为。

新增回归在修复前有 3 项失败，错误均为 `The model emitted invalid tool JSON.`。
修复后，全部 11 项输出解析测试通过。
用例覆盖完整输入、逐字符输入、Unicode、引号、反斜线、连续 JSON/XML 调用，以及无效和未完成 JSON。

## 阻塞提交测试

旧 fixture 的启动、长 token 数组构造和摘要计算可能先耗尽 150ms 请求 deadline。
测试因此没有进入它原本要检查的 submit 写入阶段。
加入 300ms 启动延迟，可以确定性复现空命令记录和等待停止超时。

新 fixture 先准备数据并报告 ready，再接收请求。
子进程读到 submit 的首字节后报告事件，测试主动允许它恢复读取。
约 115KiB 的提交帧仍能形成真实管道背压。
原有 deadline、取消时限、输出抑制、资源保留和指标断言均保留。
该模块 4 项测试通过，也通过了本轮全量库测试。

## 验证结果

| 范围 | 结果 |
| --- | --- |
| Rust serving 全量库测试 | 125 通过，0 失败，1 忽略 |
| 实际 CLI 信号与 HTTP 排空测试 | 6 通过，0 失败 |
| Python serving 测试 | 226 通过，6 跳过 |

Rust 忽略项需要读取真实主机压力传感器。
Python 跳过项需要通过 `APXINF_SERVING_URL` 指定运行中的模型服务。
Rust 库测试耗时 21.17 秒；CLI 测试耗时 22.61 秒，其中包含真实的 20 秒 HTTP 排空等待。

复现命令：

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk cargo test -p apxinf-serving --offline --locked --lib -j 2
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk cargo test -p apxinf-serving --offline --locked --test shutdown_signals -j 2
python3 -m unittest discover -s tests/python -p 'test_serving_*.py'
```

部分测试需要本机回环监听和对子进程执行 `ps`、`kill` 的权限。
新增 Rust 文件和改动文件通过格式检查。
英文契约与本地服务说明通过 STE 结构检查，未进行完整词典审查。

这些结果不验证 GPU 异常恢复，也不证明 64K–128K 上下文的内存容量或稳定性。
