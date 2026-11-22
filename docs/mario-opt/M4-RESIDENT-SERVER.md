# M4 mini：常驻 decide-server（socket 模式）

Issue #41 第四棒：把 M4 上 ANE 建立 + 形状预热的成本从「每局」变成「一次」。
授权来自 Issue #41 评论 `6714873058`（基于 #45 合并点 `a60c2c39`）。

- 测量机：Apple M4（Mac mini，16 GB，MLX arch `applegpu_g16g`），macOS 27.0.1
  (26A434)
- 本棒构建：分支 `agent/clef-column-scan-928c4a3c1597` @ `be506fed`，
  sha256 `d073b3f4…2f6`
- 运行配置：`APXINF_CLEF_COMPOSITION=gpu-ane-neurons` +
  `APXINF_CLEF_NEURON_SHARE=70`、3 形状 warm manifest（#45）、
  `--model-variant auto`
- demo：`~/mario-demo/mario-demo` 的 `evaluate.py --policy clef`（separate
  协议，3 请求/决策），经 `scripts/decide_client.py` 连常驻 socket

## 0. 结论

1. **常驻监听模式已实现**（`--listen unix:<path>` / `<host>:<port>`，默认
   OFF）：模型只加载一次，`ready` 前完成 3 形状预热；每个客户端连接收到同一
   `ready` 行后走原 JSONL。**pipe/JSONL 默认路径逐位不变**。
2. **一次启动成本**：load **100.0 s** + warm **20.3 s**（enemy 17.9 /
   obstacle 1.2 / gap 1.2）；之后每局只需 shim spawn + connect +
   ready = **20.6–24.7 ms**（对比 #45 每局 93–130 s load + 16 s warm）。
3. **同一常驻 server 连跑 2 个 12 决策 episode**：
   - ep1：wall **46.5 s**，首决策 **4 245 ms**，每决策 median 3 897 ms
     （3 349–4 319），x 40 → 213，12 决策上限；
   - ep2：wall **44.7 s**，首决策 **4 129 ms**，median 3 776 ms（3 125–4 129），
     x 40 → 213；
   - 每局启动延迟 21–25 ms，局间没有重新加载/重新预热。
4. **对比 #45**（每局一个新进程）：单局 186.1 s（含 load+warm ~110 s）→ 常驻
   单局 46.5 s（load+warm 只在会话开始时付一次）；两局合计 2 × 186 = 372 s →
   **120.3 + 46.5 + 44.7 = 211.5 s**（−43 %），每多一局再省 ~140 s。
5. **demo 零改动接入**：`APXINF_DECIDE_CMD="$PY scripts/decide_client.py
   --addr unix:/tmp/clef.sock"`；shim 转发 ready 与 JSONL，`spawn_ms` 从
   「进程启动 + ANE load」变成「shim spawn + socket connect」。
6. **pipe 模式回归**：reference（GPU-only）pipe 启动 load 2 228 ms、ready
   2 687 ms，3 个 demo crop 请求全部按 JSONL 回答（1.4–2.0 s/请求），默认路径
   未变；预热失败/坏 manifest/bind 失败都是启动错误。
7. 建议：demo runbook 改成「启动一次常驻 server + 每局 shim」；70 % share
   与 3 形状 manifest 不变；默认 OFF 不动。

## 1. 常驻监听模式

`apxinf-decide-server` 新增 `--listen unix:<path>`（或
`--listen <host>:<port>` / `tcp:<host>:<port>`，env
`APXINF_DECIDE_LISTEN`），默认不启用——**pipe/JSONL 的默认路径完全不变**。

- 进程仍然只加载一次 checkpoint，并在 `ready` 前完成 `--warm-shapes`
  （#45 的 3 形状预热）；
- `ready` 行（`load_ms`/`warm_ms`/`warm_shapes`）在启动时打到 stdout，之后
  每个客户端连接建立时再收到同一 `ready` 行；
- 连接上沿用现有 JSONL：每行一个 request，回一行 response；连接关闭后接受
  下一个连接。**单模型、单线程串行**（demo 每局一条连接），并发连接排队，
  这与「一个常驻模型服务多个 episode」的目标一致；
- unix socket：如果路径存在且**能连上**，报错「已被 live server 服务」，
  不抢占；连不上（崩溃残留）才删除后 bind；
- 预热失败或 manifest 坏仍在启动时报错，不静默退冷；监听 bind 失败也是
  启动错误；
- 每局的成本从「进程 spawn + 93–130 s ANE load + 16 s 预热」变成
  「进程 spawn + socket connect（毫秒级）」。

## 2. demo 接入：decide_client.py

`scripts/decide_client.py`（~80 行）把 stdin/stdout 原样转发到常驻 socket：

```bash
# 一次：启动常驻 server（加载 + 预热只发生一次）
apxinf-decide-server --model <ckpt> --model-variant auto \
  --listen unix:/tmp/clef.sock --warm-shapes devlocal/clef-demo-warmup/warm-shapes.json

# 每局：demo 侧零改动，只把 APXINF_DECIDE_CMD 指到 shim
export APXINF_DECIDE_CMD="$PY scripts/decide_client.py --addr unix:/tmp/clef.sock"
$PY evaluate.py --policy clef --max-decisions 12 --out <episode>
```

shim 连接（带重试/超时）、转发 server 的 `ready` 行，然后双向搬 JSONL；
stdin EOF 时半关闭写端，等 server 处理完关闭连接后退出。demo 的
`ClefLocalWorker` 只看 shim 的 stdout，所以 demo 不需要任何改动；
`APXINF_DECIDE_TIMING_FILE` 的 `spawn_ms`/`ready_ms` 现在反映的是
shim 的 spawn + connect，而不是 ANE load。

## 3. pipe 模式回归

默认 pipe 路径在本次改动后由 reference（GPU-only）快速回归（同一二进制）：

| 项 | 值 |
| --- | ---: |
| variant | reference |
| load_ms | 2 228 |
| ready（spawn→ready） | 2 687 ms |
| enemy 320×400 | 2 006 ms，noul 0.0218 |
| obstacle 384×288 | 1 432 ms，noul 0.0601 |
| gap 480×320 | 1 634 ms，noul 0.0700 |

3 个请求都按「一行 request、一行 response」返回，`answers.<name>.noul`
在 [0,1]；pipe 模式没有行为变化（单元测试 5 个：base64、媒体形式、manifest、
solid PNG、listen 地址解析）。

## 4. 测量：一次启动，两局 episode

同一常驻进程（`--listen unix:/tmp/clef-resident.sock` + 3 形状 warm +
70 % share），demo 的 `evaluate.py --policy clef --max-decisions 12` 连跑两次，
每局经 `scripts/decide_client.py` 连 socket：

| 项 | 值 |
| --- | --- |
| 常驻启动 load_ms | **100 005** |
| 常驻启动 warm_ms | **20 337**（enemy 17 880 / obstacle 1 246 / gap 1 206） |
| 常驻启动合计 | **120.3 s**（一次性，发生在第一局之前） |
| ep1 spawn → ready | **20.6 / 23.8 ms** |
| ep1 首决策 | **4 245 ms** |
| ep1 每决策 median（min–max） | **3 897 ms**（3 349–4 319） |
| ep1 总墙钟 / 决策数 / x | **46.5 s** / 12 / 40 → 213 |
| ep2 spawn → ready | **21.7 / 24.7 ms** |
| ep2 首决策 | **4 129 ms** |
| ep2 每决策 median（min–max） | **3 776 ms**（3 125–4 129） |
| ep2 总墙钟 / 决策数 / x | **44.7 s** / 12 / 40 → 213 |

- 两局都跑到 12 决策上限（decision_limit），未死亡；x 序列相同
  （41→47→56→70→87→105→123→141→159→177→195→213）。
- 局间没有重新加载/重新预热：ep2 的 spawn→ready 仍是 24.7 ms，首决策
  4.1 s（steady），与 ep1 一致。
- 常驻进程在 demo 之外管理（启动一次、结束 kill）；shim 每局由 demo spawn，
  退出时半关闭 socket，server 处理完关闭连接后继续 accept 下一局。

## 5. 与 #45（每局一个新进程）对比

| 项 | #45（每局新进程） | 常驻（本棒） |
| --- | ---: | ---: |
| 每局启动 | 93–130 s load + 16 s warm | **21–25 ms**（shim spawn + connect） |
| 第一局前一次性 | 0（每局都付） | **120.3 s**（load 100.0 + warm 20.3） |
| 单局总墙钟（12 决策） | 186.1 s | **46.5 s / 44.7 s** |
| 首决策 | 3 954 ms | **4 245 / 4 129 ms**（steady，无形状编译） |
| 每决策 median | 3 865 ms | **3 897 / 3 776 ms** |
| 两局合计 | 2 × 186.1 = 372.2 s | **211.5 s**（−43 %） |
| 三局合计 | 558.3 s | **256.2 s**（−54 %） |

首决策/每决策的差异在批内噪声（#45 的 3 954/3 865 与常驻的 4 129–4 245/
3 776–3 897 同一量级）；差别完全在「每局是否重付 ANE load + 预热」。

## 6. 限制

- **单模型、单线程串行**：并发连接排队（demo 每局一条连接）；多客户端并行
  需要多模型或 `--concurrency` 语义，内存代价是 N × 模型。
- **常驻进程生命周期**：由 launcher/runbook 管理（启动一次、结束 kill）；
  崩溃会留下 unix socket 文件——下次启动时如果连不上会删除残留再 bind，
  能连上则报错「已被服务」，不抢占。
- **内存**：常驻进程一直持有 70 % share 的 ANE program/权重（~3.4 GB cache，
  进程 RSS 更高），16 GB 主机上会长期占内存；demo 局间建议保留（否则又回到
  每局 load）。
- **shim 每局 spawn**：~20–25 ms + 一个 Python 进程；不是零成本，但相比
  93–130 s 可忽略。
- **socket 路径/权限**：unix socket 的目录权限与 demo 用户一致；跨用户/容器
  需要 tcp 模式或权限设置。
- **测量口径**：同图同语料、70 % share、3 形状 warm；两局都是 12 决策上限，
  未到旗杆；x 序列相同（同一模型/动作）。

## 7. 复现（runbook）

```bash
# 0) 编译（或复用 ~/work-clef-showcase/target/release/apxinf-decide-server）
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export MLX_ROOT="$HOME/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
cargo build --release --features mlx,coreml --bin apxinf-decide-server

# 1) 启动一次常驻 server（load + 3 形状预热只发生一次）
export APXINF_COREML_CACHE="$HOME/m4-demo/coreml-cache-resident"
export APXINF_CLEF_COMPOSITION=gpu-ane-neurons
export APXINF_CLEF_NEURON_SHARE=70
apxinf-decide-server --model <clef checkpoint> --model-variant auto   --listen unix:/tmp/clef.sock   --warm-shapes devlocal/clef-demo-warmup/warm-shapes.json

# 2) 每局：demo 零改动，只换 APXINF_DECIDE_CMD
export APXINF_DECIDE_CMD="$PY scripts/decide_client.py --addr unix:/tmp/clef.sock"
$PY evaluate.py --policy clef --max-decisions 12 --out <episode-1>
$PY evaluate.py --policy clef --max-decisions 12 --out <episode-2>

# 3) 结束
kill <server pid>; rm -f /tmp/clef.sock
```

本棒的测量由 `devlocal/clef-resident/scripts/session_resident.sh` 在
EngineTailor 双锁内一次完成（pipe 回归 → 常驻启动 → 2 局 → kill）；原始
run/trace/日志在 `devlocal/clef-resident/`（gitignored 中间产物）。

## 8. 交接（Direction / Evidence / Next candidates / Cost）

- **Direction**：demo 改成「常驻 server 一次加载 + 每局 shim 连接」；每局
  启动从 93–130 s + 16 s 降到 21–25 ms；load+warm 只在会话开始时付一次。
  pipe 默认行为不变，`--listen` 默认 OFF。
- **Evidence**：常驻 load 100.0 s + warm 20.3 s；ep1 46.5 s / 首决策 4 245 ms
  / median 3 897 ms；ep2 44.7 s / 4 129 ms / 3 776 ms；每局 spawn→ready
  21–25 ms；两局合计 211.5 s vs #45 的 372.2 s（−43 %）；reference pipe
  回归 3/3 请求通过。
- **Next candidates**：
  1. 把「常驻 server 生命周期」接进 demo 的正式 launcher（启动/健康检查/
     退出清理），而不是手工 kill；
  2. 多客户端并发（`--concurrency N` 或多模型池）——当前单线程串行；
  3. 常驻进程的 RSS/ANE residency 长期监控（16 GB 主机上 70 % share 的
     3.4 GB 权重会一直占着）；
  4. runbook 里把 socket 路径/权限/残留清理写成标准步骤。
- **Cost**：本棒 ≈ 1.5 h 墙钟（含 ~0.3 h 锁等待与编译），2 次 Core ML 编译
  轮次（reference 不需要），Opus 0 次。

Opus: 0 次。
