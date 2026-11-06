# Clef `joint` 协议 vs `separate` 协议（端到端）

"删工作"型候选：demo 每个决策问 3 个问题（enemy / obstacle / gap，各一张裁图）。

- `separate`（现状）：3 个请求 × 1 图 1 问，demo 端到端 reference 6599 ms/决策
  （[HANDOVER.md](HANDOVER.md)）
- `joint`：1 个请求 × 3 图 3 问（`MARIO_CLEF_PROTOCOL=joint`，bridge
  `examples/mario_demo/clef_local.py` 已有；`apxinf-decide-server` 单请求 ≤4 图）

测量环境：

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1
- 二进制：`apxinf-decide-server`（`--features mlx,coreml`），构建自
  `origin/showcase/clef-vision` @ `3ad92d69`，sha256 `ee6fc881e400…f26d`
- tier：**reference**（6599 ms 基线所在 tier）；时间线另录 `auto` 以回答
  ANE/GPU 重叠问题
- 证据与脚本：`devlocal/mario-opt/joint-protocol/`（git-ignored）

## 1. 环境确认（阶段 1）

`joint` **开箱可用，未改任何代码**。冒烟（每臂独立 decide-server 进程，
tape 2 决策 + 自然局 2 决策，单锁窗）4 臂全部 exit 0、Swapouts +0；joint
每决策恰好 1 个请求、3 图、3 个 `noul` 答案，bridge 正常解析。

前提修正：Tier-1 前缀缓存（默认 ON）已经让**同一进程内第一个请求之后**的
所有请求跳过 37-token chunk0。冒烟里 separate 的首请求 3308 ms，其后
2160–2436 ms/请求。所以 joint 省下的**不是** "2 × chunk0（1946.8 ms）"，而是
chunk0 之外的每请求固定成本（逐层 launch、head、sync、preprocess/encode、
请求间空隙）——由 §4 时间线量化。

冒烟已见一个决策翻转：frame `0009` 的 obstacle，separate 0.8856 → joint
0.4130。完整 parity 见 §2。

## 2. Parity（阶段 2）

待补。

## 3. 延迟（阶段 3）

待补。

## 4. before/after 算子时间线（阶段 4）

待补。

## 5. 结论

待补。
