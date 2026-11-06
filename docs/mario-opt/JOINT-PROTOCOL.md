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

输入：proto14 tape（14 帧 × 3 裁图 = 42 个答案，各帧裁图尺寸固定），经
demo 真实 bridge（`ClefLocalBackend.decide`）逐帧回放；两协议每轮各一个
独立 decide-server 进程。MLX 参考用 **bridge 逐字节相同的 payload**：
joint = `joint-demo-style-42`（既有），separate = 新建
`separate-bridge-42`（`DEFAULT_STATE` + `scene\nquestion`，42 案，补上
"demo 真实 separate payload 从未对 MLX 验过"的缺口）。标签取自
`suite/manifest.json`（`crops.*.label`），判定阈值 p > 0.5。

| 比较 | n | 决策一致 | median\|Δp\| | max\|Δp\|（worst） | >0.035 |
| --- | --- | --- | --- | --- | --- |
| A/A separate（r0 vs r1、r0 vs r2） | 42 | 42/42 | 0 | **0**（f32 逐位一致） | 0 |
| A/A joint（r0 vs r1、r0 vs r2） | 42 | 42/42 | 0 | **0**（f32 逐位一致） | 0 |
| **apxinf joint vs apxinf separate**（3 轮完全相同） | 42 | **38/42** | 0.0246 | **0.736**（0045/obstacle） | 14 |
| apxinf joint vs MLX joint | 42 | 42/42 | 0.0025 | **0.274**（0009/obstacle） | 1 |
| apxinf separate vs MLX separate-bridge | 42 | 41/42 | 0.0027 | 0.053（0012/enemy） | 3 |
| **MLX joint vs MLX separate-bridge** | 42 | **39/42** | 0.0275 | **0.748**（0009/obstacle） | 13 |

apxinf joint vs separate 的 4 个翻转（p_joint / p_separate / 标签）：

| case | joint | separate | 标签 | MLX 复现？ |
| --- | --- | --- | --- | --- |
| 0009/obstacle | 0.413 | 0.886 | True | ✓（MLX 0.139 / 0.887）→ joint **漏判障碍** |
| 0013/obstacle | 0.283 | 0.737 | True | ✓（MLX 0.312 / 0.731）→ joint **漏判障碍** |
| 0045/obstacle | 0.040 | 0.776 | False | ✓（MLX 0.042 / 0.770）→ joint 改对 |
| 0012/enemy | 0.093 | 0.526 | True | 阈值边缘：MLX separate 0.473 本身已在另一侧 |

标签准确率（tape 42 案）：apxinf separate **32/42**、apxinf joint
**30/42**、MLX separate-bridge 31/42、MLX joint 30/42（apxinf joint 与
MLX joint 错的 12 案完全相同）。

解读：

1. **三个 obstacle 翻转是模型级的**：参考实现 MLX 在同一对 payload 上
   翻转同样的 3 案、方向相同。joint 改的是模型输入（3 图同上下文 +
   "This question refers to image N of 3."），不是数值等价变换；其中 2 案
   是把真实障碍判成"无"——demo 里会直接撞管道。这一条 ApxInf 侧修不了。
2. **实现侧还有一处偏离**：apxinf joint vs MLX joint 决策 42/42 一致，但
   0009/obstacle |Δp| = 0.274，超过本线 0.035 的门。来源与未发表的前次
   研究（`signpost-timeline-candidates` worktree，未提交的
   `JOINT-PROTOCOL.md` §4）一致：vision tower 把多图 patch 行拼成一条序列做
   linear（M = 1512 而非 432），注意力按图分段；obstacle 裁图对 GEMM 形状
   带来的 ulp 级扰动病态放大（block00 0.18 % 元素不同 → `vision_rows`
   95.3 % 不同），与该图在多图中的位置无关。
3. separate 自身对 bridge-style MLX 的 max|Δp| 是 0.053（0012/enemy，阈值边缘
   翻转）——历史 0.0328 用的是 suite state，不是 demo 真实 payload。这一条
   与 joint 无关，单独记下。

**判定：parity 不过**（行为等价 38/42 = 90.5 %；实现门 0.274 > 0.035）。

## 3. 延迟（阶段 3）

单锁窗（`~/.cache/enginetailor/metal-measure.lock` → `/tmp/enginetailor-metal.lock`，
阻塞等待，持有整个窗口，取锁时无其它持有者），3 轮、奇数轮反序；每轮先
tape、后自然局（`evaluate.py --policy clef --seed 0 --max-decisions 12`，
`MARIO_CLEF_WARM_START=1` 对称暖启）。每臂独立进程、前后 `vm_stat`、每 3 s
扫描外来模型进程。**12/12 臂 exit 0、Swapouts +0、外来进程 0 → 3 轮全干净。**
各臂去掉首决策（Tier-1 填充）后配对；gain = (separate − joint) / separate，
3 轮合并，bootstrap（b = 4000，seed 42）取中位数 CI95。

| 工作负载 | 配对 | separate 中位 | joint 中位 | 省 | gain 中位 | CI95 | joint 更快 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tape（按帧配对，同输入） | 39 | 6788.1 ms | 5991.6 ms | 789.2 ms | **+11.57 %** | [11.32 %, 12.03 %] | 100 % |
| 自然局（按决策序号配对） | 30 | 6782.6 ms | 5996.2 ms | 783.0 ms | **+11.55 %** | [11.24 %, 11.76 %] | 100 % |

各轮 tape gain：+11.69 / +11.54 / +11.57 %；自然局：+11.55 / +11.75 /
+11.17 %。separate 每请求 server 中位：enemy 2180.7、obstacle 2164.8、gap
2446.1 ms；joint 单请求 5991.3 ms。separate tape 最短请求 2145 ms ⇒ 没有
Tier-2 命中，两臂不对称不成立。

自然局两协议在第 2 决策就分叉（separate 把 gap 误判为 0.63 → 跳；joint
0.04 → 跑）：separate 11 决策、x = 313、撞敌死亡（两次假阳跳反而前进更
远）；joint 跑满 12 决策、x = 213、全程未遇到任何正标签。自然局因此只用于
延迟，不能用来比准确率。

**延迟判定：远超 2 %（+11.6 %，CI 下界 +11.2 %）。**

## 4. before/after 算子时间线（阶段 4）

待补。

## 5. 结论

待补。
