# Clef `joint` / `separate` parity — 独立复核

本文件复核 `joint`（1 请求 × 3 图 3 问）与 `separate`（3 请求 × 1 图 1 问）
在**同一批裁图**上的答案一致性，并回答一个问题：`joint` 的 +11.6 % 延迟收益
能不能按现有 parity 门落地。

- 与并行测量会话的 `JOINT-PROTOCOL.md` §2 的关系：那份表由该会话产出；本文件在**另一 ticket worktree**（`clef-joint-parity-00b117478736`）
  上只读原始 tape / MLX 记录重新计算，补上逐案 ledger、唯一帧去重和
  “实现门 / 行为门”的拆分。数字与 §2 逐项一致。
- 本棒**没有**重录设备测量或时间线；tape 来自并行会话的
  `p12-campaign-20261107-061556` clean window（3 轮 × 2 协议，12/12 臂
  Swapouts +0），时间线见该会话的 `timeline-joint-before-after.png`。
- Opus：本棒 0 次。

## 1. 输入与等价性

| 输入 | 内容 |
| --- | --- |
| tape | `separate-proto14` 的 14 帧 × 3 裁图（enemy / obstacle / gap），经 demo 真实 bridge `ClefLocalBackend.decide` 回放；两协议每轮各一个独立 `apxinf-decide-server` 进程 |
| 服务端 | `apxinf-decide-server`，`--model-variant reference`，sha256 `ee6fc881e400…f26d`（构建自 `showcase/clef-vision` @ `3ad92d69`） |
| MLX joint | `baseline/results/joint-demo-style-42`：state = `Answer each question using only the picture it refers to.`，每题题面带 `This question refers to image N of 3.`，3 图一次请求 |
| MLX separate | `results/mlx-separate-bridge-42`：同一 state，1 图 1 问，42 案（补齐“demo 真实 separate payload 从未对 MLX 验过”的缺口） |
| 标签 | `suite/manifest.json` 的 `crops.*.label`（RAM predicate），判定阈值 p > 0.5 |

等价性：两协议 tape 的每一帧 `crop_sha` 逐字节相同；A/A 为同一协议跨轮
（3 个独立进程）比较。

**伪重复**：proto14 的 14 帧里 `0014` / `0060` / `0086` 与 `0000` 的裁图
逐字节相同，所以 42 条答案只有 **33 个唯一 case**。下表同时给 42-case 与
33-unique 数字；翻转集合不变，只是 `42` 的计数把 `0000` 的 3 个 case
重复了 4 次。

## 2. 结果

| 比较 | n | 决策一致 | median \|Δp\| | max \|Δp\|（worst） | >0.035 |
| --- | --- | --- | ---: | ---: | ---: |
| A/A joint（r0–r1 / r0–r2 / r1–r2） | 42 | 42/42 ×3 | 0 | **0**（f32 逐位） | 0 |
| A/A separate（同上 3 对） | 42 | 42/42 ×3 | 0 | **0**（f32 逐位） | 0 |
| **apxinf joint vs apxinf separate**（3 轮相同） | 42 | **38/42（90.5 %）** | 0.0246 | **0.7360**（0045/obstacle） | 14 |
| ↳ 去重后 | 33 | 29/33（87.9 %） | 0.0272 | 0.7360 | 13 |
| **apxinf joint vs MLX joint**（实现门） | 42 | **42/42** | 0.00245 | **0.2736**（0009/obstacle） | 1 |
| ↳ 去掉 0009/obstacle | 41 | 41/41 | — | **0.0308** | 0 |
| **apxinf separate vs MLX separate-bridge** | 42 | 41/42（97.6 %） | 0.00273 | 0.0527（0012/enemy） | 3 |
| **MLX joint vs MLX separate-bridge**（模型级对照） | 42 | **39/42（92.9 %）** | 0.0275 | 0.7478（0009/obstacle） | 13 |

标签准确率（仅作 sanity，不作为 parity 门）：apxinf joint **30/42**、
apxinf separate **32/42**、MLX joint **30/42**、MLX separate-bridge 31/42。
apxinf joint 与 MLX joint 的 12 个错案**完全相同**。

## 3. 四个翻转

| case | 标签 | apxinf sep | apxinf joint | MLX sep | MLX joint | 解释 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0009/obstacle | T | 0.8856 → **T** | 0.4130 → **F** | 0.8872 → T | 0.1394 → F | 模型级：joint 漏判真实障碍 |
| 0013/obstacle | T | 0.7372 → **T** | 0.2830 → **F** | 0.7311 → T | 0.3124 → F | 模型级：joint 漏判真实障碍 |
| 0045/obstacle | F | 0.7759 → T（错） | 0.0399 → **F**（对） | 0.7698 → T | 0.0422 → F | 模型级：joint 纠正一个假阳 |
| 0012/enemy | T | 0.5259 → **T** | 0.0933 → **F** | 0.4732 → F | 0.1055 → F | 阈值边缘：apxinf separate 与 MLX separate 本身已在两侧；apxinf joint 与 MLX joint 同侧 |

前 3 个翻转在参考实现 MLX 上方向、位置完全相同，说明它们来自 **joint 的
输入语义变化**（3 图同上下文 + “This question refers to image N of 3.”），
不是 apxinf 数值实现的问题 —— 这一条在 ApxInf 侧修不了。第 4 个是
separate 路径在阈值边缘（0.526 vs MLX 0.473）的已知敏感案，与 joint 实现无关。

## 4. 实现门的那一个超门 case

`apxinf joint vs MLX joint` 唯一超过 0.035 的是 `0009/obstacle`
（0.4130 vs 0.1394，|Δp| = 0.2736），但：

- 两者**决策一致**（都判 F），且 42/42 决策一致；
- 同案 MLX 自己的 joint 与 separate 相差 **0.748**（0.1394 vs 0.8872），
  是整批 42 案里对输入上下文最敏感的点；
- 去掉该案后其余 41 案的 max |Δp| = **0.0308 ≤ 0.035**，median 0.0025。

因此最可能的解释是：**joint 的数值实现本身是忠实的**（相对 MLX joint：
42/42 决策、41/42 概率在门内、错案集合相同）；0.274 是模型在该 case 的
敏感态上对微小数值差的放大；决策不受影响，也不是改协议能改变的行为差异。
它仍是唯一一个未解释的实现侧偏离，需要多图 vision 的逐 stage dump 才能
定论（见 §7 候选）。

## 5. 判定

按本线 PR #5 的 standing gate（**42/42 决策 + max|Δp| ≤ 0.035**）：

| 门 | 结果 | 判定 |
| --- | --- | --- |
| 实现门：apxinf joint vs MLX joint | 42/42 决策；41/42 ≤0.035；单点 0.2736 | **单点超门** |
| 行为门：apxinf joint vs apxinf separate | 38/42；median 0.0246；max 0.7360 | **不通过** |

**`joint` 不能作为 `separate` 的 drop-in 替代**：它改变 4/42 个答案
（4/33 唯一案）。其中 3 个是模型级变化（MLX 复现）：2 个真实障碍漏判、
1 个假阳纠正；第 4 个（0012/enemy）是 separate 侧阈值边缘的 1 个真实敌人
漏判（joint 与 MLX joint 同侧）。2 个障碍漏判在 demo 里会直接撞管道。+11.55 %（CI95 [11.24, 11.76]）的延迟收益
是真实的（同输入 tape、3 轮全干净），但按现有 parity 门**不可落地**；
是否接受“换协议 + 换答案”是产品决策，不是 ApxInf 的实现优化。

## 6. 证据与复现

原始证据（只读，本棒未重跑设备）：

- tape 与 A/A：`devlocal/mario-opt/joint-protocol/runs/p12-campaign-20261107-061556/r{0,1,2}-tape-{joint,separate}/tape/records.jsonl`
- MLX joint：`mario-showcase-20261103/baseline/results/joint-demo-style-42`
- MLX separate-bridge：`devlocal/mario-opt/joint-protocol/results/mlx-separate-bridge-42`
- 时间线（并行会话录制）：`devlocal/mario-opt/joint-protocol/runs/timeline-joint-before-after.png`；
  reference 窗口 6854.9 → 6039.0 ms，GPU busy 97.5 → 98.7 %，bubble 2.46 → 1.27 %。

本 ticket worktree 的复核脚本（git-ignored）：

```bash
python3 devlocal/clef-joint-parity/scripts/parity_analysis.py \
  --out devlocal/clef-joint-parity/results/parity.json
python3 devlocal/clef-joint-parity/scripts/parity_table.py \
  > devlocal/clef-joint-parity/results/parity-table.md
```

复现用的服务端（clean-window 已构建，未在本棒重建）：

```bash
/Users/haiyan/Documents/Infinity/Agent4Kernel/apxinf-worktrees/clef-joint-protocol/target/release/apxinf-decide-server \
  --model /Users/haiyan/Documents/Infinity/Agent4Kernel/models/clef-flash-4bit-b5adbe2 \
  --model-variant reference
```

## 7. 下一步候选

1. **语义保真的“半 joint”**：只合并 preprocess + vision，仍用 3 个
   separate 语言 pass（各自单图 prompt）。机制预期：省 2 次 preprocess /
   request setup，但保不住 64 次 layer 调用；预期 <2 %，falsifier = 实测
   <2 % 即 NOT-WORTH。
2. **接受 joint 为产品/协议变更**：joint 自然局跑满 12 决策且未死亡
   （separate 11 决策撞敌死亡，x=313 vs 213），但答案语义已变；需要人拍板
   “是否接受 2 个障碍漏判 + 1 个假阳纠正”，不能由 parity 自动通过。
3. **定论 0009/obstacle 的 0.274**：对 frame 0009 做多图 vs 单图的
   vision 逐 stage dump（`APXINF_CLEF_DUMP_STAGES`），确认是否已知的
   obstacle-only 多图数值放大；falsifier = vision rows 在 ulp 内、偏离出现在
   text/head，则说明还有一处未发现的实现差异。

## 附录：42 案逐案 ledger

`u=Y` 表示唯一帧（`0014/0060/0086` 与 `0000` 逐字节相同，只是重复）。
`sep/joint` 列：T/F 为 p>0.5 后的决策；`flip` 表示两协议决策不同。
`Δ(j−s) = |p_joint − p_separate|`；`Δ(j−MLXj) = |p_joint − p_MLX_joint|`。

| frame | u | q | label | sep | joint | MLX sep | MLX joint | sep/joint | Δ(j−s) | Δ(j−MLXj) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0000 | Y | enemy | F | 0.0219 | 0.0410 | 0.0239 | 0.0396 | F/F = | 0.0190 | 0.0014 |
| 0000 | Y | gap | F | 0.0775 | 0.0439 | 0.0742 | 0.0458 | F/F = | 0.0336 | 0.0019 |
| 0000 | Y | obstacle | F | 0.0562 | 0.0330 | 0.0649 | 0.0304 | F/F = | 0.0233 | 0.0026 |
| 0009 | Y | enemy | F | 0.0165 | 0.0090 | 0.0163 | 0.0087 | F/F = | 0.0075 | 0.0003 |
| 0009 | Y | gap | F | 0.0421 | 0.0460 | 0.0443 | 0.0432 | F/F = | 0.0039 | 0.0028 |
| 0009 | Y | obstacle | T | 0.8856 | 0.4130 | 0.8872 | 0.1394 | T/F **flip** | 0.4726 | 0.2736 |
| 0012 | Y | enemy | T | 0.5259 | 0.0933 | 0.4732 | 0.1055 | T/F **flip** | 0.4325 | 0.0121 |
| 0012 | Y | gap | F | 0.0651 | 0.0429 | 0.0571 | 0.0422 | F/F = | 0.0222 | 0.0006 |
| 0012 | Y | obstacle | T | 0.1541 | 0.1839 | 0.1541 | 0.1530 | F/F = | 0.0298 | 0.0308 |
| 0013 | Y | enemy | T | 0.0901 | 0.0373 | 0.0927 | 0.0372 | F/F = | 0.0528 | 0.0001 |
| 0013 | Y | gap | F | 0.0478 | 0.0570 | 0.0542 | 0.0610 | F/F = | 0.0092 | 0.0040 |
| 0013 | Y | obstacle | T | 0.7372 | 0.2830 | 0.7311 | 0.3124 | T/F **flip** | 0.4542 | 0.0294 |
| 0014 | · | enemy | F | 0.0225 | 0.0403 | 0.0239 | 0.0396 | F/F = | 0.0179 | 0.0007 |
| 0014 | · | gap | F | 0.0770 | 0.0427 | 0.0742 | 0.0458 | F/F = | 0.0342 | 0.0031 |
| 0014 | · | obstacle | F | 0.0573 | 0.0327 | 0.0649 | 0.0304 | F/F = | 0.0246 | 0.0023 |
| 0029 | Y | enemy | T | 0.1836 | 0.0814 | 0.1895 | 0.0806 | F/F = | 0.1022 | 0.0007 |
| 0029 | Y | gap | F | 0.0298 | 0.0388 | 0.0323 | 0.0338 | F/F = | 0.0090 | 0.0049 |
| 0029 | Y | obstacle | F | 0.0357 | 0.0448 | 0.0388 | 0.0390 | F/F = | 0.0092 | 0.0058 |
| 0045 | Y | enemy | F | 0.4244 | 0.2330 | 0.4707 | 0.2408 | F/F = | 0.1914 | 0.0078 |
| 0045 | Y | gap | T | 0.0223 | 0.1381 | 0.0228 | 0.1291 | F/F = | 0.1158 | 0.0090 |
| 0045 | Y | obstacle | F | 0.7759 | 0.0399 | 0.7698 | 0.0422 | T/F **flip** | 0.7360 | 0.0023 |
| 0053 | Y | enemy | F | 0.0496 | 0.1078 | 0.0399 | 0.0974 | F/F = | 0.0582 | 0.0104 |
| 0053 | Y | gap | T | 0.0233 | 0.0505 | 0.0232 | 0.0496 | F/F = | 0.0272 | 0.0009 |
| 0053 | Y | obstacle | F | 0.0179 | 0.0292 | 0.0187 | 0.0284 | F/F = | 0.0113 | 0.0008 |
| 0059 | Y | enemy | T | 0.1750 | 0.1597 | 0.1577 | 0.1478 | F/F = | 0.0152 | 0.0119 |
| 0059 | Y | gap | F | 0.0213 | 0.0337 | 0.0208 | 0.0327 | F/F = | 0.0124 | 0.0010 |
| 0059 | Y | obstacle | T | 0.3540 | 0.1567 | 0.3895 | 0.1596 | F/F = | 0.1972 | 0.0029 |
| 0060 | · | enemy | F | 0.0225 | 0.0403 | 0.0239 | 0.0396 | F/F = | 0.0179 | 0.0007 |
| 0060 | · | gap | F | 0.0770 | 0.0427 | 0.0742 | 0.0458 | F/F = | 0.0342 | 0.0031 |
| 0060 | · | obstacle | F | 0.0573 | 0.0327 | 0.0649 | 0.0304 | F/F = | 0.0246 | 0.0023 |
| 0065 | Y | enemy | T | 0.3720 | 0.1691 | 0.3675 | 0.1886 | F/F = | 0.2029 | 0.0195 |
| 0065 | Y | gap | F | 0.0321 | 0.0781 | 0.0336 | 0.0601 | F/F = | 0.0460 | 0.0180 |
| 0065 | Y | obstacle | F | 0.0254 | 0.0281 | 0.0250 | 0.0281 | F/F = | 0.0027 | 0.0000 |
| 0085 | Y | enemy | F | 0.0601 | 0.0489 | 0.0544 | 0.0467 | F/F = | 0.0112 | 0.0021 |
| 0085 | Y | gap | F | 0.0605 | 0.3473 | 0.0562 | 0.3416 | F/F = | 0.2868 | 0.0057 |
| 0085 | Y | obstacle | F | 0.0487 | 0.0706 | 0.0496 | 0.0592 | F/F = | 0.0219 | 0.0113 |
| 0086 | · | enemy | F | 0.0225 | 0.0403 | 0.0239 | 0.0396 | F/F = | 0.0179 | 0.0007 |
| 0086 | · | gap | F | 0.0770 | 0.0427 | 0.0742 | 0.0458 | F/F = | 0.0342 | 0.0031 |
| 0086 | · | obstacle | F | 0.0573 | 0.0327 | 0.0649 | 0.0304 | F/F = | 0.0246 | 0.0023 |
| 0104 | Y | enemy | T | 0.0601 | 0.0443 | 0.0619 | 0.0458 | F/F = | 0.0157 | 0.0014 |
| 0104 | Y | gap | F | 0.0373 | 0.0850 | 0.0326 | 0.0837 | F/F = | 0.0477 | 0.0013 |
| 0104 | Y | obstacle | F | 0.0211 | 0.0327 | 0.0213 | 0.0324 | F/F = | 0.0116 | 0.0003 |
