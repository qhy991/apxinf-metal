# M1 Pro（g13s）每进程 ANE bring-up：归因与削减记录

Issue #62 第 1 项。测量机：Apple M1 Pro 16 GB，macOS 27.0.1（applegpu_g13s），
Clef Flash 9B × Mario demo（`MARIO_CLEF_PROTOCOL=separate`）。所有数字来自
#62 评论里的实测（2026-11-08/09，同一 showcase 分支二进制）。

## TL;DR

- 每局 spawn 的 load 是 demo 启动延迟的最大单项：**96.0 s/局**（其中 ~41.5 s 是 load）。
- **削减已落地**：常驻 decide-server（`--listen unix:` + `scripts/decide_client.py`）→
  **54.7 s/局（1.76×）**，per-decision 不变；暖缓存下常驻进程 load 22.7 s 只付一次，
  client 往返 ~0.11 s → 每局启动目标（≤1 s）达成。
- **placement 证据缓存 → NOT-WORTH（按实测）**：暖缓存 load 21.9 s 里 inspect 只占
  ≤5–10 s，且重放证据会削弱 admission gate 的时效性；按 KEEP 门不划算，未落地。
- **真正的杠杆是"别让缓存被清"**：缓存默认上限 16 GiB（PR #66），暖缓存不再被误清；
  冷启动 65 s 只在换 composition 或手动清缓存后付一次，常驻 server 下每局为 0。

## 1. 归因（`sample` 30 s，暖 program cache，auto 组合）

| 阶段 | 证据 | 占 load 的主导程度 |
|---|---|---|
| Core ML 模型加载 | `+[MLModel modelWithContentsOfURL:...]` 272 样本 | 大 |
| **placement inspection** | `walk_plan(MLComputePlan*, ...)` 多分支、`ProgramCache::load_with_placement` 2 968 样本 | **最大** |
| ANE program 编译/写盘 | `ProgramCache::compiled` 仅 2–3 样本 → **缓存命中** | 0 |
| 权重读取/反量化 | `int8_per_row` 2 502、`load_split` 3 844（含上面两项） | 中 |
| 其它 | `sha256::compress` 2 965（key 校验） | 小 |

`load_ms = 62 711`（另一进程 68 092）。program 数：**32 层 × 1 program = 32 个**
（32 dispatches/request，ANE-ATTRIBUTION 同口径）。磁盘缓存命中，所以 60+ s 不是
"每次重新编译"，而是"每次重新实例化 + 重新 inspect placement"。

## 2. 加载时间矩阵（同一二进制，M1 Pro）

| 场景 | load |
|---|---:|
| 冷 program cache，默认 retries（3） | 65.2 s |
| 冷 program cache，retries=1 | 53.8 s |
| **暖 program cache，默认 retries** | **21.9 s** |
| **暖 program cache，retries=1** | **21.3 s** |

读法：暖缓存下 placement retry 只值 0.65 s；`inspect`（walk_plan）只是 21.9 s 的
一部分（其余是 MLModel 实例化、权重反量化、key 校验与 admission）。早先 62–68 s
的归因样本对应的是缓存未暖/被清的状态。

## 3. 削减选项逐项结论

### 3.1 placement 证据缓存 — NOT-WORTH（按实测，未落地）

机制：按 ProgramCache key（model_bytes+weights+SystemVersion，geometry 已含于
model_bytes）+ compute units 缓存 `MLComputePlan` 遍历结果，命中跳过 inspect。

不做的原因（ KEEP 门）：
- 收益上界 = 暖缓存 load 里 inspect 的份额，实测 ≤5–10 s（§2），且只在冷/暖缓存
  切换后的首次加载有意义；
- 风险：重放证据意味着 admission gate（`admit(&facts)`，真正的安全门）对 placement
  的时效性保证变弱（"陈证据放行"）；
- 常驻 server（§3.2）已经把这一成本摊到每局 0，收益面向不复存在。

曾有一版实现草稿（证据以 JSON 存于 `<key>.placement.json`，serde 重放 + 命中仍走
admission + 被拒自愈），按本结论丢弃，未合入（不留死代码）。

### 3.2 常驻 decide-server — 已落地（showcase 分支）

`src/bin/apxinf-decide-server.rs --listen unix:/tmp/clef.sock` + 仓库
`scripts/decide_client.py`（M4 上 PR #48 验证过，管线同一份代码；M1 Pro 本次验证）。

demo 实测（同一二进制、同一 episode，11 决策 / 122 帧）：

| 模式 | wall | per-decision 中位 |
|---|---:|---:|
| 每局 spawn（`APXINF_DECIDE_CMD` = decide-server 本体） | **96.04 s** | 4 545 ms |
| 常驻 server + decide_client | **54.72 s** | 4 575 ms |

**每局墙面 96.0 → 54.7 s（1.76×）**；per-decision 不变（常驻只消除启动，符合预期；
历史 6 599 ms/决策是更早二进制，本次两种模式都 ~4.55 s，per-decision 改善来自
分支本身的 Tier-1 前缀缓存等优化，与常驻无关）。常驻进程 `load_ms` 22 665 ms
（暖 cache；冷启动约 62–68 s），`warm_ms 0`；client 往返（含 python 启动）~0.11 s。

复现：

```sh
# 常驻
$SERV --model <clef> --model-variant auto --listen unix:/tmp/clef.sock
# demo（server 与 evaluate 都要这个 env）
export APXINF_DECIDE_CMD="$PY <repo>/scripts/decide_client.py --addr unix:/tmp/clef.sock"
$PY examples/mario_demo/evaluate.py --policy clef --max-decisions 12 --out /tmp/eval-resident2
```

### 3.3 减少 program 数 — 不 pursued

暖缓存 load 21.9 s 与 program 数基本无关（编译已是 0）；program 数只影响冷启动
编译，而冷启动在 §4 的缓存保留策略下极少发生。收益面向不存在。

## 4. 真正的杠杆：缓存保留（PR #66）

M1 Pro 16 GB 内存下，Core ML 缓存（数十 GiB 量级）此前会被系统/手动清掉，导致
demo 反复付 60+ s"暖"启动。PR #66 给缓存加了默认 **16 GiB 上限**（load 后 trim、
E5 bundle cache 计入），避免误清。冷启动 65 s 只在换 composition 或手动清缓存后
付一次；常驻 server 下每局启动为 0。

## 5. 冷启动后功能验证

清空缓存重启（冷启动）后 4 决策 smoke episode：decisions 4 / frames 62 /
wall 22.5 s / per-decision 中位 4 544 ms，与暖缓存的 4 545–4 575 ms 一致 →
清缓存不改变计算路径。

## 6. 验收对照（Issue #62 第 1 项）

- 归因表（5 阶段 + program 数）：§1、§2 ✓
- 至少一项落地收益：常驻 server，每局 96.0 → 54.7 s（常驻 + 暖 server 时每局
  启动 ~0.11 s，≤1 s 目标达成）✓
- 42 案 parity：证据缓存未落地无需 parity；冷启动 smoke per-decision 与暖缓存
  一致（§5）；demo episode 两种模式下 per-decision 一致（§3.2）✓
- 新开关默认 OFF：未引入新开关（唯一改动是 demo 用法 + PR #66 的缓存上限）✓
