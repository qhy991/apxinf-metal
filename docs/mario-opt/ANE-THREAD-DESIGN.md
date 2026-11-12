# ANE 专用线程 / ANE-GPU 层内流水 — Phase 0/1 设计与上界

审计 §3 #1（megakernel 线唯一从未实现的候选）。本棒只做**事实、依赖图、
Amdahl 上界、Opus 设计和失败判据**，不写实现代码。对象是 **auto（GPU+ANE
列拆分）**，reference 档不涉及 ANE。

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1，Core ML 9.0，MLX 0.32.3
- 证据：`docs/mario-opt/TIMELINE-OPERATOR.md`（单 auto 请求 3453.2 ms 的
  xctrace；本机桌面 GPU 干扰、Swapouts≠0，只作结构证据）+ 代码
- Opus：**1 次**（medium，117 s，6448 B）

## 0. 结论（先给判定）

1. **同层 ANE 与 GPU 已经并行**：`pack` 和 GPU share 先入队，host 只等
   `packed`（GPU pack），Core ML predict 期间 GPU share 继续跑；90.1 % 的
   我方 GPU busy 落在 ANE wait/predict 的 host 阻塞窗内。
2. **跨层 ANE(N+1) ∥ GPU(N) 是硬依赖，数学上做不到**：下一层入口的
   RMSNorm 需要完整 `m_N = gpu_N + ane_N`；线性延迟 add 不能缩短关键路径，
   512-token tile 也堵死了按 token 的波前流水。
3. 只改调度的上界：稳态 chunk1 的 host 间隙 **69.5 ms = 5.4 %**（首个请求
   3453 ms 的 2.0 %）；**现实能拿到的 ≤2 %**，低于 KEEP 门。**这个方向对
   >10 % 是死的**，只剩 ≤5.4 % 理论 / ≤2 % 现实的 host 间隙，而专用线程
   只能消掉其中一部分。
4. 因此**本阶段不写实现**；若仍要做，最小设计见 §6，前置判据见 §7。

## 1. 当前调度事实（auto 组合，代码）

`crates/apxinf-model/src/metal_clef/model.rs` 的 `mlp`（GPU+ANE 分支）：

```rust
let packed = AneMlp::pack(h)?;          // GPU kernel: [1,T,4096] -> FP16 512-row tiles
let gpu = c.mlp(h, m)?;                  // GPU share (6656 columns)
self.stream.async_eval(&[packed])?;      // queue pack
self.stream.async_eval(&[gpu])?;         // queue GPU share, behind pack
let share = ane.predict(&self.stream, &packed, tokens, index)?;
return gpu.add(&share);
```

`AneMlp::predict`（`ane.rs`）：

```rust
let tiles = { let _wait = signpost_interval!(ANE_WAIT, ...); packed.to_bytes_ready()? };
// ... bytes -> f16 ...
let out = { let _pred = signpost_interval!(ANE_PREDICT, ...);
            self.model.predict(&[("x", tile, ...)], ...)? };
```

- `ANE_WAIT` = host `to_bytes_ready(packed)`，**等的是 GPU 把 pack 算完并
  读回**，不是等 ANE；GPU share 在 pack 之后排队，等 pack 期间 GPU 继续跑。
- `ANE_PREDICT` = 同步 Core ML `predict`（每层一次；chunk1 266 token 仍是
  1 个 512-row tile）。host 在这里阻塞；GPU share 并行执行。
- 合并 `gpu.add(share)` 之后才轮到下一层的 mixer；所以 **GPU(N+1) 与
  ANE(N+1) 都依赖 ANE(N) 的输出**。
- `apxinf-coreml::Model` 明确 **Not `Send`：prediction 运行在拥有线程**
  （`runtime.rs:133`）；同步 predict 没有公开的并发保证，ANE 程序是否
  真正并发也未知（Opus §1）。

## 2. 时间线证据（单 auto 请求 3453.2 ms）

| 分量 | 时长 | 占请求 | 64 次 / 备注 |
| --- | ---: | ---: | --- |
| `ane_predict`（Core ML） | 1599.7 ms | 46.3 % | 64 次，median 26.8 ms |
| `ane_wait`（等 GPU pack） | 1321.6 ms | 38.3 % | 64 次，median 17.6 ms |
| layer 内 host 图构建残差 | 319.3 ms | 9.2 % | chunk0 249.7 + chunk1 69.5 |
| host stages（pre/enc/vision/head/sync） | 212.3 ms | 6.1 % | — |
| GPU busy | 1907.2 ms | 55.2 % | 硬件 command buffer |

重叠（host 窗口口径）：

| 窗口 | 覆盖我方 GPU | 说明 |
| --- | ---: | --- |
| wait 窗口（全部） | 995.1 ms（52.2 % of GPU busy） | 等 pack 时 GPU 在跑 pack/share |
| predict 窗口（全部） | 722.9 ms（37.9 %） | ANE predict 时 GPU share 在跑 |
| 合计 | **1718.0 ms = 90.1 % of GPU busy** | GPU 基本在 ANE 阻塞窗里执行 |

稳态 chunk1（32 层，1293.8 ms）：wait 562.2（**96.6 %** 有 GPU）、
predict 662.1（**73.9 %** 有 GPU）→ predict 内 GPU 无我方工作的
**172.8 ms（13.4 % of chunk1）**。chunk0（1946.8 ms）：wait 759.5（59.5 %）、
predict 937.6（24.9 %）；chunk0 是首请求冷启动，Tier-1 之后稳态不再付。

> 注：timeline 的 8.7 % “真气泡”里含 WindowServer/Ventura 的桌面 GPU 争用；
> 结构性可动的 host 间隙只有 chunk1 的 69.5 ms（§4）。§C5 明确本录制受桌面
> GPU 干扰（请求窗内 4917 条 GPU interval，我方 525），Swapouts +63980；
> 绝对时延与 busy 比例不可外推，结构（顺序、依赖、重叠窗）可信。

## 3. 依赖图（硬串行在哪）

```mermaid
flowchart LR
  A_N[ANE(N) predict] --> M_N[merge m_N = gpu_N + ane_N]
  M_N --> H_N1[h_{N+1} = h_N + r_N + m_N]
  H_N1 --> R_N1[RMSNorm / mixer(N+1) GPU]
  R_N1 --> P_N1[pack(N+1) GPU]
  P_N1 --> A_N1[ANE(N+1) predict]
  R_N1 --> G_N1[GPU share(N+1)]
  G_N1 --> M_N1[merge(N+1)]
  A_N1 --> M_N1
```

- 同层：`P_N1` 之后，`G_N1`（GPU share）与 `A_N1`（ANE）**可并行**，当前
  代码已经这样排。
- 跨层：`A_N1` 与 `G_N1` 都要等 `M_N`，而 `M_N` 要等 `A_N` → **ANE(N)
  完成之前，GPU(N+1) 与 ANE(N+1) 都不能开始**；`ANE(N+1)` 还要额外经过
  mixer(N+1)。
- 为什么不能推迟 ANE share 的加法：下一层入口 `RMSNorm(h_{N+1})` 的缩放
  系数 `1/rms(a+b)` 含 `a·b` 交叉项，必须知道完整的 `b`；线性延迟
  `W·g⊙(a+b)=W·g⊙a+W·g⊙b` 里补算 `W·g⊙b` 的 FLOPs 与整个投影同级，
  关键路径不缩短；把 `W·diag(g)·W_down_ane` 预合成会让输入从 4096 变成
  5632 维，更差。近似（用上一层 rms）会改数值，joint 的 38/42 已说明这个
  模型对漂移敏感，排除。
- 按 token 的波前（ANE(N,tile1) ∥ GPU(N+1,tile0)）被 **512-token tile**
  堵死：266 token 本来就是一个 tile，切小只会让 ANE 按 512 跑多次，总时间
  成倍增加。

## 4. Amdahl 上界（用实测数字）

### 4.1 稳态 chunk1（1293.8 ms，32 层）

| 每层分量 | ms | 能否被调度删除 |
| --- | ---: | --- |
| wait（GPU mixer+pack，ANE 输入） | 17.57 | **不能**（真实串行依赖；96.6 % 时间 GPU 在跑） |
| predict 中与 GPU share 重叠 | 15.29 | 已经并行 |
| predict 中只有 ANE（GPU 空） | 5.40 | **不能**：没有独立 GPU 工作可填；这是列配比问题（172.8 ms） |
| host 间隙（merge/encode/唤醒） | 2.17 | **只有这块可动**（69.5 ms） |
| 合计 | **40.43** | 1293.8 ms |

- **同层完全重叠**：`17.57 + max(15.29, 20.69) + 2.17 = 40.43 ms` —— 与
  现状逐位相同，**同层已经到理论上界**，没有剩余。
- **跨层完全重叠（假想可行）**：每层关键路径 = `max(GPU 32.86, ANE 20.69)`
  ≈ 32.86 → chunk1 ≈ 1051 ms，−18.7 %；但依赖图证明不可行，不能作为上界。
- **只改调度（专用线程/双缓冲）**：最多消掉 host 间隙 69.5 ms =
  **5.4 % of chunk1**、**2.0 % of 3453 ms**；其中 merge/下一层 encode 仍
  依赖 `m_N`，线程只能省唤醒/阻塞，现实 **≤2 %**。
- **首个请求的 chunk0**：layer host 残差 249.7 ms（7.2 % of 3453；其中大量是
  冷启动 + 桌面干扰），即使全消掉也只在首请求有意义，稳态不付；全部
  layer host 残差 319.3 ms = 9.2 %（chunk0 249.7 + chunk1 69.5）。

### 4.2 与审计的 10 %+ 说法的关系

审计的 "若 predict 未被掩盖：10 %+；若已掩盖：仅 gap 尾部 ~2–3 %"——
本时间线给出的答案是**已掩盖**：90.1 % 的 GPU busy 在 ANE 阻塞窗内，
chunk1 的 predict 窗口 73.9 % 有 GPU。10 %+ 需要跨层重叠，而跨层被
RMSNorm 依赖禁止。**方向对 >10 % 是死的。**

## 5. Opus 结论（1 次，medium，117 s）

- **Core ML 并发**：同步 `prediction` 不要假设线程安全，需自己串行化；
  macOS 14+ 有 async/`completionHandler` API，Apple 称可并发调用，但
  **两个不同模型能否在 ANE 上真并发没有公开保证**，应按"硬件按程序串行"
  设计；并发最多重叠 host 准备/拷贝与另一次 ANE 执行。需要用 probe 验证：
  两线程连续 predict 的吞吐 ≤1.1× 串行 → ANE 串行。
- **依赖是硬的**：RMSNorm 的 `1/rms(a+b)` 含交叉项；线性延迟 add 不缩短
  关键路径；近似会改数值（parity 敏感）；token 波前被 512 tile 堵死。
- **同层双缓冲的实际上限**：chunk1 的 **5.4 %**（69.5 ms host 间隙），
  3453 ms 的 2.0 %；现实 ≤2 %；专用线程只省唤醒延迟，省不掉 merge 和
  下一层 encode。
- **内存**：输入+输出一组 8 MB，双缓冲 16 MB，**整模型共用一组**；不要
  为并发复制 `MLModel`（每层 INT8 share ~69 MB，32 层 ~2.2 GB，16 GB 上
  很可能换页）。失败阈值：窗口内 Swapouts>0 → INCONCLUSIVE；常驻内存比
  基线多 >64 MB → 设计有误。
- **NOT-WORTH 判据（不必先实现）**：xctrace 量 chunk1 每层
  "pack GPU 结束→predict 开始" + "predict 结束→merge GPU 开始" 两段间隙，
  总和 <26 ms（chunk1 的 2 %）就直接 NOT-WORTH；并发 probe ≤1.1×；任何臂
  Swapouts≠0 不算数。

## 6. 最小设计（若仍要做，只改调度）

- 一个 ANE worker 线程，与主线程用深度 1 的 SPSC 队列；worker 拥有 24 个
  `AneMlp`/`MLModel`（`Model` 不是 `Send`，必须在 worker 线程加载或整体
  迁移）。
- 主线程：`pack` + `gpu_share` 入队 → 把 `packed` 句柄交给 worker → worker
  `to_bytes_ready` + 同步 predict → 写输出 slot → 通知主线程 → 主线程
  `gpu.add(share)`。
- 同步点只有两个：pack 完成、predict 完成；**不要给每层加第二份模型**。
- 预期：chunk1 −69.5 ms 上限（5.4 %），现实 ≤2 %；跨层仍然没有重叠，
  因为 `m_N` 依赖不变。

## 7. 建议与失败判据

- **本阶段不实现**：同层已到上界，跨层数学上不可行；调度-only 上限
  5.4 %（chunk1）/2.0 %（首请求），现实 ≤2 %，低于 KEEP 门。专用线程
  只会把 host 阻塞挪到 channel 等待，不会增加 GPU/ANE 的并行。
- 若未来仍要碰：先做 §5 的两段间隙测量；总和 ≥26 ms 才值得写最小原型；
  并发 probe 必须 >1.1× 串行吞吐；任何 Swapouts≠0 的轮次只记 INCONCLUSIVE。
- 真正可能 >2 % 的相邻方向不是线程，而是**压缩 chunk1 里 172.8 ms 的
  ANE-only 时间**：列配比（6656/5632 已配平；7680 因 Swapouts 关闭）、
  ANE 程序/ tile 服务时间、或减少 ANE 列的层数；这些都不是"调度"问题。

## 8. 证据与复现

- 代码：`crates/apxinf-model/src/metal_clef/model.rs`（`mlp` auto 分支）、
  `crates/apxinf-model/src/metal_clef/ane.rs`（`pack`/`predict`）、
  `crates/apxinf-coreml/src/runtime.rs`（`Model` Not Send）。
- 时间线：`docs/mario-opt/TIMELINE-OPERATOR.md` §C1–C5（单请求 3453.2 ms；
  wait/predict 覆盖、chunk0/chunk1 拆分）。
- Opus：`/tmp/ane-opus-brief.md` → 117 s，6448 B；日志
  `~/apxinf-runtime/opus-calls.log` 2026-11-07T04:17:40Z。
- 本机时间线带桌面干扰（C5），不作为 latency KEEP 证据；本报告只用其
  结构关系。
