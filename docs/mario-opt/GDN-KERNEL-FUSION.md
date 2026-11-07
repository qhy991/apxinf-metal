# Clef kernel-level GDN fusion — measured 1.54 % upper bound → NOT-WORTH

审计 `doc/notworth-audit.md` §3 #1/#3 的 "kernel 级 GDN 融合"（原计划 2–4 %，
`gdn_block=fused_prep` / `eight_lanes`）。本棒先核对形状契约，再在设备上对
**整条参考链**逐段计时；结论是**可融合部分的总上限 1.54 % < 2 %**，任何
kernel 级合并都不可能过门，因此 **NOT-WORTH**，不改代码。

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1，reference 组合
- 证据：`devlocal/gdn-fusion/`（git-ignored）
- Opus：0 次

## 1. 形状契约（与 D3 不同）

`clef-flash-4bit-b5adbe2` 的 text config：

| 项 | 值 |
| --- | --- |
| layers | 32 = **24 GDN (`linear_attention`) + 8 full attention**（不是 27 GDN） |
| hidden | 4096 |
| Hk / Hv | 16 / 32 |
| Dk / Dv | 128 / 128 |
| conv kernel | 4 taps |
| conv_dim | 2·Hk·Dk + Hv·Dv = **8192** |
| chunk1 L（42 案） | 262 / 266 / 286 |

**关键事实**：reference 组合**已经**用 `apxinf_gated_delta_four_values`
（`GatedDeltaRecurrence::new`），也就是 D3 线拿到 +4.27 % 的那颗 recurrence
kernel；GPU/ANE 组合才用 `eight_lanes`。所以 "把 D3 的 GDN kernel 搬过来"
在 Clef reference 上已经发生过了，剩余可做的只有 conv1d/silu/split/norm
的 prep 和 recurrent 后的 gate。

## 2. 设备实测：整条参考链逐段计时

临时基准（`devlocal/gdn-fusion/scripts/gdn_chain_bench.rs`，用与
`metal_clef::gated_delta` 完全相同的参考 op：`concat(history,mixed).conv1d`
→ `silu` → q/k `fast_rms_norm_unweighted`+BF16 scale；`four_values` /
`eight_lanes` recurrence；`y.fast_rms_norm(norm)`+`silu(z)` gate），L=266、
24 层、每段 `stream.eval` 后取 24 次平均：

| 段 | 每层 | 24 层合计 | 占 2197 ms 请求 |
| --- | ---: | ---: | ---: |
| prep（conv1d+silu+split+q/k norm+scale） | 0.907 ms | **21.8 ms** | 0.99 % |
| recurrence（four_values，现状） | 1.685 ms | **40.4 ms** | 1.84 % |
| recurrence（eight_lanes） | 1.186 ms | **28.5 ms** | 1.30 % |
| gate（rms_norm(y)·silu(z)） | 0.501 ms | **12.0 ms** | 0.55 % |
| **prep + gate（可被融合删除的部分）** | — | **33.8 ms** | **1.54 %** |

这些数字是每段单独 `eval` 的串行上界（真实请求里 24 层在一张图里背靠背，
只会更小），所以对 "融合能省多少" 是一个**保守上界**。

## 3. 为什么融合不可能过 2 %

- 融合能删掉的是 prep + gate 的 kernel 与中间读写：**总共 33.8 ms = 1.54 %**。
  即使把这段融合成 **0 成本**，也 < 2 %；conv/silu/qk norm/gate 的计算与
  读写仍然存在，实际可省只是其中一部分。
- recurrence 的 40.4 ms 是状态递推的计算，不能靠 "合并 launch" 删除；它
  已经是 D3 验证过的 four_values kernel。
- 唯一更快的 recurrence 变体是 `eight_lanes`：1.685 → 1.186 ms/层，
  **省 11.9 ms（0.54 %）**。5 案 smoke（同输入、Tier-1、reference）：
  决策 5/5 相同，但 max|Δp| 到 **0.025246**（f0012/enemy），延迟增益
  −2.4 / +1.3 / −2.6 / +1.4 / +0.5 %（均值 ≈ −0.4 %，噪声内）。
  0.025 的漂移叠加基线对 MLX 的 0.0328 头寸，42 案 parity 很可能顶门；
  即便 parity 过了，+0.54 % 也 < 2 %。
- 把 prep+gate 全部删除（1.54 %）+ eight_lanes（0.54 %）的理论拼盘是
  2.08 %，但它要求 prep/gate 的**全部计算**消失且 eight_lanes parity 不退化；
  真实融合只能省 prep/gate 的一部分，达不到。

## 4. 为什么 D3 上 +4.27 %、Clef 上不行

D3 的 +4.27 % 是**recurrence kernel 替换**（stock eager 多 kernel → 专用
four_values/eight_lanes），在 D3 的模型里 GDN recurrence 占比更高、stock
路径的中间读写/launch 更多。Clef reference 已经用了同一颗 four_values
kernel；剩下的 conv1d/silu/norm/gate 是小算子，24 层合计 1.54 %。加上
reference 请求的 GPU busy 已 98.5 %（算子间气泡只有 ~1.5 %），"少 launch /
少中间读写" 没有可拿的结构性空间。审计里的 `fused_prep 2–4 %` 是 D3/CUDA
侧的外推，不是 Clef Metal 的实测。

## 5. 判定

- **NOT-WORTH**：可融合链的实测总上界 1.54 % < 2 %；无需占用测量窗做
  3 轮 campaign，也不必落一个注定 <2 % 的 kernel。
- 不新增环境门控/代码；reference 保持 four_values（KEEP 基线），
  `eight_lanes` 只在 GPU/ANE 组合使用。
- 无 before/after 时间线：本方向没有可落地的候选（与 `ANE-COLUMN-BALANCE.md`
  §1 的负结果先例一致）；证据是上面的逐段设备计时。

## 6. 下一步候选

1. **reference 档的 Q4 GEMM**：GDN 层真正的大头是 qkv/z/out 的 Q4 GEMM
   （每层 ~7 ms 量级），不是 prep/gate；要动 2 % 只能动 GEMM/kernel 级别
   （audit §3 的 q4 GEMM / fusion 线，但已有 0.049 parity 失败记录）。
2. **auto 档**：ANE wait/predict 与 319 ms 层内 host 残差仍是唯一 >2 % 的
   结构性空间，受 42 案 Swapouts 阻塞。
3. **组合项**：eight_lanes + prep/gate 融合即使全做也只能到 ~2 %，且
   parity 风险高；不建议单独重开。

## 7. 证据与复现

```bash
# 逐段计时（临时 test；脚本已保存在 devlocal）
cp devlocal/gdn-fusion/scripts/gdn_chain_bench.rs \
   crates/apxinf-model/tests/gdn_chain_bench.rs
export MLX_ROOT="$HOME/apxinf-runtime/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
cargo test -p apxinf-model --release --features mlx,coreml \
  --test gdn_chain_bench -- --ignored --nocapture
# 输出：prep 0.907, four_values 1.685, eight_lanes 1.186, gate 0.501 ms/层
```

- 逐段输出：`devlocal/gdn-fusion/`（本次运行日志）
- eight_lanes smoke：`devlocal/gdn-fusion/smoke/{gdn4,gdn8}/records.jsonl`
  （5 案、reference、Tier-1、无 warmup；决策 5/5，max|Δp| 0.025246）
