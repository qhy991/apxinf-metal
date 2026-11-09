# ANEForge 评估：直接编译 ANE 程序 vs 我们的 Core ML ANE 路径

> 评估对象：[ANEForge](https://github.com/sbryngelson/ANEForge) 0.4.0（pip 包；绕过 Core ML，
> 直接经私有 `aned`/`e5rt` 栈编译并调度 ANE 程序）。
> 结论先行：**在我们自己的负载上，ANEForge 全面不优于现有路径**——MLP tile 慢 14–38%，
> sdpa 慢 21–52% 且 GPU 本来就更快，Qwen3-0.6B decode 慢 5.4 倍**且 28 层 fp16 前向在 M1 上
> 数值坍缩（输出全错）**，因果 sdpa 在 M1 上直接编译失败。
> 唯一的亮点是固定 dispatch 开销更低（~0.43 ms vs Core ML ~0.93 ms），但该优势只在小 shape 出现，
> 而我们的生产 shape 恰好是大 shape。**不建议集成；维持现有 Core ML + GPU 路径。**

## ⚠️ 私有 API 风险（首屏必读）

ANEForge 的整套运行时建立在 **Apple 未公开的私有框架**之上：

- 它直接调用 `AppleNeuralEngine.framework`/`E5RT` 的私有符号（`aned` 守护进程协议、e5rt
  编译/执行入口），自行构造 ANE 程序包并绕过 Core ML 的全部公开 API。
- **任何 macOS 升级都可能无声地破坏它**：Apple 不承诺这些符号的稳定性，也没有文档、没有
   deprecation 周期。本次评估在 macOS 27.0（Darwin 27，T6000/M1 Pro）上实测可用，但这只是
  一个时间点的快照。
- **已经观察到跨芯片脆弱性**：`af.sdpa` 的 native 融合注意力桥在本机（M1 Pro, h13）编译即失败
  （`verifyBundleAtPath: invalid model`，见 §3.4），而同项目文档声称在更新的芯片上可用——
  私有 ABI 的芯片间差异由用户自行承担。
- 若进入产品路径，故障模式是「系统更新后推理栈整体失效且无法向 Apple 求助」。我们的 Core ML
  路径只使用公开 API（`MLModel`/`MLComputePlan`），无此风险。
- 它额外要求 Python 运行时常驻（我们是 Rust 栈），集成只能走 sidecar 子进程，进一步放大
  运维面（进程生命周期、IPC、崩溃隔离、版本钉死）。

## 1. 评估设置

| 项 | 值 |
|---|---|
| 机器 | MacBook Pro, Apple M1 Pro (T6000), macOS 27.0.0 (Darwin 27.0.0) |
| 我们的一侧 | `apxinf-coreml`（Core ML MLProgram，`MLComputeUnitsCPUAndNeuralEngine`），探针为 `metal_clef/ane.rs::program` 的逐操作复刻（合成权重，同 shape/同 recipe） |
| ANEForge 一侧 | pip `aneforge` 0.4.0（`~/aneforge-venv`），`compile_multi`/`compile` + `compress="int8"` |
| GPU 基线 | MLX 0.32 `fast.scaled_dot_product_attention`（与 `metal_clef` GPU attention 同一算子）；Qwen3 用仓库 `metal_text_probe`（release，commit 79db7e91） |
| 数据 | 同一组种子随机权重/输入二进制（`devlocal/aneforge-eval/data`），三条路径逐字节共享 |
| 计时 | 每次 5 warmup + 21 次取中位；双锁（`~/.cache/enginetailor/metal-measure.lock` → `/tmp/enginetailor-metal.lock`）串行化，无其他测量活动 |
| parity | fp32 numpy 参考；int8 用同一组量化权重反量化后作参考，隔离 ANE 计算噪声 |
| 能效 | **未测**：本机无免密 sudo，`powermetrics` 不可用（局限已记录） |
| ANE 放置证据 | Core ML 侧逐 op `MLComputePlan` preferred device（全 ANE，见日志）；ANEForge 侧其运行时直接构造 ANE 程序 |

负载 shape（全部来自生产配置）：

- **Clef MLP tile（M1 Pro 列切分 #147）**：hidden 4096，ANE 份额 5632 行，token tile 512，
  gate/up 逐行对称 INT8 + SiLU（`0.5v(1+tanh(v/2))`）+ 行最大值归一化防溢出 + down 两段 INT8 K-split。
  生产契约：最终输出 = `y × gate_scale × up_scale`。
- **非因果 attention（Clef 几何）**：16 heads × head_dim 256，tokens ∈ {512, 1152}。
- **Qwen3-0.6B**：官方 BF16 checkpoint（HF cache 离线加载）。

## 2. 算子对比：Clef ANE MLP tile

延迟（ms/次调用，中位；越小越好）：

| shape / recipe | 我们 Core ML | ANEForge | Δ（ANEForge vs 我们） |
|---|---|---|---|
| **5632×512 INT8（生产 shape）** | **9.30**（3.81 TMAC/s） | 10.57 | **+13.7% 慢** |
| 5632×512 FP16 | **14.17**（2.50 TMAC/s） | 19.58 | **+38.2% 慢** |
| 5632×512 INT8，去掉归一化守卫¹ | **9.30** | 9.15 | −1.6%（噪声内，且无数值守卫） |
| 512×64 INT8（小 shape，看固定开销） | 1.13 | **0.43** | **−62% 快** |
| 512×64 FP16（小 shape） | 0.93 | **0.49** | **−47% 快** |

¹ ANEForge 的 idiomatic 写法（plain SwiGLU）少了我们的溢出守卫（行最大值归一化）：
该守卫是 Clef recipe 为生产 checkpoint 的极端激活值专门设计的（见
`metal_clef/ane.rs` 的 `DOWN_ABS_SUM_LIMIT`/`factor_cap` 推导），去掉它只省 ~0.15 ms
却要重新承担 fp16 溢出风险。

parity（最终输出 `y×gs×us` vs fp32 参考；int8 用同量化权重参考）：

| 路径 | max\|Δ\| | rel L2 | cosine |
|---|---|---|---|
| Core ML INT8 | 0.0129 | 1.03e-3 | 0.9999979 |
| ANEForge INT8 | 0.0892 | 1.06e-2 | 0.9999419 |
| Core ML FP16 | 0.0110 | 1.01e-3 | 0.9999984 |
| ANEForge FP16 | 0.0428 | 5.59e-3 | 0.9999834 |

两条路径都过常规 parity 门，但 **ANEForge 的数值噪声一致地是 Core ML 的 4–10 倍**
（同一 ANE 硬件、同一有效权重——差异来自编译器的累加/融合策略）。

**结论（MLP）**：生产 shape 下 Core ML 明显更快且更准。ANEForge 唯一的胜场是
**固定 dispatch 开销**（~0.43 ms vs ~0.93–1.13 ms，印证其文档的 ~0.2–0.5 ms 单 flight 下限），
但 Clef tile 是 512-token 大 shape，固定开销仅占 ~5–10%，无法翻盘。**不换。**

## 3. 算子对比：非因果 attention（`af.sdpa`）

延迟（ms/次，中位）：

| shape [1,16,T,256] | 我们 Core ML(ANE) | ANEForge(ANE) | **GPU MLX（现状）** |
|---|---|---|---|
| T=512 | 3.16 | 4.82 | **1.99** |
| T=1152 | 9.25 | 11.23 | **7.30** |

parity（vs fp32 参考，rel L2）：三者都过门（cosine ≥ 0.99999）；GPU 反而最准——
GPU 5.3e-4 / 5.4e-4（T=512 / 1152），Core ML = ANEForge 1.8e-3 / 3.8e-3。

**结论（sdpa）**：GPU 已经是三者中最快最准的；ANE（无论哪条路径）都不该接 attention。
而且注意 §3.4：**ANEForge 的因果 sdpa 在 M1 上根本不可用**，即使想为 causal 场景用也不行。

### 3.4 ANEForge 因果 sdpa 在 M1 上失败（实测）

`af.sdpa(is_causal=True)`（native 融合注意力桥，`[1,2,128,64]` 最小 shape）：

```
RuntimeError: sdpa_invoker failed (rc=4):
{"status":"compile_failed","error":"Error Domain=com.apple.appleneuralengine Code=10
 \"verifyBundleAtPath: invalid model\""}
```

非因果走分解路径可用（上表），因果在 M1/h13 上编译即失败。ANEForge 的 `load_llm` 内部
用 matmul 分解绕开了该桥，所以 LLM 路径不受此影响；但任何「直接用 `af.sdpa` 承载 Clef
因果 attention」的方案在 M1 上是死的。

## 4. 模型级对比：Qwen3-0.6B decode

| 路径 | decode | 备注 |
|---|---|---|
| 我们 GPU（`metal_qwen3` bf16-compiled, commit 79db7e91） | **95.9 tok/s 中位**（tpot 10.29–10.65 ms，3 次×128 tokens，输出连贯） | 与 owner 实测 100 tok/s 一致 |
| ANEForge 纯 ANE fp16 | **17.9 tok/s 中位**（54–59 ms/token；3 次×128 tokens） | **数值错误：输出全为 `!`** |

### 4.1 ANEForge Qwen3-0.6B 实测（本机，双锁窗口内）

`af.load_llm` 离线加载本地 HF snapshot（3.4 s），`warmup(256)` 一次性编译 19.4 s
（编译缓存持久化在 `~/Models/.aneforge-cache`），随后 3 次 greedy 生成 ×128 tokens：

| 指标 | 值 |
|---|---|
| 稳态 decode | **17.9 tok/s 中位**（rep：18.2 / 17.9 / 17.0） |
| 单 token 分解 | layers ≈ 27.5 ms + ANE lm_head ≈ 27.8 ms + sample ≈ 0.08 ms |
| 生成文本 | `!!!!!!…`（无意义；greedy 始终命中 `!`） |
| batched prefill 文本 | 同样全 `!` |

**数值崩塌（致命）**：完整 28 层 fp16 前向在本机静默坍缩——最终 hidden 全零。
分层探查（真实权重）显示激活 absmax 从 L1 的 92.9 逐层衰减到 L28 的 0.0；
2 层截断则数值自洽。这与 ANEForge 文档记载的 fp16 数值悬崖一致
（matmul 输出 ~32752 饱和；**pre-A16 芯片** slice 通道 >4094 饱和——M1 正是 pre-A16）。
`batched_prefill=False`、`opt=0`、host lm_head、减小块大都试过，无效。
**即：ANEForge 的 Qwen3-0.6B 在 M1 上不仅慢 5.4 倍，输出还是错的。**
（时间数据本身有效——计算图完整执行，只是数值无意义。）

值得一提：decode 时间的一半花在 151936 词表的 ANE lm_head（分块 compile_multi）上；
我们 GPU 路径的同一矩阵远快得多——大词表投影恰是 ANE 的弱项。

未测 int8 变体：坍缩发生在**激活**侧（fp16 饱和），权重量化无法修复，故跳过。

## 5. 混合模型可行性（只评估）：「Clef 全 ANE」

Clef Flash 9B = Qwen3.5 架构族（24 Gated-DeltaNet + 8 full attention，4096 hidden，
我们的 GPU+ANE 组合目前只把 MLP 的一部分放到 ANE）。ANEForge 的 `qwen35.py` 提供了
**同族架构**（Qwen3.5/Qwen3-Next，48 DeltaNet + 16 attn @27B）的纯 ANE decode 证据。
逐条核对后：

1. **只有 decode，没有 prefill（决定性卡点）**。`qwen35.py` 只注册
   `DECODE_MIXERS["gated_deltanet"]`；`PREFILL_MIXERS` 里只有普通 `attention`。
   GDN 的并行/chunked 扫描 prefill **不存在**——混合模型的 prompt 只能逐 token 走 decode
   程序（~57 ms/token 量级）。Clef 的负载是**单次前向的决策请求**（最长 16384 tokens），
   逐 token「prefill」意味着单个 4096-token 请求 ≈ 4 分钟。**仅此一条就判死「Clef 全 ANE」。**
2. **27B 规模在 M1 Pro 上不可行**。`load_gguf` 自己写明 64 层 fp16 ≈ 54 GB > RAM，
   靠逐层 stream+free 续命——decode 每 token 都要把 ~54 GB 权重从 SSD 过一遍，
   tok/s 不可用。Clef 9B Q4（~5.7 GB）规模可行，但没有现成 checkpoint 适配
   （需要自写 Q4→fp16 转换 + `adapt()`，工作量中等）。
3. **fp16 数值悬崖真实存在**：`load_gguf` 需要 `resid_scale=32` 把残差缩 32 倍才能不溢出
   fp16（27B 深层 outlier）。这印证了我们 Clef recipe 里行最大值归一化守卫的必要性，
   也意味着迁移时我们的 Q4/int8 parity 门要全部重新标定。
4. 因果 sdpa 的 native 桥在 M1 失败（§3.4）；`qwen35.py` 的 decode attention 用 matmul
   分解绕行，可用但效率一般。

**结论（混合模型）**：ANEForge 证明了「Qwen3.5 架构族的**逐 token decode**可以纯 ANE 跑通」
（架构可行性证据有价值），但对我们真正的负载形态（长请求单次前向决策）没有可迁移的路径。
卡点清单：GDN prefill 缺失、27B 流式权重、fp16 范围、M1 因果 sdpa 桥坏、Python sidecar 运维面。

## 6. 总结论

| 候选 | 判定 | 依据 |
|---|---|---|
| Clef ANE MLP tile 换 ANEForge | **不换** | 慢 13.7%（int8）/38%（fp16），噪声大 4–10×，私有 API 风险白担 |
| attention 上 ANE（任何路径） | **不做** | GPU 已是最快最准；ANEForge 因果 sdpa 在 M1 不可用 |
| Qwen3-0.6B decode 换 ANEForge | **不换** | 17.9 vs 95.9 tok/s（慢 5.4×）且 28 层 fp16 前向在 M1 数值坍缩，输出全错 |
| Clef 全 ANE（混合模型） | **不可行（当前）** | 无 GDN prefill；逐 token prefill ~4 分钟/请求 |
| 小 shape 固定开销优化 | 记录，不行动 | ANEForge floor 0.43 ms < Core ML 0.93 ms，但生产 shape 用不上 |
| 能效对比 | 未能测量 | 无免密 sudo（powermetrics 需要 root） |

集成成本（若未来某场景真的要换）：sidecar Python 子进程 + 权重/输入 IPC +
崩溃隔离 + aneforge 版本钉死 + macOS 版本回归测试矩阵；估计 >1 人周，且随时可能被
系统更新作废。**在当前证据下没有任何算子值得付出这个成本 + 私有 API 风险。**

## 附录 A：复现

```sh
# 数据（种子固定，三条路径共享同一份二进制）
~/aneforge-venv/bin/python devlocal/aneforge-eval/scripts/gen_data.py

# 我们的一侧（Core ML；探针是 metal_clef/ane.rs::program 的逐操作复刻，devlocal 独立 crate）
cd devlocal/aneforge-eval/probe && cargo build --release && cd ..
scripts/lock.sh ./probe/target/release/ane-probe mlp-int8 data results /tmp/aneforge-eval-coreml-cache

# ANEForge 一侧
scripts/lock.sh ~/aneforge-venv/bin/python scripts/probe_aneforge.py mlp-int8

# GPU sdpa 基线 / parity
scripts/lock.sh mlx-venv/bin/python scripts/probe_mlx_sdpa.py 512
~/aneforge-venv/bin/python scripts/parity.py all

# Qwen3-0.6B GPU 基线（仓库 release 探针）
scripts/lock.sh bash -c "APXINF_METAL_TEXT_MODEL=metal_qwen3 \
  <metal_text_probe> <Qwen3-0.6B snapshot> qwen3/corpus.json 3 none"
```

探针二进制复用自 `qwen3-probe` worktree 的 release 构建；其探针源码与本 checkout HEAD
（79db7e91）逐字节一致，构建时另含一个**默认关闭**的计时插桩补丁
（`APXINF_QWEN3_DECODE_STAGES`，本次运行未启用），探针自检元数据报
`checkout_dirty=false`。

原始证据：`devlocal/aneforge-eval/{logs,results}/`（逐次计时 JSON、placement 列表、parity 明细）。

环境说明：评估期间在 `~/aneforge-venv` 增补了 `torch`/`transformers`（`af.load_llm` 的
加载依赖，PyPI 安装），以及 `devlocal/aneforge-eval/mlx-venv`（mlx+numpy，GPU sdpa 基线）。
仓库代码零改动；所有探针脚本都在 `devlocal/`（不进 git diff）。
