# 重复图像 prepared 缓存（跳过 preprocess）

自主选向的第二个候选（`APXINF_CLEF_PREP_CACHE`）。授权同
[ANE-COLUMN-BALANCE.md](ANE-COLUMN-BALANCE.md)：PR #5 评论
`c2797f63ba9ca92530d33c441b474f68`。

- 测量机：Apple M1 Pro 16 GB，macOS 27.0.1
- tier：**reference（GPU-only）**——auto tier 的 vs-MLX parity 门
  （0.035）被基线自身的 0.0404 卡死，且 swap 污染使 KEEP 池为空
  （见 [ANE-COLUMN-BALANCE.md](ANE-COLUMN-BALANCE.md) §4）；reference
  tier 两条都干净
- gate：`APXINF_CLEF_PREP_CACHE`（`0/1`/unset，默认 OFF）

## 1. 选向与预期收益机制

clue (a)（“避免跑 chunk0 比加速 chunk0 值钱”）+ §4 气泡剖面：
**preprocess 是 Tier-2 命中窗口的整段开头气泡**（请求起始 → 第一个
`ane_wait` 之间两引擎全空，fill 录制里 68.8 ms）。

机制（测量前写下）：不改任何数值——按请求 PNG 字节做 key，命中就
原样返回上次的 `PreparedImages`，**从窗口里删掉** PNG 解码、bicubic
resize、patchify 和多 MB 的像素 key 哈希；位级保持，所以 parity 必然
与基线一致。预期收益 <2 %（这些 demo 裁图很小），按新规先写清为什么
仍值得试：**(i) 直接缩短 Tier-2 命中请求的开头真气泡；(ii) 是"重复
流量删工作"这条已验证路线（Tier-1 +8.72 %、Tier-2 +45.3 %）的下一块
拼图。** 否则直接跳过。

## 2. 实现

`crates/apxinf-model/src/metal_clef/preprocess.rs`：

- `PreparedImages` 加 `Clone`；新 `PreparedImagesCache`（RefCell +
  VecDeque，limit 8，LRU），key = 整个请求的 PNG 字节 length-prefix
  SHA-256；`prepare(images, cache)` 先读字节 + 算 key + 查表，命中
  直接返回，未命中走原路径并回填；
- `crates/apxinf-model/src/metal_clef/mod.rs`：`prep_cache_flag`（unset
  跟 `PREP_CACHE_DEFAULT_ON`，精确 `0/1`，其它报错）+ ClefRunner 字段
  `prepared_cache: Option<_>`，load 时读一次；KEEP 才翻默认常量；
- `host_probe.rs` 的计时调用点显式传 `None`（probe 要报原始代价）。

生产改动 ~155 行（含注释与测试调用点）。

单测（3 个，全过）：gate 拼写与默认相对性；命中返回**逐字节相同**
的矩阵且 limit=1 时正确淘汰（IEND 后追加字节 = 同像素不同 key）；
多图请求的 key 分离。

## 3. 测量

### 3.1 42-case parity pass（reference，1 轮）

目的：给 KEEP 提供 vs 冻结 MLX 的 parity 证据（repeat suite 的 case id
与 MLX 输出对不上，必须单独跑）。两臂 Swapouts 均 +0：

| 臂 | A/A vs base | vs MLX |
| --- | --- | --- |
| base | — | 42/42，max\|Δp\| **0.032783** ✓ |
| cache | 42/42，max\|Δp| **0.0**（bit_equal） | 42/42，max\|Δp\| **0.032783** ✓ |

42-case 上缓存**从不命中**（每案图片唯一）→ 两臂逐位相同 → parity
与基线完全一致，门通过 ✓（0.032783 复现了 Tier-2 记录里的基线值）。

### 3.2 repeat campaign（reference，3 轮 × 2 measured）

套件：`devlocal/matmul-fusion/suites/prep-cache-repeat.json`
（4 案图 × [fill + v1 + v2] = 12 案；v1/v2 = 同 PNG 不同问句）。

**3 轮全部干净**（6/6 臂次 Swapouts +0）：

| 臂次 | 墙钟 | Swapouts |
| --- | --- | --- |
| r0 base / cache | 44.2 / 44.2 s | +0 / +0 |
| r1 cache / base（反序） | 44.4 / 44.3 s | +0 / +0 |
| r2 base / cache | 44.3 / 44.2 s | +0 / +0 |

配对增益（12 案 × 2 轮 = 36 对，跨 3 个干净轮累积）：

| | |
| --- | --- |
| median | **−0.174 %** |
| bootstrap CI95 | **[−0.359 %, +0.214 %]**（含 0） |
| 更快案例占比 | 38.9 % |
| 逐轮 median | +0.263 / −0.359 / +0.302 % |

**A/A：12/12 决策相同，max|Δp| = 0.0 → bit_equal**（质量满分）。

### 3.3 为什么没有收益（逐案结构）

记录里的窗口呈**双峰**（同一 case 的两个 rep 分处两峰）：

- **~2420 ms = Tier-2 未命中**：所有 fill + rep0 的 v1；
- **~1140 ms = Tier-2 命中**：rep0 的 v2 + rep1 的全部 v1/v2。

在 Tier-2 命中 rep 上把两臂并排看：base 与 cache 的请求时长
**交错、无系统差**（如 rep1-v2：1139.6/1137.6/1127.1/1139.0 vs
1132.8/1122.7/1137.1/1130.0）。所以要么缓存命中但
**warm preprocess 在这些小裁图上本来就低于噪声**，要么根本没命中。

chrome 诊断（cache-off = 基线，有效）证实 warm preprocess **恒为
12.4–18.4 ms（median 12.7 ms，24 请求）**，包括 rep1 的全部重复案——
它就是 §4 上界推导里的那个数。

（rep0 的 v1 仍走 ~2420 ms：Tier-2 的"同一 prepared 图至少见过一次"
准入/命中在 fill 之后一个请求才生效，rep1 起全部变 ~1140 ms——这是
Tier-2 既有行为，与本候选无关。）

## 4. 结论

**NOT-WORTH**（默认 OFF、**不进主线**；实现代码已回退，working tree 只剩
eager-eval 候选）。收口依据是**硬上界**，不是噪声读数：chrome 诊断给出
warm preprocess = 12.7 ms（§3.3），命中窗 = ~1140 ms、fill 窗 = ~2420 ms。
缓存最多省下整个 preprocess → 命中案上界 12.7/1140 = **1.11 %**、fill 案
0.52 %；12 案 median 上界 ≈ **1.1 % < 2 %**。**就算缓存完全免费也到不了
KEEP 门**，所以按新规（"收益 <2% 写结论、按 NOT-WORTH、再选下一个"）
中途收口，不烧一轮 campaign。

void 说明：首次 42-case parity pass、3-round repeat campaign 与 chrome run
因 runner `arm_env` 在 `env.update` 之后又 `env.pop(env_key)` 而**作废**
（两臂都跑 cache-off）——已改名 `results/*-VOID-arm-env-bug` / 对应 log，
不得引用其数字。chrome JSON 以 `logs/prep-cache-chrome-diag-cacheOFF-baseline.json`
保留：它是 cache-off 基线，恰好是上界推导需要的有效数据。runner 已修复
（先清四个 qualified knob 的 ambient 值、再设臂 key），并以断言单测验证
arm 值在 `env_key=APXINF_CLEF_PREP_CACHE` 与 `=APXINF_CLEF_ANE_COLUMNS`
两种情况下都存活。

## 5. 归因边界

与 [TIMELINE-OPERATOR.md](TIMELINE-OPERATOR.md) §5 相同：GPU = 硬件
command buffer 级、算子标签 = host os_signpost、ANE = 软件轴
wall-clock；本候选的 parity 证据是 host 读回的概率值，不是硬件计数。

## 6. 产物与复现

维护产物：本文（negative candidate，**代码已回退、不进主线**；gate 的
`PREP_CACHE_DEFAULT_ON` 保持 false）。void 化并改名的 campaign/parity/chrome
记录、有效的 cache-off 基线 JSON、repeat 套件与脚本留在
`devlocal/matmul-fusion/`（见 §4）。

```bash
export MLX_ROOT="$HOME/apxinf-runtime/mlx-sdk"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_ROOT/lib -C link-arg=-Wl,-headerpad_max_install_names"
cargo build --release --features mlx,coreml --bin apxinf-latency-worker
python3 devlocal/matmul-fusion/scripts/make_prep_cache_suite.py --frames 4
python3 devlocal/matmul-fusion/scripts/run_column_campaign.py --run \
  --env-key APXINF_CLEF_PREP_CACHE --arms base=0 --arms cache=1 \
  --rounds 1 --measured 1 --variant reference --suite 42 \
  --run-root prep-cache-parity
python3 devlocal/matmul-fusion/scripts/run_column_campaign.py --run \
  --env-key APXINF_CLEF_PREP_CACHE --arms base=0 --arms cache=1 \
  --rounds 3 --measured 2 --variant reference \
  --suite devlocal/matmul-fusion/suites/prep-cache-repeat.json \
  --run-root prep-cache-repeat \
  --parity-summary devlocal/matmul-fusion/results/prep-cache-parity/summary.json
```
（实现代码在 campaign 后按"不进主线"回退；重跑前需恢复 §2 的改动。）
