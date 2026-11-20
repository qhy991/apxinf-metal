# 预登记：native 4-bit 知识/推理质量评测（MMLU + GSM8K）— QUEUE #8 附属门

**状态**：预登记已冻结，commit 在先，测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-94，2026-11-20 ~10:25 CST，Lead 10:20 追加任务（见 STATUS）。
**目的**：回答"原生 4-bit 能否设为主力"的知识/推理维度（与 4-bit-primary agent 门 `doc/lfm3-4bit-primary.md` 互补，是模型级、确定性、样本量充足的口径）。

## 1. Gate（冻结原文，供 PR 引用）

```
GATE（Lead 10:20，逐字）：
- 样本：MMLU 按 subject 分层抽样 500 题（固定 seed，记 sha256）；GSM8K 抽 200 题（固定 seed）。抽样脚本与样本 id 列表一并落盘。
- 臂：native-4bit（APXINF_LFM3_4BIT=1）vs native-bf16（unset），同二进制，greedy（temp=0），max_tokens=64。
- 提示模板（写死并记录）：
  - MMLU：`Answer the following multiple choice question. Reply with only the letter of the correct answer.\n\n<question>\nA) ...\nB) ...\nC) ...\nD) ...`，用正则提取首个 [ABCD]；
  - GSM8K：`Solve the problem. End with "The answer is <number>".`，取最后一个数字与 gold 比较。
- 指标：各臂 accuracy（MMLU / GSM8K 分开报）；delta（4bit − bf16）；answer flip rate（两臂给出不同答案的题占比）；同一臂两段的 accuracy spread（A/A，>5pp 要标注）；端到端 token 用量（可选）。
- 判定（现在冻结）：
  - MMLU 与 GSM8K 的 delta 都 ≥ −3pp ⇒ PASS（支持把 4-bit 设为主力）；
  - 任一 delta 在 −3 ~ −10pp ⇒ PARTIAL（4-bit 保持 opt-in，按场景选）；
  - 任一 < −10pp ⇒ FAIL（不设主力）。
- 交付：doc/lfm3-4bit-knowledge-eval.md（预登记 + 结果 + 判定）+ receipts 路径；并入文档 PR。
```

附 Lead 10:22 B/C（对本门的追加约束，逐字）：

```
B. 知识/推理评测（MMLU 500 + GSM8K 200）——这一项是模型级量化信号，用直接 greedy 提示才是干净口径（agent 包装会引入推理风格噪声）——照我 10:20 那条执行；但请额外加一个小交叉验证：取 50 道 MMLU 用 mini-swe-agent/claude 驱动回答一次，确认"直接提示"与"agent 提示"的答案分布没有系统性偏差，写进文档。
C. 记录要求：每个 harness 的调用命令行、版本/commit、任务集 sha256 都写进 receipt；判定规则不变。
```

## 2. Decision（已冻结的实现决策）

- **数据源**（已就位，M1 `~/lfm3-dig/datasets/`）：
  - `mmlu-test.parquet` 14,042 行，cols `question, subject, choices(list), answer(int64 0..3)`，文件 sha256 前 16 = `74a41822ce7d3def`；
  - `gsm8k-test.parquet` 1,319 行，cols `question, answer(string，含 `#### <gold>` 尾注)`，文件 sha256 前 16 = `ee7b8da9e381df27`。
- **抽样（round-94 实测前先定口径，抽完把实际 seed/sha 写进 §5）**：
  - MMLU：57 subjects 分层，每 subject 配额 = `floor(500 × n_s / 14042)`，余数按 subject 字典序回补到恰好 500；`random.Random(20261120).sample` 逐 subject 抽；
  - GSM8K：`random.Random(20261121).sample(range(1319), 200)`；
  - 产物：`samples/mmlu-500.jsonl`、`samples/gsm8k-200.jsonl`（字段含 `idx, subject, question, choices, gold`；gold 仅进 receipt，永不进 prompt —— grep 自检）+ 各自 sha256 + id 序列 sha256。
- **执行形态**：greedy、temp 不传（或 0）、max_tokens=64、同一 `apxinf-serve` 二进制与同一 worker 二进制（G3 同源），两臂 = `APXINF_LFM3_4BIT=1` 开/关；服务起法与 banner/census effect proof 复用 `scripts/run-g1prime.sh` 的 G3 三件套（banner `native_4bit=`、census、`find -newer` freshness）。
- **A/A**：每臂 **≥2 段**，analyzer 报告同臂两段 accuracy spread；>5pp ⇒ 按 RUNNER §4 标注该臂窗口 spread 超标，accuracy 数字标注 caveat（spread >5pp ⇒ 该臂 accuracy 视为不可靠，向 Lead 报告是否作废）。
- **answer flip rate**：按"该题两臂各自唯一答案不同"计（未提取到答案的题记为 `unparsed`，不计入分母，单列计数）。
- **50 题 agent 交叉验证**：从 500 题 MMLU 样本内按 subject 字典序取前 50 题；驱动 = mini-swe-agent（`~/ag0-lfm3-0f6bbbfaa348/.venvs/mini-swe-agent/bin/mini`）或 claude CLI，**两臂必须同驱动、同 prompt 模板适配**；只跑一次/臂（探测系统性偏差，不做主判定）；记录调用命令行与版本。若通用 agent 在 1.7B 上跑不动（D4 已有记录），退 `code_loop` 并在文档写明。
- **INVALID 范围**：G3 空臂 / 抽样 sha 与本 doc §5 冻结值不符 / A/A spread >5pp 且作废后仍超 / 两臂二进制或 seed 不一致。**跨精度答案差异不是 INVALID**（Lead 00:16 冻结）。

## 3. 实现步骤（一轮一步）

1. **抽样 + 样本冻结**（离线，M1，pyarrow 从 mini-swe-agent venv 取，`python3 -I`）：跑 `scripts/build-knowledge-samples.py` ⇒ 样本 jsonl + sha 写进本 doc §5，commit（与本预登记同一 commit 或紧随其后）。**样本 sha 未写进 §5 之前不得起服务。**
2. **评测 harness**（离线）：`scripts/run-knowledge-eval.sh`（ABBA ≥2 段/臂，每段对一个 dataset 全样本发 greedy 请求，engine basis `--engine-timings`，双 Metal 锁 fcntl 顺序等待、`pmset displaysleepnow` + `caffeinate -si`）+ `scripts/analyze-knowledge-eval.py`（accuracy、delta、flip rate、per-arm spread、unparsed 计数、verdict json）。复用 round-50–57 的 harness 教训：错误 body 必读、stderr 按 epoch 聚合、分析块用独立 .py 不用 heredoc。
3. **测量窗**（M4，锁 FREE）：两臂 × 2 dataset × 2 段，预计 ~15–20 min 锁内（按 Lead 10:20 估算上浮）。
4. **回填 + PR**：结果写进本 doc §6 + 并入文档 PR（PR #133 或 Lead 指定的新 PR）。

## 4. 机制算术（预期，非判定）

- MMLU 500 × 2 段 × 2 臂 = 2,000 请求，max 64 tok，短输出；GSM8K 200 × 2 × 2 = 800 请求；两臂 decode 时长差 ~3.3×（G4-speed），预登记估算窗内 ~12–20 min。
- token 用量：MMLU ~500×64×4 = 128k tok/臂-段量级（实际远小于上限，因答案只有 1 字母）；GSM8K 推理长度更长，按 receipt 实报。

## 5. 冻结前置校验（开窗前必须全部命中，否则 abort 不动设备）

**抽样轮实取（round-95，2026-11-20 10:25 CST，M1，`python3 -I` + mini-swe-agent venv pyarrow 26.0.1，产物 `samples/`，receipt `samples/knowledge-samples.json`）：**

- 数据集文件 sha256：
  - `mmlu-test.parquet` = `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`（14,042 行）；
  - `gsm8k-test.parquet` = `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`（1,319 行）；
- 样本文件 sha256：
  - `samples/mmlu-500.jsonl` = `dbb9d807cf276b722e0f9ab40998878a8675e2fff1b7696c965dda58515d8eda`（500 行，57 subjects，quota 全表见 receipt）；
  - `samples/gsm8k-200.jsonl` = `1ce79a9f4d7bf1b280fe0de308bb12e67d3cb015c316fb6ed737965850f98519`（200 行，id 升序）；
- 样本 id 序列 sha256：
  - MMLU `samples/mmlu-500.ids.json` = `011104644a9f092eb62c941af44c9da683ba344bba2112e311ecdbc12f707fa6`；
  - GSM8K `samples/gsm8k-200.ids.json` = `88243d86aaa248c74e2ce386c3e5a29f1b0511620254dca0d8c172016a4b9da0`；
- 抽样 seed：MMLU = **20261120**（逐 subject `random.Random(...).sample`，subject 字典序消费流）、GSM8K = **20261121**（`sample(range(1319), 200)` 后升序）；
- 样本数：MMLU = **500**（配额 `floor(500×n_s/14042)` + subject 字典序回补余数，sum=500 已校验）、GSM8K = **200**；
- 产物写死一次：`samples/mmlu-500.jsonl`、`samples/gsm8k-200.jsonl` 及 ids、receipt 均为 write-once，内容不同则 `FROZEN_SAMPLE_MISMATCH` abort（重跑须 `--force`，作废此前窗口）；
- gold 不进 prompt 自检：GSM8K 按数值字段 grep prompt = **7 题命中**（7/200，纯算术巧合，gold 不来自模型输入；MMLU gold 为 A–D 字母，模板本身含字母不适用 grep）。

## 6. 结果（测量后回填）

**round-97 冒烟窗（2026-11-20 11:08–11:10，n_mmlu=10 / n_gsm8k=10，seg=1/臂，A=native-4bit → B=native-bf16）—— 中间态，尚无有效 verdict，原因如下：**

- **analyzer 崩溃已修（round-98 离线）**：`analyze` 在 ACC 打印行用 `agg[arm][ds]["n"]`，但 `aggregate()` 返回的是 `["n_per_seg"]` ⇒ `KeyError: 'n'`、`knowledge-verdict.json` 未生成、`SMOKE_*` verdict 未产出。修复 = 该 4 处改读 `n_per_seg`（round-97 §293 口径）。**本轮未重跑窗**，下一轮 analyzer 复跑既有 segment json 即可出 verdict（不占锁、不动设备）。
- **GSM8K 口径 P1 caveat（逐字，round-97 §290，结论待 Lead 拍板）**：GSM8K **20/20 请求 `cause=length`、`completion_tokens=64`** —— 模型在 64 token 截断处尚未算出答案（reply_tail 全为推理过程），`extract_gsm8k` 取到的"末位数字"是**截断瞬间的中间量**（如 idx154 解析出 3，实际是每学期 3 课 × 周数没算完；idx974 碰巧 5=gold）。**因此 64-token 下的 GSM8K accuracy 数字无效**；MMLU 无此问题（答案短，finish=length 是回嘴继续写，答案已在前缀）。
- **数字（仅供 Lead 看方向，不进判定）**：MMLU 4bit 4/10 vs bf16 2/10（delta=+2q；**MMLU 的 +2q 源于 bf16 3 题 unparsed（idx461/8889/9237）而非答错，是解析率差，非精度差**；两臂都 parsed 的 7 题答案全同 flip=0/7）；GSM8K 4bit 1/10 vs bf16 1/10（delta=0q，但按上条 caveat 数字无效）。速度 engine basis：4bit median 10.74 ms/tok vs bf16 36.06 ⇒ 3.36×，与 G4-speed PASS 3.3× 一致。A/A：单段，spread=0.00（退化、无意义）。
- **待 Lead 拍板（bug-2，改冻结值，本轮未动）**：GSM8K `max_tokens` 64 → 256（或改 prompt 让它在 64 内给答案）。`max_tokens=64` 是 Lead 10:20 Gate 冻结值，**本 worker 无权单方面改**；改口径须 Lead 批准后另起一轮 + 重开窗。

**最终 MMLU/GSM8K accuracy、delta、flip rate、verdict：待上述两点处理后回填。**

（其他测量结果：待填）
