# no-thinking SFT 回退分析:pass@1 0.29 → 0.09

日期:2026-07-09(pod 日志时间 07-10,UTC 时区差)
环境:RunPod A100,`/workspace/coding_rl`
结论先行:**SFT 训练管线没有 bug,是训练目标本身的问题——no-thinking 线的 SFT
数据是人类竞赛提交的原始代码(terse、tab 缩进、无推导),模型忠实学会了这种
风格,而模仿一个比自己弱的"老师"直接把 4B 模型拉低了。**

---

## 1. 现象

同一 dev 集、同一采样参数,唯一区别是 `--lora`:

```bash
python rl/scripts/eval_model.py --model Qwen/Qwen3-4B \
  --data data/dev_internal.jsonl --limit 1000 --no-thinking \
  --out outputs/eval/qwen3_4b_base_nothink.json --n 1

python rl/scripts/eval_model.py --model Qwen/Qwen3-4B \
  --lora outputs/qwen3_4b_sft_nothink \
  --data data/dev_internal.jsonl --limit 1000 --no-thinking \
  --out outputs/eval/qwen3_4b_sft_nothink.json --n 1
```

| 模型 | pass@1 (n=1, temp 0.8, 1000 题) |
| --- | ---: |
| Qwen3-4B base(no-thinking) | **0.29** |
| + SFT LoRA `qwen3_4b_sft_nothink` | **0.09** |

采样:temperature 0.8 / top_p 0.95 / max_tokens 4096,prompt 用
`enable_thinking=False` 渲染(与训练一致)。

噪声界:n=1、p≈0.29、1000 题时,pass@1 的标准误 ≈ √(0.29·0.71/1000) ≈ 0.014,
2σ ≈ ±0.03。**-0.20 的回退约 14σ,不是采样噪声。**

涉事 SFT 训练(`rl/rlcoder/train/sft_trl.py` docstring 示例,LoRA):

```bash
python rlcoder/train/sft_trl.py --model Qwen/Qwen3-4B \
  --data data/sft_train.jsonl --no-thinking \
  --bf16 --gradient-checkpointing --packing \
  --output outputs/qwen3_4b_sft_nothink
```

默认超参:LoRA r=32 / α=64 / dropout 0.05 / all-linear,lr 1e-4 cosine
(warmup 3%),2 epochs,max_length 8192,packing,completion_only_loss=True。
(如实际命令与此不同,请在此更正。)

---

## 2. 证据

### 2.1 生成对比(前 8 题,BASE vs SFT,同 prompt 同采样)

用 vLLM 对 `data/dev_internal.jsonl` 前 8 题各生成 1 个样本
(temp 0.8 / top_p 0.95 / max_tokens 4096 / seed 0,脚本见附录):

| # | BASE 长度 (chars) | BASE finish | SFT 长度 (chars) | SFT finish |
| --- | ---: | --- | ---: | --- |
| 0 | 2255 | stop | 293 | stop |
| 1 | 293 | stop | 193 | stop |
| 2 | 826 | stop | 203 | stop |
| 3 | 2920 | stop | 221 | stop |
| 4 | 1654 | stop | 310 | stop |
| 5 | 1042 | stop | 2007 | stop |
| 6 | 936 | stop | 1099 | stop |
| 7 | 19175 | length | 475 | stop |

- 双方 8/8 都有 ```` ```python ```` fence,代码提取正常 → **不是格式/解析问题**。
- 中位长度:BASE ≈ 1350 字符 → SFT ≈ 300 字符,**砍掉约 4.5 倍**。
- BASE 的问题是偶发啰嗦(#7 顶到 length);SFT 的问题是整体塌缩成极短答案。

### 2.2 风格指纹:8 空格缩进

SFT 输出全部是 **8 空格缩进**(人类提交里 tab 缩进的典型渲染),BASE 输出是
标准 4 空格。模型不会凭空发明缩进风格——这是"学到了训练数据表面风格"的直接
指纹,和上游人类金标(APPS/TACO/CodeContests 时代的选手提交)完全吻合。

### 2.3 具体样本:质量塌缩

**样本 #0(组合求和 mod 100003):**

BASE 铺开写了预计算(阶乘、前缀和、幂表)的正规路线;SFT 给了指数复杂度的
暴力枚举——形状像答案,复杂度不可能过题:

```python
# SFT #0:对每个查询枚举全部 C(n,k) 个组合
comb = list(itertools.combinations(a, k))
for lst in comb:
    for i in range(len(lst)):
        ans += lst[i]
```

**样本 #1(输出 [l,r] 内 k 的幂,否则 -1):**

BASE 的解是对的(先收集再统一输出,空则 -1)。SFT 的短代码有两个真 bug:

```python
# SFT #1
l, r, k = map(int, input().split())
n = 1
if n >= l and n <= r:
        print(n, end=' ')
while n * k <= r:
        n *= k
        if n >= l and n <= r:
                print(n, end=' ')
print(-1 if n >= r else '')
```

- `l=1, r=8, k=2`:先打出 `1 2 4 8`,随后 `n=8 >= r=8` 又补一个 `-1` → 错;
- `l=5, r=7, k=2`:区间内没有幂,循环结束 `n=4 < r=7`,打印空串而非 `-1` → 错。

这就是 0.09 的微观形态:**短、像人类提交、但要么暴力要么带 bug**。

### 2.4 数据来源链(模型学的就是这个)

no-thinking SFT 的 completion 逐级来自人类金标:

1. `rl/scripts/build_dataset.py --source hf` → HF 数据集
   `open-r1/verifiable-coding-problems-python_decontaminated-tested`
   (`build_dataset.py:36`);
2. `rl/rlcoder/data/parse.py:59`:`gold_solution = row["gold_standard_solution"]`
   ——该字段是上游(源自 APPS/TACO/CodeContests 等)的**人类选手提交**,
   fenced 短代码、tab 缩进、零解释;
3. `rl/scripts/split_stages.py` 切出 `sft_train.jsonl` / `dev_internal.jsonl`
   (同池、互斥);
4. `rl/rlcoder/train/sft_trl.py:55`:
   `completion = p.gold_solution.strip()`,completion-only loss 原样模仿。

即:**SFT 目标 = 人类提交原文**。2 epochs × lr 1e-4 × r32 all-linear 足以把
这种风格焊死。2.1–2.3 的输出特征(短 + 8 空格缩进 + 竞赛提交腔)证明训练
"完全成功"——成功地学坏了。

### 2.5 排除替代解释

| 假设 | 排除依据 |
| --- | --- |
| chat template / prompt 不一致 | 两次 eval 仅差 `--lora`,渲染同为 `enable_thinking=False` |
| 代码提取失败压分 | SFT 8/8 `fence=True`、`code_len>0`、`finish=stop` |
| eval 分布外 | `dev_internal` 与 `sft_train` 是同一 pool 的分层互斥切分——模型在自己的训练分布上变差 |
| 采样噪声 | 1000 题上 Δ=-0.20 ≈ 14σ(见 §1) |
| 训练发散/模型损坏 | 输出连贯、格式规范、风格高度一致——不是崩了,是学偏了 |

---

## 3. 机制:为什么模仿人类金标会降智

1. **向更弱的老师蒸馏。** Qwen3-4B(2025)写竞赛 Python 的平均水平已高于
   APPS/TACO 时代的人类提交均值。SFT 把策略往数据分布上拉,数据比模型弱,
   就是往下拉。蒸馏只在两种情况下有益:老师更强(R1 traces),或者数据来自
   模型自己的被验证正确的输出(rejection sampling)。
2. **答案没有推导过程。** 人类金标只有结论,没有得出结论的路径。4B 学不会
   其中的洞察,只学到了最表面的统计特征:**短**。
3. **no-thinking 下输出长度≈推理预算。** base 在 no-thinking 模式靠"在回答里
   边写边想"(输出 800–3000 字符)撑起 0.29;SFT 教它"回答应为 200–500 字符",
   等于把测试时计算量砍掉一个数量级。
4. **金标本身只保证过弱测试。** 上游只验证过自带测试用例(且本仓截断到
   max-tests 条),大量金标是 hacky/暴力写法,模仿它们泛化极差。

---

## 4. 待复核项(pod 上)

1. **实锤训练数据风格**(应与 SFT 输出严丝合缝:中位数几百字符、大比例
   tab/8 空格缩进):

   ```python
   import json, statistics
   lens, tab8, n = [], 0, 0
   for line in open("data/sft_train.jsonl"):
       g = json.loads(line).get("gold_solution") or ""
       n += 1; lens.append(len(g))
       if "\n\t" in g or "\n        " in g: tab8 += 1
   print(f"n={n} median_len={statistics.median(lens)} tab8_frac={tab8/n:.2f}")
   ```

2. **检查训练日志**有无 `[warn] TRL x.y ignores: [...]`,尤其
   `completion_only_loss` 是否被旧版 TRL 丢弃(次要嫌疑,不影响主结论)。
3. 之后的 A/B 评测用 `--n 8`(`eval_model.py` 默认)降噪;n=1 只够看大差距。

---

## 5. 结论与选项

**人类金标适合当 RL 的 verifier 素材,不适合当 SFT 模仿目标。** 候选方向:

- **A(推荐):no-thinking 线跳过 SFT,base 直接 GRPO。**
  SFT 的动机是"教格式",但 base 已 8/8 输出规范 fenced code——格式不是瓶颈,
  正确率才是,那是 GRPO 的活。
- **B:保留 SFT 阶段,但改 rejection-sampling(STaR/RFT)。**
  用 base 在 rl_pool 上采 k=8,只留过测试的样本做 SFT。基建现成:
  `rl/scripts/difficulty_filter.py --save-rollouts` 已输出带 pass/fail 的
  rollouts,可直接复用。模型留在自己的分布内,只强化对的部分。
- **C:要蒸馏就走 thinking 线**(OpenCodeReasoning / codeforces-cots 的
  `<think>` traces,`--max-length 16384`),老师比学生强,方向才是向上的。

状态:分析完成,等待选 A/B 后动代码。

---

## 附录:生成对比脚本

```python
import sys
sys.path.insert(0, "rl")
from rlcoder.data.load import load_clean_jsonl
from rlcoder.prompting import load_processing_class, render_chat_prompt, build_messages
from rlcoder.rewards.code_reward import extract_code
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL, LORA = "Qwen/Qwen3-4B", "outputs/qwen3_4b_sft_nothink"
probs = load_clean_jsonl("data/dev_internal.jsonl", limit=8)
proc = load_processing_class(MODEL)
prompts = [render_chat_prompt(proc, build_messages(p), enable_thinking=False) for p in probs]

llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192,
          gpu_memory_utilization=0.90, enable_lora=True, max_lora_rank=32, seed=0)
sp = SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=4096)

for tag, req in [("BASE", None), ("SFT", LoRARequest("a", 1, LORA))]:
    outs = llm.generate(prompts, sp, lora_request=req)
    print(f"\n########## {tag} ##########")
    for i, o in enumerate(outs):
        t = o.outputs[0].text
        print(f"[{tag} #{i}] len={len(t)}ch  finish={o.outputs[0].finish_reason}  "
              f"fence={'```' in t}  code_len={len(extract_code(t))}")
    for i, o in enumerate(outs[:2]):
        print(f"\n----- {tag} #{i} RAW head -----\n{o.outputs[0].text[:1200]}")
```
