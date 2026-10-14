# my_lab：我的 MiniMind 学习实验记录

跟着 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 从零训练一个 64M 参数的 LLM，并在一台 Windows 笔记本上走完 预训练 → SFT → LoRA。这里记录实际用到的命令、参数、结果和踩过的坑。

## 环境

| 项目 | 配置 |
|---|---|
| 系统 | Windows 11 |
| 显卡 | RTX 2070 Max-Q，8 GB 显存，**不支持 bf16**，训练时持续在 85–90 °C 降频运行 |
| Python | conda 环境 `minimind`，Python 3.11，torch 2.14.0+cu126 |
| 数据 | HuggingFace `jingyaogong/minimind_dataset`：`pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`、`lora_identity.jsonl` |

Windows 上需要注意：

- `requirements.txt` 里的 `ujson==5.1.0` 没有 Python 3.11 的 Windows wheel，需要 C++ 编译器；代码里没有用到它，已注释掉。
- `load_dataset` 默认把缓存写到 C 盘，设置 `$env:HF_DATASETS_CACHE = "D:\LLM\hf_cache"` 改到 D 盘。
- 显卡不支持 bf16，所有训练都加 `--dtype float16`（会自动启用 GradScaler）。
- `--num_workers 2`，Windows 上开 8 个数据加载进程又慢又占内存。

## 1. 预训练

```bash
cd trainer
python -u train_pretrain.py --dtype float16 --batch_size 16 --accumulation_steps 16 --num_workers 2 --epochs 1 --from_resume 1
```

- batch 32 会显存不足，改成 batch 16 × 梯度累积 16 = 每次更新 256 条，和默认的 32 × 8 等效。
- 1 个 epoch，79,390 步，用时约 6 小时 40 分钟。
- loss：7.48（第 100 步）→ 2.76（1 万步）→ 2.22（3 万步）→ **1.90**（结束）。
- ⚠️ 用的是脚本默认的 `max_seq_len=340`，而 README 推荐 mini 数据用 **768**，超过 340 token 的文本被截断了。以后重跑应改成 768。

测试：`python eval_llm.py --load_from ./model --weight pretrain`。模型已有一些知识（能说出"散射、短波长"），但会接着用户的问题往下写，比如先补一个问号，或者替用户编出"，并给出一个简单的例子"；回答不会自己结束，也没有格式。原因是预训练数据里混有大量直接拼接的问答文本，却没有对话模板。

## 2. SFT

```bash
python -u train_full_sft.py --dtype float16 --batch_size 8 --accumulation_steps 2 --num_workers 2 --epochs 1 --from_resume 1
```

- 保持 `max_seq_len=768`。抽样 3000 条统计：中位数 496 token，45% 超过 512，只有 5.5% 超过 768，所以不能缩短到 512。
- batch 8 × 累积 2 = 16，等于默认值；113,215 步，总共约 12 小时。
- loss：2.44 → 约 1.7。
- 学会了：对话模板（不再续写问题）、Markdown 排版、身份（"由 jingyaogong 创建"）、会主动结束。
- 变差了：回答变长以后，重复和幻觉更多（例如"光反应"连写 4 次）。小模型的知识填不满 SFT 教给它的"长篇分点"格式。

### 坑：Windows 上保存检查点崩溃

跑到第 48,000 步时崩溃：

```
PermissionError: [WinError 5] 拒绝访问: '../checkpoints/full_sft_768.pth.tmp' -> '../checkpoints/full_sft_768.pth'
```

`os.replace` 在 Windows 上遇到目标文件被临时占用（杀毒软件或索引服务扫描刚写好的大文件）时会失败。修复方法：在 `trainer/trainer_utils.py` 里加了 `_safe_replace`，失败后按指数退避重试，最多 10 次，仍然失败就打印警告并继续训练。之后用 `--from_resume 1` 从第 47,000 步接着训练，后面没再出问题。

## 3. LoRA：给模型换身份

MiniMind 只给 `in_features == out_features` 的线性层加 LoRA，也就是每层的 q_proj 和 o_proj，共 16 个模块，rank 16，393,216 个参数，占 0.61%。

生成数据（把 MiniMind 和 Jingyao Gong 换成自己的名字，并混入普通对话）：

```bash
python my_lab/make_identity_data.py --name MomoMind --dev momomojun --n_general 300
```

三次实验，测试时都用 `--temperature 0.3`：

| 实验 | 数据 | epoch / lr | 更新次数 | 身份数据上的 loss | 你是谁 | 没见过的问法 / 英文 | 中国的首都 |
|---|---|---|---|---|---|---|---|
| ① `lora_identity_momo` | 91 条身份对话 | 10 / 1e-4 | 60 | 2.99 → 2.31 | ❌ 还是 minimind | ❌ | ✅ 北京 |
| ② `lora_try` | 91 条身份对话 | 30 / 5e-4 | 180 | 2.98 → 0.59 | ✅ MomoMind | ✅ | ❌ **也在自我介绍** |
| ③ `lora_identity_mix` | 91 身份 + 300 普通 | 30 / 5e-4 | 750 | 3.04 → 0.56 | ✅ MomoMind | ✅ | ✅ 北京 |

- ① **欠拟合**：只更新了 60 次，ΔW 只有 W 的 0.8%，压不过 SFT 里根深蒂固的旧身份。
- ② **过拟合 / 灾难性遗忘**：91 条全是自我介绍，模型学成了"问什么都自我介绍"。
- ③ **数据混合（回放）**：和 ② 相比只改了数据这一个变量，遗忘基本消失。剩下的副作用是：简单问题开头会多出"您好，"（从身份数据里学到的说话风格）；"日本首都"从答对东京变成答错。
- 判断一个错误是不是 LoRA 造成的，要和不加 LoRA 的 SFT 模型对比：诺贝尔奖、深圳这类问题，SFT 模型本来就答错。
- `SFTDataset` 会随机给对话加 system 提示，所以同一份数据每次算出的 loss 会有一点浮动（2.98 和 3.04）。

检查 LoRA 有没有学会：

```bash
python my_lab/lora_loss.py out/lora_identity_mix_768.pth --data dataset/lora_identity_momo.jsonl
```

## 4. 注意力可视化

```bash
python my_lab/attn_demo.py "小明养了一只猫，它很可爱" 6
```

打印第 6 个 token（"它"）在 8 层、每层 8 个头里，把注意力分给了前面哪些 token。第 2 层第 3 个头把 0.44 的注意力给了"猫"；整体上，大部分注意力落在逗号和自己身上。换成"小红买了一本书，她很喜欢"时，没有哪个头明显指向"小红"（"小红"被拆成了两个 token，而且模型很小）。

## 5. 工具调用

SFT 数据里已经混入了约 10 万条工具调用样本，所以不需要单独训练。`scripts/eval_toolcall.py` 的工具全是模拟的（天气永远 22°C，单位换算永远乘 0.621371）。

```bash
python my_lab/toolcall_trace.py     # 分 7 步展示：程序拼输入 → 模型逐 token 写请求 → 程序解析并执行 → 结果交还模型 → 最终回答
```

- 模型只会写字："要不要用工具"就是第 1 个 token 预测出 `<tool_call>`（概率 0.75），"用哪个工具"是第 8 个 token（`get` 0.81）。真正执行工具的是 Python 程序。
- 工具说明会占很长的输入：0 个工具 30 token，2 个 407，8 个 **1317**（超过了 SFT 训练时的 768）。
- `cd scripts; python eval_toolcall.py --weight full_sft` 的 8 个自动测试：单步任务（算乘法、换算单位、查天气）基本都对；两步任务（生成随机数再求平方）失败；有一题参数填错（汇率工具填了 `timezone`），只是因为模拟工具不看参数，答案才碰巧对；英文加多任务完全没调用工具。

## 6. DPO

```bash
python my_lab/show_dpo.py                      # 看 chosen / rejected 数据
cd trainer
python -u train_dpo.py --dtype float16 --batch_size 4 --num_workers 2 --epochs 1 --log_interval 100 --save_interval 500 --from_resume 1
python -u train_dpo.py ... --learning_rate 1e-6 --save_weight dpo_lr1e6     # 对比实验，只改学习率
cd ..
python my_lab/dpo_eval.py --weight dpo         # 偏好准确率（参考模型 full_sft）
python my_lab/repeat_eval.py                   # 重复率、长度，三个模型 × 8 题 × 3 次
```

- `dpo.jsonl` 有 17,166 条，同一个问题配一个 chosen、一个 rejected；batch 4 实际是 8 条序列 × 1024 token，加上冻结的参考模型，8 GB 刚好够；4292 步，约 50 分钟。
- 第一个 loss 是 **0.6937**，和理论值 ln 2 一致（训练开始时策略模型和参考模型相同）。
- 默认的 `save_interval=100` 每次要写 138 MB + 649 MB，改成 500。

| | SFT | DPO lr 4e-8（默认） | DPO lr 1e-6 |
|---|---|---|---|
| 训练 loss（最后 1/4 段的平均） | — | 0.648 | 0.409，波动 0.08～1.39 |
| 权重相对 SFT 的变化 | 0 | 0.0069% | — |
| chosen 的平均进步 | 0 | +0.16 | +2.31 |
| rejected 的平均进步 | 0 | −0.38 | −7.87 |
| 偏好准确率（训练数据中抽 500 条） | 50% | 74.6% | 88.8% |
| 对话 | 基线 | 几乎看不出变化（每个 token 的概率只变了约 0.15%） | 条理和格式明显变好；个别题重复和跑题更多（"多语言支持"写 6 遍、"光合作用并非标准术语"） |
| 重复 4 字片段（8 题 × 3 次，`repeat_eval.py`） | 19.6% | 18.7% | 23.2% |
| 写满 400 token 上限的比例 | 54.2% | 58.3% | **83.3%** |

- DPO 主要是在压低 rejected，而不是拉高 chosen。
- 指标好不等于回答好：偏好准确率只衡量"更偏向 chosen"，而且评估用的是训练数据。大学习率学到了标注员喜欢的"有条理"，最明确的代价是**长度偏差**（写满上限 54% → 83%），平均重复只略增。这就是作者把学习率设成 4e-8 的原因。
- 各阶段对权重的改动：随机初始化 → 预训练 106.8%，预训练 → SFT 9.56%，SFT → DPO 0.0069%；身份 LoRA 的 ΔW 是 q/o 原权重的 22.8%（存在单独的文件里，原模型不变）。

## 7. GRPO

需要奖励模型 `internlm/internlm2-1_8b-reward`（3.4 GB，放在 minimind 的同级目录，加载时 `trust_remote_code=True`，fp16 占 3.4 GB 显存）和 `rlaif.jsonl`（19,502 道只有问题的题）。

```bash
python my_lab/reward_demo.py                           # 体验裁判：规则 + 奖励模型怎么打分
python my_lab/make_rlaif_split.py --n_train 1500 --n_eval 100
cd trainer
cmd /c "python -u train_grpo.py --dtype float16 --num_workers 2 --loss_type grpo --batch_size 1 --num_generations 4 --max_gen_len 384 --data_path ../dataset/rlaif_train1500.jsonl --save_interval 100 --from_resume 1 --debug_mode --debug_interval 50 > ..\grpo.log 2>&1"
cd ..
python my_lab/grpo_eval.py --weights full_sft,grpo      # 100 道没见过的题 × 2 个回答
```

- `train_grpo.py` 的默认 `--loss_type` 是 `cispo`，要学 GRPO 得指定 `grpo`。
- **修复 1**：`train_grpo.py` 在 fp16 下没有 GradScaler，补上了（作者用 bf16 所以没遇到）。`train_ppo.py`、`train_agent.py` 有同样问题。
- **修复 2**：输出重定向到文件时 Windows 用 GBK，数据里一个落单的代理字符 `홀` 让 `Logger` 打印 `[DEBUG]` 时崩溃；改为编码失败时转义再打印。
- 每步约 10～13 秒，全量要 53 小时，所以只用 1500 道题，约 5 小时。
- `Actor Loss` 一直显示 0 是正常的（每次生成后只更新一次，ratio = 1），要看 Reward。

| | SFT | GRPO |
|---|---|---|
| 平均总奖励（100 道评估题） | −1.506 | **−1.332** |
| 奖励模型分 / 规则分 | −1.614 / +0.109 | −1.504 / +0.172 |
| 平均长度 | 187 | 218 |
| 有思考过程 | 14% | **30%** |
| 写满 384 上限 | 24% | 24% |

- 提升的 95% bootstrap 置信区间 [+0.05, +0.29]，不含 0；200 对回答 GRPO 更好 99、更差 65、相同 36。
- 但看具体回答：GRPO 主要学会了"写思考过程"（规则最多给 +1.25），思考内容是套路（"好的，用户问的是……首先，我需要……"），常常太长、挤掉回答；部分"进步"来自 SFT 那次恰好严重重复。奖励设计决定了模型学到什么。
- 奖励本身的漏洞（`reward_demo.py`）：不到 20 字的正确短答被扣分；重复惩罚按词切分，检测不到中文换着说法的重复；奖励模型几乎分不清"首都是北京"和"首都是上海"。

## 下一步

- [x] 测试 SFT 模型的工具调用能力
- [x] DPO
- [x] GRPO
- [ ] PPO / CISPO
- [ ] 知识蒸馏
- [ ] 用 `max_seq_len=768` 重跑预训练

学习笔记（概念讲解、我的问题和解答、小检验）在 [LLM学习笔记.md](LLM学习笔记.md)。
