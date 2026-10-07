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

## 下一步

- [ ] 测试 SFT 模型的工具调用能力（SFT 数据里已混入约 10 万条 Tool Call 样本）
- [ ] DPO
- [ ] GRPO / PPO / CISPO（先确认 8 GB 显存够不够）
- [ ] 知识蒸馏
- [ ] 用 `max_seq_len=768` 重跑预训练
