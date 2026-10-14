"""在训练时没见过的评估题上，比较几个模型拿到的平均奖励。打分方式和 train_grpo.py 的 calculate_rewards 相同。

每道题用和训练相同的方式构造提示（RLAIFDataset，thinking_ratio 0.9），每个模型对每道题写 N 个回答。
同一道题，所有模型用同一个随机种子，比较才公平。完整回答保存到 my_lab/results/grpo_eval.md。

用法（在仓库根目录运行，需要 GPU 和 ../internlm2-1_8b-reward）:
    python my_lab/grpo_eval.py --weights full_sft
    python my_lab/grpo_eval.py --weights full_sft,grpo
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import LMForRewardModel

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='full_sft')
parser.add_argument('--data', default=str(ROOT / 'dataset' / 'rlaif_eval100.jsonl'))
parser.add_argument('--samples', type=int, default=2, help='每道题写几个回答')
parser.add_argument('--max_gen_len', type=int, default=384, help='和训练时的 --max_gen_len 保持一致')
parser.add_argument('--thinking_ratio', type=float, default=0.9)
parser.add_argument('--reward_model', default=str(ROOT.parent / 'internlm2-1_8b-reward'))
parser.add_argument('--out', default=str(ROOT / 'my_lab' / 'results' / 'grpo_eval.md'))
args = parser.parse_args()


def rep_penalty(text, n=3, cap=0.5):   # 和 train_grpo.py 完全相同
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def reward(rm, prompt, response):
    """返回 (规则分, 奖励模型分)，逻辑同 train_grpo.calculate_rewards"""
    matches = re.findall(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", prompt, re.DOTALL)
    messages = [{"role": role, "content": content.strip()} for role, content in matches]
    answer, rule = response, 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    if '</think>' in response:
        thinking, answer_part = response.split('</think>', 1)
        rule += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        rule += 0.25 if response.count('</think>') == 1 else -0.25
        answer = answer_part.strip()
    rule -= rep_penalty(answer)
    return rule, rm.get_score(messages, answer)


tok = AutoTokenizer.from_pretrained(ROOT / 'model')
random.seed(0)   # 固定"是否开启思考"和随机 system 提示，所有模型看到完全相同的提示
ds = RLAIFDataset(args.data, tok, max_length=768, thinking_ratio=args.thinking_ratio)
prompts = [ds[i]['prompt'] for i in range(len(ds))]
print(f'{len(prompts)} 道评估题，每题 {args.samples} 个回答；正在加载奖励模型……')
rm = LMForRewardModel(args.reward_model, device='cuda', dtype=torch.float16)

weights = args.weights.split(',')
results = {}
start = time.time()
for w in weights:
    model = MiniMindForCausalLM(MiniMindConfig())
    model.load_state_dict(torch.load(ROOT / 'out' / f'{w}_768.pth'), strict=True)
    model = model.half().eval().cuda()
    rows = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        torch.manual_seed(1000 + i)
        with torch.no_grad():   # 和训练时的 rollout 一样：temperature 0.8，其余用 generate 的默认值
            out = model.generate(inputs=ids['input_ids'].repeat(args.samples, 1),
                                 attention_mask=ids['attention_mask'].repeat(args.samples, 1),
                                 max_new_tokens=args.max_gen_len, temperature=0.8,
                                 eos_token_id=tok.eos_token_id, do_sample=True)
        for k in range(args.samples):
            new_ids = out[k][ids['input_ids'].shape[1]:].tolist()
            if tok.eos_token_id in new_ids:
                new_ids = new_ids[:new_ids.index(tok.eos_token_id) + 1]
            text = tok.decode(new_ids, skip_special_tokens=True)
            rule, rm_score = reward(rm, prompt, text)
            think = text.split('</think>', 1)[0].strip() if '</think>' in text else ''
            rows.append({'i': i, 'k': k, 'text': text, 'rule': rule, 'rm': rm_score, 'total': rule + rm_score,
                         'tokens': len(new_ids), 'stopped': tok.eos_token_id in new_ids, 'thinks': bool(think)})
        if (i + 1) % 20 == 0:
            print(f'  {w}: {i + 1}/{len(prompts)}（已用 {time.time() - start:.0f} 秒）')
    results[w] = rows
    del model
    torch.cuda.empty_cache()


def mean(xs):
    return sum(xs) / len(xs)


print(f'\n{"模型":<12}{"平均总奖励":>10}{"奖励模型分":>10}{"规则分":>8}{"平均长度":>8}{"有思考过程":>10}{"写满上限":>8}')
for w in weights:
    r = results[w]
    print(f'{w:<12}{mean([x["total"] for x in r]):>+13.3f}{mean([x["rm"] for x in r]):>+13.3f}'
          f'{mean([x["rule"] for x in r]):>+10.3f}{mean([x["tokens"] for x in r]):>10.0f}'
          f'{mean([x["thinks"] for x in r]):>12.0%}{mean([not x["stopped"] for x in r]):>10.0%}')

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'# GRPO 评估：{", ".join(weights)}\n\n{len(prompts)} 道评估题 × {args.samples} 个回答，'
            f'max_gen_len={args.max_gen_len}，同一题各模型用同一个随机种子。\n\n')
    for i, prompt in enumerate(prompts):
        last_user = re.findall(r"<\|im_start\|>user\s+(.*?)<\|im_end\|>", prompt, re.DOTALL)
        f.write(f'## 第 {i + 1} 题：{last_user[-1].strip()[:80] if last_user else ""}\n\n')
        for w in weights:
            for x in [x for x in results[w] if x['i'] == i]:
                f.write(f'### {w}（第 {x["k"] + 1} 个）  总分 {x["total"]:+.2f}（规则 {x["rule"]:+.2f}，'
                        f'奖励模型 {x["rm"]:+.2f}），{x["tokens"]} token\n\n{x["text"]}\n\n')
print(f'\n完整回答已保存到 {out}')
