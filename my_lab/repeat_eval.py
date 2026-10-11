"""用数字比较几个模型的回答：重复率、长度、能不能自己停下来。

每个模型对 eval_llm.py 的 8 个固定问题各回答 N 次（默认 3 次）。
同一个问题的第 k 次回答，所有模型用同一个随机种子，比较才公平。
完整回答会保存到 my_lab/results/repeat_eval.md，方便人工判断"切不切题、对不对"。

用法（在仓库根目录运行，需要 GPU）:
    python my_lab/repeat_eval.py
    python my_lab/repeat_eval.py --weights full_sft,dpo,dpo_lr1e6 --samples 3
"""
import argparse
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

PROMPTS = [   # 和 eval_llm.py 的自动测试相同
    '你有什么特长？',
    '为什么天空是蓝色的',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '如果明天下雨，我应该如何出门',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '推荐一些中国的美食',
]

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='full_sft,dpo,dpo_lr1e6', help='逗号分隔的权重前缀')
parser.add_argument('--samples', type=int, default=3, help='每个问题回答几次')
parser.add_argument('--max_new_tokens', type=int, default=400)
parser.add_argument('--temperature', type=float, default=0.85)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--out', default=str(ROOT / 'my_lab' / 'results' / 'repeat_eval.md'))
args = parser.parse_args()


def sentences(text):
    """按句号、问号、换行等切成句子，去掉编号、加粗符号和空白，用来判断"同一句话说了两遍" """
    parts = re.split(r'[。！？!?；;\n]', text)
    cleaned = []
    for p in parts:
        p = re.sub(r'^\s*(\d+[\.、]|[-*#]+)\s*', '', p)
        p = re.sub(r'[\s*#：:，,、“”"（）()]', '', p)
        if len(p) >= 4:
            cleaned.append(p)
    return cleaned


def repeated_sentence_ratio(text):
    sents = sentences(text)
    if not sents:
        return 0.0
    seen, dup = set(), 0
    for s in sents:
        dup += s in seen
        seen.add(s)
    return dup / len(sents)


def repeated_4gram_ratio(text):
    """连续 4 个字的片段里，有多少在这个回答前面已经出现过。能抓到"粤菜是中国四大菜系之一"这种半句重复"""
    chars = re.sub(r'\s', '', text)
    grams = [chars[i:i + 4] for i in range(len(chars) - 3)]
    if not grams:
        return 0.0
    seen, dup = set(), 0
    for g in grams:
        dup += g in seen
        seen.add(g)
    return dup / len(grams)


tok = AutoTokenizer.from_pretrained(ROOT / 'model')
weights = args.weights.split(',')
results = {w: [] for w in weights}
start = time.time()

for w in weights:
    model = MiniMindForCausalLM(MiniMindConfig())
    model.load_state_dict(torch.load(ROOT / 'out' / f'{w}_768.pth'), strict=True)
    model = model.half().eval().cuda()
    for pi, prompt in enumerate(PROMPTS):
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                       add_generation_prompt=True, open_thinking=False)
        inputs = tok(text, return_tensors='pt').to('cuda')
        for k in range(args.samples):
            torch.manual_seed(1000 * pi + k)          # 同一题同一次，各模型用同一个种子
            out = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                 max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_p=args.top_p, eos_token_id=tok.eos_token_id, do_sample=True)
            new_ids = out[0][inputs['input_ids'].shape[1]:].tolist()
            stopped = tok.eos_token_id in new_ids
            reply = tok.decode(new_ids, skip_special_tokens=True).strip()
            results[w].append({'prompt': prompt, 'k': k, 'reply': reply, 'tokens': len(new_ids),
                               'stopped': stopped, 'rep_sent': repeated_sentence_ratio(reply),
                               'rep_4gram': repeated_4gram_ratio(reply)})
        print(f'  {w}: 第 {pi + 1}/{len(PROMPTS)} 题完成（已用 {time.time() - start:.0f} 秒）')
    del model
    torch.cuda.empty_cache()


def mean(xs):
    return sum(xs) / len(xs)


print(f'\n每个模型 {len(PROMPTS)} 题 × {args.samples} 次 = {len(PROMPTS) * args.samples} 个回答\n')
header = f'{"模型":<12}{"重复句比例":>10}{"重复4字片段":>12}{"平均长度(token)":>16}{"写满上限的比例":>14}'
print(header)
for w in weights:
    r = results[w]
    print(f'{w:<12}{mean([x["rep_sent"] for x in r]):>12.1%}{mean([x["rep_4gram"] for x in r]):>14.1%}'
          f'{mean([x["tokens"] for x in r]):>16.0f}{mean([not x["stopped"] for x in r]):>16.1%}')

print('\n按问题看"重复4字片段"（每格是该模型 N 次回答的平均）:')
print(f'{"问题":<22}' + ''.join(f'{w:>12}' for w in weights))
for prompt in PROMPTS:
    row = [mean([x['rep_4gram'] for x in results[w] if x['prompt'] == prompt]) for w in weights]
    print(f'{prompt[:20]:<22}' + ''.join(f'{v:>12.1%}' for v in row))

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'# 回答对比：{", ".join(weights)}\n\n')
    f.write(f'temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}, '
            f'每题 {args.samples} 次，同一题同一次各模型用同一个随机种子。\n\n')
    for prompt in PROMPTS:
        f.write(f'## {prompt}\n\n')
        for k in range(args.samples):
            for w in weights:
                x = next(x for x in results[w] if x['prompt'] == prompt and x['k'] == k)
                f.write(f'### {w}（第 {k + 1} 次）  重复句 {x["rep_sent"]:.0%}，重复4字片段 {x["rep_4gram"]:.0%}，'
                        f'{x["tokens"]} token，{"自己停下" if x["stopped"] else "写满上限"}\n\n')
                f.write(x['reply'] + '\n\n')
print(f'\n完整回答已保存到 {out}')
