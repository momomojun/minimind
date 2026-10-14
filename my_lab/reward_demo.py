"""体验 GRPO 的"裁判"：同一个问题配几个不同的回答，看奖励怎么打分。

打分方式和 trainer/train_grpo.py 的 calculate_rewards 一致（不含 <think> 部分）：
  长度规则：回答 20～800 字 +0.5，否则 −0.5
  重复惩罚：按 3 个词的片段重复程度扣分，最多 −0.5
  奖励模型：InternLM2-1.8B-Reward 打分，截断到 −3～+3

用法（在仓库根目录运行，需要 GPU，奖励模型放在 ../internlm2-1_8b-reward）:
    python my_lab/reward_demo.py              # 用内置的例子
    python my_lab/reward_demo.py --interactive  # 自己输入问题和回答
"""
import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.trainer_utils import LMForRewardModel

EXAMPLES = {
    '为什么天空是蓝色的': {
        '准确简洁': '因为瑞利散射：阳光进入大气后会被空气分子散射，波长越短的光散射得越强。蓝光波长短，被散射到天空各个方向，所以我们看到的天空是蓝色的。',
        '内容错误': '因为天空反射了海洋的颜色。海洋是蓝色的，阳光照到海面再反射到天上，所以天空也变成了蓝色。',
        '重复啰嗦': '天空是蓝色的是因为散射。天空是蓝色的是因为散射。天空是蓝色的是因为散射。天空是蓝色的是因为散射。天空是蓝色的是因为散射。',
        '答非所问': '天气晴朗的时候很适合出门散步，记得带上水和防晒霜，也可以约上朋友一起去公园野餐。',
        '太短': '散射。',
    },
    '中国的首都是哪里': {
        '正确': '中国的首都是北京。',
        '错误': '中国的首都是上海。',
        '正确但啰嗦': '中国的首都是北京。北京是中国的首都。中国的首都就是北京，北京是首都。总之，中国的首都是北京。',
    },
}


def rep_penalty(text, n=3, cap=0.5):   # 和 train_grpo.py 完全相同
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def score(rm, question, answer):
    length = 0.5 if 20 <= len(answer.strip()) <= 800 else -0.5
    rep = -rep_penalty(answer)
    model = rm.get_score([{'role': 'user', 'content': question}], answer)
    return length, rep, model, length + rep + model


parser = argparse.ArgumentParser()
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--reward_model', default=str(ROOT.parent / 'internlm2-1_8b-reward'))
args = parser.parse_args()

print('正在加载奖励模型（约 15 秒，占 3.4 GB 显存）……')
rm = LMForRewardModel(args.reward_model, device='cuda', dtype=torch.float16)


def show(question, answers):
    print(f'\n问题：{question}')
    print(f'  {"回答":<10}{"长度规则":>8}{"重复惩罚":>8}{"奖励模型":>8}{"总分":>8}   回答内容')
    for name, ans in answers.items():
        l, r, m, t = score(rm, question, ans)
        print(f'  {name:<10}{l:>+10.2f}{r:>+10.2f}{m:>+10.2f}{t:>+9.2f}   {ans[:40]}{"……" if len(ans) > 40 else ""}')


if args.interactive:
    while True:
        q = input('\n问题（直接回车退出）：').strip()
        if not q:
            break
        answers, k = {}, 1
        while True:
            a = input(f'  回答 {k}（直接回车结束输入）：').strip()
            if not a:
                break
            answers[f'回答{k}'] = a
            k += 1
        if answers:
            show(q, answers)
else:
    for q, answers in EXAMPLES.items():
        show(q, answers)
