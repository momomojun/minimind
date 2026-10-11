"""随机打印几条 DPO 数据，看看 chosen 和 rejected 有什么区别。

用法（在仓库根目录运行）:
    python my_lab/show_dpo.py           # 默认看 3 条
    python my_lab/show_dpo.py --n 5 --seed 1
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_chars', type=int, default=300, help='每段回答最多显示多少字')
args = parser.parse_args()

lines = open(ROOT / 'dataset' / 'dpo.jsonl', encoding='utf-8').readlines()
print(f'dpo.jsonl 一共 {len(lines)} 条数据\n')
random.seed(args.seed)
for k, line in enumerate(random.sample(lines, args.n), 1):
    sample = json.loads(line)
    print('=' * 70)
    print(f'第 {k} 条，包含的字段: {list(sample.keys())}')
    for key in ('chosen', 'rejected'):
        print(f'\n---------- {key} ----------')
        for msg in sample[key]:
            text = str(msg.get('content', '')).strip()
            if len(text) > args.max_chars:
                text = text[:args.max_chars] + f' ……（共 {len(text)} 字）'
            print(f'[{msg["role"]}] {text}')
    print()
