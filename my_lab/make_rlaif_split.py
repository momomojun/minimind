"""从 rlaif.jsonl 里随机抽出互不重叠的训练题和评估题。

用法（在仓库根目录运行）:
    python my_lab/make_rlaif_split.py --n_train 1500 --n_eval 100
输出:
    dataset/rlaif_train<n_train>.jsonl   GRPO 训练用
    dataset/rlaif_eval<n_eval>.jsonl     只用来评估，训练时不会见到
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('--n_train', type=int, default=1500)
parser.add_argument('--n_eval', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

lines = (ROOT / 'dataset' / 'rlaif.jsonl').read_text(encoding='utf-8').splitlines()
random.seed(args.seed)
picked = random.sample(range(len(lines)), args.n_train + args.n_eval)
eval_idx, train_idx = picked[:args.n_eval], picked[args.n_eval:]
assert not set(eval_idx) & set(train_idx)

for name, idx in [(f'rlaif_train{args.n_train}', train_idx), (f'rlaif_eval{args.n_eval}', eval_idx)]:
    out = ROOT / 'dataset' / f'{name}.jsonl'
    out.write_text('\n'.join(lines[i] for i in idx) + '\n', encoding='utf-8')
    print(f'{out.name}: {len(idx)} 道题')
print(f'（从 {len(lines)} 道题里抽取，两份互不重叠）')
