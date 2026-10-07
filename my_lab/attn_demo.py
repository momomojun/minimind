"""查看某个 token 在每一层、每个头里把注意力分给了前面哪些 token。

用法（在仓库根目录运行）:
    python my_lab/attn_demo.py "小明养了一只猫，它很可爱" 6
第一个参数是句子，第二个参数是要观察的 token 序号（先随便填一个，看打印出的序号列表再改）。
默认读取 ./minimind-3（transformers 格式的官方权重），可用 --model 换成别的目录。
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('sentence')
parser.add_argument('query', type=int, help='要观察的 token 序号')
parser.add_argument('--model', default=str(ROOT / 'minimind-3'))
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
# eager 实现才会把注意力权重返回出来；默认的 Flash Attention 算完就丢掉了
model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation='eager').eval()

ids = tok(args.sentence)['input_ids']
pieces = [tok.decode([i]) for i in ids]
print(list(enumerate(pieces)))

with torch.no_grad():
    out = model(torch.tensor([ids]), output_attentions=True)

q = args.query
print('query token:', pieces[q])
for layer, attn in enumerate(out.attentions):   # 每层 attn 形状: (1, 头数, T, T)
    p = attn[0, :, q, :q + 1]                      # 第 q 个 token 那一行，只取它能看到的位置
    avg = p.mean(0)
    print(f'L{layer} avg:', ' '.join(f'{pieces[j]}:{avg[j]:.2f}' for j in range(q + 1)))
    print('   heads->', [(pieces[int(p[h].argmax())], round(float(p[h].max()), 2)) for h in range(p.shape[0])])
