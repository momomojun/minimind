"""衡量一个模型"更喜欢 chosen 还是 rejected"：用保存好的权重，在 DPO 数据上算偏好准确率和 DPO loss。

参考模型固定是 full_sft（DPO 训练前的自己）。对每条数据：
  chosen 的进步   = log π(chosen)   − log π_ref(chosen)
  rejected 的进步 = log π(rejected) − log π_ref(rejected)
  差距           = chosen 的进步 − rejected 的进步
  差距 > 0 就算"偏好正确"

用法（在仓库根目录运行，需要 GPU）:
    python my_lab/dpo_eval.py --weight dpo
    python my_lab/dpo_eval.py --weight dpo --n 1000
注意：dpo.jsonl 也是训练数据，所以这里测的是"训练集上学到了多少"，不是泛化能力。
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import DPODataset

parser = argparse.ArgumentParser()
parser.add_argument('--weight', default='dpo', help='要评估的权重前缀，例如 dpo')
parser.add_argument('--ref', default='full_sft', help='参考模型的权重前缀')
parser.add_argument('--n', type=int, default=500, help='随机抽多少条来评估')
parser.add_argument('--beta', type=float, default=0.15)
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


def load(name):
    m = MiniMindForCausalLM(MiniMindConfig())
    m.load_state_dict(torch.load(ROOT / 'out' / f'{name}_768.pth'), strict=True)
    return m.float().eval().cuda()   # 用 fp32 计算，避免很小的差别被 fp16 的舍入误差淹没


def seq_logprob(model, x, y, mask):
    """整个回答的 log 概率：每个 token 的 log 概率，只把回答部分（mask=1）加起来"""
    logp = F.log_softmax(model(x[None].cuda()).logits.float(), dim=-1)[0]
    token_logp = logp.gather(-1, y.cuda()[:, None]).squeeze(-1)
    return (token_logp * mask.cuda()).sum().item()


random.seed(args.seed)   # DPODataset 取数据时会随机去掉空的 <think>，固定种子让结果可复现
tok = AutoTokenizer.from_pretrained(ROOT / 'model')
ds = DPODataset(str(ROOT / 'dataset' / 'dpo.jsonl'), tok, max_length=args.max_seq_len)
idx = random.sample(range(len(ds)), args.n)
policy, ref = load(args.weight), load(args.ref)

correct, losses, gain_c, gain_r = 0, [], [], []
with torch.no_grad():
    for k, i in enumerate(idx, 1):
        b = ds[i]
        dc = seq_logprob(policy, b['x_chosen'], b['y_chosen'], b['mask_chosen']) - \
             seq_logprob(ref, b['x_chosen'], b['y_chosen'], b['mask_chosen'])
        dr = seq_logprob(policy, b['x_rejected'], b['y_rejected'], b['mask_rejected']) - \
             seq_logprob(ref, b['x_rejected'], b['y_rejected'], b['mask_rejected'])
        margin = dc - dr
        correct += margin > 0
        losses.append(-F.logsigmoid(torch.tensor(args.beta * margin)).item())
        gain_c.append(dc)
        gain_r.append(dr)
        if k % 100 == 0:
            print(f'  已评估 {k}/{args.n}')

n = args.n
print(f'\n评估模型: {args.weight}    参考模型: {args.ref}    样本数: {n}')
print(f'chosen 的平均进步   : {sum(gain_c) / n:+.4f}')
print(f'rejected 的平均进步 : {sum(gain_r) / n:+.4f}')
print(f'平均差距            : {(sum(gain_c) - sum(gain_r)) / n:+.4f}')
print(f'偏好准确率          : {correct / n:.1%}   （差距 > 0 的比例）')
print(f'平均 DPO loss       : {sum(losses) / n:.4f}   （0.6931 = 完全没有区别）')
