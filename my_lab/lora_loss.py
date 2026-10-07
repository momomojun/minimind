"""比较加 LoRA 前后，模型在某份对话数据上的平均 loss。用来判断 LoRA 是欠拟合还是已经学会。

用法（在仓库根目录运行，需要 GPU）:
    python my_lab/lora_loss.py out/lora_identity_mix_768.pth --data dataset/lora_identity_momo.jsonl
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from dataset.lm_dataset import SFTDataset

parser = argparse.ArgumentParser()
parser.add_argument('lora', help='LoRA 权重路径')
parser.add_argument('--data', default=str(ROOT / 'dataset' / 'lora_identity_momo.jsonl'))
parser.add_argument('--base', default=str(ROOT / 'out' / 'full_sft_768.pth'))
parser.add_argument('--max_seq_len', type=int, default=340)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(ROOT / 'model')
ds = SFTDataset(args.data, tok, max_length=args.max_seq_len)


def avg_loss(model):
    total = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            total += model(x[None].cuda(), labels=y[None].cuda()).loss.item()
    return total / len(ds)


model = MiniMindForCausalLM(MiniMindConfig()).cuda().eval()
model.load_state_dict(torch.load(args.base), strict=True)
print(f'base only    loss: {avg_loss(model):.3f}')
apply_lora(model)
load_lora(model, args.lora)
print(f'base + LoRA  loss: {avg_loss(model):.3f}')
