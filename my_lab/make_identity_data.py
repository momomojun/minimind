"""生成自定义身份的 LoRA 数据，并可选地混入普通 SFT 对话，防止灾难性遗忘。

用法（在仓库根目录运行）:
    python my_lab/make_identity_data.py --name MomoMind --dev momomojun --n_general 300
输出:
    dataset/lora_identity_<name>.jsonl      只有身份对话（91 条）
    dataset/lora_identity_<name>_mix.jsonl  身份对话 + n_general 条普通对话，打乱顺序
依赖 dataset/lora_identity.jsonl 和 dataset/sft_t2t_mini.jsonl（从 HuggingFace 下载）。
"""
import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.lm_dataset import pre_processing_chat, post_processing_chat

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='新的助手名字，替换 MiniMind')
parser.add_argument('--dev', required=True, help='新的开发者名字，替换 Jingyao Gong')
parser.add_argument('--n_general', type=int, default=300, help='混入多少条普通对话，0 表示不混')
parser.add_argument('--max_seq_len', type=int, default=340, help='和 train_lora.py 保持一致，超长的普通对话不选')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

data_dir = ROOT / 'dataset'
identity = []
for line in open(data_dir / 'lora_identity.jsonl', encoding='utf-8'):
    sample = json.loads(line)
    for msg in sample['conversations']:
        msg['content'] = msg['content'].replace('MiniMind', args.name).replace('Jingyao Gong', args.dev)
    identity.append(json.dumps(sample, ensure_ascii=False))

out_identity = data_dir / f'lora_identity_{args.name.lower()}.jsonl'
out_identity.write_text('\n'.join(identity) + '\n', encoding='utf-8')
print(f'{out_identity.name}: {len(identity)} 条身份对话')

if args.n_general > 0:
    tok = AutoTokenizer.from_pretrained(ROOT / 'model')
    random.seed(args.seed)
    lines = open(data_dir / 'sft_t2t_mini.jsonl', encoding='utf-8').readlines()
    random.shuffle(lines)
    old_names = ['jingyao', 'Jingyao', 'minimind', 'MiniMind', '龚']
    general, skipped = [], {'旧名字': 0, '工具调用': 0, '超长': 0}
    for line in lines:
        if len(general) >= args.n_general:
            break
        if any(n in line for n in old_names):          # 避免和新身份互相矛盾
            skipped['旧名字'] += 1
            continue
        conv = json.loads(line)['conversations']
        if any(m.get('tools') or m.get('tool_calls') or m['role'] not in ('user', 'assistant') for m in conv):
            skipped['工具调用'] += 1
            continue
        prompt = post_processing_chat(tok.apply_chat_template(pre_processing_chat(conv), tokenize=False))
        if len(tok(prompt).input_ids) > args.max_seq_len:   # 超长会被截断，丢掉结尾的 <|im_end|>
            skipped['超长'] += 1
            continue
        general.append(line.rstrip('\n'))
    mixed = identity + general
    random.shuffle(mixed)
    out_mix = data_dir / f'lora_identity_{args.name.lower()}_mix.jsonl'
    out_mix.write_text('\n'.join(mixed) + '\n', encoding='utf-8')
    print(f'{out_mix.name}: {len(identity)} 身份 + {len(general)} 普通 = {len(mixed)} 条，跳过 {skipped}')
