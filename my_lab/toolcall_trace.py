"""一步一步展示工具调用的全过程：模型只负责"写字"，真正执行工具的是 Python 程序。

用法（在仓库根目录运行）:
    python my_lab/toolcall_trace.py
    python my_lab/toolcall_trace.py --prompt "北京今天天气怎么样？" --tools get_current_weather,get_current_time
每一步结束会暂停，按回车继续。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from eval_toolcall import MOCK_RESULTS, get_tools

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default='帮我算一下 256 乘以 37 等于多少')
parser.add_argument('--tools', default='calculate_math,get_current_time', help='逗号分隔的工具名')
parser.add_argument('--weight', default='full_sft')
parser.add_argument('--max_new_tokens', type=int, default=120)
args = parser.parse_args()


def pause(title):
    input(f'\n------ 按回车进入：{title} ------')
    print()


tok = AutoTokenizer.from_pretrained(ROOT / 'model')
model = MiniMindForCausalLM(MiniMindConfig())
model.load_state_dict(torch.load(ROOT / 'out' / f'{args.weight}_768.pth'), strict=True)
model = model.half().eval().cuda()
tools = get_tools(args.tools.split(','))


def generate_greedy(text, show_tokens=12):
    """每次只预测一个 token（选概率最大的），打印前几个 token 的预测过程"""
    ids = tok(text, return_tensors='pt').input_ids.cuda()
    out_ids = []
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            probs = torch.softmax(model(ids).logits[0, -1].float(), dim=-1)
            p, nxt = probs.max(dim=-1)
            if step < show_tokens:
                print(f'  第 {step + 1:2d} 个 token: {tok.decode([nxt.item()])!r:22s} 概率 {p.item():.2f}')
            if nxt.item() == tok.eos_token_id:
                print(f'  …… 第 {step + 1} 个 token 是 <|im_end|>，模型认为自己说完了，停止生成')
                break
            out_ids.append(nxt.item())
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return tok.decode(out_ids)


messages = [{'role': 'user', 'content': args.prompt}]
print('【第 1 步】Python 程序准备好两样东西：')
print('  用户的问题:', args.prompt)
print('  可用的工具:', [t['function']['name'] for t in tools])

pause('第 2 步：程序把工具说明和问题拼成一段文字，交给模型')
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
print(text)
print(f'（共 {len(tok(text).input_ids)} 个 token。模型能看到的只有这段文字）')

pause('第 3 步：模型一个 token 一个 token 地往下写')
reply = generate_greedy(text)
print('\n模型写出的完整内容：')
print(reply)

pause('第 4 步：Python 程序用正则表达式在这段文字里找 <tool_call>')
calls = []
for m in re.findall(r'<tool_call>(.*?)</tool_call>', reply, re.DOTALL):
    try:
        calls.append(json.loads(m.strip()))
    except json.JSONDecodeError:
        print('  这一段不是合法的 JSON，跳过:', m.strip())
if not calls:
    print('  没找到 <tool_call>，说明模型选择直接回答，流程结束。')
    sys.exit()
for c in calls:
    print(f'  找到一个请求 → 工具名: {c.get("name")}   参数: {c.get("arguments")}')

pause('第 5 步：Python 程序真正去执行工具（模型完全不参与）')
messages.append({'role': 'assistant', 'content': reply})
for c in calls:
    fn = MOCK_RESULTS.get(c.get('name'))
    result = fn(c.get('arguments', {})) if fn else {'error': f'未知工具: {c.get("name")}'}
    print(f'  执行 {c.get("name")}({c.get("arguments")}) → {result}')
    messages.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)})

pause('第 6 步：程序把执行结果接在对话后面，再交给模型一次')
text2 = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
print('（前面和第 2 步一样，省略。新增的部分是：）')
print(text2[len(text):])

pause('第 7 步：模型看到工具结果，写出最终回答')
final = generate_greedy(text2, show_tokens=6)
print('\n最终回答：', final.split('</think>')[-1].strip())
