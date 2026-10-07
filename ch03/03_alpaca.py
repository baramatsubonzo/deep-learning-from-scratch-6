import json
from codebot.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

with open('codebot/tiny_codes_sft.json') as f:
    data = json.load(f)

item = data[0]
print(item)

text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)

ids = tokenizer.encode(text)
print(ids)
