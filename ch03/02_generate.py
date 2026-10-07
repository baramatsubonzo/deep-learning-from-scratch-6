import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import torch
import torch.nn.functional as F

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()

    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

device = get_device()
model_path = 'codebot/model_pretrain.pt'
tokenizer_path = 'codebot/merge_rules.pkl'
prompt = "def"
max_new_tokens = 200
temperature = 1.0

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

for i in range(5):
    print(f"---サンプル {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(generated_text)
    print()
