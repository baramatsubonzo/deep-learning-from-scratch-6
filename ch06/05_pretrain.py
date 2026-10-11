import os
import sys
from importlib import import_module

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from tqdm import tqdm

from codebot.utils import get_device
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer

# 数字で始まるモジュール名は通常のfrom文で指定できないため。
get_lr = import_module('ch06.03_lr_scheduler').get_lr


train_data = np.memmap('storybot/tiny_stories_train.bin', dtype=np.uint16, mode='r')

def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)
        ix = ix[ix + context_len + 1 < len(data)]
        if len(ix) == 0:
            return None, None

    x = torch.stack(
        [torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])

    y = torch.stack(
        [torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)

def evaluate(model, val_data, context_len, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_start = len(val_data) - context_len -1
    num_batches = (max_start // context_len) // batch_size + 1

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Validation"):
            offset = batch_idx * batch_size * context_len
            x, y = get_batch(val_data, context_len, batch_size, device, random=False, offset=offset)
            if x is None:
                break

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += y.numel()

        model.train()
        return total_loss / total_tokens

device = get_device()
data_path = 'storybot/tiny_stories_train.bin'
val_data_path = 'storybot/tiny_stories_valid.bin'
tokenizer_path = 'storybot/merge_rules.pkl'
model_save_path = 'storybot/model_pretrain.pt'

context_len = 256
vocab_size = 10000
batch_size = 32
learning_rate = 0.001
warmup_iters = 200
max_iters = 40000
embed_dim = 512
n_head = 16
n_layer = 4
ff_dim = 1344
theta = 10000
eval_iters = 500
grad_clip = 1.0
save_iters = [500, 5000]

train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT(
    vocab_size, context_len, embed_dim, n_head, n_layer, ff_dim, theta
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(range(max_iters))

val_loss = float('inf')
val_losses = []
val_iters = []

for i in pbar:
    lr = get_lr(i, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_x, batch_y = get_batch(train_data, context_len, batch_size, device)

    optimizer.zero_grad()

    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if i in save_iters:
        save_path = f'storybot/model_iter_{i}.pt'
        model.save(save_path)
        print(f"\nModel saved at iteration {i}: {save_path}")

    if (i % eval_iters) == 0 or i == max_iters -1:
        val_loss = evaluate(model, val_data, context_len, batch_size, device)
        val_losses.append(val_loss)
        val_iters.append(i)
    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val_loss': f'{val_loss:.6f}'})

plt.figure(figsize=(10, 6))
plt.plot(val_iters, val_losses)
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.savefig('loss_val.png')

model.save(model_save_path)
