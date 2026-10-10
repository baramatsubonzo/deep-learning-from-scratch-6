import time

start = time.perf_counter()

import torch

large = torch.tensor(1000.0, dtype=torch.float16)
small = torch.tensor(0.01, dtype=torch.float16)

print(large + small)

tiny = torch.tensor(1e-8, dtype=torch.float16)
print(tiny)

huge = torch.tensor(70000.0, dtype=torch.float16)
print(huge)

tiny_fp16 = torch.tensor(1e-8, dtype=torch.float16)
print(tiny_fp16)

tiny_bf16 = torch.tensor(1e-8, dtype=torch.bfloat16)
print(tiny_bf16)

huge_fp16 = torch.tensor(70000.0, dtype=torch.float16)
print(huge_fp16)

huge_bf16 = torch.tensor(70000.0, dtype=torch.bfloat16)
print(huge_bf16)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# cpuの場合の実行時間結果 -> 実行時間: 0.878秒

device = 'cuda' if torch.cuda.is_available() else 'mps'
# mpsの場合の実行時間結果 -> 実行時間: 0.711秒

a = torch.randn(1000, 1000, device=device)

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    b = a @ a
    c = a.sum()
    print(b.dtype)
    print(c.dtype)

if device == 'mps':
    torch.mps.synchronize()
elif device == 'cuda':
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print(f"実行時間: {elapsed:.3f}秒")