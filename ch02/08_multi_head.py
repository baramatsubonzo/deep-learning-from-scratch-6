import torch
import torch.nn as nn
import torch.nn.functional as F

# B = 2
# C = 4
# E = 16
# H = 3
# D = 8

# x = torch.randn(B, C, E)

# W_q = nn.Linear(E, H*D, bias=False)
# W_k = nn.Linear(E, H*D, bias=False)
# W_v = nn.Linear(E, H*D, bias=False)

# Q = W_q(x)
# K = W_k(x)
# V = W_v(x)

# Q = Q.view(B, C, H, D).transpose(1, 2)
# K = K.view(B, C, H, D).transpose(1, 2)
# V = V.view(B, C, H, D).transpose(1, 2)

# scores = torch.matmul(Q, K.transpose(-2, -1))
# scores = scores / (D ** 0.5)

# mask = torch.tril(torch.ones(C, C, device=scores.device))
# scores = scores.masked_fill(mask == 0, float('-inf'))

# weights = F.softmax(scores, dim=-1)
# hidden = torch.matomul(weights, V)

# hidden = hidden.transpose(1,2)
# hidden = hidden.contiguous().view(B, C, H*D)

# W_o = nn.Linear(H*D, E, bias=False)
# output = W_o(hidden)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate = 0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D ** 0.5)

        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim = -1 )
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V)
        hidden = hidden.transpose(1,2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        output = self.output_dropout(output)

        return output

embed_dim = 512
n_head = 8
head_dim = 64

mha = MultiHeadAttention(embed_dim, n_head, head_dim)

batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

output = mha(x)
print(f"入力形式: {x.shape}")
print(f"出力形式: {output.shape}")