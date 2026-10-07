import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')


import torch
import torch.nn as nn
from codebot.model import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, dropout_rate)
            for _ in range(n_layer)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        self.embed.weight = self.unembed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, C = ids.shape
        device = ids.device

        pos = torch.arange(0, C, dtype=torch.long, device=device)
        emb = self.embed(ids)
        pos_emb = self.pos_embed(pos)

        x = self.dropout(emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model

vocab_size = 1000
max_context_len = 256
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1

model = GPT(vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate)

dummy_input = torch.randint(0, vocab_size, (1, max_context_len))
logits = model(dummy_input)
print(f"出力形状: {logits.shape}")
    