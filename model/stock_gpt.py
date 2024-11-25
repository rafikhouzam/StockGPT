import torch
import torch.nn as nn

class StockGPT(nn.Module):
    def __init__(self, vocab_size=402, embed_size=256, num_heads=8, num_layers=6, block_size=20, dropout=0.4):
        super(StockGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.positional_embedding(positions)

        for transformer in self.transformer_blocks:
            x = transformer(x)
            x = self.layer_norm(x)  # Apply LayerNorm after each transformer block

        logits = self.fc(x)
        return logits
