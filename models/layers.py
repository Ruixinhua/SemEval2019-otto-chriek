import torch
import torch.nn as nn


class AttLayer(nn.Module):

    def __init__(self, word_emb_dim, attention_hidden_dim):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(word_emb_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        attention_weight = torch.unsqueeze(self.attention(x), 2)
        y = torch.sum(x * attention_weight, dim=1)
        return y