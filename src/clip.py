import torch
import torch.nn as nn
import torch.nn.functional as F 
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        '''vocabulary size: 49408, embedding dimension: 768, number of tokens: 77'''
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.ModuleList([ 
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)  
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long) 

        # (batch_size, seq_len) -> (batch_size, seq_len, dim) 
        state = self.embedding(tokens)
        # Apply encoder layers similar to the Transformer's encoder
        for layer in self.layers:
            state = layer(state) 
        output = self.layernorm(state) 

        # (batch_size, seq_len, dim)
        return output
    
class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, num_dim: int, num_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, num_dim)
        # A learnable weight matrix for positional information of each token
        self.position_embedding = nn.Parameter(torch.zeros((num_token, num_dim)))
    
    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim) 
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, num_embed: int):
        super().__init__()

        # Pre-attention layer
        self.layernorm_1 = nn.LayerNorm(num_embed) 
        # Self attention  
        self.attention = SelfAttention(num_heads, num_embed)
        # Pre-FNN layer
        self.layernorm_2 = nn.LayerNorm(num_embed) 
        # Feedforward layer
        self.linear_1 = nn.Linear(num_embed, 4 * num_embed)
        self.linear_2 = nn.Linear(4 * num_embed, num_embed) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # (batch_size, seq_len, dim) 
        residue = x

        ### SELF ATTENTION ### 
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        ### FEEDFORWARD NETWORK ###
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        x = self.linear_2(x)
        x += residue
        # (batch_size, seq_len, dim)
        return x
    




