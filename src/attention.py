import torch 
import torch.nn as nn
from torch.nn import functional as F
import math 

class SelfAttention(nn.Module):
    """
        Self attention layer: implement of multi-head self attention 
    """
    def __init__(self, num_heads: int, dim_embed: int, \
                 in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()

        self.in_proj = nn.Linear(dim_embed, 3 * dim_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.num_heads = num_heads 
        self.dim_heads = dim_embed // num_heads
        assert dim_embed % num_heads == 0, "dim_embed must be divisible by num_heads"

    def forward(self, x: torch.Tensor, casual_mask=False):
        input_shape = x.shape
        batch_size, seq_len, dim_embed = input_shape

        intermim_shape = (batch_size, seq_len, self.num_heads, self.dim_heads) 

        # (batch_size, seq_len, Dim) -> (batch_size, seq_len, Dim * 3) -> 3 tensors
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # (N, seq_len, Dim) -> (N, seq_len, n_heads, d_heads) -> (N, n_heads, seq_len, d_heads)
        q = q.view(intermim_shape).permute(0, 2, 1, 3)
        k = k.view(intermim_shape).permute(0, 2, 1, 3)
        v = v.view(intermim_shape).permute(0, 2, 1, 3)

        weights = q @ k.transpose(-2, -1)
        # weights = torch.matmul(q, k.transpose(-2, -1))

        if casual_mask:
            # Mask where the upper tranigle(above the principal diagonal) is made up of 1.
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, -torch.inf) # Fill the upper triangle with -inf
        
        weights  /= math.sqrt(self.dim_heads)
        weights = F.softmax(weights, dim=-1) 
        
        # (N, n_heads, seq_len, d_heads) @ (N, n_heads, seq_len, d_heads) 
        output = weights @ v 
        output = output.transpose(1, 2).reshape(input_shape) # a.reshape = a.view() + a.contiguous().view()
        # output = output.transpose(1, 2).contiguous().view(input_shape) # view只能处理连续张量，reshape都可以
        # output = output.concat()
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, num_heads, dim_embed, dim_cross, \
                 in_proj_bias=True, out_proj_bias=True):
          super().__init__()

          self.q_proj = nn.Linear(dim_embed, dim_embed, bias=in_proj_bias)
          self.k_proj = nn.Linear(dim_cross, dim_embed, bias=in_proj_bias)
          self.v_proj = nn.Linear(dim_cross, dim_embed, bias=in_proj_bias) 
          self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
          self.num_heads = num_heads
          self.dim_heads = dim_embed // num_heads

    def forward(self, x, y):
        """
            x (latent) : (batch_size, seq_len_Q, dim_Q) 
            y (context) : (batch_size, seq_len_KV, dim_KV) = (batch_size, 77, 768) 
        """

        input_shape = x.shape
        batch_size, seq_len, dim_embed = input_shape
        # divide each embedding of Q into multiple heads such that dim_heads * num_heads = dim_embed
        interm_shape = (batch_size, -1, self.num_heads, self.dim_heads)

        # (BS, seq_len, d_embed) -> (BS, seq_len, H, d_embed / H) -> (BS, H, seq_len, d_embed / H)
        q = self.q_proj(x) 
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).permute(0, 2, 1, 3) #  = .transpose(2, 1)
        k = k.view(interm_shape).permute(0, 2, 1, 3) 
        v = v.view(interm_shape).permute(0, 2, 1, 3)  

        # (BS, H, seq_len_Q, n_heads) @ (BS, H, n_heads, seq_len_KV) -> (BS, H, seq_len_Q, seq_len_KV)
        weights = q @ k.transpose(-1, -2)
        weights /= math.sqrt(self.dim_heads)
        weights = F.softmax(weights, dim=-1)

        # (BS, H, seq_len_Q, seq_len_KV) @ (BS, H, seq_len_KV, n_heads) -> (BS, H, seq_len_Q, n_heads)
        # -> (BS, seq_len_Q, H, n_heads) -> (BS, seq_len_Q, d_embed)
        output = weights @ v
        output = output.transpose(2, 1).contiguous().view(input_shape) 
        output = self.out_proj(output)
        # output : (batch_size, seq_len_Q, dim_embed_Q)
        return output
     




