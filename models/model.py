import torch
import torch.nn as nn
import numpy as np



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn




class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_inputs, d_inputs_emb, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_inputs = d_inputs
        self.d_inputs_emb = d_inputs_emb

        self.w_inputs_emb = nn.Linear(d_inputs, d_inputs_emb, bias=False)
        self.w_multiHead_emb = nn.Linear(d_inputs_emb, n_head * d_inputs_emb, bias=False)

        self.fc = nn.Linear(n_head * d_inputs_emb, d_inputs_emb, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_inputs_emb ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, inputs):

        n_head, d_inputs_emb = self.n_head, self.d_inputs_emb
        sz_b, len_obs = inputs.size(0), inputs.size(1)


        # embed inputs  [sz_b, len_obs, 6] ->  [sz_b, len_obs, d_emb]
        inputs_emb = self.w_inputs_emb(inputs)
        residual = inputs_emb[:,-1,:].unsqueeze(1)
        
        # Pass through the pre-attention projection: b x len_obs x (n*d_emb)
        # Separate different heads: b x len_obs x n x d_emb
        q = self.w_multiHead_emb(inputs_emb).view(sz_b, len_obs, n_head, d_inputs_emb)
        k = self.w_multiHead_emb(inputs_emb).view(sz_b, len_obs, n_head, d_inputs_emb)
        v = self.w_multiHead_emb(inputs_emb).view(sz_b, len_obs, n_head, d_inputs_emb)

        


        # Transpose for attention dot product: b x n x len_obs x d_emb
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Get the query b x n x len_obs[-1] x d_emb
        q = q[:,:,-1,:].unsqueeze(2)

        output, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x 1 x n x d_emb
        # Combine the last two dimensions to concatenate all the heads together: b x 1 x (n*d_emb)
        output = output.transpose(1, 2).contiguous().view(sz_b, 1, -1)
        output = self.fc(output)
        # output = self.dropout(self.fc(output))
        output += residual

        # output = self.layer_norm(output)

        return output, attn
