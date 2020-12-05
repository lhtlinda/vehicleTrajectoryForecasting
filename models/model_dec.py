import torch
import torch.nn as nn
import numpy as np
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, qmask):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        # mask attn
        if qmask != None:
            mask = (1. - qmask) * (-1e30)
            attn = torch.add(attn, mask)
            
            
        attn = self.dropout(nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class FeedForward(nn.Module):
    ''' FeedForward module '''

    def __init__(self, n_head, d_inputs_emb):
        super().__init__()
        self.fc = nn.Linear(d_inputs_emb, d_inputs_emb, bias=False)
        self.layer_norm = nn.LayerNorm(d_inputs_emb, eps=1e-6)


    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output += residual
        output = self.layer_norm(output)
    
        return output



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
        self.layer_norm = nn.LayerNorm(d_inputs_emb, eps=1e-6)


    def forward(self, Q, K, V, QMask = None):

        n_head, d_inputs_emb = self.n_head, self.d_inputs_emb
        sz_b, len_obs = K.size(0), K.size(1)
        q_len = Q.shape[1]

        # embed inputs  [sz_b, len_obs, 6] ->  [sz_b, len_obs, d_emb]
        Q_emb = self.w_inputs_emb(Q)
        K_emb = self.w_inputs_emb(K)
        V_emb = self.w_inputs_emb(V)
        residual = Q_emb
        
        # Pass through the pre-attention projection: b x len_obs x (n*d_emb)
        # Separate different heads: b x len_obs x n x d_emb
        q = self.w_multiHead_emb(Q_emb).view(sz_b, q_len, n_head, d_inputs_emb)
        k = self.w_multiHead_emb(K_emb).view(sz_b, len_obs, n_head, d_inputs_emb)
        v = self.w_multiHead_emb(V_emb).view(sz_b, len_obs, n_head, d_inputs_emb)

    
        # Transpose for attention dot product: b x n x len_obs x d_emb
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
     
        output, attn = self.attention(q, k, v, QMask)

       
        # Combine the last two dimensions to concatenate all the heads together: b x len_obs x (n*d_emb)
        output = output.view(sz_b, q_len, -1)
    
        output = self.fc(output)
        output += residual
        output = self.layer_norm(output)
        
        return output



class transformer(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_inputs, d_inputs_emb, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_inputs = d_inputs
        self.d_inputs_emb = d_inputs_emb

        #transformer pipeline
        self.encoder_attn = MultiHeadAttention(n_head, d_inputs, d_inputs_emb)
        self.encoder_ffn = FeedForward(n_head, d_inputs_emb)
        
        self.decoder_attn1 = MultiHeadAttention(n_head, 2, d_inputs_emb)
        self.decoder_attn2 = MultiHeadAttention(n_head, d_inputs_emb, d_inputs_emb)
        self.decoder_ffn = FeedForward(n_head, d_inputs_emb)
        
        self.fc = nn.Sequential(
            nn.Linear(d_inputs_emb, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 2, bias=False),
        )



    def forward(self, inputs, outputs):
        batch_size = inputs.shape[0]
        output_len = outputs.shape[1]
        output_dim = outputs.shape[2]
        # output shifted right by 1
        outputs = torch.cat([torch.zeros((batch_size, 1, output_dim)), outputs[:,:output_len-1,:]],1)
        
        # encoder
        encoded_attn = self.encoder_attn(inputs, inputs, inputs)
        encoded_ffn = self.encoder_ffn(encoded_attn)
        # K, V = encoder output
        encoded_K = encoded_ffn
        encoded_V = encoded_ffn
        
        # mask shifted output
        output_mask = np.triu(np.ones((outputs.shape[1],outputs.shape[1])), k=1).astype('uint8')
        output_mask = torch.from_numpy(output_mask)
        
        #decoder
        encoded_Q = self.decoder_attn1(outputs, outputs, outputs, output_mask )
        decoded = self.decoder_attn2(encoded_Q, encoded_K, encoded_V)
        
        #output pred
        output = self.fc(decoded)

        
        return output
