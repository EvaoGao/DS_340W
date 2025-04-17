# all imports here
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.nn import GCNConv
from torch.autograd import Variable

from datetime import datetime
from tqdm import tqdm
import sklearn
from copy import deepcopy


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional = False):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.seq_length = SEQ_LENGTHs
        self.bidrectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional = bidirectional)
        
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        #h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(ula)
        
        return out

import torch.nn as nnu
import math

device = 'cpu'

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights

class MultiHeadAttentionCosformerNew(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttentionCosformerNew, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q).permute(0,2,1,3)  # (B, S, H, D)
        k = self.split_heads(k).permute(0,2,1,3)  # (B, S, H, D)
        v = self.split_heads(v).permute(0,2,1,3)  # (B, S, H, D)
        B = q.shape[0]
        S = q.shape[1]

        q = torch.nn.functional.elu(q) + 1 # Sigmoid torch.nn.ReLU()
        k = torch.nn.functional.elu(k) + 1 # Sigmoid torch.nn.ReLU()

        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos = (torch.cos(1.57*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).to(device)
        sin = (torch.sin(1.57*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).to(device)
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q_cos, k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + 1e-5)
        # denominator -> [batch_size, seq_len, n_heads]

        O = torch.einsum('bsnz,bsn->bsnz', qkv_cos +
                              qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        concat_attention = self.concat_heads(O.permute(0,2,1,3)) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, None

class MultiHeadAttentionCosSquareformerNew(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttentionCosSquareformerNew, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q).permute(0,2,1,3)  # (B, S, H, D)
        k = self.split_heads(k).permute(0,2,1,3)  # (B, S, H, D)
        v = self.split_heads(v).permute(0,2,1,3)  # (B, S, H, D)
        B = q.shape[0]
        S = q.shape[1]

        q = torch.nn.functional.elu(q) + 1 # Sigmoid torch.nn.ReLU()
        k = torch.nn.functional.elu(k) + 1 # Sigmoid torch.nn.ReLU()

        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos = (torch.cos(3.1415*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).to(device)
        sin = (torch.sin(3.1415*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).to(device)
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        kv = torch.einsum('bsnx,bsnz->bnxz', k, v)
        # kv -> [batch_size, n_heads, d_head, d_head]
        qkv = torch.einsum('bsnx,bnxz->bsnz', q, kv)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q, k.sum(axis=1)) + torch.einsum('bsnd,bnd->bsn', q_cos, k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + 1e-5)
        # denominator -> [batch_size, seq_len, n_heads]

        O = torch.einsum('bsnz,bsn->bsnz', qkv + qkv_cos +
                              qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        concat_attention = self.concat_heads(O.permute(0,2,1,3)) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, None


class MultiHeadAttentionCosSquareformer(torch.nn.Module):
    """
    Multi-head self-attention with cos^2-based re-weighting.
    If causal_mask is provided, it should be shape (S, S) with 
    1 = positions not allowed to attend (above diagonal),
    0 = allowed positions. We'll fill those with -inf in the attention logits.
    """
    def __init__(self, D, H):
        super().__init__()
        self.D = D  # hidden dimension
        self.H = H  # number of heads

        # Learnable linear transforms for queries, keys, values
        self.wq = torch.nn.Linear(D, D*H)
        self.wk = torch.nn.Linear(D, D*H)
        self.wv = torch.nn.Linear(D, D*H)

        # Project the multi-head output back to D
        self.dense = torch.nn.Linear(D*H, D)

        # Example final projection to scalar
        self.out_proj = torch.nn.Linear(D, 1)

    def split_heads(self, x):
        """
        (B, S, D*H) -> (B, H, S, D)
        """
        B, S, D_H = x.shape
        assert D_H == self.D*self.H
        x = x.view(B, S, self.H, self.D)
        x = x.permute(0, 2, 1, 3)  # (B,H,S,D)
        return x

    def concat_heads(self, x):
        """
        (B, H, S, D) -> (B, S, D*H)
        """
        B, H, S, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B,S,H,D)
        x = x.view(B, S, H*D)                   # (B,S,D*H)
        return x

    def forward(self, x, causal_mask=None):
        """
        x: (B, S, D)
        causal_mask: (S, S), with 1's where positions should not attend
                     (i.e. above the diagonal for look-ahead).
                     None if no masking is required.
        return: (B, S, 1)
        """
        device = x.device
        B, S, D = x.shape
        assert D == self.D

        # Project to Q,K,V
        q = self.wq(x)  # (B,S,D*H)
        k = self.wk(x)  # (B,S,D*H)
        v = self.wv(x)  # (B,S,D*H)

        #Split into heads
        q = self.split_heads(q)  # (B,H,S,D)
        k = self.split_heads(k)
        v = self.split_heads(v)

        #Scale by sqrt(d_head)
        d_head = float(self.D)
        q = q * (d_head ** -0.5)
        k = k * (d_head ** -0.5)

        # Nonlinearities if desired (like elu+1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        #  Build cos^2, sin^2 factors over time
        cos_vals = torch.cos(math.pi * torch.arange(S, device=device) / S) ** 2
        sin_vals = torch.sin(math.pi * torch.arange(S, device=device) / S) ** 2

        #shape (S,) -> (B,S) by expand
        cos_vals = cos_vals.unsqueeze(0).expand(B, S)
        sin_vals = sin_vals.unsqueeze(0).expand(B, S)

        # Weighted q, k by cos^2, sin^2
        q_cos = torch.einsum("bhjd,bs->bhjd", q, cos_vals)
        q_sin = torch.einsum("bhjd,bs->bhjd", q, sin_vals)
        k_cos = torch.einsum("bhjd,bs->bhjd", k, cos_vals)
        k_sin = torch.einsum("bhjd,bs->bhjd", k, sin_vals)

        eps = 1e-6

        # [B,H,S,D] -> [B,H,S,1,D]
        q_5d     = q.unsqueeze(3)      # (B,H,S,1,D)
        k_5d     = k.unsqueeze(2)      # (B,H,1,S,D)
        attn1    = (q_5d * k_5d).sum(dim=-1)  # (B,H,S,S)

        q_cos_5d = q_cos.unsqueeze(3)
        k_cos_5d = k_cos.unsqueeze(2)
        attn2    = (q_cos_5d * k_cos_5d).sum(dim=-1) # (B,H,S,S)

        q_sin_5d = q_sin.unsqueeze(3)
        k_sin_5d = k_sin.unsqueeze(2)
        attn3    = (q_sin_5d * k_sin_5d).sum(dim=-1)

        # attention_scores shape: (B,H,S,S)
        attention_scores = attn1 + attn2 + attn3

        # Optionally mask out future positions:
        if causal_mask is not None:
            # causal_mask shape: (S,S) with 1= block, 0= keep
            # broadcast to (B,H,S,S)
            attention_scores = attention_scores.masked_fill(causal_mask.bool(), float('-inf'))

        #Convert to valid probabilities
        attn_weights = F.softmax(attention_scores, dim=-1)  # shape (B,H,S,S)

        # Multiply attn_weights by V for each position
        # V shape: (B,H,S,D). We'll do something like:
        # (B,H,S,S) x (B,H,S,D) => need to broadcast the second to (B,H,S,S,D)?
        # We'll do a standard batch matmul approach:
        # We can do:
        # out = torch.einsum('bhss,bhsd->bhsd', attn_weights, v)
        out = torch.einsum('bhss,bhsd->bhsd', attn_weights, v)

        # Combine heads => (B, S, D)
        out = self.concat_heads(out)

        # Final linear to dimension D, then a single scalar
        out = self.dense(out)       # (B, S, D)
        out = self.out_proj(out)    # (B, S, 1)

        return out, attn_weights



class MultiHeadAttentionCosSquareformerWithProj(nn.Module):
    '''Multi-head self-attention module with output projection layer'''
    def __init__(self, D, H, out_features=1):
        super(MultiHeadAttentionCosSquareformerWithProj, self).__init__()
        self.attention = MultiHeadAttentionCosSquareformerNew(D, H)
        self.proj = nn.Linear(D, out_features)
    
    def forward(self, x, mask):
        out, _ = self.attention(x, mask)
        out = self.proj(out)
        return out, None

# Positional encodings
def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates


def positional_encoding(D, position=20, dim=3, device=device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return torch.tensor(pos_encoding, device=device)

class TransformerLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate, attention_type='cosine_square'):
        super(TransformerLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mlp_hidden = nn.Linear(D, hidden_mlp_dim)
        self.mlp_out = nn.Linear(hidden_mlp_dim, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        if attention_type == 'cosine':
          self.mha = MultiHeadAttentionCosformerNew(D, H)
        elif attention_type == 'cosine_square':
          self.mha = MultiHeadAttentionCosSquareformerNew(D, H)
        else:
          self.mha = MultiHeadAttention(D,H)

    def forward(self, x, look_ahead_mask):
        
        attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn) # (B,S,D)
        attn = self.layernorm1(attn + x) # (B,S,D)

        mlp_act = torch.relu(self.mlp_hidden(attn))
        mlp_act = self.mlp_out(mlp_act)
        mlp_act = self.dropout2(mlp_act)
        
        output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights
  
class Transformer(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, attention_type='cosine_square', SL=20):
        super(Transformer, self).__init__()
        self.attention_type = attention_type
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D) # multivariate input
        self.output_projection = nn.Linear(D, out_features) # multivariate output
        self.pos_encoding = positional_encoding(D, position=SL)
        self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim, 
                                        dropout_rate=dropout_rate, attention_type=self.attention_type
                                       ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        B, S, D = x.shape
        # attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D
        
        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, _ = self.dec_layers[i](x=x,
                                          look_ahead_mask=mask)
            # attention_weights['decoder_layer{}'.format(i + 1)] = block
        
        x = self.output_projection(x)
        
        return x, None # attention_weights # (B,S,S)

class TransLSTM(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, LSTM_module, attention_type='regular'):
        super(TransLSTM, self).__init__()
        self.attention_type = attention_type
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D) # multivariate input
        self.output_projection = nn.Linear(D, 4) # multivariate output
        self.fc = nn.Linear(4*2, out_features)
        self.pos_encoding = positional_encoding(D)
        self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim, 
                                        dropout_rate=dropout_rate, attention_type=self.attention_type
                                       ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.LSTM = LSTM_module

    def forward(self, x, mask):
        x_l = self.LSTM(x)
        B, S, D = x.shape
        attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D
        
        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x=x,
                                          look_ahead_mask=mask)
            attention_weights['decoder_layer{}'.format(i + 1)] = block
        
        x = self.output_projection(x)

        x = torch.cat((x,x_l),axis=2)

        x = self.fc(x)
        
        return x, attention_weights # (B,S,S)
    








class CosSquareAttention(nn.Module):
    """
    Multi-head self-attention with cos^2-based re-weighting (single layer).
    """
    def __init__(self, D, H):
        super().__init__()
        self.D = D  # hidden dimension
        self.H = H  # number of heads

        # Q, K, V
        self.wq = nn.Linear(D, D * H)
        self.wk = nn.Linear(D, D * H)
        self.wv = nn.Linear(D, D * H)

        # Combine heads back to dimension D
        self.dense = nn.Linear(D * H, D)

    def split_heads(self, x):
        B, S, D_H = x.shape
        assert D_H == self.D * self.H
        x = x.view(B, S, self.H, self.D)  # (B,S,H,D)
        x = x.permute(0, 2, 1, 3)         # (B,H,S,D)
        return x

    def concat_heads(self, x):
        B, H, S, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B,S,H,D)
        x = x.view(B, S, H * D)                 # (B,S,D*H)
        return x

    def forward(self, x, causal_mask=None):
        """
        x: (B,S,D)
        causal_mask: (S,S) with 1 where future positions are blocked, 0 otherwise
        returns: (B,S,D)
        """
        B, S, D = x.shape

        # 1) Compute Q,K,V
        q = self.wq(x)  # (B,S,D*H)
        k = self.wk(x)  
        v = self.wv(x)

        # 2) Reshape
        q = self.split_heads(q)  # (B,H,S,D)
        k = self.split_heads(k)  
        v = self.split_heads(v)

        # 3) Scale by sqrt(d_head)
        d_head = float(self.D)
        q = q * (d_head ** -0.5)
        k = k * (d_head ** -0.5)

        # 4) Activation (ELU+1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 5) Build cos^2, sin^2 weighting across the sequence dimension S
        device = x.device
        cos_vals = torch.cos(math.pi * torch.arange(S, device=device)/S) ** 2
        sin_vals = torch.sin(math.pi * torch.arange(S, device=device)/S) ** 2
        # shape (S,) -> broadcast to (B,S)
        cos_vals = cos_vals.unsqueeze(0).expand(B, S)
        sin_vals = sin_vals.unsqueeze(0).expand(B, S)

        # Weighted Q,K
        q_cos = torch.einsum('bhjd,bs->bhjd', q, cos_vals)
        q_sin = torch.einsum('bhjd,bs->bhjd', q, sin_vals)
        k_cos = torch.einsum('bhjd,bs->bhjd', k, cos_vals)
        k_sin = torch.einsum('bhjd,bs->bhjd', k, sin_vals)

        # 6) Convert to attention scores
        # We'll do a standard batch matmul approach: shape (B,H,S,S)

        # Expand Q, K to 5D so we can sum over D:
        q_5d     = q.unsqueeze(3)      # (B,H,S,1,D)
        k_5d     = k.unsqueeze(2)      # (B,H,1,S,D)
        attn1    = (q_5d * k_5d).sum(dim=-1)  # (B,H,S,S)

        q_cos_5d = q_cos.unsqueeze(3)
        k_cos_5d = k_cos.unsqueeze(2)
        attn2    = (q_cos_5d * k_cos_5d).sum(dim=-1)

        q_sin_5d = q_sin.unsqueeze(3)
        k_sin_5d = k_sin.unsqueeze(2)
        attn3    = (q_sin_5d * k_sin_5d).sum(dim=-1)

        # attention_scores => (B,H,S,S)
        attention_scores = attn1 + attn2 + attn3

        # 7) Optional causal masking
        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask.bool(), float('-inf'))

        # 8) Softmax
        attn_weights = F.softmax(attention_scores, dim=-1)  # (B,H,S,S)

        # 9) Weighted sum over V => (B,H,S,D)
        out = torch.einsum('bhss,bhsd->bhsd', attn_weights, v)

        # 10) Combine heads => (B,S,D)
        out = self.concat_heads(out)

        # 11) Final linear
        out = self.dense(out)  # (B,S,D)

        return out
    

class CosSquareFormerEncoderLayer(nn.Module):
    """
    One encoder block with:
    - CosSquare self-attention
    - Residual + LN
    - Feed-forward MLP
    - Another residual + LN
    """
    def __init__(self, D, H, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = CosSquareAttention(D, H)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

        # Simple feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(D, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, D),
            nn.Dropout(dropout),
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B,S,D)
        
        # 1) Pre-norm approach: LN -> attn -> residual
        x_ln = self.norm1(x)
        attn_out = self.attn(x_ln, mask)  # (B,S,D)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out

        # 2) LN -> MLP -> residual
        x_ln = self.norm2(x)
        mlp_out = self.mlp(x_ln)  # (B,S,D)
        x = x + mlp_out
        return x

class CosSquareFormerEncoder(nn.Module):
    """
    Stacked CosSquareFormer layers, with optional positional encoding.
    """
    def __init__(self, D, H, N_layers=4, ff_dim=256, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.layers = nn.ModuleList([
            CosSquareFormerEncoderLayer(D, H, ff_dim, dropout) 
            for _ in range(N_layers)
        ])
        self.norm_final = nn.LayerNorm(D)

        # Optional positional embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, D))
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        """
        x: (B,S,D)
        mask: (B,H,S,S) or (S,S) broadcast
        returns: (B,S,D)
        """
        B, S, D = x.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")
        # Add positional embedding
        x = x + self.pos_embed[:, :S, :]

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask=mask)

        # Final LN
        x = self.norm_final(x)
        return x
    
class CosSquareFormerModel(nn.Module):
    """
    Full model with:
    - Optional input projection from raw features
    - Stacked CosSquareFormer layers
    - Final projection to 1 dimension
    """
    def __init__(self, 
                 input_dim,   # number of raw input features
                 D,           # internal hidden dimension
                 H,           # number of heads
                 N_layers=4, 
                 ff_dim=256, 
                 dropout=0.1, 
                 max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D)
        self.encoder = CosSquareFormerEncoder(D, H, N_layers, ff_dim, dropout, max_seq_len)
        self.out_proj = nn.Linear(D, 1)

    def forward(self, x, mask=None):
        """
        x: (B,S,input_dim)
        returns: (B,S,1)
        """
        # 1) Project to internal dimension
        x = self.input_proj(x)  # (B,S,D)

        # 2) Pass through stacked cosSquareFormer layers
        x = self.encoder(x, mask=mask)  # (B,S,D)

        # 3) Final projection
        out = self.out_proj(x)  # (B,S,1)
        return out, None
    

class CosSquareFormerForecastModel(nn.Module):
    """
    Modified CosSquareFormer model to output a 7-day forecast.
    It first encodes the 14-day input, then uses the last hidden state
    to predict the next 7 days.
    """
    def __init__(self, 
                 input_dim,  # number of raw input features
                 D,          # internal hidden dimension
                 H,          # number of heads
                 N_layers=4, 
                 ff_dim=256, 
                 dropout=0.1, 
                 max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, D)
        self.encoder = CosSquareFormerEncoder(D, H, N_layers, ff_dim, dropout, max_seq_len)
        # New projection layer for forecasting 7 days from the context vector.
        self.future_proj = nn.Linear(D, 7)

    def forward(self, x, mask=None):
        """
        x: (B, S, input_dim) with S=14 days input.
        returns: (B, 7, 1) forecast for the next 7 days.
        """
        # 1) Project input.
        x = self.input_proj(x)  # (B, S, D)
        # 2) Encode the sequence.
        x = self.encoder(x, mask=mask)  # (B, S, D)
        # 3) Use the last time step as context.
        context = x[:, -1, :]  # (B, D)
        # 4) Forecast 7 days.
        forecast = self.future_proj(context)  # (B, 7)
        # Reshape to have a final singleton dimension.
        forecast = forecast.unsqueeze(-1)  # (B, 7, 1)
        return forecast, None


class GCNForecast(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=7, dropout=0.3):
        super(GCNForecast, self).__init__()
        # Two GCN layers and one linear layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # Final MLP head
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.dropout(x)

        x = self.lin(x)  # Final output shape: [num_nodes, out_channels]
        return x

