import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pdb

class TemporalRelationalLayer(nn.Module):
    def __init__(self, d_model, h, qk_fc, v_fc, out_fc, topK, dr_rate=0):
        super(TemporalRelationalLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qk_fc) # (C+d_p, d_model)
        self.k_fc = copy.deepcopy(qk_fc) # (C+d_p, d_model)
        self.v_fc = copy.deepcopy(v_fc) # (C, d_model)
        self.out_fc = out_fc              # (d_model, C)
        self.dropout = nn.Dropout(p=dr_rate)
        self.mask = self._make_mask(topK)
        self.mask = self.mask[None, None, :, :].repeat(1, self.h, 1, 1).cuda()
        
        
    def calculate_attention(self, query, key, value, mask):
        # d_k = d_model/h
        # query, key, value: (B, h, 2k, d_k)
        # mask: (2k, 2k) 
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T (B, h, 2k, 2k)
        attention_score = attention_score + mask
        attention_score = attention_score / math.sqrt(d_k)
        
        attention_prob = F.softmax(attention_score, dim=-1) # (B, h, 2k, 2k)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value) # (B, h, 2k, d_k)
        
        return out
    
    def forward(self, query, key, value):
        # query, key: (B, 2k, C+d_p)
        # value: (B, 2k, C)
        # return value: (B, 2k, C)
        
        B = query.size(0)
        mask = self.mask.repeat(B, 1, 1, 1)
        def transform(x, fc): # (B, 2k, C+d_p) or (B, 2k, C)
            out = fc(x)       # (B, 2k, d_model)
            out = out.view(B, -1, self.h, self.d_model//self.h) # (B, 2k, h, d_k)
            out = out.transpose(1 ,2) # (B, h, 2k, d_k)
            return out
        
        query = transform(query, self.q_fc) # (B, h, 2k, d_k)
        key = transform(key, self.k_fc) # (B, h, 2k, d_k)
        value = transform(value, self.v_fc) # (B, h, 2k, d_k)
        
        out = self.calculate_attention(query, key, value, mask) # (B, h, 2k, d_k)
        out = out.transpose(1, 2) # (B, 2k, h, d_k)
        out = out.contiguous().view(B, -1, self.d_model) # (B, 2k, d_model)
        out = self.out_fc(out) # (B, 2k, C)
        return out
    
    def _make_mask(self, k):

        all_one = torch.ones((k,k))
        all_zero = torch.zeros((k,k))
        one_zero = torch.cat((all_one, all_zero), dim=1)
        zero_one = torch.cat((all_zero, all_one), dim=1)
        eye_mat = torch.eye(2*k)
        sigma = -1e+10
        mask = torch.cat((one_zero, zero_one), dim=0)
        
        mask = sigma*(mask - eye_mat) # (2k, 2k)
        mask.requires_grad = False
        return mask