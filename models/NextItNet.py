# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:43:41 2021

@author: lpott
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class DilatedResBlock(nn.Module):
    def __init__(self,dilation,channel,max_len):
        super(DilatedResBlock,self).__init__()
        self.dilation = dilation
        self.channel = channel
        self.half_channel = int(channel/2)
        self.max_len = max_len
        
        self.reduce = nn.Conv1d(channel,self.half_channel,1)
        self.masked = nn.Conv1d(self.half_channel,self.half_channel,3,dilation=dilation)
        self.increase = nn.Conv1d(self.half_channel,channel,1)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=[max_len])#channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=[max_len])#self.half_channel)
        """
        self.reduce_norm = nn.LayerNorm(normalized_shape=channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        
    def forward(self,x):
        y = self.reduce_norm(x.permute(0,2,1)).permute(0,2,1)
        #y = self.reduce_norm(x)

        y = F.leaky_relu(x)
        y = self.reduce(y)
        
                
        y = self.masked_norm(y.permute(0,2,1)).permute(0,2,1)
        y = F.leaky_relu(y)
        y = F.pad(y,pad=(2 + (self.dilation-1)*2,0),mode='constant')
        y = self.masked(y)
      
        
        y = self.increase_norm(y.permute(0,2,1)).permute(0,2,1)
        #y = self.increase_norm(y)
        y = F.leaky_relu(y)
        y = self.increase(y)
        
        return x+y
        

class NextItNet(nn.Module):
    """

    """
    def __init__(self,item_num, args, hidden_layers=2,dilations=[1,2,4,8],pad_token=0):
        super(NextItNet,self).__init__()
        self.embedding_dim = args.node_feat_dim
        self.channel = args.node_feat_dim
        self.output_dim = item_num
        self.pad_token = pad_token
        self.max_len = args.max_input_sequence_length
        self.dev = args.device
        
        self.loss_fct = nn.CrossEntropyLoss()
    
        self.item_embedding = nn.Embedding(self.output_dim+1,self.embedding_dim,padding_idx=pad_token)
        
        self.hidden_layers = nn.Sequential(*[nn.Sequential(*[DilatedResBlock(d,self.embedding_dim,self.max_len) for d in dilations])\
            for _ in range(hidden_layers)])

        self.final_layer = nn.Linear(self.embedding_dim, self.output_dim)

    
    def log2feats(self,x):
        x = self.item_embedding(torch.LongTensor(x).to(self.dev)).permute(0,2,1)
        x = self.hidden_layers(x)
        x = self.final_layer(x.permute(0,2,1))  
        return x
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # shape: [batch_size, max_len, feature_dim]
        log_seqs = log_seqs[:, :, 0]
        pos_seqs = pos_seqs[:, :, 0]
        neg_seqs = neg_seqs[:, :, 0]
        
        log_feats = self.log2feats(log_seqs)

        return log_feats, None

    def predict(self, user_ids, log_seqs, item_indices, k=100):
        log_seqs = log_seqs[:, :, 0]
        item_indices = item_indices[:, :, 0]
        
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = final_feat[:, torch.LongTensor(item_indices).to(self.dev)]

        logits = item_embs

        values, indices = torch.topk(logits, k=k, dim=-1)
        
        batch_indices = np.arange(item_indices.shape[0])[:, None] 
        ordered_indices = item_indices[batch_indices, indices.cpu()]
        
        return ordered_indices
