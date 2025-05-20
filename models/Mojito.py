import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import math
import torch.nn.functional as F
from models.modules import AdmixMultiHeadAttentionBlocks, BasisTimeEncoder, GlobalAttentionModule

class Mojito(nn.Module):
    """
    Mixture of item & context attention based on the work:
    Nguyen et al. "Improving transformer with an admixture of attention heads."
    Neurips 2022.
    a pytorch version from https://github.com/deezer/sigir23-mojito/
    """
    def __init__(self, user_num, item_num, args):
        super(Mojito, self).__init__()
        
        self.item_num = item_num
        self.user_num = user_num
        self.dev = args.device
        
        self.lambda_trans_seq = 0.5
        self.beta = 1
        self.lambda_global = 0.1
        self.tempo_linspace = 8
        self.tempo_embedding_dim = 16
        self.expand_dim = 3
        # node feature dim
        self.embedding_dim = 64
    
        self.item_emb = torch.nn.Embedding(self.item_num + 2, self.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 2, self.embedding_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_input_sequence_length, self.embedding_dim)
    
        self.time_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'hour']
        self.time_encoders = nn.ModuleDict()
        for feature in self.time_features:
            self.time_encoders[feature] = BasisTimeEncoder(
                time_dim=self.tempo_embedding_dim,
                expand_dim=self.expand_dim,
                tempo_linspace=self.tempo_linspace
            )
            
        self.time_linear = nn.Linear(len(self.time_features) * self.tempo_embedding_dim, self.embedding_dim)
        nn.init.normal_(self.time_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.time_linear.bias)
        self.time_dropout = nn.Dropout(p=args.dropout)
        
        self.sigma_noise = nn.Parameter(0.1 * torch.ones(args.num_heads, dtype=torch.float32))
        self.emb_dropout = nn.Dropout(p=args.dropout)
        self.mha_blocks = AdmixMultiHeadAttentionBlocks(num_blocks=args.num_layers, dim_head=self.embedding_dim, num_heads=args.num_heads, \
                                                    dim_output=self.embedding_dim*2, dropout_rate=args.dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim * 2, eps=1e-8)
        
        self.global_attention = GlobalAttentionModule(self.beta, self.lambda_trans_seq)
        
        self.loss_fct = MojitoLoss(gamma=self.lambda_global)
        
    def log2feats(self, log_seqs):
        timestamps = log_seqs[:, :, 1]
        log_seqs = log_seqs[:, :, 0]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        ctx_emb = self.get_ctx_emb(timestamps) 
        ctx_seqs = ctx_emb * self.item_emb.embedding_dim ** 0.5
        ctx_seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        concat_seq_emb = torch.cat([seqs, ctx_seqs], dim=-1)
        concat_seq_emb = self.emb_dropout(concat_seq_emb)
        timeline_mask = ~torch.BoolTensor(log_seqs == 0).to(self.dev)
        concat_seq_emb *= timeline_mask.unsqueeze(-1)
        
         # Expand and repeat sigma_noise
        sigma_noise = self.sigma_noise.unsqueeze(0)
        sigma_noise = sigma_noise.repeat(log_seqs.shape[0], 1)
        seqs = self.mha_blocks(
            seq=concat_seq_emb,
            context_seq=ctx_emb,
            sigma_noise=sigma_noise,
            mask=timeline_mask,
            causality=True
        )
        # Apply layer normalization
        # batch_size, seq_len, embedding_dim * 2
        seqs = self.layer_norm(seqs)
        return seqs
        
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, fism_ids):
        pos_timestamps = pos_seqs[:, :, 1]
        pos_seqs = pos_seqs[:, :, 0]
        neg_seqs = neg_seqs[:, :, 0]
        
        user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        fism_embs = self.item_emb(torch.LongTensor(fism_ids).to(self.dev))
        # batch_size, seq_len, fism_num + 1
        fism_embs = torch.cat([user_embs.unsqueeze(1), fism_embs], dim=1)
        
        # batch_size, seq_len, 2 * embedding_dim
        seq_emb = self.log2feats(log_seqs)
        
        pos_ctx_emb = self.get_ctx_emb(pos_timestamps) 
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # batch_size, seq_len, 2 * embedding_dim
        pos_emb_local = torch.cat([pos_embs, pos_ctx_emb], dim=-1)
        neg_emb_local = torch.cat([neg_embs, pos_ctx_emb], dim=-1)

        pos_logits_local = torch.sum(pos_emb_local * seq_emb, dim=-1)
        neg_logits_local = torch.sum(neg_emb_local * seq_emb, dim=-1)
        
        # log2feats positive/negative??
        pos_attn_vecs = self.global_attention(pos_embs, fism_embs)
        neg_attn_vecs = self.global_attention(neg_embs, fism_embs)

        pos_logits_global = torch.sum(pos_attn_vecs * pos_embs, dim=-1)
        neg_logits_global = torch.sum(neg_attn_vecs * neg_embs, dim=-1)
        
        return [pos_logits_local, pos_logits_global], [neg_logits_local, neg_logits_global]
    
    def predict(self,  user_ids, log_seqs, item_indices, fism_ids, k=100):
        item_timestamp = np.repeat(item_indices[:, 0, 1].reshape(-1, 1), item_indices.shape[1], axis=1)
        item_indices = item_indices[:, :, 0]
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        ctx_emb = self.get_ctx_emb(item_timestamp) 
        test_embs = torch.cat([item_embs, ctx_emb], dim=-1)
        
        log_feats = self.log2feats(log_seqs)
        loc_logits = torch.matmul(log_feats, test_embs.transpose(-1, -2))[:, -1, :]
        
        user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        fism_embs = self.item_emb(torch.LongTensor(fism_ids).to(self.dev))
        fism_embs = torch.cat([user_embs.unsqueeze(1), fism_embs], dim=1)
        seq_embs = self.item_emb(torch.LongTensor(log_seqs[:, :, 0]).to(self.dev))
        attn_seq = self.global_attention(seq_embs, fism_embs)
        
        # Calculate global sequence vector
        glob_seq_vecs = seq_embs * (1.0 - self.lambda_trans_seq) + \
                    (seq_embs * attn_seq) * self.lambda_trans_seq
        glob_seq_vecs = torch.sum(glob_seq_vecs[:, 1:, :], dim=1, keepdim=True)

        # Calculate test item attention vectors
        glob_test_atts = self.global_attention(item_embs, fism_embs)
        glob_test_logits = item_embs * (1.0 - self.lambda_trans_seq) + \
                        (item_embs * glob_test_atts) * self.lambda_trans_seq
        glob_test_logits = (glob_test_logits + glob_seq_vecs) / log_seqs.shape[1]
        glob_test_logits = torch.sum(glob_test_logits * item_embs, dim=-1)
        
        logits = loc_logits + self.lambda_global * glob_test_logits
        
        values, indices = torch.topk(logits, k=k, dim=-1)
        batch_indices = np.arange(item_indices.shape[0])[:, None] 
        ordered_indices = item_indices[batch_indices, indices.cpu()]
        
        return ordered_indices
        
    def get_ctx_emb(self, timestamp):
        total_detailed_time = self.get_detailed_time(timestamp)
        seq_detailed_time = []
        for time_type in self.time_features:
            seq_detailed_time.append(self.time_encoders[time_type](total_detailed_time[time_type]))
        seq_detailed_time = torch.cat(seq_detailed_time, dim=-1)

        ctx_seq = self.time_linear(seq_detailed_time)
        # Apply dropout (automatically enabled during training, disabled during evaluation)
        ctx_seq = self.time_dropout(ctx_seq)
        
        # batch_size, seq_len, embedding_dim
        return ctx_seq
    
    def get_detailed_time(self, timestamps):
        # batch_size, max_seq
        # Ensure input is a numpy array for processing
        batch_size, max_seq = timestamps.shape
        
        # Initialize result arrays
        year = np.zeros_like(timestamps, dtype=np.int32)
        month = np.zeros_like(timestamps, dtype=np.int32)
        day = np.zeros_like(timestamps, dtype=np.int32)
        dayofweek = np.zeros_like(timestamps, dtype=np.int32)
        dayofyear = np.zeros_like(timestamps, dtype=np.int32)
        week = np.zeros_like(timestamps, dtype=np.int32)
        hour = np.zeros_like(timestamps, dtype=np.int32)
        
        # Process each batch and sequence position
        for i in range(batch_size):
            for j in range(max_seq):
                if timestamps[i, j] > 0:  # Assume 0 or negative values are padding
                    dt = datetime.fromtimestamp(timestamps[i, j])
                    year[i, j] = dt.year
                    month[i, j] = dt.month
                    day[i, j] = dt.day
                    dayofweek[i, j] = dt.weekday()  # 0-6, 0 is Monday
                    dayofyear[i, j] = dt.timetuple().tm_yday
                    week[i, j] = dt.isocalendar()[1]
                    hour[i, j] = dt.hour
        
        # If the original input is a PyTorch tensor, convert results to tensors
        device = self.dev
        year = torch.tensor(year, dtype=torch.long, device=device)
        month = torch.tensor(month, dtype=torch.long, device=device)
        day = torch.tensor(day, dtype=torch.long, device=device)
        dayofweek = torch.tensor(dayofweek, dtype=torch.long, device=device)
        dayofyear = torch.tensor(dayofyear, dtype=torch.long, device=device)
        week = torch.tensor(week, dtype=torch.long, device=device)
        hour = torch.tensor(hour, dtype=torch.long, device=device)
        
        # Return all features
        return {
            'year': year,
            'month': month,
            'day': day,
            'dayofweek': dayofweek,
            'dayofyear': dayofyear,
            'week': week,
            'hour': hour
        }
        
class MojitoLoss(nn.Module):

    def __init__(self, gamma=1e-2):
        super(MojitoLoss, self).__init__()
        self.gamma = gamma
        
    def compute_loss(self, pos_logits, neg_logits):
        # Calculate global loss
        # Note: In PyTorch, we need to ensure self.istarget is an appropriate mask
        # Assume self.istarget is a boolean or float tensor, with value 1 indicating target items
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        ) / len(pos_logits)
        
        return loss

    def forward(self, pos_logits, neg_logits, pos_logits_global, neg_logits_global):
        # reg loss => weight_decay:
        loc_loss = self.compute_loss(pos_logits, neg_logits)
        global_loss = self.compute_loss(pos_logits_global, neg_logits_global)
        loss = loc_loss + self.gamma * global_loss
        return loss