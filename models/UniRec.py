import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import math
import torch.nn.functional as F
from models.modules import AdmixMultiHeadAttentionBlocks, BasisTimeEncoder, GlobalAttentionModule

class UniRec(nn.Module):
    
    def __init__(self, user_num, item_num, args):
        super(UniRec, self).__init__()
        
        self.item_num = item_num
        self.user_num = user_num
        self.seq_len = args.max_input_sequence_length
        self.dev = args.device
        
        self.lambda_trans_seq = 0.5
        self.beta = 1
        self.lambda_global = 0.1
        self.tempo_linspace = 8
        self.tempo_embedding_dim = 16
        self.expand_dim = 3
        self.embedding_dim = 64
        self.local_output_dim = self.embedding_dim * 2
    
        self.item_emb = torch.nn.Embedding(self.item_num + 2, self.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 2, self.embedding_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_input_sequence_length, self.embedding_dim)
        self.timestamp_emb = torch.nn.Embedding(8761, self.embedding_dim)
    
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
        self.admix_mha_blocks = AdmixMultiHeadAttentionBlocks(num_blocks=args.num_layers, dim_head=self.embedding_dim, num_heads=args.num_heads, \
                                                    dim_output=self.local_output_dim, dropout_rate=args.dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.local_output_dim, eps=1e-8)
        
        self.global_attention = GlobalAttentionModule(self.beta, self.lambda_trans_seq)
        
        self.loss_fct = UniRecLoss(gamma=self.lambda_global)
        
        self.user_branch_linear = nn.Linear(self.local_output_dim, self.local_output_dim, bias=False)
        nn.init.xavier_uniform_(self.user_branch_linear.weight)
        
        self.item_branch_linear = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.item_tail_linear = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        
    def compute_admix_embs(self, seqs, ctx_seqs, timeline_mask):
        concat_seq_emb = torch.cat([seqs, ctx_seqs], dim=-1)
        concat_seq_emb = self.emb_dropout(concat_seq_emb)
        concat_seq_emb *= timeline_mask.unsqueeze(-1)
        
         # Expand and repeat sigma_noise
        sigma_noise = self.sigma_noise.unsqueeze(0)
        sigma_noise = sigma_noise.repeat(seqs.shape[0], 1)
        seqs = self.admix_mha_blocks(
            seq=concat_seq_emb,
            context_seq=ctx_seqs,
            sigma_noise=sigma_noise,
            mask=timeline_mask,
            causality=True
        )
        # Apply layer normalization
        # batch_size, seq_len, embedding_dim * 2
        seqs = self.layer_norm(seqs)
        return seqs
        
    def log2feats(self, log_seqs, uneven_seqs, timestamp_id):
        timestamps = log_seqs[:, :, 1]
        log_seqs = log_seqs[:, :, 0]
        
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        ctx_emb = self.get_ctx_emb(timestamps) 
        ctx_seqs = ctx_emb * self.item_emb.embedding_dim ** 0.5
        ctx_seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        ctx_seq_ts_emb = self.timestamp_emb(torch.LongTensor(timestamp_id).to(self.dev))
        ctx_seq_ts = ctx_seq_ts_emb * self.item_emb.embedding_dim ** 0.5
        ctx_seq_ts += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        uneven_seq_embs = self.item_emb(torch.LongTensor(uneven_seqs).to(self.dev))
        uneven_seqs = uneven_seq_embs * self.item_emb.embedding_dim ** 0.5 + self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        timeline_mask = ~torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs_branch1 = self.compute_admix_embs(seqs=seqs, ctx_seqs=ctx_seqs, timeline_mask=timeline_mask)
        seqs_branch2 = self.compute_admix_embs(seqs=seqs, ctx_seqs=ctx_seq_ts, timeline_mask=timeline_mask)
        un_seqs = self.compute_admix_embs(seqs=uneven_seqs, ctx_seqs=ctx_seqs, timeline_mask=timeline_mask)
        
        return seqs_branch1, seqs_branch2, un_seqs
        
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, fism_ids, **kwargs):
        log_timestamps = log_seqs[:, :, 1]
        pos_timestamps = pos_seqs[:, :, 1]
        pos_seqs = pos_seqs[:, :, 0]
        neg_seqs = neg_seqs[:, :, 0]
        
        user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
        fism_embs = self.item_emb(torch.LongTensor(fism_ids).to(self.dev))
        # batch_size, seq_len, fism_num + 1
        fism_embs = torch.cat([user_embs.unsqueeze(1), fism_embs], dim=1)
        uneven_seqs = kwargs["train_data"]["uneven_seq"].numpy()
        
        # batch_size, seq_len
        log_timestamps_pad = np.concatenate([log_timestamps[:, 0:1], log_timestamps], axis=-1)
        timestamp_id = np.clip((log_timestamps_pad[:, 1:] - log_timestamps_pad[:, :-1]) // 3600, a_min=0, a_max=365*24)
        pos_timestamps_pad = np.concatenate([pos_timestamps[:, 0:1], pos_timestamps], axis=-1)
        pos_timestamp_id = np.clip((pos_timestamps_pad[:, 1:] - pos_timestamps_pad[:, :-1]) // 3600, a_min=0, a_max=365*24)
        
        # batch_size, seq_len, 2 * embedding_dim
        seq_emb, seq_emb_branch2, un_seq_emb = self.log2feats(log_seqs=log_seqs, uneven_seqs=uneven_seqs, timestamp_id=timestamp_id)
        
        pos_ctx_emb = self.get_ctx_emb(pos_timestamps) 
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # batch_size, seq_len, 2 * embedding_dim
        pos_emb_local = torch.cat([pos_embs, pos_ctx_emb], dim=-1)
        neg_emb_local = torch.cat([neg_embs, pos_ctx_emb], dim=-1)
        pos_logits_local = torch.sum(pos_emb_local * seq_emb, dim=-1)
        neg_logits_local = torch.sum(neg_emb_local * seq_emb, dim=-1)
        
        pos_ctx_emb_branch2 = self.timestamp_emb(torch.LongTensor(pos_timestamp_id).to(self.dev))
        pos_emb_local_branch2 = torch.cat([pos_embs, pos_ctx_emb_branch2], dim=-1)
        neg_emb_local_branch2 = torch.cat([neg_embs, pos_ctx_emb_branch2], dim=-1)
        pos_logits_local_branch2 = torch.sum(pos_emb_local_branch2 * seq_emb_branch2, dim=-1)
        neg_logits_local_branch2 = torch.sum(neg_emb_local_branch2 * seq_emb_branch2, dim=-1)
        
        pos_attn_vecs = self.global_attention(pos_embs, fism_embs)
        neg_attn_vecs = self.global_attention(neg_embs, fism_embs)
        pos_logits_global = torch.sum(pos_attn_vecs * pos_embs, dim=-1)
        neg_logits_global = torch.sum(neg_attn_vecs * neg_embs, dim=-1)
        
        # user_branch loss
        u_value, epoch_num, uneven_item_num = kwargs["train_data"]["u_value"].numpy(), kwargs["epoch_num"], kwargs["train_data"]["uneven_item_num"].numpy()
        # # [batch_size, seqlen, 1]
        u_value = torch.LongTensor(u_value).unsqueeze(-1).expand(-1, self.seq_len).unsqueeze(-1).to(self.dev)
        # [batch_size, seqlen, local_output_dim]
        u_value = u_value.expand(-1, -1, self.local_output_dim).float()
        # Calculate weight w_u
        w_u = (np.pi / 2) * ((epoch_num - 100) / 120) + (np.pi / 2) * ((uneven_item_num - 3) / 50)
        w_u = torch.abs(torch.sin(torch.tensor(w_u, device=self.dev, dtype=torch.float32)))
        w_u = w_u.reshape(log_seqs.shape[0], 1, 1)
        user_distance = ((self.user_branch_linear(un_seq_emb) - seq_emb) ** 2)
        user_branch_loss = u_value * w_u * user_distance
        user_branch_loss = torch.mean(user_branch_loss) * 0.2
        
        # item_branch loss
        item_neighbors_seqs, item_head_tail_values, item_var_values = kwargs["train_data"]["item_neighbors_seqs"].numpy(), \
            kwargs["train_data"]["item_head_tail_values"].numpy(), kwargs["train_data"]["item_var_values"].numpy()
        item_embs = self.item_emb(torch.LongTensor(log_seqs[:, :, 0]).to(self.dev))
        item_neighbors_seqs = item_neighbors_seqs.reshape(-1)
        item_neighbors_embs = self.item_emb(torch.LongTensor(item_neighbors_seqs).to(self.dev)).reshape(log_seqs.shape[0], -1, 3, self.embedding_dim)
        item_branch_loss = self.get_item_branch_loss(item_embeddings=item_embs, neighbor_embeddings=item_neighbors_embs, \
                            item_head_tail_values=item_head_tail_values, epoch_num=epoch_num, item_var_values=item_var_values)
        
        # tail item loss
        tail_item_loss = self.get_tail_item_loss(item_embeddings=item_embs, neighbor_embeddings=item_neighbors_embs, \
                            item_head_tail_values=item_head_tail_values, epoch_num=epoch_num)
        
        return [pos_logits_local, pos_logits_local_branch2, pos_logits_global], [neg_logits_local, neg_logits_local_branch2, neg_logits_global], \
                [user_branch_loss, item_branch_loss, tail_item_loss]
    
    def predict(self,  user_ids, log_seqs, item_indices, fism_ids, k=100, **kwargs):
        item_timestamp = np.repeat(item_indices[:, 0, 1].reshape(-1, 1), item_indices.shape[1], axis=1)
        log_timestamps = log_seqs[:, :, 1]
        log_timestamps_pad = np.concatenate([log_timestamps[:, 0:1], log_timestamps], axis=-1)
        log_timestamp_id = np.clip((log_timestamps_pad[:, 1:] - log_timestamps_pad[:, :-1]) // 3600, a_min=0, a_max=365*24)
        item_timestamp_id = np.clip((item_indices[:, 0, 1] - log_timestamps[:, -1]) // 3600, a_min=0, a_max=365*24)
        item_indices = item_indices[:, :, 0]
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        ctx_emb = self.get_ctx_emb(item_timestamp) 
        test_embs = torch.cat([item_embs, ctx_emb], dim=-1)
        ctx_seq_ts = self.timestamp_emb(torch.LongTensor(item_timestamp_id).to(self.dev)).unsqueeze(1).repeat(1, item_indices.shape[1], 1)
        test_embs_branch2 = torch.cat([item_embs, ctx_seq_ts], dim=-1)
        
        seq_emb, seq_emb_branch2, _ = self.log2feats(log_seqs=log_seqs, uneven_seqs=kwargs["eval_data"]["uneven_seq"].numpy(), timestamp_id=log_timestamp_id)
        
        loc_logits = torch.matmul(seq_emb, test_embs.transpose(-1, -2))[:, -1, :]
        loc_logits_branch2 = torch.matmul(seq_emb_branch2, test_embs_branch2.transpose(-1,-2))[:, -1, :]
        
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
        
        logits = loc_logits + loc_logits_branch2 + self.lambda_global * glob_test_logits
        
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
    
    def get_item_branch_loss(self, item_embeddings, neighbor_embeddings, item_head_tail_values, epoch_num, item_var_values):
        # Calculate attention weights
        expanded_item_embeddings = item_embeddings.unsqueeze(dim=2)
        dot_product = torch.sum(expanded_item_embeddings * neighbor_embeddings, dim=-1)  # [batch_size, seq_len, 3]
        dot_product_max = torch.max(dot_product, dim=-1, keepdim=True)[0]
        shifted_dot_product = dot_product - dot_product_max
        
        exp_dot_product = torch.exp(shifted_dot_product)
        weights = exp_dot_product / torch.sum(exp_dot_product, dim=-1, keepdim=True)  # Normalized weights
        
        # Weighted combination of neighbor embeddings
        weighted_neighbor_embeddings = torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=2)  # [batch_size, seq_len, embedding_dim]
        
        # Concatenate original item embeddings and weighted neighbor embeddings
        all_embeddings = torch.cat([item_embeddings, weighted_neighbor_embeddings], dim=-1)  # [batch_size, seq_len, 2*embedding_dim]
        
        head_item_mask = torch.Tensor(item_head_tail_values).unsqueeze(-1).float().to(self.dev)
        masked_combined_embeddings = all_embeddings * head_item_mask
        final_embeddings = self.item_branch_linear(masked_combined_embeddings)
        
        # Calculate head_item embeddings
        head_item_embeddings = item_embeddings * head_item_mask
        
        # Calculate coefficients
        coefficients = ((np.pi / 2) * (epoch_num - 100.0) / 120.0) + \
                    ((np.pi / 2) * (100.0 - item_var_values) / 100.0)  # [batch_size, self.seqlen]
        
        sin_coefficients = torch.sin(torch.tensor(coefficients, device=self.dev))
        w_i = sin_coefficients.unsqueeze(-1)
        
        # Calculate loss
        item_branch_loss = w_i * ((final_embeddings - head_item_embeddings) ** 2)
        loss = torch.mean(item_branch_loss) * 0.3
        
        return loss
    
    def get_tail_item_loss(self, item_embeddings, neighbor_embeddings, item_head_tail_values, epoch_num):
        # Get combined neighbor embeddings
        self.neighbors_embedding = self.item_neighbors_combine(item_embeddings=item_embeddings, neighbor_embeddings=neighbor_embeddings)  
        final_embeddings = self.item_tail_linear(self.neighbors_embedding)
        
        # Update sequence based on head/tail item flags
        # Note: PyTorch's where function parameter order is different from TensorFlow
        item_head_tail_mask = torch.tensor(item_head_tail_values).unsqueeze(-1).expand(-1, -1, self.embedding_dim).to(self.dev)
        updated_seq = torch.where(item_head_tail_mask.bool(), item_embeddings, final_embeddings)
        
        # Calculate tail item loss
        item_tail_loss = (updated_seq - item_embeddings) ** 2
        loss = torch.mean(item_tail_loss) * 0.1
        
        # Apply sin weight
        sin_w = torch.sin(torch.tensor(np.pi/2) * (epoch_num / 220))
        loss = loss * sin_w  
        return loss
    
    def item_neighbors_combine(self, item_embeddings, neighbor_embeddings):
        # Reshape neighbor embeddings
        batch_size, seq_len = item_embeddings.shape[0], item_embeddings.shape[1]
        neighbor_embeddings = neighbor_embeddings.reshape(batch_size, seq_len, 3, self.embedding_dim)
        
        # Expand item embeddings dimension for dot product with neighbor embeddings
        expanded_item_embeddings = item_embeddings.unsqueeze(2)  # [batch_size, seq_len, 1, embedding_dim]
        
        # Calculate attention weights
        dot_product = torch.sum(expanded_item_embeddings * neighbor_embeddings, dim=-1)  # [batch_size, seq_len, 3]
        dot_product_max = torch.max(dot_product, dim=-1, keepdim=True)[0]
        shifted_dot_product = dot_product - dot_product_max
        
        exp_dot_product = torch.exp(shifted_dot_product)
        weights = exp_dot_product / torch.sum(exp_dot_product, dim=-1, keepdim=True)  # Normalized weights
        
        # Weighted combination of neighbor embeddings
        weighted_neighbor_embeddings = torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=2)  # [batch_size, seq_len, embedding_dim]
        
        # Concatenate original item embeddings and weighted neighbor embeddings
        all_embeddings = torch.cat([item_embeddings, weighted_neighbor_embeddings], dim=-1)  # [batch_size, seq_len, 2*embedding_dim]
        
        return all_embeddings

class UniRecLoss(nn.Module):

    def __init__(self, gamma=1e-2):
        super(UniRecLoss, self).__init__()
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


    def forward(self, pos_logits, neg_logits, pos_logits_branch2, neg_logits_branch2, pos_logits_global, neg_logits_global, other_loss):
        # reg loss => weight_decay:
        user_branch_loss, item_branch_loss, item_tail_loss = other_loss
        
        loc_loss = self.compute_loss(pos_logits, neg_logits)
        loc_loss += self.compute_loss(pos_logits_branch2, neg_logits_branch2)
        global_loss = self.compute_loss(pos_logits_global, neg_logits_global)
        loss = loc_loss + self.gamma * global_loss
        
        loss += user_branch_loss + item_branch_loss + item_tail_loss
        
        return loss