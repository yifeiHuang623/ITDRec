import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BasisTimeEncoder(nn.Module):
    """Mercer's time encoding as a PyTorch module"""
    
    def __init__(self, time_dim, expand_dim, tempo_linspace):
        super(BasisTimeEncoder, self).__init__()
        
        # Initialize parameters
        self.time_dim = time_dim
        self.expand_dim = expand_dim
        self.tempo_linspace = tempo_linspace
        
        # Initialize periodic basis
        init_period_base = np.linspace(0, tempo_linspace, time_dim)
        self.period_var = nn.Parameter(torch.tensor(init_period_base, dtype=torch.float32))
        
        # Initialize expansion variables (similar to glorot/xavier initialization)
        bound = 1 / math.sqrt(time_dim)
        self.basis_expan_var = nn.Parameter(
            torch.empty(time_dim, 2 * expand_dim).uniform_(-bound, bound)
        )
        
        # Initialize bias
        self.basis_expan_var_bias = nn.Parameter(torch.zeros(time_dim))
    
    def forward(self, inputs):
        """
            inputs: [batch_size, max_len] 
            outputs: [batch_size, max_len, time_dim]
        """
        # Expand inputs
        expand_input = inputs.unsqueeze(2).expand(-1, -1, self.time_dim)  # [batch_size, max_len, time_dim]
        
        # Apply 10.0 exponent
        period_var = 10.0 ** self.period_var
        
        # Expand period variables
        period_var = period_var.unsqueeze(1).expand(-1, self.expand_dim)  # [time_dim, expand_dim]
        
        # Create expansion coefficients
        expand_coef = torch.arange(1, self.expand_dim + 1, dtype=torch.float32, device=inputs.device).reshape(1, -1)
        
        # Calculate frequency variables
        freq_var = 1 / period_var
        freq_var = freq_var * expand_coef
        
        # Calculate sine and cosine encodings
        sin_enc = torch.sin(expand_input.unsqueeze(-1) * freq_var.unsqueeze(0).unsqueeze(0))
        cos_enc = torch.cos(expand_input.unsqueeze(-1) * freq_var.unsqueeze(0).unsqueeze(0))
        
        # Concatenate sine and cosine encodings, and apply weights
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1) * self.basis_expan_var.unsqueeze(0).unsqueeze(0)
        
        # Sum and add bias
        time_enc = time_enc.sum(dim=-1) + self.basis_expan_var_bias.unsqueeze(0).unsqueeze(0)
        
        return time_enc
        
class GlobalAttentionModule(nn.Module):
    def __init__(self, beta=1.0, lambda_trans_seq = 0.5):
        super(GlobalAttentionModule, self).__init__()
        self.beta = beta
        self.lambda_trans_seq = lambda_trans_seq
    
    def forward(self, seq, fism_items):
        # Calculate attention scores
        w_ij = torch.matmul(seq, fism_items.transpose(1, 2))
        max_wij = torch.max(w_ij, dim=-1, keepdim=True)[0]  # Find maximum value in each row
        shifted_wij = w_ij - max_wij  # Shift values
        exp_wij = torch.exp(shifted_wij)  # Calculate exponent after shifting, greatly reducing overflow risk
        
        exp_sum = torch.sum(exp_wij, dim=-1, keepdim=True)

        # Apply beta parameter (if not equal to 1.0)
        if self.beta != 1.0:
            exp_sum = torch.pow(exp_sum, torch.tensor(self.beta, dtype=torch.float32))
       
        # Calculate attention weights and apply
        att = exp_wij / (exp_sum + 1e-24)
        att_vecs = torch.matmul(att, fism_items)
        
        if self.lambda_trans_seq < 1:
            att_fism_seq = seq * (1.0 - self.lambda_trans_seq) + \
                           (seq * att_vecs) * self.lambda_trans_seq
        else:
            att_fism_seq = seq * att_vecs
        
        return att_fism_seq  
        
"""
This code is based on the original pytorch code found in
https://github.com/minhtannguyen/FishFormer
Reference: Nguyen, Tan, et al. "Improving transformer with an admixture of attention heads." 
Advances in Neural Information Processing Systems 35 (2022): 27937-27952.
"""

class HDPNet(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(HDPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.activation = F.relu if activation == 'relu' else getattr(F, activation)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class AdmixMultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, dim_head=16, dim_emb=8, dim_ctx=8 ,dropout_rate=0, 
                 residual_type='add'):
        super(AdmixMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        
        self.dim_emb = dim_emb
        self.dim_ctx = dim_ctx
        self.dropout_rate = dropout_rate
        self.residual_type = residual_type
        
        # Define projection layers
        self.q_it_proj = nn.Linear(dim_emb, self.dim_head)
        self.k_it_proj = nn.Linear(dim_emb, dim_head)
        self.q_ctx_proj = nn.Linear(dim_ctx, dim_head)
        self.k_ctx_proj = nn.Linear(dim_ctx, dim_head)
        self.v_proj = nn.Linear(num_heads * dim_head, num_heads * dim_head)
        
        # HDP network
        self.hdp_net = HDPNet(input_dim=2, output_dim=num_heads)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, queries_list, keys_list, sigma_noise, causality=False, with_att=False):
        queries_it, queries_ctx = queries_list
        keys_it, keys_ctx = keys_list
        
        # Linear projections
        Q_glob_it = self.q_it_proj(queries_it)  # (N, T_q, dim_head)
        K_glob_it = self.k_it_proj(keys_it)     # (N, T_k, dim_head)
        Q_glob_ctx = self.q_ctx_proj(queries_ctx)  # (N, T_q, dim_head)
        K_glob_ctx = self.k_ctx_proj(keys_ctx)     # (N, T_k, dim_head)
        V = self.v_proj(keys_it)  # (N, T_k, num_heads*dim_head)
        
        # Split and reshape V
        batch_size, seq_len_k, _ = V.size()
        V_ = V.view(batch_size, seq_len_k, self.num_heads, self.dim_head)
        V_ = V_.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len_k, self.dim_head)
        
        # Calculate attention scores
        mean_att_scores_it = torch.matmul(Q_glob_it, K_glob_it.transpose(-1, -2))  # (N, T_q, T_k)
        mean_att_scores_ctx = torch.matmul(Q_glob_ctx, K_glob_ctx.transpose(-1, -2))  # (N, T_q, T_k)
        
        # Concatenate attention scores
        seq_len_q = Q_glob_it.size(1)
        mean_att_scores = torch.cat([mean_att_scores_it.unsqueeze(1), 
                                     mean_att_scores_ctx.unsqueeze(1)], dim=1)  # (N, 2, T_q, T_k)
        
        # Add noise
        sigma_noise = sigma_noise.view(-1, self.num_heads, 1, 1)  # (num_global_heads, 1, 1, 1)
        noise = torch.randn_like(mean_att_scores) * (sigma_noise ** 2)
        att_scores = mean_att_scores + noise
        
        # Reshape and pass through HDPNet
        att_scores = att_scores.permute(0, 2, 3, 1).contiguous()  # (N, T_q, T_k, 2)
        att_scores = att_scores.view(-1, 2)  # (N*T_q*T_k, 2)
        att_scores = self.hdp_net(att_scores)  # (N*T_q*T_k, num_heads)
        att_scores = att_scores.view(batch_size, seq_len_q, seq_len_k, self.num_heads)
        att_scores = att_scores.permute(0, 3, 1, 2).contiguous()  # (N, num_heads, T_q, T_k)
        att_scores = att_scores.view(batch_size * self.num_heads, seq_len_q, seq_len_k)
        
        # Scale
        att_scores = att_scores / (self.dim_head ** 0.5)
        
        # Key Masking
        key_masks = torch.sign(torch.sum(torch.abs(keys_it), dim=-1))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, seq_len_q, 1)  # (h*N, T_q, T_k)
        
        att_scores = att_scores.masked_fill(key_masks == 0, -1e9)  # (h*N, T_q, T_k)
        
        # Causality = Future blinding
        if causality:
            diag_vals = torch.ones_like(att_scores[0])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = tril.unsqueeze(0).repeat(att_scores.size(0), 1, 1)  # (h*N, T_q, T_k)
            att_scores = att_scores.masked_fill(masks == 0, -1e9)  # (h*N, T_q, T_k)
        
        # Activation
        att_scores = F.softmax(att_scores, dim=-1)
        
        # Query Masking
        query_masks = torch.sign(torch.sum(torch.abs(queries_it), dim=-1))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, seq_len_k)  # (h*N, T_q, T_k)
        att_scores = att_scores * query_masks  # broadcasting. (h*N, T_q, T_k)
        
        # Dropouts
        att_scores = self.dropout(att_scores)
        
        # Weighted sum
        outputs = torch.matmul(att_scores, V_)  # (h*N, T_q, dim_head)
        
        # Restore shape
        outputs = outputs.view(self.num_heads, batch_size, seq_len_q, self.dim_head)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len_q, -1)  # (N, T_q, num_heads*dim_head)
        
        # Residual connection
        if self.residual_type == 'add':
            outputs = outputs + queries_it
        elif self.residual_type == 'mult':
            outputs = outputs * queries_it
        else:
            raise ValueError(f'Not support residual type {self.residual_type}')
        
        if with_att:
            return outputs, att_scores
        else:
            return outputs

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(input_dim, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        if inputs.shape[-1] == outputs.shape[-1]:
            outputs += inputs
        return outputs

class AdmixMultiHeadAttentionBlocks(nn.Module):
    def __init__(self, num_blocks, dim_head, num_heads, dim_output, dropout_rate=0.1,
                 output_dim=-1, residual_type='add'):
        super(AdmixMultiHeadAttentionBlocks, self).__init__()
        self.num_blocks = num_blocks
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.embedding_dim = num_heads * dim_head
        self.output_dim = output_dim
        self.residual_type = residual_type
        
        # Create attention layers and feed-forward network layers
        self.attention_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm_ctx_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        self.feed_forward_layers = nn.ModuleList()
        
        for i in range(num_blocks):
            if i == 0:
                self.norm1_layers.append(nn.LayerNorm(self.dim_head*2))
                self.norm_ctx_layers.append(nn.LayerNorm(self.dim_head))
                self.attention_layers.append(
                    AdmixMultiHeadAttention(num_heads=num_heads, dim_head=dim_head, dim_emb=self.dim_head*2, dim_ctx=self.dim_head,
                                    dropout_rate=dropout_rate, residual_type=residual_type)
                )
            else:
                self.norm1_layers.append(nn.LayerNorm(self.embedding_dim))
                self.norm_ctx_layers.append(nn.LayerNorm(self.embedding_dim - self.dim_head))     
                self.attention_layers.append(
                    AdmixMultiHeadAttention(num_heads=num_heads, dim_head=dim_head, dim_emb=self.embedding_dim, dim_ctx=self.embedding_dim - self.dim_head,
                                    dropout_rate=dropout_rate, residual_type=residual_type)
                )
            self.norm2_layers.append(nn.LayerNorm(self.embedding_dim))
            
            if i == (num_blocks - 1):
                hidden_units = dim_output
            else:
                hidden_units = self.embedding_dim
                
            self.feed_forward_layers.append(
                PointWiseFeedForward(input_dim=self.embedding_dim, hidden_units=hidden_units,
                                      dropout_rate=dropout_rate)
            )
    
    def forward(self, seq, context_seq, sigma_noise, mask, causality=True):
        for i in range(self.num_blocks):
            # Self-attention
            queries_list = [self.norm1_layers[i](seq), self.norm_ctx_layers[i](context_seq)]
            keys_list = [seq, context_seq]
            
            seq = self.attention_layers[i](
                queries_list=queries_list,
                keys_list=keys_list,
                sigma_noise=sigma_noise,
                causality=causality
            )
            # Feed forward
            seq = self.feed_forward_layers[i](self.norm2_layers[i](seq))
            # Apply mask
            seq = seq * mask.unsqueeze(-1)
            # Update context_seq for next block
            context_seq = seq[:, :, self.dim_head:]
            
        return seq