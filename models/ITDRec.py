import numpy as np
import torch
import torch.nn as nn

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
    
class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output
    
class ITDRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(ITDRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)
        
        # initialize
        self.item_emb = torch.nn.Embedding(self.item_num + 2, args.node_feat_dim, padding_idx=0)       
        
        self.time_encoder = TimeEncoder(time_dim=args.time_feat_dim)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.node_dim = args.node_feat_dim + args.time_feat_dim * 2

        self.last_layernorm = torch.nn.LayerNorm(self.node_dim, eps=1e-8)
        self.args =args
        
        for _ in range(args.num_layers):
            new_attn_layernorm = torch.nn.LayerNorm(self.node_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.node_dim,
                                                            args.num_heads,
                                                            args.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.node_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.node_dim, args.dropout)
            self.forward_layers.append(new_fwd_layer)
            
        # self.time_predictor = TemporalPredictor(d_model=self.node_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.node_dim,      
            nhead=4,          
            dim_feedforward=self.node_dim,   
            dropout=0.1,       
            activation='relu'  
        )

        self.time_predictor = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.time_predictor_type = args.time_predictor_type
        self.time_bin = args.time_bin
        if self.time_predictor_type == "regression":
            self.regressor = nn.Sequential(
                nn.Linear(self.node_dim, 80),
                nn.ReLU(),
                nn.Linear(80, 1)
            )
        elif self.time_predictor_type == 'binary_classifier':
            self.regressor = nn.Sequential(
                nn.Linear(self.node_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 100)
            )
            self.classifiers = nn.ModuleList([
                nn.Linear(100, 1) for _ in range(self.time_bin)
            ])
        elif self.time_predictor_type == 'heatmap':
            self.regressor = nn.Sequential(
                nn.Linear(self.node_dim, self.time_bin),
                nn.ReLU(),
                nn.Linear(self.time_bin, self.time_bin)
            )
        
        self.linear = nn.Sequential(
            nn.Linear(self.node_dim + args.node_feat_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )
        
        self.pos_emb = torch.nn.Embedding(args.max_input_sequence_length, args.time_feat_dim)
        

    def log2feats(self, log_seqs):
        item_seqs, time_seqs = log_seqs[:, :, 0], log_seqs[:, :, 1]
        seqs = self.item_emb(torch.LongTensor(item_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        time_start = np.array([(np.min(batch[batch > 0]) if batch[batch > 0].size > 0 else 0) for batch in time_seqs])
        absolute_positions = (time_seqs - time_start[:, None, ]) / (60*60*24)
        absolute_positions = self.time_encoder(timestamps=torch.from_numpy(absolute_positions).float().to(self.dev))
        
        relative_positions = log_seqs[:, :, 2]
        relative_positions = self.time_encoder(timestamps=torch.from_numpy(relative_positions).float().to(self.dev))
        
        # batch_size, max_len, node_feat_dim + time_feat_dim * 2
        seqs = torch.cat([seqs, absolute_positions, relative_positions], dim=-1)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(item_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # shape: [batch_size, max_len, feature_dim]
        log_feats = self.log2feats(log_seqs)
        
        log_feats_item = log_feats[:, :, :self.item_emb.embedding_dim]
        
        log_feats_time = log_feats[:, :, self.item_emb.embedding_dim:]
        
        pos_seqs_item = pos_seqs[:, :, 0]
        neg_seqs_item = neg_seqs[:, :, 0]
            
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs_item).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs_item).to(self.dev))

        pos_logits = (log_feats_item * pos_embs).sum(dim=-1)
        neg_logits = (log_feats_item * neg_embs).sum(dim=-1)
        
        pos_embs_time = self.time_encoder(timestamps=torch.from_numpy(np.zeros_like(log_seqs[:, :, 1])).float().to(self.dev))
        neg_embs_time = self.time_encoder(timestamps=torch.from_numpy(np.zeros_like(log_seqs[:, :, 1])).float().to(self.dev))
        
        pos_embs_time = torch.cat([pos_embs, pos_embs_time, pos_embs_time], dim=-1)
        neg_embs_time = torch.cat([neg_embs, neg_embs_time, neg_embs_time], dim=-1)
        
        log_feats_mask = torch.triu(torch.ones(log_feats.shape[0], log_feats.shape[0]) * float('-inf'), diagonal=1).bool()
        pos_time = self.time_predictor(log_feats, pos_embs_time, log_feats_mask, None)
        pos_time = self.regressor(pos_time)
        neg_time = self.time_predictor(log_feats, neg_embs_time, log_feats_mask, None)
        neg_time = self.regressor(neg_time)
        
        if self.time_predictor_type == 'binary_classifier':
            pos_time_list, neg_time_list = [], []
            for classifier in self.classifiers:
                output = torch.sigmoid(classifier(pos_time)).squeeze()
                pos_time_list.append(output)
                output = torch.sigmoid(classifier(neg_time)).squeeze()
                neg_time_list.append(output)
            pos_time = torch.stack(pos_time_list, dim=-1)
            neg_time = torch.stack(neg_time_list, dim=-1)
        
        return pos_logits, neg_logits, pos_time, neg_time

    def predict(self, user_ids, log_seqs, item_indices, time_weight, k=100):
        item_indices = item_indices[:, :, 0]
        
        log_feats = self.log2feats(log_seqs)
        
        log_feats_item = log_feats[:, :, :self.item_emb.embedding_dim]

        final_feat_item = log_feats_item[:, -1, :]
        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        item_embs_time = self.time_encoder(timestamps=torch.from_numpy(np.zeros_like(item_indices)).float().to(self.dev))
        item_embs_time = torch.cat([item_embs, item_embs_time, item_embs_time], dim=-1)

        logits = item_embs.matmul(final_feat_item.unsqueeze(-1)).squeeze(-1)
        
        final_feat_repeat = final_feat.reshape(log_feats.shape[0], 1, -1).repeat(1, item_embs.shape[1], 1)
        final_feats_mask = torch.triu(torch.ones(log_feats.shape[0], log_feats.shape[0]) * float('-inf'), diagonal=1).bool()
        predict_time = self.time_predictor(final_feat_repeat, item_embs_time, final_feats_mask, None)
        predict_time = self.regressor(predict_time)
        
        if self.time_predictor_type == 'binary_classifier':
            outputs = [torch.sigmoid(classifier(predict_time)).squeeze() for classifier in self.classifiers]
            predict_time = torch.stack(outputs, dim=-1)
            threshold_mask = (predict_time >= 0.5)
            first_triggered = threshold_mask.int().argmax(dim=-1)
            no_trigger_mask = ~threshold_mask.any(dim=-1)
            predict_time_result = first_triggered * (365 // self.time_bin)
            predict_time_result[no_trigger_mask] = 365 * 2
            predict_time = predict_time_result.to(self.dev)
            
        elif self.time_predictor_type == 'heatmap':
            max_indices = torch.argmax(predict_time, dim=-1)  # (batch_size, seq_len)
            predict_time = max_indices * (2 * 365 / self.time_bin)
        
        logits = logits * torch.exp(-time_weight*predict_time.squeeze())
        # make sure that time weight is proper
        assert torch.all(torch.max(logits, dim=1).values > 0),  'time weight is too large'
    
        values, indices = torch.topk(logits, k=k, dim=-1)
        
        batch_indices = np.arange(item_indices.shape[0])[:, None] 
        
        ordered_indices = item_indices[batch_indices, indices.cpu()]
    
        predict_time = predict_time[batch_indices, indices.cpu()]
        
        return ordered_indices, predict_time
