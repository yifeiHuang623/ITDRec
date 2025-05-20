import torch
import numpy as np
from torch import nn
import numpy as np
import torch
    
class TimeLSTM(torch.nn.Module):
    def __init__(self, item_num, args):
        super(TimeLSTM, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 2, args.node_feat_dim, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)
        
        self.hidden_size = args.node_feat_dim

        self.W_all = nn.Linear(args.node_feat_dim, self.hidden_size * 4)
        self.U_all = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.W_d = nn.Linear(self.hidden_size, self.hidden_size)

    def log2feats(self, log_seqs, timestamps):
        timestamps = torch.LongTensor(timestamps).to(self.dev)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        batch_size, seq_len, _ = seqs.size()
        outputs = []
        h = torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(self.dev)
        c = torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(self.dev)
        for s in range(seq_len):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)  # discounted short term mem
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(seqs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            
        log_feats = torch.stack(outputs, 1)
       
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # shape: [batch_size, max_len, feature_dim]
        timestamp = log_seqs[:, :, 1]
        log_seqs = log_seqs[:, :, 0]
        
        pos_seqs = pos_seqs[:, :, 0]
        neg_seqs = neg_seqs[:, :, 0]
        
        log_feats = self.log2feats(log_seqs, timestamp)
            
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, k=100):
        timestamp = log_seqs[:, :, 1]
        log_seqs = log_seqs[:, :, 0]
        item_indices = item_indices[:, :, 0]
        
        log_feats = self.log2feats(log_seqs, timestamp)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        values, indices = torch.topk(logits, k=k, dim=-1)
        
        batch_indices = np.arange(item_indices.shape[0])[:, None] 
        ordered_indices = item_indices[batch_indices, indices.cpu()]
        
        return ordered_indices