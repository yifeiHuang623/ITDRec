r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""
import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

def gather_indexes(output, gather_index):
    """Gathers the vectors at the spexific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)


class GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, n_items, args):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.args = args
        self.n_items = n_items
        self.embedding_size = args.node_feat_dim
        self.hidden_size = args.node_feat_dim
        self.loss_type = 'BPR'
        self.num_layers = args.num_layers
        self.dropout_prob = args.dropout

        self.dev = args.device

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)

    def forward(self, u, item_seq, pos_seq, neg_seq):
        batch, seq_len = item_seq.shape[0], item_seq.shape[1]

        item_seq_emb = self.item_embedding(torch.LongTensor(item_seq[:, :, 0]).to(self.dev))
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = gru_output

        pos_items_emb = self.item_embedding(torch.LongTensor(pos_seq[:, :, 0]).to(self.dev))
        neg_items_emb = self.item_embedding(torch.LongTensor(neg_seq[:, :, 0]).to(self.dev))
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]

        return pos_score, neg_score

    def predict(self, u, seq, item_indices, k=100):
        item_indices = item_indices[:, :, 0]
    
        item_seq = torch.LongTensor(seq[:, :, 0]).to(self.dev)
        item_indices = torch.LongTensor(item_indices).to(self.dev)

        batch, seq_len = item_seq.shape[0], item_seq.shape[1]

        seq_len = torch.full((batch,), seq_len, dtype=torch.long).to(self.dev)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = gather_indexes(gru_output, seq_len - 1)

        test_item_emb = self.item_embedding(item_indices)
        scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(1, 2)).squeeze(1)  # [B n_items]

        values, indices = torch.topk(scores, k=k, dim=-1)

        batch_indices = np.arange(item_indices.shape[0])[:, None]
        ordered_indices = item_indices[batch_indices, indices.cpu()]
        return ordered_indices
