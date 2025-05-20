import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args, ac_conv='relu', ac_fc='relu'):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.max_input_sequence_length
        dims = self.args.node_feat_dim
        self.n_h = 4
        self.n_v = 16
        self.drop_ratio = self.args.dropout
        self.ac_conv = activation_getter[ac_conv]
        self.ac_fc = activation_getter[ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users + 1, dims)
        self.item_embeddings = nn.Embedding(num_items + 1, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items + 1, dims+dims)
        self.b2 = nn.Embedding(num_items + 1, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

        self.dev = self.args.device
        
        self.loss_fct = CaserLoss()

    def forward(self, user_var, seq_var, pos_var, neg_var):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        seq = torch.LongTensor(seq_var[:, :, 0]).to(self.dev)
        user = torch.LongTensor(user_var).to(self.dev)
        pos = torch.LongTensor(pos_var[:, :, 0]).to(self.dev)
        neg = torch.LongTensor(neg_var[:, :, 0]).to(self.dev)

        item = torch.cat((pos, neg), 1)

        item_embs = self.item_embeddings(seq).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item)
        b2 = self.b2(item)

        res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        
        (targets_prediction,
                     negatives_prediction) = torch.split(res,
                                                         [pos_var.shape[1],
                                                          neg_var.shape[1]], dim=1)

        return targets_prediction, negatives_prediction

    def predict(self, user_var, seq_var, item_var, k=100):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """
        item_var = item_var[:, :, 0]

        seq = torch.LongTensor(seq_var[:, :, 0]).to(self.dev)
        user = torch.LongTensor(user_var).to(self.dev)
        item = torch.LongTensor(item_var).to(self.dev)

        # Embedding Look-up
        item_embs = self.item_embeddings(seq).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item)
        b2 = self.b2(item)

        w2 = w2.squeeze()
        b2 = b2.squeeze()

        res = (w2 @ x.unsqueeze(2)).sum(1) + b2

        values, indices = torch.topk(res, k=k, dim=-1)

        batch_indices = np.arange(item_var.shape[0])[:, None]
        ordered_indices = item_var[batch_indices, indices.cpu()]

        return ordered_indices
    
class CaserLoss(nn.Module):
    def __init__(self):
        super(CaserLoss, self).__init__()
        
    def forward(self, pos_logits, neg_logits):
       positive_loss = -torch.mean(
                        torch.log(torch.sigmoid(pos_logits)))
       negative_loss = -torch.mean(
            torch.log(1 - torch.sigmoid(neg_logits)))
       loss = positive_loss + negative_loss
       
       return loss