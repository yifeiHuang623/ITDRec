from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
import os

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class TrainDataset(Dataset):
    def __init__(self, user_sequence: dict, max_sequence_length: int, item_num: int, time_threshold: float):
        super(TrainDataset, self).__init__()
        self.user_id = list(user_sequence.keys())
        self.user_sequence = list(user_sequence.values())
        
        self.max_len = max_sequence_length
        self.item_num = item_num
        self.time_threshold = time_threshold
    
    def __len__(self):
        return len(self.user_sequence)
    
    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        user_sequence = self.user_sequence[idx]
        
        seq = np.zeros((self.max_len, 3))
        pos = np.zeros((self.max_len, 3))
        neg = np.zeros((self.max_len, 3))
        
        nxt = user_sequence[-1]
        pos_items = set(item[0] for item in user_sequence)
        
        for i, item in enumerate(reversed(user_sequence[:-1])):
            if i >= self.max_len:
                break
            idx = self.max_len - 1 - i
            seq[idx] = [item[0], item[1], item[2]]
            pos[idx] = [nxt[0], nxt[1], nxt[2]]
            if nxt != 0:  neg[idx, 0] = random_neq(1, self.item_num + 1, pos_items)
            nxt = item

        return {
            "user_id": user_id,
            "seq": seq.tolist(),
            "pos": pos.tolist(),
            "neg": neg.tolist()
        }
    
class EvalDataset(Dataset):
    def __init__(self, user_history_sequence: dict, user_target_sequence: dict, max_sequence_length: int, \
                    time_threshold: float, item_num: int, eval_count: int=100):
        super(EvalDataset, self).__init__()    
        self.user_history_sequence = user_history_sequence
        self.user_id = list(user_target_sequence.keys())
        self.target_sequence = list(user_target_sequence.values())
        
        self.max_len = max_sequence_length
        self.time_threshold = time_threshold
        self.eval_count = eval_count
        self.item_num = item_num
        
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        target_sequence = self.target_sequence[idx]
        history_sequence = self.user_history_sequence[user_id]
        
        seq = np.zeros((self.max_len, 3))
        if len(history_sequence) > self.max_len:
            seq = history_sequence[-self.max_len:]
        else:
            seq[-len(history_sequence):] = history_sequence
            
        neg_items = []
        neg_count = self.eval_count - len(target_sequence)
        history_items = set([item[0] for item in history_sequence + target_sequence])
        for _ in range(neg_count): 
            neg_item = random_neq(1, self.item_num + 1, history_items)
            neg_items.append(neg_item)
            
        # change target time to time interval
        start_time = seq[-1][1]
        target_sequence_interval = []
        for target_item in target_sequence:
            target_sequence_interval.append((target_item[0], min((target_item[1] - start_time) / (60*60*24), self.time_threshold), target_item[2]))
            
        neg_sequence = np.array([[neg_item, self.time_threshold + random.randint(0, self.time_threshold), \
            self.time_threshold + random.randint(0, self.time_threshold)] for neg_item in neg_items])
        
        return {
            "user_id": user_id,
            "history_seq": seq, 
            "label_seq": target_sequence_interval,
            "neg_seq": neg_sequence
        }
    
def collate_fn_train(batch):
    new_batch = {
        "user_id": torch.stack([torch.tensor(sample['user_id']) for sample in batch], dim=0),
        "seq": torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
        "pos": torch.stack([torch.tensor(sample['pos']) for sample in batch], dim=0),
        "neg": torch.stack([torch.tensor(sample['neg']) for sample in batch], dim=0)
    }
    return new_batch
    
def collate_fn_eval(batch):
    new_batch = {
        "user_id": torch.stack([torch.tensor(sample['user_id']) for sample in batch], dim=0),
        "history_seq": torch.stack([torch.tensor(sample['history_seq']) for sample in batch], dim=0),
        # concat the label and neg sequence into a sequence of the same length and record the label count
        "total_seq": torch.stack([torch.cat([torch.tensor(sample["label_seq"]), \
            torch.tensor(sample["neg_seq"])], dim=0) for sample in batch], dim=0),
        "label_num": torch.tensor([len(sample["label_seq"]) for sample in batch])
    }
    return new_batch
        
def get_event_prediction_data_loader(dataset_name: str, max_sequence_length: int, batch_size: int):
    dataset_path = './processed_data/{}/ml_{}_cluster.csv'.format(dataset_name, dataset_name)
    if os.path.exists(dataset_path):
        graph_df = pd.read_csv(dataset_path)
    else:
        # load total data not include cluster data
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        # get time cluster
        clusters_list = []
        for u_group in graph_df.groupby('u'):
            u_id = u_group[0]
            u_ts = np.array(u_group[1]['ts']).reshape(-1,1)
            # max time interval in a cluster is 1 day 
            clusters = DBSCAN(eps=60*60*24, min_samples=1).fit_predict(u_ts) 
            clusters_df = pd.DataFrame({
                'u': u_id,  
                'ts': u_ts.flatten(),  
                'cluster': clusters  
            })
            # time interval in the cluster
            clusters_df['last_ts_in_cluster'] = clusters_df.groupby('cluster')['ts'].transform('last')
            
            # time interval between the cluster
            prev_last_ts = clusters_df.groupby('cluster')['last_ts_in_cluster'].last().shift(1)
            clusters_df['prev_cluster_last_ts'] = clusters_df['cluster'].map(prev_last_ts)
            clusters_df['between_cluster'] = (clusters_df['ts'] - clusters_df['prev_cluster_last_ts']).fillna(0) / (60*60*24)
            
            clusters_list.append(clusters_df)
            
        clusters_df = pd.concat(clusters_list, ignore_index=True)
        graph_df = pd.merge(graph_df, clusters_df, on=['u', 'ts'], how='left')
        graph_df.to_csv(dataset_path)

    user_num = graph_df.u.max() 
    item_num = graph_df.i.max() - graph_df.u.max()

    # we set the time_interval_max to 365 days
    time_interval_max = 365
    
    train_sequence, valid_sequence, test_sequence, train_valid_sequence = {}, {}, {}, {}
    for user_group in graph_df.groupby("u"):
        u_id = user_group[0]
        cluster_id = user_group[1].groupby("cluster")
        total_cluster = []
        for c in cluster_id:
            cluster_i, cluster_ts, cluster_interval = c[1]['i'], c[1]['ts'], c[1]['between_cluster']
            total_cluster.append([[i, ts, interval] for i, ts, interval in zip(cluster_i - user_num + 1, cluster_ts, cluster_interval)])
            
        if len(total_cluster) < 3:
            train_sequence.update({u_id: [item for cluster in total_cluster for item in cluster]})
        else:
            train_sequence.update({u_id: [item for cluster in total_cluster[:-2] for item in cluster]})
            valid_sequence.update({u_id: total_cluster[-2]})
            train_valid_sequence.update({u_id: [item for cluster in total_cluster[:-1] for item in cluster]})
            test_sequence.update({u_id: total_cluster[-1]})
    
    # combine them as the Dataset
    train_dataset = TrainDataset(user_sequence=train_sequence, max_sequence_length=max_sequence_length, item_num=item_num, time_threshold=time_interval_max)
    valid_dataset = EvalDataset(user_history_sequence=train_sequence, user_target_sequence=valid_sequence, max_sequence_length=max_sequence_length, time_threshold=time_interval_max, item_num=item_num)
    test_dataset = EvalDataset(user_history_sequence=train_valid_sequence, user_target_sequence=test_sequence, max_sequence_length=max_sequence_length, time_threshold=time_interval_max, item_num=item_num)
    
    # get data_loader
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_eval)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_eval)
    
    print("The dataset has {} different users, {} different items".format(user_num, item_num))
    print("The training dataset has {} sequences,".format(len(train_dataset)))
    print("The validation dataset has {} sequences,".format(len(valid_dataset)))
    print("The test dataset has {} sequences".format(len(test_dataset)))

    return train_data_loader, valid_data_loader, test_data_loader, user_num, item_num, time_interval_max