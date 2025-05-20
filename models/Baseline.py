import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

class Baseline(torch.nn.Module):
    def __init__(self, model_name, time_threshold):
        super().__init__()
        self.model_name = model_name
        self.time_threshold = time_threshold
        
    def predict(self, user_ids, log_seqs, item_indices, time_weight):
        ordered_indices = torch.tensor(item_indices[:, :, 0])
        if self.model_name == "random":
            predict_time = torch.rand_like(torch.tensor(ordered_indices)) * self.time_threshold
        elif self.model_name == "mean":
            # batch_size, seq_length
            predict_time = []
            for id in range(len(log_seqs)):
                all_time = log_seqs[:, :, 1][id]
                total_time = all_time[all_time > self.time_threshold * 2].reshape(-1, 1)
                clusters = DBSCAN(eps=86400, min_samples=1).fit_predict(total_time) 
                unique_clusters = set(clusters)
                centroids = []
                for cluster_id in unique_clusters:
                    cluster_points = total_time[clusters == cluster_id]
                    centroids.append(np.mean(cluster_points, axis=0).item())
                
                if len(centroids) > 1:
                    avg_distance = np.mean((np.array(centroids) - np.array([0] + centroids[:-1]))[1:] / (60 * 60 * 24))
                else:
                    avg_distance = self.time_threshold / 2  
                predict_time.append([avg_distance] * len(item_indices[id]))

            predict_time = torch.tensor(predict_time)
        
        return ordered_indices, predict_time