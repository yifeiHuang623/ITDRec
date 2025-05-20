import torch
import numpy as np
import copy

def calculate_dcg(pos, indicator):
    return np.sum((indicator / np.log2(pos + 1)))

def calculate_ndcg(predict, labels, k=None):
    pos = np.arange(1, k+1)
    dcg = calculate_dcg(pos=pos, indicator=np.array([(1 if int(item_id) in labels else 0) for item_id in predict[:k]]))
    
    n_rel = min(k, len(labels))
    ideal_indicator = np.ones(n_rel)
    if n_rel < k:
        ideal_indicator = np.pad(ideal_indicator, (0, k-n_rel), 'constant')
    idcg = calculate_dcg(pos=pos, indicator=ideal_indicator)
    
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg

def get_recommendation_item_prediction_metrics(predict: torch.Tensor, labels: np.ndarray):
    eval_metrics_indices = [
        "hit@5",
        "hit@10",
        "hit@20",
        
        "recall@5",
        "recall@10",
        "recall@20",
        
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
    ]
    
    eval_metrics = {metric: 0 for metric in eval_metrics_indices}
    
    for id in range(predict.shape[0]):
        label_id = [item[0] for item in labels[id]]
        for metric in eval_metrics_indices:
            index, k = metric.split("@")
            if index == "ndcg":
                eval_metrics[metric] += calculate_ndcg(predict=predict[id], labels=label_id, k=int(k))
            elif index == "hit":
                if len(set(label_id) & set(predict[id][:int(k)].tolist())) != 0:
                    eval_metrics[metric] += 1 
            else:
                eval_metrics[metric] += len(set(label_id) & set(predict[id][:int(k)].tolist())) / len(labels[id])
    
    for metric in eval_metrics_indices:
        eval_metrics[metric] /= predict.shape[0]
    
    return eval_metrics

def get_recommendation_event_prediction_metrics(predict_item: torch.Tensor, predict_time: torch.Tensor, labels: np.ndarray, predict_time_mean: np.array=None):
    eval_metrics = get_recommendation_item_prediction_metrics(predict=predict_item, labels=labels)
    
    eval_metrics_indices_time = [
        "absolute time@1",
        "absolute time@3",
        "absolute time@7",
        "absolute time@15",
        "absolute time@30"
    ]
    
    labels_time_list = []
    predict_time_list = []
    predict_time_mean_list = []
    for id in range(predict_item.shape[0]):
        target_id = [item[0] for item in labels[id]]
        mask = np.in1d(predict_item[id], target_id)
        indices = np.where(mask)[0]
        labels_time_list.extend([item[1] for item in labels[id]])
        predict_time_list.extend(predict_time[id][indices].cpu().detach().numpy().flatten())
        predict_time_mean_list.extend(predict_time_mean[id][indices])
    
    eval_metrics_time = {}
    predict_time_list_update = copy.deepcopy(predict_time_list)
    for metric_key in eval_metrics_indices_time:
        index, k = metric_key.split("@")
        correct, total, positive, pos_predict = 0, 0, 0, 0
        for id in range(len(labels_time_list)):
            total += 1
            if labels_time_list[id] >= 365:
                positive += 1
            if predict_time_list[id] >= 365:
                pos_predict += 1
            
            if predict_time_list[id] >= 365:
                predict_time = predict_time_mean_list[id]
                predict_time_list_update[id] = predict_time_mean_list[id]
            else:
                predict_time = predict_time_list[id]
            
            if index == "absolute time":
                if abs(labels_time_list[id] - predict_time) <= float(k):
                    correct += 1
            else:
                if max(predict_time / labels_time_list[id], labels_time_list[id] / predict_time) <= float(k):
                    correct += 1
                
        eval_metrics_time[metric_key] = correct / total
        
    eval_metrics.update(eval_metrics_time)
    
    return eval_metrics
    

