import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.Baseline import Baseline

from utils.metrics import get_recommendation_event_prediction_metrics, get_recommendation_item_prediction_metrics

def evaluate_model_item_prediction(model: nn.Module, evaluate_data_loader: DataLoader, model_name='TiSASRec'):
    model.eval()
    with torch.no_grad():
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120)
        for batch_idx, evaluate_data in enumerate(evaluate_idx_data_loader_tqdm):
            u, history_seq, total_seq, label_num = evaluate_data["user_id"].numpy(), evaluate_data["history_seq"].numpy(), evaluate_data["total_seq"].numpy(), evaluate_data["label_num"].numpy()
            
            labels =  [total_seq[idx][:label_num[idx]] for idx in range(len(total_seq))]
            
            if model_name == 'Mojito':
                predict = model.predict(u, history_seq, total_seq, evaluate_data["fism"].numpy())
            elif model_name == 'UniRec':
                predict = model.predict(u, history_seq, total_seq, evaluate_data["fism"].numpy(), eval_data = evaluate_data)
            else:
                predict = model.predict(u, history_seq, total_seq)
            
            evaluate_metrics.append(get_recommendation_item_prediction_metrics(predict=predict, labels=labels))
            
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch')
            
    return evaluate_metrics

def evaluate_model_event_prediction(model: nn.Module, evaluate_data_loader: DataLoader, time_threshold: float, time_weight: float):
    model.eval()
    with torch.no_grad():
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120)
        for batch_idx, evaluate_data in enumerate(evaluate_idx_data_loader_tqdm):
            u, history_seq, total_seq, label_num = evaluate_data["user_id"].numpy(), evaluate_data["history_seq"].numpy(), evaluate_data["total_seq"].numpy(), evaluate_data["label_num"].numpy()

            labels = [total_seq[idx][:label_num[idx]] for idx in range(len(total_seq))]
            
            predict_item, predict_time = model.predict(u, history_seq, total_seq, time_weight)
            predict_time = torch.clamp(predict_time, min=0, max=time_threshold)

            baseline = Baseline(model_name="mean", time_threshold=time_threshold)
            _, predict_time_mean = baseline.predict(u, history_seq, total_seq, 0)
            
            evaluate_metrics.append(get_recommendation_event_prediction_metrics(predict_item=predict_item, predict_time=predict_time, labels=labels, predict_time_mean=predict_time_mean))
            
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch')
            
    return evaluate_metrics