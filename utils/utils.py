import random
import torch
import torch.nn as nn
import numpy as np

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer

def convert_time(label: torch.Tensor, time_type: str, time_bin: int):
    if time_type == 'binary_classifier':
        labels_time = []
        for event_time_seq in label:
            pos_labels_seq = []
            for event_time in event_time_seq:
                label = torch.zeros(time_bin)
                event_time = int(event_time.item())
                if event_time < 365:
                    label[event_time // (365 // time_bin): ] = 1.0  
                pos_labels_seq.append(label)
            labels_time.append(torch.stack(pos_labels_seq, dim=0))
        labels_time = torch.stack(labels_time, dim=0)
        return labels_time
    elif time_type == 'heatmap':
        delta_max = 365
        sigma = 30
        smoothing_factor = 0.1
        batch_size, seq_len = label.shape
        
        heatmap_prototypes = torch.zeros((batch_size, seq_len, time_bin), device=label.device)
        indices = torch.arange(time_bin, device=label.device).float()
        indices = indices.view(1, 1, -1)  # [1, 1, time_bin]
        tau = label.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]

        scale_factor = 2 * delta_max / time_bin
        gaussian = torch.exp(-0.5 * ((scale_factor * indices - tau) / sigma)**2)

        sums = gaussian.sum(dim=2, keepdim=True)
        mask = (sums > 0)
        normalized_gaussian = torch.where(
            mask,
            gaussian / (sums + 1e-10),
            gaussian
        )

        # label smoothing
        uniform_distribution = torch.ones_like(normalized_gaussian) / time_bin
        smoothed_labels = (1 - smoothing_factor) * normalized_gaussian + smoothing_factor * uniform_distribution
        heatmap_prototypes = smoothed_labels  
        
        return heatmap_prototypes
    else:
        return label