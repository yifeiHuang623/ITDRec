import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch

from models.Caser import Caser
from models.GRU4Rec import GRU4Rec
from models.LightSANs import LightSANs
from models.SASRec import SASRec
from models.TiSASRec import TiSASRec
from models.NextItNet import NextItNet
from models.Bert4Rec import Bert4Rec
from models.TimeLSTM import TimeLSTM
from models.Mojito import Mojito
from models.UniRec import UniRec

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.DataLoader import get_event_prediction_data_loader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_event_prediction_args
from evaluate_models_utils import evaluate_model_item_prediction

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_event_prediction_args(is_evaluation=False)

    # get dataloader for training, validation and testing
    train_data_loader, valid_data_loader, test_data_loader, user_num, item_num, _ = \
        get_event_prediction_data_loader(dataset_name=args.dataset_name, batch_size=args.batch_size, max_sequence_length=args.max_input_sequence_length, model_name=args.pretrain_model_name)
    
    val_metric_all_runs, test_metric_all_runs, = [], []

    for run in range(args.num_runs):
        
        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.pretrain_model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.pretrain_model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.pretrain_model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create pretrain model
        if args.pretrain_model_name == 'SASRec':
            model = SASRec(user_num, item_num, args).to(args.device)
        elif args.pretrain_model_name == 'TiSASRec':
            model = TiSASRec(user_num=user_num, item_num=item_num, args=args).to(args.device)
        elif args.pretrain_model_name == 'Caser':
            model = Caser(num_users=user_num, num_items=item_num, model_args=args).to(args.device)
        elif args.pretrain_model_name == 'LightSANs':
            model = LightSANs(n_items=item_num, args=args).to(args.device)
        elif args.pretrain_model_name == 'GRU4Rec':
            model = GRU4Rec(n_items=item_num, args=args).to(args.device)
        elif args.pretrain_model_name == 'NextItNet':
            model = NextItNet(item_num=item_num, args=args).to(args.device)
        elif args.pretrain_model_name == 'Bert4Rec':
            model = Bert4Rec(num_items=item_num, args=args).to(args.device)
        elif args.pretrain_model_name == 'TimeLSTM':
            model = TimeLSTM(item_num=item_num, args=args)
        elif args.pretrain_model_name == 'Mojito':
            model = Mojito(user_num=user_num, item_num=item_num, args=args)
        elif args.pretrain_model_name == 'UniRec':
            model = UniRec(user_num=user_num, item_num=item_num, args=args)

        model = convert_to_gpu(model, device=args.device)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.pretrain_model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    
        
        save_model_folder = f"./saved_models/{args.pretrain_model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=f"{args.save_model_name}", logger=logger, model_name=args.pretrain_model_name)
        
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(args.num_epochs):
            model.train()
            
            torch.autograd.set_detect_anomaly(True)
            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_data_loader_tqdm = tqdm(train_data_loader, ncols=120)
            for batch_idx, train_data in enumerate(train_data_loader_tqdm):
                u, seq, pos, neg = train_data["user_id"].numpy(), train_data["seq"].numpy(), train_data["pos"].numpy(), train_data["neg"].numpy()
                
                if args.pretrain_model_name == 'Mojito':
                    pos_logits, neg_logits = model(u, seq, pos, neg, train_data["fism"].numpy())
                elif args.pretrain_model_name == 'UniRec':
                    pos_logits, neg_logits, other_loss = model(u, seq, pos, neg, train_data["fism"].numpy(), epoch_num=epoch, train_data=train_data)
                else:
                    pos_logits, neg_logits = model(u, seq, pos, neg)
                indices = np.where(pos[:, :, 0] != 0)
                
                if args.pretrain_model_name in ['LightSANs', 'GRU4Rec', 'Caser']:
                    loss = model.loss_fct(pos_logits[indices], neg_logits[indices])        
                elif args.pretrain_model_name in ['NextItNet', 'Bert4Rec']:
                    loss = model.loss_fct(pos_logits[indices], torch.LongTensor(pos[:, :, 0][indices]).to(args.device))
                elif args.pretrain_model_name == 'Mojito':
                    loss = model.loss_fct(pos_logits[0][indices], neg_logits[0][indices], pos_logits[1][indices], neg_logits[1][indices])
                elif args.pretrain_model_name == 'UniRec':
                    loss = model.loss_fct(pos_logits[0][indices], neg_logits[0][indices], pos_logits[1][indices], neg_logits[1][indices], \
                                         pos_logits[2][indices], neg_logits[2][indices], other_loss)
                else:
                    pos_labels, neg_labels = torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                
            val_metrics = evaluate_model_item_prediction(model=model, evaluate_data_loader=valid_data_loader, model_name=args.pretrain_model_name)
            
            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_metrics = evaluate_model_item_prediction(model=model, evaluate_data_loader=test_data_loader, model_name=args.pretrain_model_name)

                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                if metric_name in ["rmse", "mae"]:
                    higher = False
                else:
                    higher = True
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), higher))
            early_stop = early_stopping.step(val_metric_indicator, model, important_indices=["hit@5", "recall@5", "hit@10", "hit@20", "recall@10", "recall@20"])

            if early_stop:
                break
        
        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        test_metrics = evaluate_model_item_prediction(model=model, evaluate_data_loader=test_data_loader, model_name=args.pretrain_model_name)
        
        # store the evaluation metrics at the current run
        test_metric_dict = {}

        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.pretrain_model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        
    sys.exit()
