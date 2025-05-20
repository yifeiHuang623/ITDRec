import logging
import time
import sys
import os
import numpy as np
import warnings
import json

from models.ITDRec import ITDRec
from models.Baseline import Baseline
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from evaluate_models_utils import evaluate_model_event_prediction
from utils.DataLoader import get_event_prediction_data_loader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_event_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_event_prediction_args(is_evaluation=True)

    # get data for training, validation and testing
    train_data_loader, valid_data_loader, test_data_loader, user_num, item_num, time_threshold = \
        get_event_prediction_data_loader(dataset_name=args.dataset_name, batch_size=args.batch_size, max_sequence_length=args.max_input_sequence_length, model_name=args.model_name)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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

        # create model
        if args.model_name in ["random", "mean"]:
            model = Baseline(model_name=args.model_name, time_threshold=time_threshold)
        else:
            model = ITDRec(user_num=user_num, item_num=item_num, args=args)
            logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            # load the saved model
            load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
            early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                            save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
            early_stopping.load_checkpoint(model, map_location='cpu')

            model = convert_to_gpu(model, device=args.device)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        test_metrics = evaluate_model_event_prediction(model=model, evaluate_data_loader=test_data_loader, time_threshold=time_threshold, time_weight=args.time_weight)
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

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
