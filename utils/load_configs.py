import argparse
import sys
import torch

def get_event_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the event prediction task')
    parser.add_argument('--model_name', type=str, help='model name', default='ITDRec', choices=['ITDRec', 'random', 'mean'])
    parser.add_argument('--pretrain_model_name', type=str, help='model name', default='', choices=['SASRec', 'TiSASRec', 'Caser', 'LightSANs', 'GRU4Rec', \
                                                                                                    'NextItNet', 'Bert4Rec', 'TimeLSTM', 'Mojito', 'UniRec'])
    
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='pets', choices=['pets', 'books', 'beauty'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    
    parser.add_argument('--gpu', type=int, default=3, help='number of gpu to use')
    
    parser.add_argument('--loss_alpha', type=float, default=0.005, help='dimension of the node embedding')
    parser.add_argument('--time_weight', type=float, default=0.01, help='time weight', choices=[0.01, 0.005, 0.001, 0])
    parser.add_argument('--time_predictor_type', type=str, default='regression', help='the type of time predictor', choices=['regression', 'binary_classifier','heatmap'])
    parser.add_argument('--time_bin', type=int, default=52, help='the number of time bins')
    
    parser.add_argument('--node_feat_dim', type=int, default=172, help='dimension of the node embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    
    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        
    except:
        parser.print_help()
        sys.exit()

    return args
                