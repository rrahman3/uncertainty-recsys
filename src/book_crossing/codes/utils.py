from codes.Interactions import Interactions, InstanceSpliter
from codes.Models import NextItNet_DistributedWeights, NextItNet_Dropout
import codes.Models as Models
import argparse
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--seqlen', type=int, default=100,
                        help='Enter Sequence Length.')
    parser.add_argument('--tarlen', type=int, default=1,
                        help='Enter Target Length.')
    parser.add_argument('--setseed', type=int, default=2022,
                        help='Set a seed for random generator.')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--embed_dims', type=int, default=32,
                        help='Embedding Dimension')
    parser.add_argument('--residual_channels', type=int, default=32,
                        help='CNN Residual Channels')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='CNN Kernel Size')
    parser.add_argument('--dilations', type=int, default=[1, 2, 4],
                        help='CNN Dilations')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def init(args, k_fold=5):
    set_seed(args.setseed)

    # load dataset
    instances = Interactions(args)
    instances.process_dataset()
    instances.create_instances()

    # get_target_samples(instances, num_negative_samples=1)

    instance_spliter = InstanceSpliter(data=instances, k_fold=k_fold)
    instance_spliter.generate_folds()

    return instances, instance_spliter