import argparse


def parse_args_distribution_matching():
    """Argument Parser for Distribution matching"""
    acceptable_opts = ['adam', 'sgd']
    acceptable_dists = ['mmd', 'w2']

    parser = argparse.ArgumentParser(
        description="Dataset Distillation through Distribution Matching")
    parser.add_argument('--n_iter',
                        type=int,
                        default=30,
                        help='Number of iterations')
    parser.add_argument('--batch_size',
                        type=int,
                        default=250,
                        help='Number of iterations')
    parser.add_argument('--lr',
                        type=float,
                        default=1,
                        help='Learning rate of synthetic dataset')
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help=('optimizer for synthetic dataset.'
                              f' Must belong to {acceptable_opts}'))
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--metric',
                        type=str,
                        default='mmd',
                        help=('Metric between probability distributions.'
                              f' Must belong to {acceptable_dists}'))
    parser.add_argument('--multi',
                        type=str,
                        default='true',
                        help="Whether or not to treat domains independently")
    parser.add_argument('--reg',
                        type=float,
                        default=0.0,
                        help=("Amount of entropic regularization in OT."
                              "Only used if --metric is"
                              "'sinkhorn' or 'joint_sinkhorn'."))
    parser.add_argument('--n_runs_model',
                        type=int,
                        default=5,
                        help=('Number of times clf is runned'
                              ' for performance calculation'))
    parser.add_argument('--ipc_max',
                        type=int,
                        default=50,
                        help='Maximum number of instances per class')
    args = parser.parse_args()

    args.multi = True if args.multi.lower() == 'true' else False

    assert args.metric.lower() in acceptable_dists
    assert args.optimizer.lower() in acceptable_opts

    return args


def parse_args_dadil():
    r"""Argument parser for DaDiL."""
    acceptable_opts = ['adam', 'sgd']
    acceptable_dists = ['mmd', 'w2', 'joint_w2']

    parser = argparse.ArgumentParser(
        description="Dataset Distillation through Distribution Matching")
    parser.add_argument('--n_samples',
                        type=int,
                        default=500,
                        help='Number of Samples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=200,
                        help='Number of Samples')
    parser.add_argument('--n_components',
                        type=int,
                        default=5,
                        help='Number of Components')
    parser.add_argument('--n_iter',
                        type=int,
                        default=40,
                        help='Number of iterations')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-1,
                        help='Learning rate of synthetic dataset')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help=('optimizer for synthetic dataset.'
                              f' Must belong to {acceptable_opts}'))
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--metric',
                        type=str,
                        default='joint_w2',
                        help=('Metric between probability distributions.'
                              f' Must belong to {acceptable_dists}'))
    parser.add_argument('--multi',
                        type=str,
                        default='true',
                        help="Whether or not to treat domains independently")
    parser.add_argument('--reg',
                        type=float,
                        default=0.0,
                        help=("Amount of entropic regularization in OT."
                              "Only used if --metric is"
                              "'sinkhorn' or 'joint_sinkhorn'."))
    parser.add_argument('--n_runs_model',
                        type=int,
                        default=5,
                        help='Number of times clf is runned for performance'
                             'calculation')
    parser.add_argument('--ipc_max',
                        type=int,
                        default=50,
                        help='Maximum number of instances per class')
    args = parser.parse_args()

    args.multi = True if args.multi.lower() == 'true' else False

    assert args.metric.lower() in acceptable_dists
    assert args.optimizer.lower() in acceptable_opts

    return args


def parse_args_fed_dadil():
    r"""Argument parser for FedDaDiL."""
    acceptable_opts = ['adam', 'sgd']
    acceptable_agg = ['random', 'avg', 'wbary']
    acceptable_dists = ['mmd', 'w2', 'joint_w2']

    parser = argparse.ArgumentParser(
        description="Federated Dataset Dictionary Learning")
    parser.add_argument('--n_samples',
                        type=int,
                        default=500,
                        help='Number of Samples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=200,
                        help='Number of Samples')
    parser.add_argument('--n_components',
                        type=int,
                        default=5,
                        help='Number of Components')
    parser.add_argument('--n_iter',
                        type=int,
                        default=50,
                        help='Number of iterations')
    parser.add_argument('--n_client_it',
                        type=int,
                        default=5,
                        help='Number of iterations')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate of synthetic dataset')
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help=('optimizer for synthetic dataset.'
                              f' Must belong to {acceptable_opts}'))
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--metric',
                        type=str,
                        default='joint_w2',
                        help=('Metric between probability distributions.'
                              f' Must belong to {acceptable_dists}'))
    parser.add_argument('--aggregation',
                        type=str,
                        default='random',
                        help=('How to aggregate dictionary versions.'
                              f' Must belong to {acceptable_agg}'))
    parser.add_argument('--reg',
                        type=float,
                        default=0.0,
                        help=("Amount of entropic regularization in OT."
                              "Only used if --metric is"
                              "'sinkhorn' or 'joint_sinkhorn'."))
    args = parser.parse_args()

    assert args.metric.lower() in acceptable_dists
    assert args.optimizer.lower() in acceptable_opts
    assert args.aggregation.lower() in acceptable_agg

    return args
