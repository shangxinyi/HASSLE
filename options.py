# -*-coding:utf-8-*-
import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_fmnist', type=str, default=os.path.join(path_dir, 'data/FMNIST/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_labeled_clients', type=int, default=10)
    parser.add_argument('--num_unlabeled_clients', type=int, default=19)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs_label_training', type=int, default=10)  #
    parser.add_argument('--num_epochs_unlabel_training', type=int, default=10)
    parser.add_argument('--batch_size_local_labeled', type=int, default=128)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
    # distillation
    parser.add_argument('--num_epochs_label_distillation', type=int, default=100)  #
    parser.add_argument('--num_epochs_unlabel_distillation', type=int, default=100)
    parser.add_argument('--batch_size_label_distillation', type=int, default=128)
    parser.add_argument('--batch_size_unlabel_distillation', type=int, default=128)
    parser.add_argument('--num_labeled', type=int, default=500)
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--lr_distillation_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sampling_annotation_heterogeneity', type=bool, default=True)
    args = parser.parse_args()
    return args
