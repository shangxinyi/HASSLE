import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes):
    dict_per_client_labeled = []

    for client, indices in enumerate(zip(clients_indices_labeled, clients_indices_unlabeled)):
        nums_data_labeled = [0 for _ in range(num_classes)]
        nums_data_unlabeled = [0 for _ in range(num_classes)]
        idx_labeled, idx_unlabeled = indices
        for idx in idx_labeled:
            label = dataset[idx][1]
            nums_data_labeled[label] += 1
        dict_per_client_labeled.append(nums_data_labeled)
        for idx in idx_unlabeled:
            label = dataset[idx][1]
            nums_data_unlabeled[label] += 1
        print(f'{client}: {nums_data_labeled}')
        print(f'{client}: {nums_data_unlabeled}')
    return dict_per_client_labeled


def compute_clients_labeled_data_distribution(dataset, clients_indices_labeled, num_classes):
    dict_per_client_labeled = []
    nums_data_labeled = [0 for _ in range(num_classes)]
    for idx in clients_indices_labeled:
        label = dataset[idx][1]
        nums_data_labeled[label] += 1
    dict_per_client_labeled.append(nums_data_labeled)
    return dict_per_client_labeled


class Indices2Dataset_labeled(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        self.label_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])

        idx = self.indices[idx]
        image, label = self.dataset[idx]
        image = self.label_trans(image)

        return image, label

    def __len__(self):
        return len(self.indices)


def sampling_labeled_data_non_iid(args, data_local_training, list_label2indices_labeled, num_labeled_client, alpha):
    list_choose_labeled = []
    list_choose_labeled_client1 = []
    list_rest_label2indices_labeled = []
    random_state = np.random.RandomState(0)
    list_choose_labeled_non_iid = clients_indices(list_label2indices=list_label2indices_labeled, num_classes=args.num_classes,
                                                  num_clients=2, non_iid_alpha=alpha, seed=0)
    client1_sampling = compute_clients_labeled_data_distribution(data_local_training,
                                                                 list_choose_labeled_non_iid[0], args.num_classes)
    client1_sampling = client1_sampling[0]
    for class_idx, list_index in enumerate(list_label2indices_labeled):
        new_data = set(random_state.choice(list_index, client1_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_choose_labeled_client1.extend(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_rest_label2indices_labeled.append(list_index)
    list_choose_labeled.append(list_choose_labeled_client1)
    # the rest of clients
    list_choose_labeled_rest_client = clients_indices_unlabel(list_label2indices=list_rest_label2indices_labeled, num_classes=args.num_classes,
                                                                num_clients=(num_labeled_client-1),
                                                                non_iid_alpha=alpha, seed=10)
    list_choose_labeled.extend(list_choose_labeled_rest_client)
    return list_choose_labeled


def sampling_unlabeled_data_non_iid(args, list_label2indices_unlabeled, num_unlbeled_client, alpha):
    list_choose_unlabeled = []
    list_unlabeled_part1 = []
    list_unlabeled_part2 = []
    random_state = np.random.RandomState(0)
    class_sampling = [2000] * 10
    for class_idx, list_index in enumerate(list_label2indices_unlabeled):
        new_data = set(random_state.choice(list_index, class_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_unlabeled_part1.append(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_unlabeled_part2.append(list_index)
    list_client_part1 =clients_indices_unlabel(list_label2indices=list_unlabeled_part1,
                                            num_classes=args.num_classes, num_clients=9,
                                            non_iid_alpha=alpha, seed=1000)
    list_client_part2 = clients_indices_unlabel(list_label2indices=list_unlabeled_part2, num_classes=args.num_classes, num_clients=10,
                                            non_iid_alpha=alpha, seed=10000)
    list_choose_unlabeled.append([])
    list_choose_unlabeled.extend(list_client_part1)
    list_choose_unlabeled.extend(list_client_part2)

    return list_choose_unlabeled

