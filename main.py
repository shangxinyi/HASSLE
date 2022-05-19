from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled
import numpy as np
from torch import max, eq, no_grad
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from Model.resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import softmax, log_softmax
from Dataset.dataset import sampling_unlabeled_data_non_iid, \
    sampling_labeled_data_non_iid


class Global(object):
    def __init__(self):
        args = args_parser()
        self.labeled_model = ResNet_cifar(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.labeled_model.to(args.device)
        self.unlabeled_model = ResNet_cifar(resnet_size=8, scaling=4,
                                          save_activations=False, group_norm_num_groups=None,
                                          freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.unlabeled_model.to(args.device)

        self.labeled_res = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.labeled_res.to(args.device)
        self.unlabeled_res = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.unlabeled_res.to(args.device)

        self.device = args.device
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, list_dicts_local_labeled_params, list_nums_local_labeled_data,
                                    list_dicts_local_unlabeled_params, list_nums_local_unlabeled_data):
        fedavg_global_labeled_params = copy.deepcopy(list_dicts_local_labeled_params[0])
        fedavg_global_unlabeled_params = copy.deepcopy(list_dicts_local_unlabeled_params[0])
        for name_param in list_dicts_local_labeled_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_labeled_params, list_nums_local_labeled_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_labeled_data)
            fedavg_global_labeled_params[name_param] = value_global_param

        # unlabeled params
        for name_param in list_dicts_local_unlabeled_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_unlabeled_params, list_nums_local_unlabeled_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_unlabeled_data)
            fedavg_global_unlabeled_params[name_param] = value_global_param
        return fedavg_global_labeled_params, fedavg_global_unlabeled_params

    def aggregation_res(self, list_dicts_local_labeled_res, list_nums_local_labeled_data,
                        list_dicts_local_unlabeled_res, list_nums_local_unlabeled_data):
        global_labeled_res = copy.deepcopy(list_dicts_local_labeled_res[0])
        for name_param in list_dicts_local_labeled_res[0]:
            list_values_param = []
            for dict_local_res, num_local_data in zip(list_dicts_local_labeled_res, list_nums_local_labeled_data):
                list_values_param.append(dict_local_res[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_labeled_data)
            global_labeled_res[name_param] = value_global_param

        global_unlabeled_res = copy.deepcopy(list_dicts_local_unlabeled_res[0])
        for name_param in list_dicts_local_unlabeled_res[0]:
            list_values_param = []
            for dict_local_res, num_local_data in zip(list_dicts_local_unlabeled_res, list_nums_local_unlabeled_data):
                list_values_param.append(dict_local_res[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_unlabeled_data)
            global_unlabeled_res[name_param] = value_global_param
        return global_labeled_res, global_unlabeled_res

    def fedavg_labeled_eval(self, fedavg_labeled_params, fedavg_labeled_res, data_test, batch_size_test):
        self.labeled_model.load_state_dict(fedavg_labeled_params)
        self.labeled_model.eval()
        self.labeled_res.load_state_dict(fedavg_labeled_res)
        self.labeled_res.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs1 = self.labeled_model(images)
                _, outputs2 = self.labeled_res(images)
                outputs_all = outputs1 + outputs2
                _, predicts = max(outputs_all, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def fedavg_unlabeled_eval(self, fedavg_unlabled_params, fedavg_unlabled_res, data_test, batch_size_test):
        self.unlabeled_model.load_state_dict(fedavg_unlabled_params)
        self.unlabeled_model.eval()
        self.unlabeled_res.load_state_dict(fedavg_unlabled_res)
        self.unlabeled_res.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs1 = self.unlabeled_model(images)
                _, outputs2 = self.unlabeled_res(images)
                outputs_all = outputs1 + outputs2
                _, predicts = max(outputs_all, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def global_eval(self, fedavg_labeled_params, fedavg_labeled_res, fedavg_unlabeled_params, fedavg_unlabeled_res,
                    data_test, batch_size_test):
        self.labeled_model.load_state_dict(fedavg_labeled_params)
        self.labeled_model.eval()
        self.labeled_res.load_state_dict(fedavg_labeled_res)
        self.labeled_res.eval()
        self.unlabeled_model.load_state_dict(fedavg_unlabeled_params)
        self.unlabeled_model.eval()
        self.unlabeled_res.load_state_dict(fedavg_unlabeled_res)
        self.unlabeled_res.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects_label = 0
            num_corrects_unlabel = 0
            num_corrects_ensemble = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs1_label = self.labeled_model(images)
                _, outputs2_label = self.labeled_res(images)
                outputs_label = outputs1_label + outputs2_label
                _, predicts_label = max(outputs_label, -1)
                num_corrects_label += sum(eq(predicts_label.cpu(), labels.cpu())).item()
                # unlabel
                _, outputs1_unlabel = self.unlabeled_model(images)
                _, outputs2_unlabel = self.unlabeled_res(images)
                outputs_unlabel = outputs1_unlabel + outputs2_unlabel
                _, predicts_unlabel = max(outputs_unlabel, -1)
                num_corrects_unlabel += sum(eq(predicts_unlabel.cpu(), labels.cpu())).item()
                # ensemble
                ensemble_logits = (outputs_label + outputs_unlabel) / 2
                _, predicts_ensemble = max(ensemble_logits, -1)
                num_corrects_ensemble += sum(eq(predicts_ensemble.cpu(), labels.cpu())).item()
            accuracy_label = num_corrects_label / len(data_test)
            accuracy_unlabel = num_corrects_unlabel / len(data_test)
            accuracy_ensemble = num_corrects_ensemble / len(data_test)
        return accuracy_label, accuracy_unlabel, accuracy_ensemble

    def download_params(self):
        return copy.deepcopy(self.labeled_model.state_dict()), copy.deepcopy(self.labeled_res.state_dict()), \
               copy.deepcopy(self.unlabeled_model.state_dict()), copy.deepcopy(self.unlabeled_res.state_dict())


class Local_Model(object):
    def __init__(self):
        args = args_parser()
        self.local_labeled_model = ResNet_cifar(resnet_size=8, scaling=4,
                                                save_activations=False, group_norm_num_groups=None,
                                                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_labeled_model.to(args.device)
        self.local_unlabeled_model = ResNet_cifar(resnet_size=8, scaling=4,
                                                    save_activations=False, group_norm_num_groups=None,
                                                    freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_unlabeled_model.to(args.device)

        # for learning residual knowledge
        self.local_labeled_res = ResNet_cifar(resnet_size=8, scaling=4,
                                                save_activations=False, group_norm_num_groups=None,
                                                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_labeled_res.to(args.device)
        self.local_unlabeled_res = ResNet_cifar(resnet_size=8, scaling=4,
                                                save_activations=False, group_norm_num_groups=None,
                                                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_unlabeled_res.to(args.device)

        self.device = args.device
        self.random_state = np.random.RandomState(0)

        self.criterion = CrossEntropyLoss().to(args.device)
        self.labeled_optimizer = SGD(self.local_labeled_model.parameters(), lr=args.lr_local_training, momentum=0.9)
        self.unlabeled_optimizer = SGD(self.local_unlabeled_model.parameters(), lr=args.lr_local_training, momentum=0.9)
        self.labeled_res_optimizer = SGD(self.local_labeled_res.parameters(), lr=args.lr_distillation_training, momentum=0.9, weight_decay=0.0001)
        self.unlabeled_res_optimizer = SGD(self.local_unlabeled_res.parameters(), lr=args.lr_distillation_training, momentum=0.9, weight_decay=0.0001)

    def personalized_regularize_train(self, args, data_client_labeled,
                                      global_labeled_params, global_unlabeled_params):
        self.local_labeled_model.load_state_dict(global_labeled_params)
        self.local_labeled_model.train()
        for epoch in range(args.num_epochs_label_training):
            data_loader = DataLoader(dataset=data_client_labeled,
                                     batch_size=args.batch_size_local_labeled,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.local_labeled_model(images)
                loss1 = self.criterion(outputs, labels)
                loss2 = 0.0
                for name_param, params in self.local_labeled_model.named_parameters():
                    loss2 += torch.sum(((params - global_unlabeled_params[name_param]) ** 2) * 1)
                loss = loss1 + 0.001 * loss2
                self.labeled_optimizer.zero_grad()
                loss.backward()
                self.labeled_optimizer.step()
        return copy.deepcopy(self.local_labeled_model.state_dict())

    def pseudo_regularize_training(self, args, data_client_unlabeled, global_labeled_params,
                                   global_labeled_res, global_unlabeled_params):
        self.local_labeled_model.load_state_dict(global_labeled_params)
        self.local_labeled_model.eval()
        self.local_labeled_res.load_state_dict(global_labeled_res)
        self.local_labeled_res.eval()
        self.local_unlabeled_model.load_state_dict(global_unlabeled_params)
        self.local_unlabeled_model.train()
        for epoch in range(args.num_epochs_unlabel_training):
            data_loader = DataLoader(dataset=data_client_unlabeled,
                                     batch_size=args.batch_size_local_unlabeled,
                                     shuffle=True)
            for data_batch in data_loader:
                images, _ = data_batch
                images = images.to(self.device)
                # pseudo_label
                _, outputs_u = self.local_labeled_model(images)
                _, outputs_u_res = self.local_labeled_res(images)
                outputs_all = outputs_u + outputs_u_res
                pseudo_label = torch.softmax(outputs_all.detach() / args.T, dim=-1)
                _, targets_u = torch.max(pseudo_label, dim=-1)
                _, outputs = self.local_unlabeled_model(images)
                loss1 = self.criterion(outputs, targets_u)
                loss2 = 0.0
                for name_param, params in self.local_unlabeled_model.named_parameters():
                    loss2 += torch.sum(((params - global_labeled_params[name_param]) ** 2) * 1)
                loss = loss1 + 0.001 * loss2
                self.unlabeled_optimizer.zero_grad()
                loss.backward()
                self.unlabeled_optimizer.step()
        return copy.deepcopy(self.local_unlabeled_model.state_dict())

    def unlabel_residual_res(self, args, data_client_unlabeled, global_labeled_params, global_labeled_res,
                             global_unlabeled_params, global_unlabeled_res):
        self.local_labeled_model.load_state_dict(global_labeled_params)
        self.local_labeled_model.eval()
        self.local_labeled_res.load_state_dict(global_labeled_res)
        self.local_labeled_res.eval()
        # distillation
        self.local_unlabeled_model.load_state_dict(global_unlabeled_params)
        self.local_unlabeled_model.eval()
        self.local_unlabeled_res.load_state_dict(global_unlabeled_res)
        self.local_unlabeled_res.train()
        for epoch in range(args.num_epochs_unlabel_distillation):
            total_indices_unlabeled = [i for i in range(len(data_client_unlabeled))]
            batch_indices_unlabeled = self.random_state.choice(total_indices_unlabeled, args.batch_size_unlabel_distillation,
                                                             replace=False)
            images = []
            for idx in batch_indices_unlabeled:
                image, _ = data_client_unlabeled[idx]
                images.append(image)
            images = torch.stack(images, dim=0)
            images = images.to(self.device)
            # compute residual logits
            _, logits1 = self.local_labeled_model(images)
            _, logits2 = self.local_labeled_res(images)
            _, logits3 = self.local_unlabeled_model(images)
            logits_teacher = logits1 + logits2 - logits3
            _, logits_student = self.local_unlabeled_res(images)
            # hard loss
            outputs_all = logits1 + logits2
            pseudo_label = torch.softmax(outputs_all.detach() / args.T, dim=-1)
            _, targets_u = torch.max(pseudo_label, dim=-1)
            # model training
            # hard_loss = (F.cross_entropy(logits_student, targets_u, reduction='none') * mask).mean()
            hard_loss = self.criterion(logits_student + logits3.detach(), targets_u)
            # soft_loss
            x = log_softmax(logits_student, dim=1)
            y = softmax(logits_teacher, dim=1)
            soft_loss = F.kl_div(x, y.detach(), reduction='batchmean')
            loss = soft_loss + hard_loss
            self.unlabeled_res_optimizer.zero_grad()
            loss.backward()
            self.unlabeled_res_optimizer.step()
        return copy.deepcopy(self.local_unlabeled_res.state_dict())

    def label_residual_res(self, args, data_client_labeled, global_labeled_params, global_labeled_res,
                           global_unlabeled_params, global_unlabeled_res):
        self.local_labeled_model.load_state_dict(global_labeled_params)
        self.local_labeled_model.eval()
        self.local_labeled_res.load_state_dict(global_labeled_res)
        self.local_labeled_res.train()
        # distillation
        self.local_unlabeled_model.load_state_dict(global_unlabeled_params)
        self.local_unlabeled_model.eval()
        self.local_unlabeled_res.load_state_dict(global_unlabeled_res)
        self.local_unlabeled_res.eval()
        for epoch in range(args.num_epochs_label_distillation):
            total_indices_labeled = [i for i in range(len(data_client_labeled))]
            batch_indices_labeled = self.random_state.choice(total_indices_labeled,
                                                             args.batch_size_label_distillation, replace=False)
            images = []
            labels = []
            for idx in batch_indices_labeled:
                image, label = data_client_labeled[idx]
                images.append(image)
                label = torch.tensor(label)
                labels.append(label)
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)
            images, labels = images.to(self.device), labels.to(self.device)

            _, logits1 = self.local_unlabeled_model(images)
            _, logits2 = self.local_unlabeled_res(images)
            _, logits3 = self.local_labeled_model(images)
            logits_teacher = logits1 + logits2 - logits3
            # soft_loss
            _, logits_student = self.local_labeled_res(images)
            hard_loss = self.criterion(logits_student+logits3.detach(), labels)
            x = log_softmax(logits_student / 2, dim=1)
            y = softmax(logits_teacher / 2, dim=1)
            soft_loss = F.kl_div(x, y.detach(), reduction='batchmean')
            loss = soft_loss + hard_loss
            self.labeled_res_optimizer.zero_grad()
            loss.backward()
            self.labeled_res_optimizer.step()
        return copy.deepcopy(self.local_labeled_res.state_dict())


def partition_train(list_label2indices: list, ipc):
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []

    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)
        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled


def HASSLE():
    args = args_parser()
    random_state = np.random.RandomState(args.seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    list_client2indices_labeled = [[]] * args.num_clients
    list_client2indices_unlabeled = [[]] * args.num_clients
    # only labeled data
    list_labeled = sampling_labeled_data_non_iid(args, data_local_training, list_label2indices_labeled,
                                                 args.num_labeled_clients, args.non_iid_alpha)
    for idx, list_one_labeled in enumerate(list_labeled):
        list_client2indices_labeled[idx] = list_one_labeled
    # labeled + unlabel
    list_unlabeled = sampling_unlabeled_data_non_iid(args, list_label2indices_unlabeled,
                                                     args.num_unlabeled_clients, args.non_iid_alpha)
    for idx, list_one_unlabeled in enumerate(list_unlabeled):
        list_client2indices_unlabeled[idx] = list_one_unlabeled
    show_clients_data_distribution(data_local_training, list_client2indices_labeled,
                                   list_client2indices_unlabeled, args.num_classes)
    # 两个分支
    global_model = Global()
    local_model = Local_Model()

    total_clients = list(range(args.num_clients))
    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_labeled(data_local_training)

    labeled_fedavg_acc = []
    unlabeled_fedavg_acc = []
    ensemble_fedavg_acc = []

    # the first residual model
    for r in tqdm(range(1, 2), desc='Server'):
        dict_global_labeled_params, dict_global_labeled_res, \
        dict_global_unlabeled_params, dict_global_unlabeled_res = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        # two model
        list_dicts_local_labeled_res = []
        list_dicts_local_unlabeled_res = []
        list_nums_local_labeled_data = []
        list_nums_local_unlabeled_data = []
        for client in online_clients:
            # labeled data
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            # unlabeled data
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled
            # global unlabeled res model
            if len(data_client_unlabeled) != 0:
                list_nums_local_unlabeled_data.append(len(data_client_unlabeled))
                local_unlabeled_res = local_model.unlabel_residual_res(args, data_client_unlabeled,
                                                                       copy.deepcopy(dict_global_labeled_params),
                                                                       copy.deepcopy(dict_global_labeled_res),
                                                                       copy.deepcopy(dict_global_unlabeled_params),
                                                                       copy.deepcopy(dict_global_unlabeled_res))

                list_dicts_local_unlabeled_res.append(copy.deepcopy(local_unlabeled_res))
            # global labeled res model
            if len(data_client_labeled) != 0:
                list_nums_local_labeled_data.append(len(data_client_labeled))
                local_labeled_res = local_model.label_residual_res(args, data_client_labeled,
                                                                   copy.deepcopy(dict_global_labeled_params),
                                                                   copy.deepcopy(dict_global_labeled_res),
                                                                   copy.deepcopy(dict_global_unlabeled_params),
                                                                   copy.deepcopy(dict_global_unlabeled_res))
                list_dicts_local_labeled_res.append(copy.deepcopy(local_labeled_res))
        # aggregation
        fedavg_labeled_res, fedavg_unlabeled_res = global_model.aggregation_res(list_dicts_local_labeled_res,
                                                                                list_nums_local_labeled_data,
                                                                                list_dicts_local_unlabeled_res,
                                                                                list_nums_local_unlabeled_data)
    # save global labeled/unlabeled params
    last_round_labeled_params = copy.deepcopy(dict_global_labeled_params)
    last_round_unlabeled_params = copy.deepcopy(dict_global_unlabeled_params)
    global_model.labeled_res.load_state_dict(fedavg_labeled_res)
    global_model.unlabeled_res.load_state_dict(fedavg_unlabeled_res)
    # training global model
    for r in tqdm(range(1, args.num_rounds + 1), desc='Server'):
        # two model
        dict_global_labeled_params, dict_global_labeled_res,\
        dict_global_unlabeled_params, dict_global_unlabeled_res = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        # two branch
        list_dicts_local_labeled_params = []
        list_dicts_local_unlabeled_params = []
        list_dicts_local_labeled_res = []
        list_dicts_local_unlabeled_res = []

        list_nums_local_labeled_data = []
        list_nums_local_unlabeled_data = []
        for client in online_clients:
            # labeled data
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            # unlabeled data
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled
            # local training
            if len(data_client_unlabeled) != 0:
                list_nums_local_unlabeled_data.append(len(data_client_unlabeled))
                local_unlabeled_res = local_model.unlabel_residual_res(args, data_client_unlabeled,
                                                                       copy.deepcopy(last_round_labeled_params),
                                                                       copy.deepcopy(dict_global_labeled_res),
                                                                       copy.deepcopy(dict_global_unlabeled_params),
                                                                       copy.deepcopy(dict_global_unlabeled_res))
                list_dicts_local_unlabeled_res.append(copy.deepcopy(local_unlabeled_res))
                local_unlabeled_params = local_model.pseudo_regularize_training(args, data_client_unlabeled,
                                                                                copy.deepcopy(last_round_labeled_params),
                                                                                copy.deepcopy(dict_global_labeled_res),
                                                                                copy.deepcopy(dict_global_unlabeled_params))

                list_dicts_local_unlabeled_params.append(copy.deepcopy(local_unlabeled_params))

            if len(data_client_labeled) != 0:
                list_nums_local_labeled_data.append(len(data_client_labeled))
                local_labeled_res = local_model.label_residual_res(args, data_client_labeled,
                                                                   copy.deepcopy(dict_global_labeled_params),
                                                                   copy.deepcopy(dict_global_labeled_res),
                                                                   copy.deepcopy(last_round_unlabeled_params),
                                                                   copy.deepcopy(dict_global_unlabeled_res))
                list_dicts_local_labeled_res.append(copy.deepcopy(local_labeled_res))

                local_labeled_params = local_model.personalized_regularize_train(args, data_client_labeled,
                                                                                 copy.deepcopy(dict_global_labeled_params),
                                                                                 copy.deepcopy(dict_global_unlabeled_params))
                list_dicts_local_labeled_params.append(copy.deepcopy(local_labeled_params))

        fedavg_labeled_res, fedavg_unlabeled_res = global_model.aggregation_res(list_dicts_local_labeled_res,
                                                                                list_nums_local_labeled_data,
                                                                                list_dicts_local_unlabeled_res,
                                                                                list_nums_local_unlabeled_data)

        fedavg_labeled_params, fedavg_unlabeled_params = global_model.initialize_for_model_fusion(list_dicts_local_labeled_params,
                                                                                                  list_nums_local_labeled_data,
                                                                                                  list_dicts_local_unlabeled_params,
                                                                                                  list_nums_local_unlabeled_data)
        # eval
        one_labeled_fedavg_acc, one_unlabeled_fedavg_acc, \
        one_ensemble_fedavg_acc = global_model.global_eval(copy.deepcopy(last_round_labeled_params),
                                                           copy.deepcopy(fedavg_labeled_res),
                                                           copy.deepcopy(last_round_unlabeled_params),
                                                           copy.deepcopy(fedavg_unlabeled_res),
                                                           data_global_test, args.batch_size_test)
        labeled_fedavg_acc.append(one_labeled_fedavg_acc)
        unlabeled_fedavg_acc.append(one_unlabeled_fedavg_acc)
        ensemble_fedavg_acc.append(one_ensemble_fedavg_acc)

        global_model.labeled_model.load_state_dict(fedavg_labeled_params)
        global_model.unlabeled_model.load_state_dict(fedavg_unlabeled_params)

        # save model params for the next round
        last_round_labeled_params = copy.deepcopy(dict_global_labeled_params)
        last_round_unlabeled_params = copy.deepcopy(dict_global_unlabeled_params)

        if r % 10 == 0:
            print(labeled_fedavg_acc)
            print(unlabeled_fedavg_acc)
            print(ensemble_fedavg_acc)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    HASSLE()
