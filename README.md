# Federated Semi-Supervised Learning with Annotation Heterogeneity

**Abstract:** Federated Semi-Supervised Learning (FSSL) aims to learn a global model from different clients in an environment with both labeled and unlabeled data.  Most of the existing FSSL work generally assumes that both types of data are available on each client. In this paper, we consider a novel problem of FSSL formally defined as \emph{annotation heterogeneity}, where each client can hold an arbitrary percentage $(0\%{\text -}100\%)$ of labeled data. To address the problem of annotation heterogeneity, we propose a new FSSL framework called Heterogeneously Annotated Semi-Supervised LEarning (HASSLE).  Specifically, it is a dual-model framework with two models trained separately on labeled and unlabeled data such that it can be simply applied to any client.  Furthermore, a mutual learning strategy called Supervised-Unsupervised Mutual Alignment (SUMA) is proposed for the dual models in HASSLE with global residual alignment and model proximity alignment. With this strategy, the dual models can implicitly learn from both types of data across different clients, although each dual model is only trained locally on a single type of data. Experiments verify that the dual models in HASSLE learned by SUMA can mutually learn from each other and thereby effectively utilize the information of unlabeled data across different clients.



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- Fashion-MNIST
- CIFAR-10
- CIFAR-100



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                        | Description                                            |
| ------------------------------- | ------------------------------------------------------ |
| `num_classes`                   | Number of classes                                      |
| `num_clients`                   | Number of all clients.                                 |
| `num_online_clients`            | Number of participating local clients.                 |
| `num_labeled_clients`           | Number of fully-labeled and partially-labeled clients. |
| `num_unlabeled_clients`         | Number of unlabeled clients.                           |
| `num_rounds`                    | Number of communication rounds.                        |
| `num_epochs_local_training`     | Number of local epochs.                                |
| `batch_size_local_training`     | Batch size of local training.                          |
| `num_epochs_local_distillation` | Number of local distillation epochs.                   |
| `batch_size_local_distillation` | Batch size of local distillation.                      |
| `num_labeled_per_class`         | Number of labeled data per class.                      |
| `lr_local_distillation`         | Learning rate of distilling.                           |
| `lr_local_training`             | Learning rate of client updating.                      |
| `non_iid_alpha`                 | Control the degree of heterogeneity.                   |



### Usage

Here is an example to run HASSLE on CIFAR-10 with $\alpha=1$:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_labeled_clients=9 \
--num_unlabeled_clients=10 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=128 \
--num_epochs_local_distillation=10 \
--batch_size_local_distillation=128 \
--num_labeled_per_class=500 \
--lr_local_distillation=0.1 \
--lr_local_training=0.1 \
--non-iid_alpha=1 \
```



