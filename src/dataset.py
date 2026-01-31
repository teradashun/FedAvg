import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def get_mnist_datasets():
    train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
    return train_dataset, test_dataset

def split_dataset(dataset, num_clients, dirichlet_alpha):
    num_classes = 10
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    client_indices = {i: [] for i in range(num_clients)}

    for indices in class_indices.values():
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)
    
    subsets = [Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets