from xmlrpc.client import Boolean
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import Tensor
from torch.utils.data import Sampler
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

"""
Dataset. Create imbalanced/balanced Gaussian clusters. 
Two datasets using scikit learn, one is balanced, one has heavily imbalanced clusters.
"""

balanced_cluster_centre = [(-5, -5), (5, 5)]
balanced_cluster_std = [3.5, 5]
balanced_training_examples = [1000, 1000]

imbalanced_cluster_centre = [(-5, -5), (5, 5)]
imbalanced_cluster_std = [3.75, 5]
imbalanced_training_examples = [500, 3000]

test_set_size = 0.3
rand_state = 1
epochs = 5
batch_size = 64


X_balanced, y_balanced = make_blobs(n_samples=[balanced_training_examples[0], balanced_training_examples[1]], 
                                    cluster_std=balanced_cluster_std, 
                                    centers=balanced_cluster_centre, 
                                    n_features=2, random_state=rand_state)
X_imbalanced, y_imbalanced = make_blobs(n_samples=[imbalanced_training_examples[0], imbalanced_training_examples[1]], 
                                    cluster_std=imbalanced_cluster_std, 
                                    centers=imbalanced_cluster_centre, 
                                    n_features=2, random_state=rand_state)

# For now, just use a train test split
X_balanced_train, X_balanced_test, y_balanced_train, y_balanced_test = train_test_split(
    X_balanced, y_balanced, test_size=test_set_size, random_state=rand_state
)
X_imbalanced_train, X_imbalanced_test, y_imbalanced_train, y_imbalanced_test = train_test_split(
    X_imbalanced, y_imbalanced, test_size=test_set_size, random_state=rand_state
)

""" Now set up the data for usage """

class ClassificationDataset(Dataset):
    def __init__(self, 
                cluster_centres : list, 
                cluster_stds : list, 
                cluster_examples : list,
                transform = None,
                n_features : int = 2,
                random_state : int = 1) -> None:
        self.cluster_centres = cluster_centres
        self.cluster_stds = cluster_stds
        self.cluster_examples = cluster_examples
        self.n_features = n_features
        self.random_state = random_state
        self.transform = transform
        self.X, self.y = make_blobs(n_samples = self.cluster_examples, 
                            cluster_std = self.cluster_stds,
                            centers = self.cluster_centres,
                            random_state = self.random_state,
                            n_features = self.n_features)
        self.X = torch.from_numpy(self.X).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(torch.float32)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx) -> int:
        return self.X[idx], self.y[idx]
        
# Create data loaders.
balanced_train_data, balanced_test_data = torch.utils.data.random_split(ClassificationDataset(
                                                                                                [(-5, -5), (5, 5)],
                                                                                                [3.5, 5],
                                                                                                [1000,1000]
                                                                                            ), [1500, 500])



balanced_train_dataloader = DataLoader(balanced_train_data, batch_size=batch_size)
balanced_test_dataloader = DataLoader(balanced_test_data, batch_size=batch_size)

for X, y in balanced_test_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

""" Design the neural network for the imbalanced/balanced classification task """

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

""" Neural Network Training """

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(data_loader, model, loss_function, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Training: ")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(balanced_train_dataloader, model, loss_fn, optimizer)
    test(balanced_test_dataloader, model, loss_fn)
print("Done!")



def plot():
    plt.subplot(2,1,1)
    plt.title("Balanced Dataset")
    plt.scatter(X_balanced[y_balanced == 0, 0], X_balanced[y_balanced == 0, 1], color="red", s=10, label="Cluster1")
    plt.scatter(X_balanced[y_balanced == 1, 0], X_balanced[y_balanced == 1, 1], color="blue", s=10, label="Cluster2")
    plt.subplot(2,1,2)
    plt.title("Imbalanced Dataset")
    plt.scatter(X_imbalanced[y_imbalanced == 0, 0], X_imbalanced[y_imbalanced == 0, 1], color="red", s=10, label="Cluster1")
    plt.scatter(X_imbalanced[y_imbalanced == 1, 0], X_imbalanced[y_imbalanced == 1, 1], color="blue", s=10, label="Cluster2")

    plt.tight_layout()
    plt.show()

plot()