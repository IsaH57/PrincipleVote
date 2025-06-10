"""VotingMLP: A Multi-Layer Perceptron for Voting-Based Classification"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VotingMLP(nn.Module):
    def __init__(self, input_shape: tuple, num_candidates: int, **kwargs):
        """Initializes the VotingMLP model.

        Args:
            input_shape (tuple): Shape of the input data (num_voters, num_alternatives, num_alternatives).
            num_candidates (int): Number of candidates to classify.

        Attributes:
            flatten (nn.Flatten): Layer to flatten the input data.
            layers (nn.Sequential): Sequential layers of the MLP.
            train_loader (DataLoader): Placeholder for the training data loader.
            criterion (nn.Module): Placeholder for the loss function.
            optimizer (torch.optim.Optimizer): Placeholder for the optimizer.

        """
        super(VotingMLP, self).__init__()
        flattened_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_candidates),
            nn.Softmax(dim=1)
        )
        self.train_loader = None
        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_voters, num_alternatives, num_alternatives).
        """
        x = self.flatten(x)
        return self.layers(x)

    def set_train_loader(self, train_loader: DataLoader):
        """Sets the training data loader for the model.

        Args:
            train_loader (DataLoader): DataLoader containing the training data.
        """
        self.train_loader = train_loader

    def set_criterion(self, criterion):
        """Sets the loss function for the model.

        Args:
            criterion (nn.Module): The loss function to be used.
        """
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        """Sets the optimizer for the model.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be used.
        """
        self.optimizer = optimizer

    def train_model(self, num_epochs: int):
        """Trains the model on the provided training data.

        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(self.train_loader):.4f}")

    def evaluate_model(self, X_test: torch.Tensor, y_test: torch.Tensor):
        """Evaluates the model on the provided test data.

        Args:
            X_test (torch.Tensor): Test input data of shape (num_samples, num_voters, num_alternatives, num_alternatives).
            y_test (torch.Tensor): Test target labels of shape (num_samples, num_candidates).

        Returns:
            float: Accuracy of the model on the test data.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_test, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy
