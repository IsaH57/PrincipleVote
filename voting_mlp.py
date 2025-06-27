"""VotingMLP: A Multi-Layer Perceptron for Voting-Based Classification"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List


class VotingMLP(nn.Module):
    def __init__(self, input_size: int, train_loader: DataLoader, num_classes: int,
                 criterion=None, optimizer=None):
        """Initializes the VotingMLP model.

        Args:
            input_size (int): Size of flattened input features (mmax² × nmax).
            train_loader (DataLoader): DataLoader for training data.
            num_classes (int): Number of candidates to classify.
            criterion (nn.Module): Loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer): Optimizer. Defaults to AdamW with learning rate 0.001.
        """
        super(VotingMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),  # first hidden layer: 128 neurons
            nn.ReLU(),
            nn.Linear(128, 128),  # second hidden layer: 128 neurons
            nn.ReLU(),
            nn.Linear(128, 128),  # third hidden layer: 128 neurons
            nn.ReLU(),
            nn.Linear(128, num_classes),  # output layer: number of candidates
        )

        self.train_loader = train_loader
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
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

    def train_model(self, num_gradient_steps: int, seed: int = 42, plot: bool = False):
        """Train the model using AdamW optimizer with cosine annealing scheduler.

        Args:
            num_gradient_steps (int): Number of gradient steps to perform.
            seed (int): Random seed for reproducibility. Defaults to 42.
            plot (bool): Whether to plot the training loss. Defaults to False.
        """
        # Set fixed seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

        # AdamW optimizer
        optimizer = self.optimizer

        # Cosine Annealing with Warm Restarts scheduler
        # T_0 = initial restart period, T_mult = factor for increasing restart periods
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )

        self.train()
        step_count = 0

        # Loss tracking
        losses = []
        steps = []

        while step_count < num_gradient_steps:
            for batch_X, batch_y in self.train_loader:
                if step_count >= num_gradient_steps:
                    break

                optimizer.zero_grad()  # Reset gradients
                outputs = self(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                scheduler.step()  # Update learning rate

                # Track loss and steps
                losses.append(loss.item())
                steps.append(step_count)

                step_count += 1

                if step_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {step_count}/{num_gradient_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if plot:
            self.plot_training_loss(steps, losses)

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

    def plot_training_loss(self, steps: List[int], losses: List[float]):
        """Plots Training Loss.

        Args:
            steps (List[int]): List of training steps.
            losses (List[float]): List of loss values corresponding to the steps.
        """
        plt.figure(figsize=(12, 6))

        # Plot raw losses
        plt.plot(steps, losses, 'b-', alpha=0.6, linewidth=0.5, label='Raw Loss')

        # Plot moving average of losses
        window_size = 50
        if len(losses) > window_size:
            moving_avg = []
            for i in range(window_size, len(losses)):
                moving_avg.append(sum(losses[i - window_size:i]) / window_size)
            plt.plot(steps[window_size:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()
