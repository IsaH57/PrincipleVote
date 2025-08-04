"""VotingCNN: A CNN for Voting-Based Classification"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import numpy as np


class VotingCNN(nn.Module):
    def __init__(self, train_loader: DataLoader, m_max=5, n_max=55, conv_channels=[32, 64], output_dim=5):
        """Initializes the VotingCNN model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            m_max (int): Maximum number of candidates (alternatives).
            n_max (int): Maximum number of voters.
            conv_channels (List[int]): Number of channels for convolutional layers.
            output_dim (int): Number of output classes (candidates).

        Attributes:
            m_max (int): Maximum number of candidates (alternatives).
            n_max (int): Maximum number of voters.
            output_dim (int): Number of output classes (candidates).
            conv1 (nn.Conv2d): First convolutional layer.
            conv2 (nn.Conv2d): Second convolutional layer.
            flattened_dim (int): Flattened dimension after convolutional layers.
            fc1, fc2, fc3 (nn.Linear): Fully connected layers.
            train_loader (DataLoader): DataLoader for training data.
            criterion (nn.Module): Loss function used for training.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        """
        super(VotingCNN, self).__init__()

        self.m_max = m_max
        self.n_max = n_max
        self.output_dim = output_dim

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=m_max, out_channels=conv_channels[0], kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=(1, 5))

        # Compute flattened dim after conv layers
        self.flattened_dim = conv_channels[1] * (m_max - 4) * (n_max - 4)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim) # TODO check if extra output layer is needed

        self.train_loader = train_loader
        self.criterion =  nn.BCEWithLogitsLoss()  # TODO check if loss function is correct. paper uses CrossEntropyLoss
        self.optimizer =  optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, m_max, m_max, n_max).
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = F.relu(self.conv1(x))  # -> (batch, 32, m_max-4, n_max)
        x = F.relu(self.conv2(x))  # -> (batch, 64, m_max-4, n_max-4)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # shape: (batch_size, output_dim)
        return logits

    def train_model(self, num_steps: int, seed: int = 42, plot: bool = False):
        """Trains the CNN model.
        Args:
            num_steps (int): Number of training steps.
            seed (int): Random seed for reproducibility.
            plot (bool): Whether to plot training loss.
        """
        # Set fixed seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

        self.train()

        optimizer = self.optimizer
        criterion = self.criterion
        # Cosine Annealing scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )

        step_count = 0

        # Loss tracking
        losses = []
        steps = []

        while step_count < num_steps:
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = criterion(logits, batch_y.float())
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Track loss and steps
                losses.append(loss.item())
                steps.append(step_count)

                step_count += 1

                if step_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if plot:
            self.plot_training_loss(steps, losses)

    def predict(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Predicts the winners for the given input.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, m_max, m_max, n_max).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - winner_mask (torch.Tensor): Binary mask indicating winners.
                - probs (torch.Tensor): Probabilities of each candidate being the winner.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            winner_mask = probs > 0.5
            return winner_mask.int(), probs

    def evaluate_model(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluates the model on the test set.
        Args:
            X_test (torch.Tensor): Test input data of shape (num_samples, m_max, m_max, n_max).
            y_test (torch.Tensor): Test target labels of shape (num_samples, output_dim).
        Returns:
            float: Accuracy of the model on the test data.
        """

        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            preds = torch.sigmoid(outputs) > 0.5
            correct = ((preds.int() == y_test.int()).all(dim=1)).sum().item()
            total = y_test.size(0)
            accuracy = correct / total
            print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_training_loss(self, steps: List[int], losses: List[float]):
        """Plots the training loss over time.
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
        plt.title('CNN Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()
