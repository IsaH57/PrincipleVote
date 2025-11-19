"""VotingCNN: A CNN for Voting-Based Classification"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List

from axioms import set_training_axiom, check_anonymity, check_neutrality, check_condorcet, check_pareto, \
    check_independence
from synth_data import SynthData


class VotingCNN(nn.Module):
    """A Convolutional Neural Network (CNN) for Voting-Based Classification.

    Attributes:
            name (str): Name of the model.
            max_cand (int): Maximum number of candidates (alternatives).
            max_vot (int): Maximum number of voters.
            conv1 (nn.Conv2d): First convolutional layer.
            conv2 (nn.Conv2d): Second convolutional layer.
            flattened_dim (int): Flattened dimension after convolutional layers.
            fc1, fc2, fc3 (nn.Linear): Fully connected layers.
            train_loader (DataLoader): DataLoader for training data.
            criterion (nn.Module): Loss function used for training.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    """

    def __init__(self, train_loader: DataLoader, max_candidates: int, max_voters: int, conv_channels=[32, 64]):
        """Initializes the VotingCNN model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            max_candidates (int): Maximum number of candidates (alternatives).
            max_voters (int): Maximum number of voters.
            conv_channels (List[int]): Number of channels for convolutional layers.
        """
        super(VotingCNN, self).__init__()
        self.name = "cnn"

        self.max_cand = max_candidates
        self.max_vot = max_voters

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=max_candidates, out_channels=conv_channels[0],
                               kernel_size=(5, 1))  # TODO check if this kernel size works for 77/7
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[0],
                               kernel_size=(1, 5))  # TODO check effect of conv_channels

        # Compute flattened dim after conv layers
        self.flattened_dim = conv_channels[0] * (max_candidates - 4) * (max_voters - 4)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.max_cand)

        self.train_loader = train_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, m_max, m_max, n_max).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # shape: (batch_size, output_dim)
        return logits

    def train_model(self, num_steps: int, seed: int = 42, plot: bool = False, axiom: str = "default"):
        """Trains the CNN model.

        Args:
            num_steps (int): Number of training steps.
            seed (int): Random seed for reproducibility.
            plot (bool): Whether to plot training loss.
            axiom (str): Axiom to enforce during training.
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
            for batch_x, batch_y, prof in self.train_loader:
                optimizer.zero_grad()
                logits = self.forward(batch_x)

                loss = criterion(logits, batch_y.float())
                loss += set_training_axiom(self, batch_x, prof, axiom)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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

    def evaluate_model_hard(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluates the model using hard accuracy, meaning that the predicted set of winners must match exactly with the true set: F(P)=S.

        Args:
            X_test (torch.Tensor): Test input tensor of shape (num_samples, m_max, m_max, n_max).
            y_test (torch.Tensor): True labels tensor of shape (num_samples, num_classes).

        Returns:
            float: Hard accuracy as a fraction of correct predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()

            # hard accuracy: predicted set must match exactly
            correct = 0
            for pred, true in zip(predicted, y_test.int()):
                if torch.equal(pred, true):
                    correct += 1

            print(f"Hard Accuracy: {correct / len(y_test)}")
            return correct / len(y_test)

    def evaluate_model_soft(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluates the model using soft accuracy, meaning that there is at least one overlap between predicted winners and true winners: F(P) ⊆ S.

        Args:
            X_test (torch.Tensor): Test input tensor of shape (num_samples, m_max, m_max, n_max).
            y_test (torch.Tensor): True labels tensor of shape (num_samples, num_classes).

        Returns:
            float: Soft accuracy as a fraction of correct predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()

            correct = 0
            for pred, true in zip(predicted, y_test.int()):
                # Check if there is any overlap between predicted winners and true winners
                if (pred & true).any():
                    correct += 1

            print(f"Soft Accuracy: {correct / len(y_test)}")
            return correct / len(y_test)

    AXIOM_SAT_FUNCTIONS = {
        "anonymity": check_anonymity,
        "neutrality": check_neutrality,
        "condorcet": check_condorcet,
        "pareto": check_pareto,
        "independence": check_independence,
    }

    def evaluate_axiom_satisfaction_old(self, data: SynthData, axiom: str) -> float:
        """Evaluates the model's axiom satisfaction rate on the test set.

        Args:
            data (SynthData): The synthetic data containing test profiles and winners.
            axiom (str): The axiom to evaluate. Must be one of the keys in AXIOM_SAT_FUNCTIONS.

        Returns:
            float: Axiom satisfaction rate as a fraction of samples satisfying the axiom.
        """
        axiom_fun = self.AXIOM_SAT_FUNCTIONS.get(axiom)

        X_test, y_test = data.get_encoded_cnn()
        profiles = data.get_raw_profiles()

        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()

            satisfied = 0
            for profile, pred in zip(profiles, predicted):
                satisfied += axiom_fun(profile, pred, data.cand_max, data.winner_method)
            print(satisfied)
            print(f"Axiom ({axiom}) Satisfaction Rate: {satisfied / len(y_test)}")

        return satisfied / len(y_test)

    def evaluate_axiom_satisfaction(self, data: SynthData, axiom: str):
        """Evaluate axiom satisfaction on the test set and return the same metrics
        as train_and_eval.axiom_satisfaction (conditional & absolute satisfaction,
        percent applicable).

        Note: axiom_fun must return -1 (violation), 0 (not applicable), or 1 (satisfied).
        """
        axiom_fun = self.AXIOM_SAT_FUNCTIONS.get(axiom)
        if axiom_fun is None:
            raise ValueError(f"Axiom '{axiom}' not found in AXIOM_SAT_FUNCTIONS")

        X_test, y_test = data.get_encoded_cnn()
        profiles = data.get_raw_profiles()

        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()

            iteration = len(predicted)
            applicable = 0
            satisfied = 0

            for profile, pred in zip(profiles, predicted):
                sat = int(axiom_fun(profile, pred, data.cand_max, data.winner_method))
                if sat == 1:
                    satisfied += 1
                    applicable += 1
                elif sat == -1:
                    applicable += 1
                # sat == 0 -> not applicable, don't increment applicable or satisfied

            # avoid division by zero
            if applicable > 0:
                cond_satisfaction = satisfied / applicable
            else:
                cond_satisfaction = float("nan")

            absolute_satisfaction = (satisfied + (iteration - applicable)) / iteration
            percent_applicable = applicable / iteration

            print(f"Axiom ({axiom}) conditional satisfaction: {cond_satisfaction}")
            print(f"Axiom ({axiom}) absolute satisfaction: {absolute_satisfaction}")
            print(f"Axiom ({axiom}) percent applicable: {percent_applicable}")

        return {
            "cond_satisfaction": cond_satisfaction,
            "absolute_satisfaction": absolute_satisfaction,
            "percent_applicable": percent_applicable,
        }

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
