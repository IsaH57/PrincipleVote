"""VotingMLP: A Multi-Layer Perceptron for Voting-Based Classification"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List

from principle_vote.axioms import set_training_axiom, check_anonymity, check_neutrality, check_condorcet, check_pareto, \
    check_independence
from principle_vote.synth_data import SynthData


class VotingMLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) for Voting-Based Classification.

    Attributes:
            name (str): Name of the model.
            input_size (int): Size of the input layer, calculated as max_candidates² * max_voters.
            max_cand (int): Maximum number of candidates (alternatives).
            max_vot (int): Maximum number of voters.
            layers (nn.Sequential): Sequential container of the MLP layers.
            train_loader (DataLoader): DataLoader for training data.
            criterion (nn.Module): Loss function used for training.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    """

    def __init__(self, train_loader: DataLoader, max_candidates: int, max_voters: int, encoding_type: str = "pairwise"):
        """Initializes the VotingMLP model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            max_candidates (int): Maximum number of candidates to consider.
            max_voters (int): Maximum number of voters to consider.
            encoding_type (str): Type of encoding to use. Either "pairwise" or "onehot". Defaults to "pairwise".
        """
        super(VotingMLP, self).__init__()
        self.name = "mlp"
        self.encoding_type = encoding_type
        self.max_cand = max_candidates
        self.max_vot = max_voters

        # Calculate input size based on encoding type
        if encoding_type == "pairwise":
            # Input size: upper triangle of symmetric pairwise comparison matrix
            # matrix[i,j] = (votes for i over j - votes for j over i) / total [-1,1]
            self.input_size = max_candidates * (max_candidates - 1) // 2
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 512),  # input layer: 512 neurons
                nn.ReLU(),
                nn.Linear(512, 256),  # first hidden layer: 256 neurons
                nn.ReLU(),
                nn.Linear(256, 256),  # second hidden layer: 256 neurons
                nn.ReLU(),
                nn.Linear(256, 128),  # third hidden layer: 256 neurons
                nn.ReLU(),
                nn.Linear(128, self.max_cand),  # output layer: number of candidates
            )
        elif encoding_type == "onehot":
            # Input size: one-hot encoding of rankings
            self.input_size = max_candidates * max_candidates * max_voters
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 128),  # input layer: 128 neurons
                nn.ReLU(),
                nn.Linear(128, 128),  # first hidden layer: 128 neurons
                nn.ReLU(),
                nn.Linear(128, 128),  # second hidden layer: 128 neurons
                nn.ReLU(),
                nn.Linear(128, 128),  # third hidden layer: 128 neurons
                nn.ReLU(),
                nn.Linear(128, self.max_cand),  # output layer: number of candidates
            )
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}. Use 'pairwise' or 'onehot'.")

        self.train_loader = train_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.layers(x)

    def train_model(self, num_steps: int, seed: int = 42, plot: bool = False, axiom: str = "default"):
        """Train the model using the set optimizer with cosine annealing scheduler.

        Args:
            num_steps (int): Number of gradient steps to perform.
            seed (int): Random seed for reproducibility. Defaults to 42.
            plot (bool): Whether to plot the training loss. Defaults to False.
            axiom (str): Axiom to enforce during training. Defaults to "default".
        """
        # Set fixed seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

        optimizer = self.optimizer

        # Cosine Annealing with Warm Restarts scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )

        self.train()
        step_count = 0

        # Loss tracking
        losses = []
        steps = []

        while step_count < num_steps:
            for batch_X, batch_y, prof in self.train_loader:
                if step_count >= num_steps:
                    break

                # Move data to the same device as the model
                batch_X = batch_X.to(next(self.parameters()).device)
                batch_y = batch_y.to(next(self.parameters()).device)

                optimizer.zero_grad()  # Reset gradients
                outputs = self(batch_X)  # Forward pass

                loss = self.criterion(outputs, batch_y.float())
                loss += set_training_axiom(self, batch_X, prof, axiom)

                loss.backward()  # Backward pass
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()  # Update weights
                scheduler.step()  # Update learning rate

                # Track loss and steps
                losses.append(loss.item())
                steps.append(step_count)

                step_count += 1

                if step_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if plot:
            self.plot_training_loss(steps, losses)

    def evaluate_model_hard(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluates the model using hard accuracy, meaning that the predicted set of winners must match exactly with the true set: F(P)=S.

        Args:
            X_test (torch.Tensor): Test input tensor of shape (num_samples, input_size).
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
            X_test (torch.Tensor): Test input tensor of shape (num_samples, input_size).
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

    def evaluate_axiom_satisfaction(self, data: SynthData, axiom: str) -> float:
        """Evaluates the model's axiom satisfaction rate on the test set.

        Args:
            data (SynthData): The synthetic data containing test profiles and winners.
            axiom (str): The axiom to evaluate. Must be one of the keys in AXIOM_SAT_FUNCTIONS.

        Returns:
            float: Axiom satisfaction rate as a fraction of samples satisfying the axiom.
        """
        axiom_fun = self.AXIOM_SAT_FUNCTIONS.get(axiom)

        X_test, y_test = data.get_encoded_mlp()
        profiles = data.get_raw_profiles()

        # Move data to the same device as the model
        device = next(self.parameters()).device
        X_test = X_test.to(device)

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

    def predict(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Predicts the winners for the given input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

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

    def plot_training_loss(self, steps: List[int], losses: List[float]):
        """Plots the Training Loss.

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
        plt.title('MLP Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()
