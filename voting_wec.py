from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import numpy as np
from matplotlib import pyplot as plt
from pref_voting.generate_profiles import generate_profile
from torch import optim

from synth_data import SynthData


class VotingWEC(nn.Module):
    def __init__(self, num_alternatives, pre_trained_embeddings=None):
        """Initializes the VotingWEC model.

        Args:
            num_alternatives (int): Number of candidate alternatives (classes).
            pre_trained_embeddings (torch.Tensor, optional): Pre-trained embeddings to use. If None, train embeddings from scratch.
        Attributes:
            word_to_idx (dict): Mapping from ranking strings to indices.
            vocab_size (int): Size of the vocabulary.
            num_alternatives (int): Number of candidate alternatives.
            embeddings (nn.Embedding): Embedding layer for word embeddings.
            avg (nn.AvgPool2d): Averaging layer to pool over voters.
            fc1, fc2, fc3, out (nn.Linear): Fully connected layers for classification.
            criterion (nn.Module): Loss function used for training.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        """
        super().__init__()
        self.word_to_idx = None
        self.vocab_size = None
        self.num_alternatives = num_alternatives

        if pre_trained_embeddings is not None:
            # use pretrained embeddings if available
            self.embeddings = nn.Embedding.from_pretrained(
                pre_trained_embeddings,
                freeze=False,
            )
        else:
            self.pre_train_embeddings(corpus_size=100000, embedding_dim=200, window=7, prob_model="IC")

        # averaging layer
        self.avg = nn.AvgPool1d(kernel_size=200, stride=1)

        # 3 liner layers with 128 neurons each
        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_alternatives)

        self.criterion = nn.BCEWithLogitsLoss()  # TODO check if loss function is correct. paper uses CrossEntropyLoss
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the VotingWEC model.
        Args:
            x (List[pref_voting.profiles.Profile]): List of pref_voting.Profile objects.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_alternatives).
        """
        batch_size = len(x)

        # convert profiles to sequences of indices
        sequences = []
        for prof in x:
            sequence = []
            for ranking, count in zip(prof.rankings, prof.counts):
                ranking_str = ''.join(map(str, ranking))
                for _ in range(count):
                    if ranking_str in self.word_to_idx:
                        sequence.append(self.word_to_idx[ranking_str])
                    else:
                        sequence.append(self.word_to_idx["<unk>"])

            # padding to ensure all sequences have the same length
            while len(sequence) < 200:  # embedding dimension
                sequence.append(self.word_to_idx["<pad>"])

            sequences.append(sequence[:200])

        # convert sequences to tensor
        input_tensor = torch.tensor(sequences, dtype=torch.long)

        # Embedding layer
        embedded = self.embeddings(input_tensor)  # [batch_size, seq_len, embed_dim]

        # Averaging Layer (average over voters)
        x = self.avg(embedded.permute(0, 2, 1)).squeeze(-1)  # [batch_size, embed_dim]

        # Linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x

    def pre_train_embeddings(self, corpus_size=100000, embedding_dim=200, window=8, prob_model="IC"):
        """
        Pre-train embeddings using Word2Vec on a synthetic corpus.

        Args:
            corpus_size (int): Number of profiles to generate for training
            embedding_dim (int): Dimension of the embedding
            window (int): Context window size for Word2Vec
            prob_model (str): Probability model to use for generating profiles.

        Returns:
            (torch.Tensor, dict): Embeddings matrix and mapping from words to indices.
        """
        if prob_model not in SynthData.SUPPORTED_PROB_MODELS:
            raise ValueError(f"Probability model not supported: {prob_model}")

        print(f"Create corpus of {corpus_size} profiles with {prob_model}...")

        # create empty corpus and vocabulary
        corpus = []
        vocabulary = set(["<pad>", "<unk>"])  # special tokens for padding and unknown words

        # Generate profiles
        for i in range(corpus_size):
            # set number of candidates and voters
            num_candidates = self.num_alternatives
            num_voters = np.random.randint(1, 20)  # random number of voters between 1 and 20

            # generate profile
            profile = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            # make profile to sentence
            sentence = []
            for ranking, count in zip(profile.rankings, profile.counts):
                ranking_str = ''.join(map(str, ranking))
                vocabulary.add(ranking_str)

                # add ranking to sentence multiple times according to count
                for _ in range(count):
                    sentence.append(ranking_str)

            corpus.append(sentence)

            if i % 10000 == 0 and i > 0:
                print(f"{i} Profiles generated...")

        print(f"Vocab size: {len(vocabulary)}")

        print("Train Word2Vec model...")
        model = Word2Vec(
            sentences=corpus,
            vector_size=embedding_dim,
            window=window,
            min_count=1,
            workers=4,
            sg=1
        )

        # create embedding matrix and word_to_idx mapping
        embed = torch.zeros(len(vocabulary), embedding_dim)
        word_to_idx = {}

        # Set special tokens
        word_to_idx["<pad>"] = 0
        word_to_idx["<unk>"] = 1

        # Fill embedding matrix with pre-trained vectors
        idx = 2
        for word in vocabulary:
            if word not in ["<pad>", "<unk>"]:
                if word in model.wv:
                    embed[idx] = torch.FloatTensor(model.wv[word].copy())

                word_to_idx[word] = idx
                idx += 1

        # Set special tokens in the embedding matrix
        embed[0] = torch.zeros(embedding_dim)  # <pad> as zero vector
        embed[1] = torch.randn(embedding_dim)  # <unk> as random vector

        # create nn.Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(embed, freeze=False)
        self.vocab_size = len(vocabulary)
        self.word_to_idx = word_to_idx

        print(f"Trained embeddings: {embed.shape}")

        torch.save(embed, "embeddings.pt")
        with open("word_to_idx.json", "w") as f:
            import json
            json.dump(word_to_idx, f)

        return embed, word_to_idx

    def train_model(self, num_steps, train_loader, seed=42, plot=False):
        """
        Trains the VotingWEC model using Cosine Annealing with Warm Restarts scheduler.
        Args:
            num_steps(int): Number of training steps.
            train_loader (DataLoader): DataLoader for training data.
            seed (int): Random seed for reproducibility.
            plot (bool): Whether to plot training loss.
        Returns:
            VotingWEC: The trained model.
        """
        torch.manual_seed(seed)

        self.train()

        # Loss tracking
        losses = []
        steps = []

        optimizer = self.optimizer
        criterion = self.criterion

        # Cosine Annealing scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )

        step_count = 0
        while step_count < num_steps:
            for batch_x, batch_y in train_loader:
                if step_count >= num_steps:
                    break

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()  # update learning rate

                # Track loss and steps
                losses.append(loss.item())
                steps.append(step_count)

                step_count += 1

                if step_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Schritt {step_count}/{num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if plot:
            self.plot_training_loss(steps, losses)

        return self

    def predict(self, x):
        """
        Predicts the winners for the given input profiles.
        Args:
            x (List[pref_voting.profiles.Profile]): List of pref_voting.Profile objects.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - winners (torch.Tensor): Binary mask indicating winners.
                - probs (torch.Tensor): Probabilities of each candidate being the winner.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probs = torch.sigmoid(outputs)
            winners = probs > 0.5

        return winners, probs

    def evaluate_model(self, x_test, y_test):
        """
        Evaluates the model on the provided test data.
        Args:
            x_test (List[pref_voting.profiles.Profile]): List of pref_voting.Profile objects for testing.
            y_test (torch.Tensor): Test target labels of shape (num_samples, num_alternatives).
        Returns:
            float: Accuracy of the model on the test data.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x_test)
            predicted = torch.sigmoid(outputs) > 0.5
            accuracy = (predicted == y_test).float().mean()
            print(f"WEC Accuracy: {accuracy:.4f}")
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
