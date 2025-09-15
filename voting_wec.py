"""VotingWEC: An implementation of the Voting with Embedding Classifier (WEC) model for preference voting."""
import pref_voting
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
import numpy as np
from pref_voting.generate_profiles import generate_profile

from axioms import set_training_axiom
from synth_data import SynthData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import List


class VotingWEC(nn.Module):
    """VotingWEC: A Voting Classifier using Word Embeddings and a Feedforward Neural Network.

        Attributes:
            name (str): Name of the model.
            word_to_idx (dict): Mapping from ranking strings to indices.
            vocab_size (int): Size of the vocabulary.
            max_cand (int): Maximum number of candidates.
            max_vot (int): Maximum number of voters.
            corpus_size (int): Size of the synthetic corpus for pre-training embeddings.
            embed_dim (int): Dimension of the word embeddings.
            window_size (int): Context window size for Word2Vec.
    """

    def __init__(self, max_candidates: int, max_voters: int, corpus_size: int, embed_dim: int, window_size: int,
                 pre_trained_embeddings=None):
        """Initializes the VotingWEC model.

        Args:
            max_candidates (int): Maximum number of candidate alternatives.
            max_voters (int): Maximum number of voters.
            corpus_size (int): Size of the synthetic corpus for pre-training embeddings.
            embed_dim (int): Dimension of the word embeddings.
            window_size (int): Context window size for Word2Vec.
            pre_trained_embeddings (torch.Tensor, optional): Pre-trained embeddings to use instead of training new ones.
        """
        super().__init__()
        self.name = "wec"

        self.word_to_idx = None
        self.vocab_size = None
        self.max_cand = max_candidates
        self.max_vot = max_voters
        self.corpus_size = corpus_size
        self.embed_dim = embed_dim
        self.window_size = window_size

        # TODO given embedding must match the number of alternatives. add check for this
        if pre_trained_embeddings is not None:
            # use pretrained embeddings if available
            self.embeddings = nn.Embedding.from_pretrained(
                pre_trained_embeddings,
                freeze=False,
            )
        else:
            self.pre_train_embeddings(corpus_size=self.corpus_size, embedding_dim=self.embed_dim,
                                      window=self.window_size, prob_model="IC")

        # averaging layer
        self.avg = nn.AvgPool1d(kernel_size=self.embed_dim, stride=1)

        # 3 liner layers with 128 neurons each
        self.fc1 = nn.Linear(self.embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, self.max_cand)  # TODO check if extra output layer is needed

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)

    def pre_train_embeddings(self, corpus_size: int, embedding_dim: int, window: int,
                             prob_model="IC") -> (torch.Tensor, dict):
        """Pre-train embeddings using Word2Vec on a synthetic corpus.

        Args:
            corpus_size (int): Number of profiles to generate for training
            embedding_dim (int): Dimension of the embedding
            window (int): Context window size for Word2Vec
            prob_model (str): Probability model to use for generating profiles.

        Returns:
            (torch.Tensor, dict): Embedding matrix and mapping from words to indices.
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
            num_candidates = np.random.randint(1, self.max_cand + 1)
            num_voters = np.random.randint(1, self.max_vot + 1)

            # generate profile
            profile = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            # make profile to sentence
            sentence = []
            for ranking, count in zip(profile.rankings, profile.counts):
                ranking_str = ''.join(map(str, ranking))
                vocabulary.add(ranking_str)

                # add ranking to sentence multiple times according to count #TODO check if this is correct
                for _ in range(count):
                    sentence.append(ranking_str)

            # pad the sentence to the maximum length
            while len(sentence) < embedding_dim:
                sentence.append('<pad>')

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
            while len(sequence) < self.embed_dim:
                sequence.append(self.word_to_idx["<pad>"])

            sequences.append(sequence[:self.embed_dim])

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

    def train_model(self, num_steps, train_loader, seed=42, plot=False, axiom: str = "default"):
        """
        Trains the VotingWEC model using Cosine Annealing with Warm Restarts scheduler.

        Args:
            num_steps(int): Number of training steps.
            train_loader (DataLoader): DataLoader for training data.
            seed (int): Random seed for reproducibility.
            plot (bool): Whether to plot training loss.
            axiom (str): Axiom to enforce during training.

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
                loss += set_training_axiom(self, batch_x, batch_x, axiom)
                loss.backward()
                optimizer.step()
                scheduler.step()  # update learning rate

                # Track loss and steps
                losses.append(loss.item())
                steps.append(step_count)

                step_count += 1

                if step_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {step_count}/{num_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if plot:
            self.plot_training_loss(steps, losses)

        return self

    def predict(self, x: List[pref_voting.profiles.Profile]) -> (torch.Tensor, torch.Tensor):
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
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            winner_mask = probs > 0.5
            return winner_mask.int(), probs

    def evaluate_model_hard(self, x_test, y_test) -> float:
        """Evaluates the WEC model using hard accuracy, meaning that the predicted set of winners must match exactly with the true set: F(P)=S.

        Args:
            x_test (torch.Tensor): Test input tensor of shape (num_samples, input_size).
            y_test (torch.Tensor): True labels tensor of shape (num_samples, num_classes).

        Returns:
            float: Hard accuracy as a fraction of correct predictions.
        """
        self.eval()
        hard_correct = 0

        with torch.no_grad():
            outputs = self(x_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()

            for pred_row, true_row in zip(predicted, y_test.int()):
                if torch.equal(pred_row, true_row):
                    hard_correct += 1

        accuracy = hard_correct / len(y_test)
        print(f"WEC Hard Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_model_soft(self, x_test, y_test) -> float:
        """Evaluates the WEC model using soft accuracy, meaning that there is at least one overlap between predicted winners and true winners: F(P) ⊆ S.

        Args:
            x_test (torch.Tensor): Test input tensor of shape (num_samples, input_size).
            y_test (torch.Tensor): True labels tensor of shape (num_samples, num_classes).

        Returns:
            float: Soft accuracy as a fraction of correct predictions.
        """
        self.eval()
        total_score = 0.0

        with torch.no_grad():
            outputs = self(x_test)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            y_test_int = y_test.int()

            for pred_row, true_row in zip(predicted, y_test_int):
                true_winners = (true_row == 1).nonzero(as_tuple=True)[0]
                if len(true_winners) == 0:
                    continue  # skip edge cases with no winners
                correctly_predicted = (pred_row[true_winners] == 1).sum().item()
                total_score += correctly_predicted / len(true_winners)

        accuracy = total_score / len(y_test)
        print(f"WEC Soft Accuracy: {accuracy:.4f}")
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
