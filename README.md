# Neural Plackett-Luce Ranking

A minimal, educational implementation of **Neural Plackett-Luce** models in PyTorch.

This repository demonstrates how to learn ranking functions from pairwise comparisons (e.g., "I prefer Restaurant A over Restaurant B"). It contrasts two approaches:

1.  **Tabular Model**: Learns a specific score for each item ID. (Memorization)
2.  **Neural Model**: Learns to map item *features* (text embeddings) to a score. (Generalization)

## The Problem: Zero-Shot Ranking

In many real-world scenarios (e.g., recommending new products, restaurants, or news articles), we constantly encounter **new items** that we have no historical data for (the "Cold Start" problem).

-   A **Tabular** model (like standard Matrix Factorization or Elo) fails here because it needs to see an item ID during training to learn its score.
-   A **Neural** model can predict a score for a new item immediately based on its description/features.

## The Math

We assume the probability that item $i$ beats item $j$ follows the **Bradley-Terry** model (a special case of Plackett-Luce for pairs):

$$ P(i > j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}} = \sigma(s_i - s_j) $$

Where $s_i$ is the "score" or utility of item $i$.

-   **Tabular**: $s_i = \text{Embedding}(i)$
-   **Neural**: $s_i = \text{MLP}(\text{Features}(i))$

We minimize the Negative Log Likelihood:
$$ \mathcal{L} = -\log P(\text{winner} > \text{loser}) = \text{softplus}(s_{\text{loser}} - s_{\text{winner}}) $$

## Quick Start

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the training demo**:
    ```bash
    python train.py
    ```

## Expected Output

You will see the training progress. Notice how the **Tabular** model achieves high accuracy on the *Training* set (memorizing the specific items), but the **Neural** model is the only one that can rank the *Test* sets (unseen items from different years).

```
Epoch 1/5 | Loss Tab: 0.676 | Loss Neu: 0.677 | Train Rho (Tab/Neu): 0.96/0.93 | Test 2023: 0.63 | Test 2025: 0.49
...
```

## RLHF (DPO vs PPO) Experiment

We include scripts to fine-tune a small LLM (GPT-2) using both **Direct Preference Optimization (DPO)** and **Proximal Policy Optimization (PPO)**.

### DPO (Direct Preference Optimization)
Aligns the model directly from pairwise data without a separate reward model.
```bash
python train_dpo.py
```

### PPO (Proximal Policy Optimization)
The classical RLHF pipeline:
1.  Train a **Reward Model** on pairwise comparisons.
2.  Optimize the **Policy** using PPO to maximize the reward.
```bash
python train_ppo.py
```

This allows for a direct comparison of the two methods on the same dataset.

## Project Structure

-   `model.py`: Clean PyTorch implementations of TabularPL and NeuralPL.
-   `data.py`: Handles data loading and synthetic preference generation.
-   `train.py`: Main training loop for NeuralPL.
-   `train_dpo.py`: DPO fine-tuning script.
-   `train_ppo.py`: PPO fine-tuning script (from scratch).
-   `data/`: Contains the restaurant datasets.

## License

MIT
