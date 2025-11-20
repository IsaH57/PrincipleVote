import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
Plackett-Luce Model Implementation for Pairwise Comparisons (Bradley-Terry Model)

Scientific Context:
The Plackett-Luce model is a probability distribution over rankings of items. 
When applied to pairwise comparisons (rankings of size 2), it reduces to the Bradley-Terry model.
The fundamental assumption is that each item $i$ has a latent strength or utility $s_i$.
The probability that item $i$ beats item $j$ is given by the softmax function:

    P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j)) = 1 / (1 + exp(s_j - s_i))

This is equivalent to a logistic function of the score difference.

Optimization Objective:
We wish to maximize the likelihood of the observed pairwise comparisons.
Equivalently, we minimize the Negative Log-Likelihood (NLL).
For a single observation where item $w$ (winner) beats item $l$ (loser):

    Loss = -log P(w > l) 
         = -log (exp(s_w) / (exp(s_w) + exp(s_l)))
         = -s_w + log(exp(s_w) + exp(s_l))

Practical Considerations:
1. **Online Learning**: The user requested "runningly update". This implies Stochastic Gradient Descent (SGD).
   We update the model parameters after each observation (or small batch).
2. **Identifiability**: The probabilities depend only on the difference of scores. 
   Adding a constant to all scores doesn't change predictions. In the tabular case, we might want to fix one item's score or center them, but for SGD it usually drifts harmlessly or can be regularized.
3. **Exploration vs Exploitation**: This implementation focuses on *estimation* (learning values). 
   If this were a bandit setting, we would need an exploration strategy (e.g., Thompson Sampling, UCB). 
   Here we assume the comparisons are provided by an external process.
"""

class TabularPlackettLuce:
    """
    A simple object-based Plackett-Luce model.
    Maintains a lookup table of scores for each unique item ID.
    """
    def __init__(self, learning_rate=0.01, use_virtual_win=True):
        self.scores = {} # Maps item_id -> score (float)
        self.learning_rate = learning_rate
        self.use_virtual_win = use_virtual_win
        
    def get_score(self, item_id):
        """Returns the current score for an item, initializing to 0.0 if new."""
        if item_id not in self.scores:
            self.scores[item_id] = 0.0
        return self.scores[item_id]
    
    def predict_proba(self, item_a, item_b):
        """
        Predicts the probability that item_a beats item_b.
        P(a > b) = 1 / (1 + exp(-(s_a - s_b)))
        """
        s_a = self.get_score(item_a)
        s_b = self.get_score(item_b)
        
        # Logistic function
        # Using numpy's exp to handle potential overflow gracefully if needed, 
        # though scores usually stay within reasonable bounds with small LR.
        try:
            prob = 1.0 / (1.0 + np.exp(s_b - s_a))
        except OverflowError:
            prob = 0.0 if s_b > s_a else 1.0
            
        return prob

    def update(self, winner_id, loser_id):
        """
        Performs a single SGD step based on the observation that winner_id > loser_id.
        
        Gradient Derivation:
        L = -s_w + log(exp(s_w) + exp(s_l))
        dL/ds_w = -1 + exp(s_w)/(exp(s_w) + exp(s_l)) = -1 + P(w>l) = -P(l>w)
        dL/ds_l = exp(s_l)/(exp(s_w) + exp(s_l)) = P(l>w)
        
        Virtual Win Regularization (if enabled):
        We assume each item also beats a virtual item 0 (score 0).
        L_reg = -log P(w > 0) - log P(l > 0)
              = log(1 + exp(-s_w)) + log(1 + exp(-s_l))
        dL_reg/ds_w = -exp(-s_w)/(1 + exp(-s_w)) = -1/(1 + exp(s_w)) = -P(0 > w)
        dL_reg/ds_l = -P(0 > l)
        
        Update rule: s <- s - lr * grad
        s_w <- s_w + lr * (P(l>w) + P(0>w))
        s_l <- s_l - lr * (P(l>w) - P(0>l))  <-- Wait, dL/ds_l is positive P(l>w), dL_reg/ds_l is negative
        Total grad_l = P(l>w) - P(0>l)
        s_l <- s_l - lr * (P(l>w) - P(0>l)) = s_l - lr*P(l>w) + lr*P(0>l)
        """
        s_w = self.get_score(winner_id)
        s_l = self.get_score(loser_id)
        
        # Probability that loser would have won (the "surprise" factor)
        # P(l > w) = 1 / (1 + exp(s_w - s_l))
        prob_loser_wins = 1.0 / (1.0 + np.exp(s_w - s_l))
        
        grad_w = -prob_loser_wins
        grad_l = prob_loser_wins
        
        if self.use_virtual_win:
            # P(0 > w) = 1 / (1 + exp(s_w))
            prob_0_beats_w = 1.0 / (1.0 + np.exp(s_w))
            # P(0 > l) = 1 / (1 + exp(s_l))
            prob_0_beats_l = 1.0 / (1.0 + np.exp(s_l))
            
            grad_w -= prob_0_beats_w
            grad_l -= prob_0_beats_l
        
        # Apply updates
        self.scores[winner_id] -= self.learning_rate * grad_w
        self.scores[loser_id] -= self.learning_rate * grad_l
        
        return {
            "winner_score": self.scores[winner_id],
            "loser_score": self.scores[loser_id],
            "surprise": prob_loser_wins
        }

class NeuralPlackettLuce(nn.Module):
    """
    A neural network based Plackett-Luce model.
    Instead of learning a score per ID, we learn a function f(features) -> score.
    This allows generalization to unseen items based on their attributes.
    """
    def __init__(self, input_dim, hidden_dim=64, learning_rate=0.001, use_virtual_win=True):
        super(NeuralPlackettLuce, self).__init__()
        
        # A simple MLP to map features to a scalar score
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output is a single scalar score
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.use_virtual_win = use_virtual_win
        
    def forward(self, x):
        """Computes the score for a batch of items."""
        return self.network(x)
    
    def predict_proba(self, features_a, features_b):
        """
        Predicts P(a > b) given feature vectors.
        """
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            s_a = self.forward(features_a)
            s_b = self.forward(features_b)
            # Sigmoid of difference is equivalent to the softmax formulation for 2 items
            prob = torch.sigmoid(s_a - s_b)
        return prob.item()
    
    def update(self, winner_features, loser_features):
        """
        Performs a single SGD step.
        winner_features: Tensor of shape (1, input_dim) or (input_dim,)
        loser_features: Tensor of shape (1, input_dim) or (input_dim,)
        """
        self.train() # Set to training mode
        self.optimizer.zero_grad()
        
        # Ensure inputs are 2D tensors (batch_size, input_dim)
        if winner_features.dim() == 1:
            winner_features = winner_features.unsqueeze(0)
        if loser_features.dim() == 1:
            loser_features = loser_features.unsqueeze(0)
            
        s_w = self.forward(winner_features)
        s_l = self.forward(loser_features)
        
        # Loss: -log P(w > l)
        loss = torch.nn.functional.softplus(s_l - s_w)
        
        if self.use_virtual_win:
            # Regularization: Maximize log P(i > virtual_0) => Minimize softplus(-s_i)
            # We apply this to both items involved in the comparison
            reg_loss = torch.nn.functional.softplus(-s_w) + torch.nn.functional.softplus(-s_l)
            loss += reg_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class EMPlackettLuce:
    """
    Expectation-Maximization (EM) algorithm for Plackett-Luce (specifically Bradley-Terry).
    Also known as Hunter's MM algorithm.
    
    This is a BATCH algorithm, unlike the SGD-based online models above.
    It finds the Maximum Likelihood Estimate (MLE) of the scores given the entire history.
    
    Parameters:
    - gamma_i: The "strength" of item i. score_i = log(gamma_i).
    
    Update Rule:
    gamma_i <- W_i / Sum_{j!=i} (N_ij / (gamma_i + gamma_j))
    where W_i is total wins for i, N_ij is total comparisons between i and j.
    
    Virtual Zero Element Trick (Regularization):
    We can add a virtual comparison where every item beats a virtual item 0 (with gamma_0 = 1).
    This adds +1 to W_i and adds a term 1/(gamma_i + 1) to the denominator sum.
    This prevents gammas from exploding and ensures connectivity.
    """
    def __init__(self, use_virtual_win=True):
        self.wins = {} # item_id -> count
        self.adjacency = {} # item_id -> {opponent_id: count}
        self.gammas = {} # item_id -> gamma (positive float)
        self.item_ids = set()
        self.use_virtual_win = use_virtual_win
        
    def add_comparison(self, winner_id, loser_id, **kwargs):
        """
        Add a comparison to the dataset.
        kwargs can contain 'persona_id' or other attributes, currently ignored by the global EM model.
        """
        self.item_ids.add(winner_id)
        self.item_ids.add(loser_id)
        
        # Update Wins
        self.wins[winner_id] = self.wins.get(winner_id, 0) + 1
        if loser_id not in self.wins:
            self.wins[loser_id] = 0 # Ensure key exists
            
        # Update Adjacency (N_ij)
        if winner_id not in self.adjacency: self.adjacency[winner_id] = {}
        if loser_id not in self.adjacency: self.adjacency[loser_id] = {}
        
        self.adjacency[winner_id][loser_id] = self.adjacency[winner_id].get(loser_id, 0) + 1
        self.adjacency[loser_id][winner_id] = self.adjacency[loser_id].get(winner_id, 0) + 1
        
        # Initialize gammas if new
        if winner_id not in self.gammas: self.gammas[winner_id] = 1.0
        if loser_id not in self.gammas: self.gammas[loser_id] = 1.0

    def step(self):
        """Performs one iteration of the MM update."""
        new_gammas = {}
        
        for i in self.item_ids:
            W_i = self.wins.get(i, 0)
            
            # Virtual Win Regularization
            if self.use_virtual_win:
                W_i += 1
                
            denominator = 0.0
            
            # Sum over all j that i has compared with
            if i in self.adjacency:
                for j, n_ij in self.adjacency[i].items():
                    g_i = self.gammas[i]
                    g_j = self.gammas[j]
                    denominator += n_ij / (g_i + g_j)
            
            # Virtual Win Denominator Term (comparison with virtual item 0, gamma_0=1)
            if self.use_virtual_win:
                g_i = self.gammas[i]
                denominator += 1.0 / (g_i + 1.0)
            
            if denominator > 0:
                new_gammas[i] = W_i / denominator
            else:
                # If denominator is 0, it means i has never been compared (or W_i is 0 and logic holds)
                # If W_i > 0 but denominator 0 (impossible if connected), keep old
                new_gammas[i] = self.gammas[i]
        
        # Normalization (Geometric mean = 1 or Sum = 1)
        # Let's use Sum = len(items) so average gamma is 1
        total_gamma = sum(new_gammas.values())
        target_sum = len(new_gammas)
        if total_gamma > 0:
            factor = target_sum / total_gamma
            for i in new_gammas:
                new_gammas[i] *= factor
                
        self.gammas = new_gammas
        
    def fit(self, max_iter=10, tol=1e-4):
        """Runs EM steps until convergence or max_iter."""
        for t in range(max_iter):
            old_gammas = self.gammas.copy()
            self.step()
            
            # Check convergence (max relative change)
            max_diff = 0.0
            for i in self.gammas:
                diff = abs(self.gammas[i] - old_gammas[i])
                max_diff = max(max_diff, diff)
            
            if max_diff < tol:
                break
                
    def get_score(self, item_id):
        """Returns log(gamma) to be comparable with other models' scores."""
        return np.log(self.gammas.get(item_id, 1.0))

    def predict_proba(self, item_a, item_b):
        """P(a > b) = gamma_a / (gamma_a + gamma_b)"""
        g_a = self.gammas.get(item_a, 1.0)
        g_b = self.gammas.get(item_b, 1.0)
        return g_a / (g_a + g_b)

class GeneralizedPlackettLuce(nn.Module):
    """
    Generalized Plackett-Luce Model that learns both Utility (u) and Rationality (beta).
    
    Scientific Context:
    Standard BT model assumes P(i > j) = 1 / (1 + exp(-(u_i - u_j))).
    This model assumes P(i > j | k) = 1 / (1 + exp(-beta_k * (u_i - u_j))).
    
    - u_i: Intrinsic utility of item i.
    - beta_k: Rationality/Temperature of voter k. 
      High beta -> Deterministic (always picks higher utility).
      Low beta -> Noisy/Random.
      
    Optimization:
    - Can use pure SGD for both u and beta.
    - Can use a hybrid approach: MM algorithm for u, SGD for beta (Newman's algorithm).
    """
    def __init__(self, num_items, num_voters, lr=1e-3, sgd_all=False, n_iterations=1, n_sgd_steps=1):
        super(GeneralizedPlackettLuce, self).__init__()
        self.M = num_items
        self.K = num_voters
        self.lr = lr
        self.sgd_all = sgd_all
        self.n_iterations = n_iterations
        self.n_sgd_steps = n_sgd_steps
        
        # Parameters
        # u: Utility vector (log-scores)
        self.u = nn.Parameter(torch.zeros(self.M), requires_grad=True)
        
        # beta: Rationality vector (one per voter)
        # We initialize to 1.0
        self.beta = nn.Parameter(torch.ones(self.K), requires_grad=True)
        
        # Optimizer
        if self.sgd_all:
            self.optimizer = optim.Adam([self.u, self.beta], lr=lr)
        else:
            # If using MM for u, we only need SGD for beta
            # Note: The provided snippet fixes beta[0]=1.0 for identifiability in the loss function
            # We will handle that in the loss calculation.
            self.optimizer = optim.Adam([self.beta], lr=lr)
            
        # Data storage for MM and SGD
        # w[k, i, j] count of k preferring i over j
        self.w = torch.zeros(self.K, self.M, self.M)
        self.pairs_list = [[] for _ in range(self.K)]
        
    def add_comparison(self, winner_id, loser_id, persona_id=0):
        """
        Add a comparison observation.
        persona_id corresponds to the voter index k.
        """
        k = int(persona_id)
        if k >= self.K:
            # Handle case where persona_id might be out of bounds if we didn't set K correctly
            # For now, assume K is correct.
            return
            
        # Update counts for MM
        self.w[k, winner_id, loser_id] += 1
        
        # Update list for SGD
        # Store as [loser, winner] because the loss function expects pairs[:, 1] - pairs[:, 0]
        # The snippet's loss is log(1 + exp(beta * (u[loser] - u[winner])))
        # Wait, let's check the snippet's loss:
        # loss = torch.sum( torch.log(1.0 + torch.exp(beta * (self.u[pairs[:,1]]-self.u[pairs[:,0]]))) )
        # If pairs is [winner, loser], then u[loser] - u[winner] is negative (if winner is better).
        # exp(negative) is small. log(1+small) is small.
        # If pairs is [loser, winner], then u[winner] - u[loser] is positive.
        # exp(positive) is large. log(1+large) is large.
        # We want to MINIMIZE loss.
        # -log P(w > l) = -log (1 / (1 + exp(-beta(u_w - u_l)))) = log(1 + exp(-beta(u_w - u_l)))
        # = log(1 + exp(beta(u_l - u_w)))
        # So we want the term inside exp to be beta * (u_loser - u_winner).
        # If pairs[:, 1] is u_loser and pairs[:, 0] is u_winner, then (u_l - u_w) is correct.
        # So we should store [winner, loser].
        self.pairs_list[k].append([winner_id, loser_id])

    def _loss_func(self, pairs, k):
        # pairs is tensor of shape (N, 2) where each row is [winner, loser]
        # We want log(1 + exp(beta * (u_loser - u_winner)))
        
        u_winner = self.u[pairs[:, 0]]
        u_loser = self.u[pairs[:, 1]]
        
        if k == 0:
            # Fix 0-th trainer's beta=1.0 for scale identifiability
            b = 1.0
        else:
            b = self.beta[k]
            
        # Loss = log(1 + exp(b * (u_l - u_w)))
        loss = torch.sum(torch.log(1.0 + torch.exp(b * (u_loser - u_winner))))
        return loss

    def step(self):
        """Performs one iteration of updates (MM + SGD)."""
        
        # 1. MM Update for Utility (u)
        if not self.sgd_all:
            with torch.no_grad():
                # Newman's MM algorithm update for u
                # This assumes we are maximizing likelihood w.r.t u
                for m in range(self.M):
                    num = 0.0
                    den = 0.0
                    
                    # We iterate over all voters
                    for k in range(self.K):
                        b = self.beta[k] if k != 0 else 1.0
                        u_m = self.u[m]
                        
                        # The update rule from the snippet:
                        # num += b / (exp(b*u_m) + 1) + sum(b * w_kmi * exp(b*u) / (exp(b*u_m) + exp(b*u)))
                        # den += b * exp((b-1)*u_m) * ...
                        
                        # Let's try to implement the exact logic from the snippet provided
                        # Note: The snippet iterates over all items to compute sum. This is O(M^2).
                        # We can optimize using matrix operations if needed, but let's stick to the loop for correctness first.
                        
                        # Precompute exp(b*u) for all items
                        exp_bu = torch.exp(b * self.u)
                        exp_bu_m = torch.exp(b * u_m)
                        
                        # Term 1 of Num
                        # This term looks like a regularization or prior? Or part of the derivative?
                        # In the snippet: num += b/(torch.exp(b*self.u[m]) + 1)
                        # This looks like P(0 > m) if 0 has score 0.
                        num += b / (exp_bu_m + 1.0)
                        
                        # Term 2 of Num: Wins of m over others
                        # w[k, m, :] is vector of wins of m against others
                        # sum( b * w_kmj * exp(b*u_j) / (exp(b*u_m) + exp(b*u_j)) )
                        wins_m = self.w[k, m, :]
                        term2 = torch.sum(b * wins_m * exp_bu / (exp_bu_m + exp_bu))
                        num += term2
                        
                        # Denominator
                        # den += b * exp((b-1)*u_m) * ( ... )
                        # Inside parens:
                        # 1/(exp(b*u_m)+1) + sum( b * exp((b-1)*u_m) * w_kim / (exp(b*u_m) + exp(b*u_i)) )
                        # Wait, the snippet has `w[k,:,m]` which is losses of m (others beating m)
                        losses_m = self.w[k, :, m]
                        
                        # The snippet's denominator logic seems complex.
                        # Let's trust the snippet's formula structure.
                        
                        term_den_1 = 1.0 / (exp_bu_m + 1.0)
                        
                        # The snippet has `b * torch.exp((b-1)*self.u[m])` inside the sum?
                        # "torch.sum(b*torch.exp((b-1)*self.u[m])*self.w[k,:,m]/(torch.exp(b*self.u[m]) + torch.exp(b*self.u) ))"
                        # This simplifies. exp((b-1)u_m) = exp(b*u_m) / exp(u_m).
                        
                        term_den_2 = torch.sum(b * torch.exp((b-1)*u_m) * losses_m / (exp_bu_m + exp_bu))
                        
                        den += b * torch.exp((b-1)*u_m) * (term_den_1 + term_den_2)
                    
                    if den > 1e-9:
                        self.u[m] = torch.log(num / den)
        
        # 2. SGD Update for Beta (and u if sgd_all)
        if self.optimizer is not None:
            # Sample a batch of pairs
            # We need to flatten the list of pairs or sample from each k
            # The snippet samples `sgd_sample_size` per trainer?
            # "sampled_pairs_list = [ self.pairs_list[k][random...] for k in range(K) ]"
            
            batch_loss = 0
            self.optimizer.zero_grad()
            
            for k in range(self.K):
                pairs = self.pairs_list[k]
                if len(pairs) == 0:
                    continue
                
                # Convert to tensor if not already
                # Optimization: Keep them as lists and convert batch only
                n_samples = min(len(pairs), 100) # Use small batch size
                indices = np.random.choice(len(pairs), n_samples, replace=False)
                batch_pairs = torch.tensor([pairs[i] for i in indices], dtype=torch.long)
                
                loss = self._loss_func(batch_pairs, k)
                batch_loss += loss
            
            if isinstance(batch_loss, torch.Tensor):
                batch_loss.backward()
                self.optimizer.step()
        
    def predict_proba(self, item_a, item_b, persona_id=0):
        """
        Predict P(a > b) for a specific persona.
        """
        with torch.no_grad():
            u_a = self.u[item_a]
            u_b = self.u[item_b]
            
            if persona_id == 0:
                b = 1.0
            else:
                b = self.beta[persona_id] if persona_id < self.K else 1.0
                
            # P(a > b) = 1 / (1 + exp(-b * (u_a - u_b)))
            prob = 1.0 / (1.0 + torch.exp(-b * (u_a - u_b)))
            return prob.item()


# --- Example Usage ---
if __name__ == "__main__":
    print("=== Testing Tabular Plackett-Luce ===")
    tabular_model = TabularPlackettLuce(learning_rate=0.1)
    
    # Scenario: Item A is generally better than Item B
    items = ["ItemA", "ItemB"]
    
    # Simulate some data where A beats B 80% of the time
    np.random.seed(42)
    history = []
    for i in range(20):
        if np.random.rand() < 0.8:
            winner, loser = "ItemA", "ItemB"
        else:
            winner, loser = "ItemB", "ItemA"
            
        update_info = tabular_model.update(winner, loser)
        history.append(update_info['surprise'])
        
    print(f"Final Scores: {tabular_model.scores}")
    print(f"Predicted P(A > B): {tabular_model.predict_proba('ItemA', 'ItemB'):.4f}")
    
    print("\n=== Testing Neural Plackett-Luce ===")
    # Scenario: Items have 1 feature. Higher feature value = better item.
    # Feature A = [1.0], Feature B = [-1.0]
    neural_model = NeuralPlackettLuce(input_dim=1, learning_rate=0.05)
    
    feat_a = torch.tensor([1.0])
    feat_b = torch.tensor([-1.0])
    
    print("Initial Prediction P(A > B):", neural_model.predict_proba(feat_a, feat_b))
    
    # Train online
    for i in range(50):
        # A beats B
        loss = neural_model.update(feat_a, feat_b)
        
    print("Final Prediction P(A > B):", neural_model.predict_proba(feat_a, feat_b))
