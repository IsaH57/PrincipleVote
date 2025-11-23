import random
import itertools
import copy
from collections import defaultdict

# --- 1. DATA STRUCTURES & UTILS ---

class MockProfile:
    def __init__(self, rankings, num_cands=5):
        self.rankings = rankings
        self.num_cands = num_cands
        self.candidates = list(range(num_cands))
        self.n_voters = len(rankings)
        self.pairwise_matrix = {} 

    def get_pairwise_margin(self, a, b):
        if (a, b) in self.pairwise_matrix: return self.pairwise_matrix[(a, b)]
        a_wins = sum(1 for r in self.rankings if r.index(a) < r.index(b))
        margin = a_wins - (self.n_voters - a_wins)
        self.pairwise_matrix[(a, b)] = margin
        self.pairwise_matrix[(b, a)] = -margin
        return margin
    
    def condorcet_winner(self):
        for a in self.candidates:
            if all(self.get_pairwise_margin(a, b) > 0 for b in self.candidates if a != b):
                return a
        return None

# --- 2. VOTING RULES ---

def rule_plurality(profile):
    scores = {c: 0 for c in profile.candidates}
    for r in profile.rankings: scores[r[0]] += 1
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def rule_borda(profile):
    scores = {c: 0 for c in profile.candidates}
    n = len(profile.candidates)
    for r in profile.rankings:
        for i, c in enumerate(r): scores[c] += (n - 1 - i)
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def rule_copeland(profile):
    scores = {c: 0 for c in profile.candidates}
    for a, b in itertools.combinations(profile.candidates, 2):
        m = profile.get_pairwise_margin(a, b)
        if m > 0: scores[a] += 1
        elif m < 0: scores[b] += 1
        else: scores[a] += 0.5; scores[b] += 0.5
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def get_smith_set(profile):
    cands = set(profile.candidates)
    smith = set(rule_copeland(profile)) 
    while True:
        new_members = set()
        for s in smith:
            for c in cands - smith:
                if profile.get_pairwise_margin(c, s) > 0:
                    new_members.add(c)
        if not new_members: break
        smith.update(new_members)
    return list(smith)

def rule_instant_runoff(profile):
    current_cands = set(profile.candidates)
    while len(current_cands) > 1:
        scores = {c: 0 for c in current_cands}
        for r in profile.rankings:
            for c in r:
                if c in current_cands: scores[c] += 1; break
        min_score = min(scores.values())
        losers = [c for c, s in scores.items() if s == min_score]
        # Deterministic tie-break (remove highest index) to preserve Anonymity/Neutrality in simulation
        if len(losers) < len(current_cands): current_cands -= {max(losers)} 
        else: return list(current_cands)
    return list(current_cands)

def rule_baldwin(profile):
    current_cands = set(profile.candidates)
    while len(current_cands) > 1:
        scores = {c: 0 for c in current_cands}
        n = len(current_cands)
        for r in profile.rankings:
            filtered_r = [c for c in r if c in current_cands]
            for i, c in enumerate(filtered_r): scores[c] += (n - 1 - i)
        min_score = min(scores.values())
        losers = [c for c, s in scores.items() if s == min_score]
        if len(losers) < len(current_cands): current_cands -= set(losers)
        else: return list(current_cands)
    return list(current_cands)

def rule_kemeny_young(profile):
    max_score = -1
    winners = set()
    margins = {}
    for a in profile.candidates:
        for b in profile.candidates:
            if a != b: margins[(a,b)] = sum(1 for r in profile.rankings if r.index(a) < r.index(b))
    
    for perm in itertools.permutations(profile.candidates):
        score = 0
        for i in range(len(perm)):
            for j in range(i+1, len(perm)):
                score += margins.get((perm[i], perm[j]), 0)
        if score > max_score:
            max_score = score
            winners = {perm[0]}
        elif score == max_score:
            winners.add(perm[0])
    return list(winners)

def rule_stable_voting_proxy(profile):
    smith = get_smith_set(profile)
    if len(smith) == 1: return smith
    scores = {}
    for c in smith:
        max_loss = -9999
        for opp in smith:
            if c == opp: continue
            m = profile.get_pairwise_margin(opp, c)
            if m > max_loss: max_loss = m
        scores[c] = max_loss
    min_max_loss = min(scores.values())
    return [c for c, s in scores.items() if s == min_max_loss]

def rule_minimax(profile):
    max_defeats = {}
    for a in profile.candidates:
        worst_margin = -9999
        for b in profile.candidates:
            if a == b: continue
            margin_loss = profile.get_pairwise_margin(b, a) 
            if margin_loss > worst_margin: worst_margin = margin_loss
        max_defeats[a] = worst_margin
    min_worst = min(max_defeats.values())
    return [c for c, s in max_defeats.items() if s == min_worst]

# --- 3. AXIOM CHECKERS ---

def check_anonymity(profile, winners, rule_func):
    # Shuffle voters, result should be same
    rankings = copy.deepcopy(profile.rankings)
    random.shuffle(rankings)
    new_prof = MockProfile(rankings)
    new_winners = set(rule_func(new_prof))
    return 1 if set(winners) == new_winners else 0

def check_neutrality(profile, winners, rule_func):
    # Permute candidates
    perm = list(profile.candidates)
    random.shuffle(perm)
    perm_map = {old: new for old, new in zip(profile.candidates, perm)}
    
    new_rankings = []
    for r in profile.rankings:
        new_rankings.append([perm_map[c] for c in r])
    
    new_prof = MockProfile(new_rankings)
    new_winners = set(rule_func(new_prof))
    expected_winners = {perm_map[w] for w in winners}
    
    return 1 if new_winners == expected_winners else 0

def check_condorcet(profile, winners):
    cw = profile.condorcet_winner()
    if cw is None: return 1 # Vacuously satisfied
    return 1 if set(winners) == {cw} else 0

def check_pareto(profile, winners):
    winner_set = set(winners)
    for w in winner_set:
        for opp in profile.candidates:
            if w == opp: continue
            if all(r.index(opp) < r.index(w) for r in profile.rankings): return 0
    return 1

def check_independence_fast(profile, winners, rule_func, n_samples=6):
    winner_set = set(winners)
    if not winner_set: return 1
    winner = list(winner_set)[0]
    losers = [c for c in profile.candidates if c != winner]
    
    for loser in losers:
        for _ in range(n_samples):
            new_rankings = []
            for r in profile.rankings:
                base = list(profile.candidates)
                random.shuffle(base)
                w_idx, l_idx = base.index(winner), base.index(loser)
                if (r.index(winner) < r.index(loser)) != (w_idx < l_idx):
                    base[w_idx], base[l_idx] = base[l_idx], base[w_idx]
                new_rankings.append(base)
            
            new_w = rule_func(MockProfile(new_rankings))
            if loser in new_w: return 0
    return 1

# --- 4. SYNTHESIS ANALYSIS ---

def run_analysis():
    print(f"--- SYNTHESIS ANALYSIS (N=55, M=5, Trials=250) ---")
    print("Methods: Plurality, Borda, Copeland, TopCycle, Stable, IRV, Baldwin, Kemeny, Minimax")
    
    methods = [
        ("Plurality", rule_plurality),
        ("Borda", rule_borda),
        ("Copeland", rule_copeland),
        ("Top Cycle", get_smith_set), 
        ("Stable Voting", rule_stable_voting_proxy),
        ("Instant Runoff", rule_instant_runoff),
        ("Baldwin", rule_baldwin),
        ("Kemeny-Young", rule_kemeny_young),
        ("Minimax", rule_minimax)
    ]
    
    N = 55
    M = 5
    TRIALS = 500 
    
    # Storage for aggregated stats: [Anon, Neut, Par, Cond, Indep]
    method_scores = {name: [0, 0, 0, 0, 0] for name, _ in methods}
    oracle_scores = [0, 0, 0, 0, 0] 
    
    for t in range(TRIALS):
        if t % 50 == 0: print(f"Processing trial {t}...")
        rankings = [random.sample(range(M), M) for _ in range(N)]
        prof = MockProfile(rankings, M)
        
        # Track best result for THIS profile
        # We want to pick the result that maximizes the sum of axioms satisfied.
        # Tie-breaker priority: Indep > Condorcet > Anon > Neut > Pareto
        best_outcome = [-1, -1, -1, -1, -1] 
        best_score = -1
        
        for name, func in methods:
            w = func(prof)
            
            # Check Axioms
            a = check_anonymity(prof, w, func)
            n = check_neutrality(prof, w, func)
            p = check_pareto(prof, w)
            c = check_condorcet(prof, w)
            i = check_independence_fast(prof, w, func, n_samples=6)
            
            current_outcome = [a, n, p, c, i]
            
            # Update Method Stats
            for idx in range(5):
                method_scores[name][idx] += current_outcome[idx]
            
            # Oracle Logic
            score_sum = sum(current_outcome)
            
            if score_sum > best_score:
                best_score = score_sum
                best_outcome = current_outcome
            elif score_sum == best_score:
                # Break ties: Prioritize Independence (index 4), then Condorcet (index 3)
                if current_outcome[4] > best_outcome[4]:
                    best_outcome = current_outcome
                elif current_outcome[4] == best_outcome[4] and current_outcome[3] > best_outcome[3]:
                    best_outcome = current_outcome
                    
        # Add best result to Oracle
        for idx in range(5):
            oracle_scores[idx] += best_outcome[idx]

    # --- REPORTING ---
    print("\n" + "="*85)
    headers = ["Anon", "Neut", "Pareto", "Condorcet", "Indep(IIA)"]
    print(f"{'Voting Rule':<20} | {' | '.join([f'{h:<9}' for h in headers])}")
    print("-" * 85)
    
    # Print Standard Methods
    for name, scores in method_scores.items():
        pcts = [s / TRIALS * 100 for s in scores]
        print(f"{name:<20} | {pcts[0]:9.1f} | {pcts[1]:9.1f} | {pcts[2]:9.1f} | {pcts[3]:9.1f} | {pcts[4]:9.1f}")
        
    print("-" * 85)
    
    # Print Oracle
    o_pcts = [s / TRIALS * 100 for s in oracle_scores]
    print(f"{'THEORETICAL ORACLE':<20} | {o_pcts[0]:9.1f} | {o_pcts[1]:9.1f} | {o_pcts[2]:9.1f} | {o_pcts[3]:9.1f} | {o_pcts[4]:9.1f}")
    print("=" * 85)
    print("Interpretation:")
    print("- Anon/Neut/Pareto should be ~100% for all valid rules.")
    print("- The 'Oracle' Indep score represents the upper bound.")
    print("- If Oracle Indep > Best Rule Indep, a hybrid rule could theoretically exist.")

if __name__ == "__main__":
    run_analysis()