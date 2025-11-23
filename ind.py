import random
import itertools
import copy

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

# --- 2. STANDARD VOTING RULES ---

def rule_plurality(profile):
    scores = {c: 0 for c in profile.candidates}
    for r in profile.rankings:
        scores[r[0]] += 1
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def rule_borda(profile):
    scores = {c: 0 for c in profile.candidates}
    n = len(profile.candidates)
    for r in profile.rankings:
        for i, c in enumerate(r):
            scores[c] += (n - 1 - i)
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def rule_anti_plurality(profile):
    scores = {c: 0 for c in profile.candidates}
    for r in profile.rankings:
        scores[r[-1]] += 1 
    min_s = min(scores.values())
    return [c for c, s in scores.items() if s == min_s]

def rule_copeland(profile):
    scores = {c: 0 for c in profile.candidates}
    for a, b in itertools.combinations(profile.candidates, 2):
        m = profile.get_pairwise_margin(a, b)
        if m > 0: scores[a] += 1
        elif m < 0: scores[b] += 1
        else: scores[a] += 0.5; scores[b] += 0.5
    max_s = max(scores.values())
    return [c for c, s in scores.items() if s == max_s]

def rule_llull(profile):
    scores = {c: 0 for c in profile.candidates}
    for a, b in itertools.combinations(profile.candidates, 2):
        m = profile.get_pairwise_margin(a, b)
        if m >= 0: scores[a] += 1
        if m <= 0: scores[b] += 1
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

def rule_uncovered_set(profile):
    uncovered = []
    cands = profile.candidates
    for b in cands:
        is_covered = False
        for a in cands:
            if a == b: continue
            if profile.get_pairwise_margin(a, b) > 0: 
                a_covers_b = True
                for c in cands:
                    if c == a or c == b: continue
                    if profile.get_pairwise_margin(b, c) > 0:
                        if not profile.get_pairwise_margin(a, c) > 0:
                            a_covers_b = False; break
                if a_covers_b: is_covered = True; break
        if not is_covered: uncovered.append(b)
    return uncovered

def rule_blacks(profile):
    cw = profile.condorcet_winner()
    if cw is not None: return [cw]
    return rule_borda(profile)

def rule_instant_runoff(profile):
    current_cands = set(profile.candidates)
    while len(current_cands) > 1:
        scores = {c: 0 for c in current_cands}
        for r in profile.rankings:
            for c in r:
                if c in current_cands: scores[c] += 1; break
        min_score = min(scores.values())
        losers = [c for c, s in scores.items() if s == min_score]
        if len(losers) < len(current_cands): current_cands -= {max(losers)} # Deterministic tie-break
        else: return list(current_cands)
    return list(current_cands)

def rule_plurality_runoff(profile):
    scores = {c: 0 for c in profile.candidates}
    for r in profile.rankings: scores[r[0]] += 1
    sorted_cands = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    top2 = sorted_cands[:2]
    if len(top2) < 2: return top2
    a, b = top2[0], top2[1]
    if profile.get_pairwise_margin(a, b) > 0: return [a]
    elif profile.get_pairwise_margin(b, a) > 0: return [b]
    return [a, b]

def rule_coombs(profile):
    current_cands = set(profile.candidates)
    while len(current_cands) > 1:
        last_scores = {c: 0 for c in current_cands}
        for r in profile.rankings:
            for c in reversed(r):
                if c in current_cands: last_scores[c] += 1; break
        max_last = max(last_scores.values())
        losers = [c for c, s in last_scores.items() if s == max_last]
        if len(losers) < len(current_cands): current_cands -= set(losers)
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

def rule_nanson(profile):
    current_cands = set(profile.candidates)
    while len(current_cands) > 1:
        scores = {c: 0 for c in current_cands}
        n = len(current_cands)
        for r in profile.rankings:
            filtered_r = [c for c in r if c in current_cands]
            for i, c in enumerate(filtered_r): scores[c] += (n - 1 - i)
        avg_score = sum(scores.values()) / len(current_cands)
        survivors = {c for c, s in scores.items() if s > avg_score}
        if not survivors or len(survivors) == len(current_cands): # Fallback
             min_s = min(scores.values())
             survivors = {c for c, s in scores.items() if s > min_s}
             if not survivors: return list(current_cands)
        current_cands = survivors
    return list(current_cands)

def rule_kemeny_young(profile):
    # Brute force Kemeny for M=5
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

def rule_stable_voting(profile):
    # Proxy using Minimax within Smith Set
    smith = get_smith_set(profile)
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
    # Helper for the God Algorithm
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

# --- 3. THE GOD ALGORITHM (Maximizing IIA) ---

def rule_god_brute_force(profile):
    """
    1. Finds Smith Set (to ensure Condorcet & Pareto).
    2. Runs internal simulations on Smith Set members.
    3. Picks the member that is most robust to 'irrelevant' shuffles.
    """
    candidates = get_smith_set(profile)
    if len(candidates) == 1: return candidates
    
    best_candidate = candidates[0]
    best_score = -1
    
    # Heuristic: Minimax is usually the most stable, use it as the 'truth' 
    # for checking if a candidate survives a shuffle.
    
    for cand in candidates:
        stability_score = 0
        loser_pool = [c for c in profile.candidates if c != cand]
        
        # PROBE: 10 random permutations
        for _ in range(1000): 
            loser = random.choice(loser_pool)
            new_rankings = []
            for r in profile.rankings:
                base = list(profile.candidates)
                random.shuffle(base)
                # Enforce order
                try: orig_wins = r.index(cand) < r.index(loser)
                except: orig_wins = True
                
                c_idx, l_idx = base.index(cand), base.index(loser)
                curr_wins = c_idx < l_idx
                if orig_wins != curr_wins: base[c_idx], base[l_idx] = base[l_idx], base[c_idx]
                new_rankings.append(base)
            
            perm_prof = MockProfile(new_rankings)
            
            # Does 'cand' survive? Using Minimax as the 'stable' reference rule for the probe
            if cand in rule_minimax(perm_prof):
                stability_score += 1
                
        if stability_score > best_score:
            best_score = stability_score
            best_candidate = cand
            
    return [best_candidate]

# --- 4. AXIOM CHECKERS ---

def check_anonymity(profile, winners, rule_func):
    rankings = copy.deepcopy(profile.rankings)
    random.shuffle(rankings)
    return 1 if set(winners) == set(rule_func(MockProfile(rankings))) else 0

def check_neutrality(profile, winners, rule_func):
    perm = list(profile.candidates)
    random.shuffle(perm)
    perm_map = {old: new for old, new in zip(profile.candidates, perm)}
    new_rankings = [[perm_map[c] for c in r] for r in profile.rankings]
    new_winners = set(rule_func(MockProfile(new_rankings)))
    expected_winners = {perm_map[w] for w in winners}
    return 1 if new_winners == expected_winners else 0

def check_condorcet(profile, winners):
    cw = profile.condorcet_winner()
    if cw is None: return 1 
    return 1 if set(winners) == {cw} else 0

def check_pareto(profile, winners):
    winner_set = set(winners)
    for w in winner_set:
        for opp in profile.candidates:
            if w == opp: continue
            if all(r.index(opp) < r.index(w) for r in profile.rankings): return 0
    return 1

def check_independence(profile, winners, rule_func, n_samples=5):
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
            
            # VIOLATION CHECK
            new_w = rule_func(MockProfile(new_rankings))
            if loser in new_w: return 0
    return 1

# --- 5. MAIN ---

def run_comparison():
    print(f"--- SIMULATION (N=55, M=5, Trials=200) ---")
    print(f"{'Method':<20} | {'Anon':<5} | {'Neut':<5} | {'Par':<5} | {'Cond':<5} | {'Indep (IIA)':<10}")
    print("-" * 75)
    
    methods = [
        ("Plurality", rule_plurality),
        ("Borda", rule_borda),
        ("Anti-Plurality", rule_anti_plurality),
        ("Copeland", rule_copeland),
        ("Llull", rule_llull),
        ("Uncovered Set", rule_uncovered_set),
        ("Top Cycle", get_smith_set), 
        ("Stable Voting", rule_stable_voting),
        ("Blacks", rule_blacks),
        ("Instant Runoff", rule_instant_runoff),
        ("Plurality Runoff", rule_plurality_runoff),
        ("Coombs", rule_coombs),
        ("Baldwin", rule_baldwin),
        ("Weak Nanson", rule_nanson),
        ("Kemeny-Young", rule_kemeny_young),
        ("GOD ALGO (Brute)", rule_god_brute_force)
    ]
    
    N = 55
    M = 5
    TRIALS = 500 # Reduced for Brute Force speed
    
    for name, func in methods:
        stats = [0, 0, 0, 0, 0] # Anon, Neut, Par, Cond, Ind
        
        for _ in range(TRIALS):
            rankings = [random.sample(range(M), M) for _ in range(N)]
            prof = MockProfile(rankings, M)
            w = func(prof)
            
            stats[0] += check_anonymity(prof, w, func)
            stats[1] += check_neutrality(prof, w, func)
            stats[2] += check_pareto(prof, w)
            stats[3] += check_condorcet(prof, w)
            stats[4] += check_independence(prof, w, func, n_samples=8)
            
        stats = [x/TRIALS * 100 for x in stats]
        print(f"{name:<20} | {stats[0]:5.1f} | {stats[1]:5.1f} | {stats[2]:5.1f} | {stats[3]:5.1f} | {stats[4]:5.1f}")

if __name__ == "__main__":
    run_comparison()