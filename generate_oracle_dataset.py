"""
Generate a dataset of oracle winner predictions from voting profiles.
Each profile has rankings from voters, and we compute the "oracle winner" 
as the alternative that maximizes axiom satisfaction across multiple voting rules.

Output: CSV with columns for profile features, pairwise comparisons, and oracle winner.
"""

import random
import itertools
import json
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from dataclasses import dataclass

# --- 1. VOTING PROFILE & UTILS ---

class VotingProfile:
    def __init__(self, rankings: List[List[int]], num_cands: int = 5):
        self.rankings = rankings
        self.num_cands = num_cands
        self.candidates = list(range(num_cands))
        self.n_voters = len(rankings)
        self.pairwise_matrix = {}
        
    def get_pairwise_margin(self, a: int, b: int) -> int:
        """Return margin of a over b (positive if a preferred)."""
        if (a, b) in self.pairwise_matrix:
            return self.pairwise_matrix[(a, b)]
        a_wins = sum(1 for r in self.rankings if r.index(a) < r.index(b))
        margin = a_wins - (self.n_voters - a_wins)
        self.pairwise_matrix[(a, b)] = margin
        self.pairwise_matrix[(b, a)] = -margin
        return margin
    
    def condorcet_winner(self) -> int:
        """Return Condorcet winner if exists, else -1."""
        for a in self.candidates:
            if all(self.get_pairwise_margin(a, b) > 0 for b in self.candidates if a != b):
                return a
        return -1
    
    def get_plurality_scores(self) -> Dict[int, int]:
        """Count first-place votes."""
        scores = {c: 0 for c in self.candidates}
        for r in self.rankings:
            scores[r[0]] += 1
        return scores
    
    def get_borda_scores(self) -> Dict[int, float]:
        """Compute Borda scores."""
        scores = {c: 0 for c in self.candidates}
        n = len(self.candidates)
        for r in self.rankings:
            for i, c in enumerate(r):
                scores[c] += (n - 1 - i)
        return scores
    
    def get_copeland_scores(self) -> Dict[int, float]:
        """Compute Copeland scores (wins - losses in pairwise)."""
        scores = {c: 0 for c in self.candidates}
        for a, b in itertools.combinations(self.candidates, 2):
            m = self.get_pairwise_margin(a, b)
            if m > 0:
                scores[a] += 1
            elif m < 0:
                scores[b] += 1
            else:
                scores[a] += 0.5
                scores[b] += 0.5
        return scores
    
    def check_pareto_dominance(self, a: int, b: int) -> bool:
        """Return True if a Pareto dominates b."""
        a_better = 0
        b_better = 0
        for r in self.rankings:
            if r.index(a) < r.index(b):
                a_better += 1
            else:
                b_better += 1
        return a_better == self.n_voters and b_better == 0

# --- 2. SIMPLIFIED VOTING RULES ---

def get_winners(scores: Dict[int, float]) -> List[int]:
    """Return all candidates with max score."""
    max_score = max(scores.values())
    return [c for c, s in scores.items() if s == max_score]

def rule_plurality(profile: VotingProfile) -> List[int]:
    return get_winners(profile.get_plurality_scores())

def rule_borda(profile: VotingProfile) -> List[int]:
    return get_winners(profile.get_borda_scores())

def rule_copeland(profile: VotingProfile) -> List[int]:
    return get_winners(profile.get_copeland_scores())

# --- 3. AXIOM CHECKS ---

def check_condorcet(profile: VotingProfile, winners: List[int]) -> bool:
    """Check if Condorcet winner is selected (if exists)."""
    cw = profile.condorcet_winner()
    if cw == -1:
        return True  # No CW exists, axiom vacuously satisfied
    return cw in winners

def check_pareto(profile: VotingProfile, winners: List[int]) -> bool:
    """Check if no winner is Pareto dominated by a non-winner."""
    for w in winners:
        for c in profile.candidates:
            if c == w:
                continue
            if profile.check_pareto_dominance(c, w):
                return False  # Non-winner dominates winner
    return True

# --- 4. ORACLE WINNER COMPUTATION ---

@dataclass
class OracleResult:
    winner: int
    axiom_scores: Dict[str, float]  # Per rule
    best_score: int
    all_rule_winners: Dict[str, List[int]]

def compute_oracle_winner(profile: VotingProfile) -> OracleResult:
    """
    Compute oracle winner as the alternative that appears most frequently
    as winner across multiple voting rules, with tie-breaking by axiom satisfaction.
    """
    rules = {
        'plurality': rule_plurality,
        'borda': rule_borda,
        'copeland': rule_copeland,
    }
    
    # Count how often each candidate wins across rules
    winner_counts = {c: 0 for c in profile.candidates}
    all_rule_winners = {}
    axiom_scores = {}
    
    for rule_name, rule_func in rules.items():
        winners = rule_func(profile)
        all_rule_winners[rule_name] = winners
        
        # Check axioms for this rule
        condorcet_ok = check_condorcet(profile, winners)
        pareto_ok = check_pareto(profile, winners)
        axiom_score = int(condorcet_ok) + int(pareto_ok)
        axiom_scores[rule_name] = axiom_score
        
        for w in winners:
            winner_counts[w] += 1
    
    # Oracle winner: most frequent across rules
    max_count = max(winner_counts.values())
    oracle_candidates = [c for c, cnt in winner_counts.items() if cnt == max_count]
    
    # Tie-break by Condorcet winner if exists
    cw = profile.condorcet_winner()
    if cw in oracle_candidates:
        oracle_winner = cw
    else:
        # Tie-break by Copeland score
        copeland = profile.get_copeland_scores()
        oracle_winner = max(oracle_candidates, key=lambda c: copeland[c])
    
    return OracleResult(
        winner=oracle_winner,
        axiom_scores=axiom_scores,
        best_score=sum(axiom_scores.values()),
        all_rule_winners=all_rule_winners
    )

# --- 5. DATASET GENERATION ---

def generate_profile_features(profile: VotingProfile) -> Dict:
    """Extract features from a voting profile."""
    features = {}
    
    # Basic stats
    features['num_voters'] = profile.n_voters
    features['num_candidates'] = profile.num_cands
    
    # Per-candidate features
    plurality = profile.get_plurality_scores()
    borda = profile.get_borda_scores()
    copeland = profile.get_copeland_scores()
    
    for c in profile.candidates:
        features[f'plurality_{c}'] = plurality[c]
        features[f'borda_{c}'] = borda[c]
        features[f'copeland_{c}'] = copeland[c]
    
    # Pairwise margins (flattened upper triangle)
    for a, b in itertools.combinations(profile.candidates, 2):
        margin = profile.get_pairwise_margin(a, b)
        features[f'margin_{a}_{b}'] = margin
    
    # Condorcet winner exists
    features['has_condorcet'] = int(profile.condorcet_winner() >= 0)
    
    return features

def generate_dataset(num_profiles: int = 1000, min_voters: int = 5, max_voters: int = 55, 
                     num_candidates: int = 5, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate dataset of voting profiles with oracle winners.
    Uses variable number of voters per profile.
    
    Returns:
        profile_data: List of dicts with features + oracle winner
        pairwise_data: List of pairwise comparisons (winner/loser based on oracle)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    profile_data = []
    pairwise_data = []
    
    print(f"Generating {num_profiles} voting profiles...")
    print(f"  Voters per profile: {min_voters}-{max_voters}")
    print(f"  Candidates: {num_candidates}")
    
    for profile_id in range(num_profiles):
        if (profile_id + 1) % 100 == 0:
            print(f"  Progress: {profile_id + 1}/{num_profiles}")
        
        # Generate random rankings with variable number of voters
        num_voters = random.randint(min_voters, max_voters)
        rankings = [random.sample(range(num_candidates), num_candidates) 
                   for _ in range(num_voters)]
        profile = VotingProfile(rankings, num_candidates)
        
        # Compute oracle winner
        oracle_result = compute_oracle_winner(profile)
        
        # Extract features
        features = generate_profile_features(profile)
        features['profile_id'] = profile_id
        features['oracle_winner'] = oracle_result.winner
        features['oracle_score'] = oracle_result.best_score
        
        # Add rule winners as features
        for rule_name, winners in oracle_result.all_rule_winners.items():
            for c in profile.candidates:
                features[f'{rule_name}_wins_{c}'] = int(c in winners)
        
        profile_data.append(features)
        
        # Generate pairwise comparisons: oracle winner beats all others
        oracle_winner = oracle_result.winner
        for loser in profile.candidates:
            if loser == oracle_winner:
                continue
            
            # Also compute margin as a feature
            margin = profile.get_pairwise_margin(oracle_winner, loser)
            
            pairwise_data.append({
                'profile_id': profile_id,
                'winner': oracle_winner,
                'loser': loser,
                'margin': margin,
                'winner_plurality': features[f'plurality_{oracle_winner}'],
                'loser_plurality': features[f'plurality_{loser}'],
                'winner_borda': features[f'borda_{oracle_winner}'],
                'loser_borda': features[f'borda_{loser}'],
                'winner_copeland': features[f'copeland_{oracle_winner}'],
                'loser_copeland': features[f'copeland_{loser}'],
            })
    
    return profile_data, pairwise_data

def save_datasets(profile_data: List[Dict], pairwise_data: List[Dict], 
                  output_dir: str = "data"):
    """Save datasets to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save profile data
    profile_file = os.path.join(output_dir, "oracle_profiles.csv")
    if profile_data:
        with open(profile_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=profile_data[0].keys())
            writer.writeheader()
            writer.writerows(profile_data)
        print(f"Saved {len(profile_data)} profiles to {profile_file}")
    
    # Save pairwise data
    pairwise_file = os.path.join(output_dir, "oracle_pairwise.csv")
    if pairwise_data:
        with open(pairwise_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pairwise_data[0].keys())
            writer.writeheader()
            writer.writerows(pairwise_data)
        print(f"Saved {len(pairwise_data)} pairwise comparisons to {pairwise_file}")
    
    # Save metadata with voter range info
    if profile_data:
        voter_counts = [p['num_voters'] for p in profile_data]
        metadata = {
            'num_profiles': len(profile_data),
            'num_pairwise': len(pairwise_data),
            'min_voters': min(voter_counts),
            'max_voters': max(voter_counts),
            'avg_voters': np.mean(voter_counts),
            'num_candidates': profile_data[0]['num_candidates'],
        }
        metadata_file = os.path.join(output_dir, "oracle_dataset_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate oracle winner dataset')
    parser.add_argument('--num-profiles', type=int, default=5000, 
                       help='Number of voting profiles to generate per configuration')
    parser.add_argument('--min-voters', type=int, default=None,
                       help='Minimum number of voters per profile (overrides config defaults)')
    parser.add_argument('--max-voters', type=int, default=None,
                       help='Maximum number of voters per profile (overrides config defaults)')
    parser.add_argument('--num-candidates', type=int, default=None,
                       help='Number of candidates (overrides config defaults)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='both', choices=['config1', 'config2', 'both'],
                       help='Which configuration to run: config1 (5 alt, 5-55 voters), config2 (7 alt, 7-77 voters), or both')
    
    args = parser.parse_args()
    
    # Define configurations
    configs = []
    if args.config in ['config1', 'both']:
        configs.append({
            'name': 'config1_5alt_5-55voters',
            'min_voters': args.min_voters if args.min_voters is not None else 5,
            'max_voters': args.max_voters if args.max_voters is not None else 55,
            'num_candidates': args.num_candidates if args.num_candidates is not None else 5
        })
    if args.config in ['config2', 'both']:
        configs.append({
            'name': 'config2_7alt_7-77voters',
            'min_voters': args.min_voters if args.min_voters is not None else 7,
            'max_voters': args.max_voters if args.max_voters is not None else 77,
            'num_candidates': args.num_candidates if args.num_candidates is not None else 7
        })
    
    # Generate datasets for each configuration
    for config in configs:
        print("\n" + "=" * 60)
        print(f"ORACLE WINNER DATASET GENERATION: {config['name']}")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Profiles: {args.num_profiles}")
        print(f"  Voters per profile: {config['min_voters']}-{config['max_voters']}")
        print(f"  Candidates: {config['num_candidates']}")
        print(f"  Seed: {args.seed}")
        print("=" * 60)
        
        profile_data, pairwise_data = generate_dataset(
            num_profiles=args.num_profiles,
            min_voters=config['min_voters'],
            max_voters=config['max_voters'],
            num_candidates=config['num_candidates'],
            seed=args.seed
        )
        
        # Save with configuration-specific filenames
        output_dir = f"{args.output_dir}/{config['name']}"
        # Save with configuration-specific filenames
        output_dir = f"{args.output_dir}/{config['name']}"
        save_datasets(profile_data, pairwise_data, output_dir=output_dir)
        
        print("\n" + "=" * 60)
        print(f"SUMMARY - {config['name']}")
        print("=" * 60)
        print(f"Generated {len(profile_data)} voting profiles")
        print(f"Generated {len(pairwise_data)} pairwise comparisons")
        print(f"Average oracle score: {np.mean([p['oracle_score'] for p in profile_data]):.2f}")
        
        # Distribution of oracle winners
        winner_dist = {}
        for p in profile_data:
            w = p['oracle_winner']
            winner_dist[w] = winner_dist.get(w, 0) + 1
        print(f"\nOracle winner distribution:")
        for c in sorted(winner_dist.keys()):
            print(f"  Candidate {c}: {winner_dist[c]} ({100*winner_dist[c]/len(profile_data):.1f}%)")
    
    print("\n" + "=" * 60)
    print("ALL CONFIGURATIONS COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()
