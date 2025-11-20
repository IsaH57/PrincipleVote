"""Code to calculate axiom alignment for different voting rules and profiles."""

import json

def evaluate_axiom_satisfaction_from_json(json_data: list) -> dict:
    """Evaluate axiom satisfaction from JSON data and return conditional & absolute satisfaction,
    percent applicable for each axiom.
    
    Args:
        json_data (list): List of dictionaries containing profile results with axiom satisfaction values.
                         Each axiom value should be: -1 (violation), 0 (not applicable), or 1 (satisfied).
    
    Returns:
        dict: Dictionary with axiom names as keys and metrics as values:
              - cond_satisfaction: satisfied / applicable
              - absolute_satisfaction: (satisfied + not_applicable) / total
              - percent_applicable: applicable / total
    """
    # Initialize counters for each axiom
    axiom_names = ["anonymity", "neutrality", "condorcet", "pareto", "independence"]
    results = {}
    
    for axiom in axiom_names:
        iteration = len(json_data)
        applicable = 0
        satisfied = 0
        
        for profile_data in json_data:
            sat = profile_data["axioms"][axiom]
            
            if sat == 1:
                satisfied += 1
                applicable += 1
            elif sat == -1:
                applicable += 1
            # sat == 0 -> not applicable, don't increment applicable or satisfied
        
        # Avoid division by zero
        if applicable > 0:
            cond_satisfaction = satisfied / applicable
        else:
            cond_satisfaction = float("nan")
        
        absolute_satisfaction = (satisfied + (iteration - applicable)) / iteration
        percent_applicable = applicable / iteration
        
        print(f"Axiom ({axiom}) conditional satisfaction: {cond_satisfaction}")
        print(f"Axiom ({axiom}) absolute satisfaction: {absolute_satisfaction}")
        print(f"Axiom ({axiom}) percent applicable: {percent_applicable}")
        
        results[axiom] = {
            "cond_satisfaction": cond_satisfaction,
            "absolute_satisfaction": absolute_satisfaction,
            "percent_applicable": percent_applicable,
        }
    
    return results

if __name__ == "__main__":
    file_path = r"/home/h/hansi/PycharmProjects/PrincipleVote/ranking_models/ranking_results/human_voting_analysis_axioms.json"
    
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Evaluate axiom satisfaction
    axiom_metrics = evaluate_axiom_satisfaction_from_json(data)
    
    # Optionally save the aggregated results
    output_file = "/home/h/hansi/PycharmProjects/PrincipleVote/ranking_models/ranking_results/axiom_satisfaction_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(axiom_metrics, f, indent=2)
    
    print(f"\nAggregated results saved to {output_file}")