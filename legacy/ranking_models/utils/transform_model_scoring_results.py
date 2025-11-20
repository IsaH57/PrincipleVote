"""This script creates a ranking of images based on individual scoring results from a model.
It e.g. converts: "scores": [22.43886947631836,23.623538970947266,25.061115264892578,22.73494529724121,21.826229095458984,18.462697982788086,25.00086212158203,23.2099552154541,24.1268367767334] to "relative_rankings": [2,6,8,1,7,3,0,4,5]
"""

import json
import re


def convert_ranked_images_to_relative_numbers(input_file, output_file):
    """Convert ranked_images from filenames to relative position numbers.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        ranked_images = item["ranked_images"]

        # Extract numbers from filenames and convert to relative positions
        relative_rankings = []
        for filename in ranked_images:
            # Extract number from filename (e.g., "00002.jpg" -> 2)
            match = re.search(r'(\d+)\.jpg', filename)
            if match:
                image_number = int(match.group(1))
                # Find the minimum number in this prompt's images to calculate relative position
                all_numbers = []
                for fname in ranked_images:
                    num_match = re.search(r'(\d+)\.jpg', fname)
                    if num_match:
                        all_numbers.append(int(num_match.group(1)))

                min_number = min(all_numbers)
                relative_position = image_number - min_number
                relative_rankings.append(relative_position)

        # Add relative rankings to the item
        item["relative_rankings"] = relative_rankings

    # Write the updated data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Relative rankings added and saved to {output_file}")


if __name__ == "__main__":
    input_file = "../ranking_results/model_scoring_results_raw_output.json"
    output_file = "../ranking_results/model_scoring_results.json"

    convert_ranked_images_to_relative_numbers(input_file, output_file)