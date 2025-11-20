from datasets import load_from_disk

try:
    personas = load_from_disk("/home/ra63hik/dppo/data/nemotron_personas")
    print("Personas columns:", personas['train'].column_names)
    print("Personas example:", personas['train'][0])
except Exception as e:
    print("Error loading personas:", e)

try:
    restaurants = load_from_disk("/home/ra63hik/dppo/data/nytimes_best_restaurants_2024")
    print("Restaurants columns:", restaurants['train'].column_names)
    print("Restaurants example:", restaurants['train'][0])
except Exception as e:
    print("Error loading restaurants:", e)
