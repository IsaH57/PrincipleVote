import os
import torch
import json
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.evaluation import evaluate_rank
from hpsv2.src.training.data import RankingDataset, collate_rank

MODEL_NAME = "ViT-H-14"
PRECISION = "amp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "PATH/TO/HPS_v2_compressed.pt"
DATA_PATH = "PATH/TO/FOLDER/HOLDING/TEST.JSON/AND/IMAGE/FOLDER"
IMAGE_FOLDER = "PATH/TO/IMAGES"
BATCH_SIZE = 20

def load_model():
    model, _, preprocess_val = create_model_and_transforms(
        MODEL_NAME,
        None,
        precision=PRECISION,
        device=DEVICE,
        jit=False,
        output_dict=True,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    tokenizer = get_tokenizer(MODEL_NAME)
    return model, preprocess_val, tokenizer

if __name__ == "__main__":
    model, preprocess_val, tokenizer = load_model()
    # produces rankings and prints accuracy, and writes all_rankings to logs/hps_rank.json
    evaluate_rank(DATA_PATH, IMAGE_FOLDER, model, BATCH_SIZE, preprocess_val, tokenizer, DEVICE)
    with open('logs/hps_rank.json', 'r') as f:
        rankings = json.load(f)
    print("Relative rankings per prompt (from logs/hps_rank.json):")
    for idx, rank in enumerate(rankings):
        print(f"Prompt {idx}: {rank}")
    with open('model_ranking_results_raw_output.json', 'w') as f:
        json.dump({"relative_rankings": rankings}, f, indent=4)