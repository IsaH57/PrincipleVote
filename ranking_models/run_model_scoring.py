import os
import torch
import json
from PIL import Image
from tqdm import tqdm

# Import HPSv2 modules
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.src.training.train import inversion_score
from hpsv2.src.training.data import collate_rank
import numpy as np

# ------ CONFIGURATION ------
MODEL_NAME = "ViT-H-14"
PRECISION = "amp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "PATH/TO/HPS_v2_compressed.pt"
DATA_PATH = "PATH/TO/FOLDER/HOLDING/TEST.JSON/AND/IMAGE/FOLDER"
IMAGE_FOLDER = "PATH/TO/IMAGES"
with open('PATH/TO/TEST.JSON', 'r') as f:
    data = json.load(f)

#prompts = [item['prompt'] for item in data]

BATCH_SIZE = 20
# ---------------------------

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

def rank_images_for_prompt(model, preprocess_val, tokenizer, prompt, image_names):
    # Load and preprocess images
    images = []
    for img_name in image_names:
        img_path = os.path.join(IMAGE_FOLDER, os.path.basename(img_name))
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess_val(img)
        images.append(img_tensor)
    images = torch.stack(images).to(DEVICE)

    # Tokenize the prompt (repeat for every image)
    text_tokens = tokenizer([prompt] * len(image_names)).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(images, text_tokens)
            image_features, text_features, logit_scale = (
                outputs["image_features"],
                outputs["text_features"],
                outputs["logit_scale"]
            )
            logits_per_image = logit_scale * image_features @ text_features.T
            # logits_per_image shape: [num_images, num_images]
            scores = torch.diagonal(logits_per_image).cpu().numpy()

    # Rank images according to their scores (higher is better)
    ranking_indices = np.argsort(-scores)
    ranked_images = [image_names[i] for i in ranking_indices]
    return ranked_images, scores

def main():
    model, preprocess_val, tokenizer = load_model()

    results = []
    for entry in tqdm(data):
        prompt = entry["prompt"]
        image_names = entry["image_path"]
        ranked_images, scores = rank_images_for_prompt(
            model, preprocess_val, tokenizer, prompt, image_names
        )
        results.append({
            "prompt": prompt,
            "ranked_images": ranked_images,
            "scores": [float(s) for s in scores]
        })
        print(f"Prompt: {prompt}")
        for img, score in zip(ranked_images, sorted(scores, reverse=True)):
            print(f"  {img}: {score:.4f}")

    # save ranking_results
    with open("ranking_results/model_scoring_results_raw_output.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()