import torch
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from late_fusion_model.modeling_fusion import MultimodalFusionLayer
import os
import torch.nn.functional as F

def compute_scores(clip_model, clip_processor, images, prompts):
    # Prepare inputs for the CLIP model using the processor
    processed_inputs = clip_processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(clip_model.device)

    # Extract image and text embeddings without calculating gradients
    with torch.no_grad():
        model_outputs = clip_model(**processed_inputs)
        image_embeddings = model_outputs.image_embeds
        text_embeddings = model_outputs.text_embeds

    # Calculate cosine similarity using PyTorch's functional API
    cosine_similarities = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1)

    # Replace negative similarities with zero
    positive_similarities = F.relu(cosine_similarities)

    return positive_similarities

def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Charge le modèle à partir d'un checkpoint"""
    # Initialisation du modèle et du tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration et modèle
    model_config = AutoConfig.from_pretrained("gpt2")
    model = MultimodalFusionLayer.from_pretrained(
        "gpt2",
        config=model_config,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True  # Suppresses the warning
    ).to(device)
    
    # Chargement du checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Passer en mode évaluation
    
    return model, tokenizer



if __name__ == "__main__":
    # Exemple d'utilisation
    checkpoint_path = "outputs/epoch_10"  # Chemin vers votre checkpoin
    # Charger le modèle
    model, tokenizer = load_checkpoint(checkpoint_path)
    
