from transformers import AutoProcessor, AutoModel
from omegaconf import DictConfig
import hydra
import torch

@torch.no_grad()
@hydra.main(config_path="./", config_name="image_encoder.yaml")
def siglip_inference(cfg: DictConfig, image):
    model = AutoModel.from_pretrained(cfg.image_encoder.model_path.size)
    processor = AutoProcessor.from_pretrained(cfg.image_encoder.model_path.size)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state