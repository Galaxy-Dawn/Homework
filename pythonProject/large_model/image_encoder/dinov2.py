from transformers import AutoImageProcessor, AutoModel
from omegaconf import DictConfig
import torch

@torch.no_grad()
def dinov2_inference(cfg: DictConfig, image):
    processor = AutoImageProcessor.from_pretrained(cfg.image_encoder_config.model_path[cfg.image_encoder_size])
    model = AutoModel.from_pretrained(cfg.image_encoder_config.model_path[cfg.image_encoder_size])
    model = model.to(cfg.device)
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(cfg.device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
