from transformers import AutoProcessor, AutoModel
from omegaconf import DictConfig
import torch

@torch.no_grad()
def siglip_inference(cfg: DictConfig, image):
    model = AutoModel.from_pretrained(cfg.image_encoder_config.model_path[cfg.image_encoder_size])
    model = model.to(cfg.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.image_encoder_config.model_path[cfg.image_encoder_size])
    inputs = processor(images=image, return_tensors="pt").to(cfg.device)
    outputs = model(**inputs)
    return outputs.last_hidden_state