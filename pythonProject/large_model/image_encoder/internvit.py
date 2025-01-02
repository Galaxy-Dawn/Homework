from transformers import AutoModel, CLIPImageProcessor
from omegaconf import DictConfig
import torch

@torch.no_grad()
def internvit_inference(cfg: DictConfig, image):
    model = (AutoModel.from_pretrained(
        cfg.image_encoder_config.model_path[cfg.image_encoder_size],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True))
    model = model.to(cfg.device)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(cfg.image_encoder_config.model_path[cfg.image_encoder_size])
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).to(cfg.device)
    outputs = model(pixel_values)
    return outputs.last_hidden_state