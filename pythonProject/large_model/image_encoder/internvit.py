from transformers import AutoModel, CLIPImageProcessor
from omegaconf import DictConfig
import hydra
import torch

@torch.no_grad()
@hydra.main(config_path="./", config_name="image_encoder.yaml")
def internvit_inference(cfg: DictConfig, image):
    model = AutoModel.from_pretrained(
        cfg.image_encoder.model_path.size,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda().eval()

    image_processor = CLIPImageProcessor.from_pretrained(cfg.image_encoder.model_path.size)
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    outputs = model(pixel_values)
    return outputs.last_hidden_state