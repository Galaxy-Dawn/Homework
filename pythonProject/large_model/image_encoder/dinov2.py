from transformers import AutoImageProcessor, AutoModel
from omegaconf import DictConfig
import hydra
import torch

@torch.no_grad()
def dinov2_inference(cfg: DictConfig, image):
    """
    :param cfg: 输入的config
    :param image: 一张RGB格式的图片
    :return:最后一层隐藏层的输出
    """
    name_of_model = cfg.image_encoder.name
    size = cfg.size
    processor = AutoImageProcessor.from_pretrained(cfg.image_encoder.model_path[size])
    model = AutoModel.from_pretrained(cfg.image_encoder.model_path[size])
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
