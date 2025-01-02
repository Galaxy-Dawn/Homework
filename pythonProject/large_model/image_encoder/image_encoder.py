from .internvit import internvit_inference
from .siglip import siglip_inference
from .dinov2 import dinov2_inference
from omegaconf import DictConfig

def image_encoder(cfg: DictConfig, image):
    """
    :param cfg: config/config.yaml 管理的参数
    :details
        dinov2 small hidden size=384 Params=22.1M
        dinov2 base hidden size=768 Params=86.6M
        dinov2 large hidden size=1024 Params=304M
        dinov2 giant hidden size=1536 Params=1.14B
        Internvit 300M hidden size = 1024
        Internvit 6B hidden size = 3200
        SigLIP base  203M  hidden_size = 768
        SigLIP large  652M  hidden_size = 1024
        SigLIP so400m  878M  hidden_size = 1152
    :param image: np.array [batch,H,W,C]
    :return: embedding,type = tensor, device=cuda
    """
    if cfg.image_encoder_config.name == "dinov2":
        return dinov2_inference(cfg, image)
    elif cfg.image_encoder_config.name == "siglip":
        return siglip_inference(cfg, image)
    elif cfg.image_encoder_config.name == "internvit":
        return internvit_inference(cfg, image)
    else:
        raise NotImplementedError
