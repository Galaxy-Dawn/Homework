from .wave2vec import wave2vec_inference
from .hubert import hubert_inference
from omegaconf import DictConfig

def speech_encoder(cfg: DictConfig, wave):
    """
    :param cfg: config/config.yaml 管理的参数
    :details
        wave2vec base 95M embedding dim 768
        wave2vec large 317M embedding dim 1024
        hubert base 95M embedding dim 768
        hubert large 317M embedding dim 1024
    :param wave: np.array [batch,语音采样点数,声道数],多声道会取平均值
    :return: embedding,type = tensor, device=cuda
    """
    if cfg.speech_encoder_config.name == "wave2vec":
        return wave2vec_inference(cfg, wave)
    elif cfg.speech_encoder_config.name == "hubert":
        return hubert_inference(cfg, wave)
    else:
        raise NotImplementedError