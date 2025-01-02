from omegaconf import DictConfig
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
import numpy as np

@torch.no_grad()
def hubert_inference(cfg: DictConfig, wave):
    if wave.ndim() == 2:
        wave = np.mean(wave, axis=-1)
    assert wave.ndim == 1, wave.ndim

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.speech_encoder_config.model_path[cfg.speech_encoder_size])
    model = HubertModel.from_pretrained(cfg.speech_encoder_config.model_path[cfg.speech_encoder_size])

    model = model.to(cfg.device)
    model = model.half()
    model.eval()

    input_values = feature_extractor(wave, return_tensors="pt").input_values
    input_values = input_values.half()
    input_values = input_values.to(cfg.device)
    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state