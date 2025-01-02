from omegaconf import DictConfig
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
import numpy as np
@torch.no_grad()
def wave2vec_inference(cfg: DictConfig, wave):
    if wave.shape[-1] == 2:
        wave = np.mean(wave, axis=-1)
    assert wave.ndim in [1, 2], wave.ndim

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.speech_encoder_config.model_path[cfg.speech_encoder_size])
    model = Wav2Vec2Model.from_pretrained(cfg.speech_encoder_config.model_path[cfg.speech_encoder_size])
    model = model.to(cfg.device)
    model = model.half()
    model.eval()

    input_values = feature_extractor(wave, return_tensors="pt").input_values
    input_values = input_values.half()
    input_values = input_values.to(cfg.device)

    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state