from omegaconf import DictConfig
import hydra
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)

@torch.no_grad()
@hydra.main(config_path="./", config_name="speech_encoder.yaml")
def wave2vec_inference(cfg: DictConfig, wave):
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.image_encoder.model_path.size)
    model = Wav2Vec2Model.from_pretrained(cfg.image_encoder.model_path.size)
    model = model.to(device)
    model = model.half()
    model.eval()

    input_values = feature_extractor(wave, return_tensors="pt").input_values
    input_values = input_values.half()
    input_values = input_values.to(device)

    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state