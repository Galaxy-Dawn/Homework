from http.client import responses

from image_encoder import image_encoder
from speech_encoder import speech_encoder
from text_encoder import text_encoder
from llm import llm_inference
import soundfile as sf
import numpy as np
import hydra
import cv2
import torch

@hydra.main(config_path='./config', config_name='config.yaml',version_base='1.1')
def main(cfg):
    # batch_size = 4
    # image = cv2.imread(cfg.example_image_path)
    # print("Shape of the input image: ", image.shape)
    # image_list = np.array([image for _ in range(batch_size)])
    # print(image_encoder(cfg,image_list).shape)
    #
    # wave, _ = sf.read(cfg.example_audio_path)
    # print("Shape of the input audio: ", wave.shape)
    # wave_list = np.array([wave for _ in range(batch_size)])
    # print(speech_encoder(cfg, wave_list).shape)
    #
    # print(text_encoder(cfg, "hello world").shape)

    llm_inference(cfg)



if __name__ == "__main__":
    main()


