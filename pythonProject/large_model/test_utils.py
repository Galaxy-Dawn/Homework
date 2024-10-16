from image_encoder import dinov2_inference
from image_encoder import internvit_inference
from image_encoder import siglip_inference
from speech_encoder import wave2vec_inference
from speech_encoder import hubert_inference
from PIL import Image
import soundfile as sf
import numpy as np
import hydra

image = Image.open('Lena.jpg').convert('RGB')
wave,_ = sf.read('water.wav')

@hydra.main(config_path='./', config_name='config.yaml')
def main(cfg):
    print("Shape of the input image: ", np.array(image).shape)
    print("Shape of the input audio: ", wave.shape)
    print(dinov2_inference(cfg.image_encoder,image).shape)
    # print(internvit_inference(image).shape)
    # print(siglip_inference(image).shape)
    # print(wave2vec_inference(wave).shape)
    # print(hubert_inference(wave).shape)

main()

