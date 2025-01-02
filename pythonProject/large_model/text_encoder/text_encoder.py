from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
def text_encoder(cfg: DictConfig, text):
    """
    :param cfg: config/config.yaml 管理的参数
    :details
        conan_embedding_v1 params=326M dim=1792
        xiaobu_embedding_v2 params=326M dim=1792
        gte_qwen2_7B_instruct params=7B dim=3584
        zpoint_large_embedding_zh params=326M dim=1792
        dmeta_embedding params=103M dim=768
        xiaobu_embedding params=326M dim=1024
        alime_embedding_large_zh params=326M dim=1024
        gte_large_zh params=326M dim=1024
    :param text 文本
    :return: embedding,type = tensor, device=cuda
    """
    model = SentenceTransformer(cfg.text_encoder_config.model_path).to(cfg.device)
    embeddings = model.encode(text, normalize_embeddings=True)
    return embeddings
