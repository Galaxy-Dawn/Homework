from huggingface_hub import snapshot_download
# snapshot_download(repo_id="TencentGameMate/chinese-hubert-base",cache_dir="/data/share/storage/checkpoint/speech_encoder/HuBERT")
# snapshot_download(repo_id="TencentGameMate/chinese-hubert-large",cache_dir="/data/share/storage/checkpoint/speech_encoder/HuBERT")
# snapshot_download(repo_id="TencentGameMate/chinese-wav2vec2-base",cache_dir="/data/share/storage/checkpoint/speech_encoder/wave2vec2.0")
# snapshot_download(repo_id="TencentGameMate/chinese-wav2vec2-large",cache_dir="/data/share/storage/checkpoint/speech_encoder/wave2vec2.0")
#
# snapshot_download(repo_id="OpenGVLab/InternViT-300M-448px",cache_dir="/data/share/storage/checkpoint/image_encoder/InternViT")
# snapshot_download(repo_id="OpenGVLab/InternViT-6B-448px-V1-5",cache_dir="/data/share/storage/checkpoint/image_encoder/InternViT")
#
# snapshot_download(repo_id="google/siglip-base-patch16-224",cache_dir="/data/share/storage/checkpoint/image_encoder/SigLIP")
# snapshot_download(repo_id="google/siglip-large-patch16-384",cache_dir="/data/share/storage/checkpoint/image_encoder/SigLIP")
# snapshot_download(repo_id="google/siglip-so400m-patch14-384",cache_dir="/data/share/storage/checkpoint/image_encoder/SigLIP")
#
# snapshot_download(repo_id="facebook/dinov2-small",cache_dir="/data/share/storage/checkpoint/image_encoder/Dinov2")
# snapshot_download(repo_id="facebook/dinov2-base",cache_dir="/data/share/storage/checkpoint/image_encoder/Dinov2")
# snapshot_download(repo_id="facebook/dinov2-large",cache_dir="/data/share/storage/checkpoint/image_encoder/Dinov2")
# snapshot_download(repo_id="facebook/dinov2-giant",cache_dir="/data/share/storage/checkpoint/image_encoder/Dinov2")
#

# snapshot_download(repo_id="TencentBAC/Conan-embedding-v1",cache_dir="/data/share/storage/checkpoint/text_encoder/conan_embedding_v1")
# snapshot_download(repo_id="lier007/xiaobu-embedding-v2",cache_dir="/data/share/storage/checkpoint/text_encoder/xiaobu_embedding_v2")
# snapshot_download(repo_id="lier007/xiaobu-embedding",cache_dir="/data/share/storage/checkpoint/text_encoder/xiaobu_embedding")
# snapshot_download(repo_id="Alibaba-NLP/gte-Qwen2-7B-instruct",cache_dir="/data/share/storage/checkpoint/text_encoder/gte_qwen2_7B_instruct")
# snapshot_download(repo_id="iampanda/zpoint_large_embedding_zh",cache_dir="/data/share/storage/checkpoint/text_encoder/zpoint_large_embedding_zh")
# snapshot_download(repo_id="DMetaSoul/Dmeta-embedding-zh",cache_dir="/data/share/storage/checkpoint/text_encoder/dmeta_embedding")
# snapshot_download(repo_id="Pristinenlp/alime-embedding-large-zh",cache_dir="/data/share/storage/checkpoint/text_encoder/alime_embedding_large_zh")
# snapshot_download(repo_id="thenlper/gte-large-zh",cache_dir="/data/share/storage/checkpoint/text_encoder/gte_large_zh")

# snapshot_download(repo_id="Qwen/Qwen2.5-0.5B",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5")
# snapshot_download(repo_id="Qwen/Qwen2.5-1.5B",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5")
# snapshot_download(repo_id="Qwen/Qwen2.5-3B",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5")
# snapshot_download(repo_id="Qwen/Qwen2.5-7B",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5")
# snapshot_download(repo_id="Qwen/Qwen2.5-14B",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5")
snapshot_download(repo_id="Qwen/Qwen2.5-7B-Instruct",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5",max_workers=1)
snapshot_download(repo_id="Qwen/Qwen2.5-14B-Instruct",cache_dir="/data/share/storage/checkpoint/llm/Qwen2.5",max_workers=1)

# snapshot_download(repo_id="01-ai/Yi-1.5-6B",cache_dir="/data/share/storage/checkpoint/llm/Yi-1.5")
# snapshot_download(repo_id="01-ai/Yi-1.5-9B",cache_dir="/data/share/storage/checkpoint/llm/Yi-1.5")
snapshot_download(repo_id="01-ai/Yi-1.5-6B-Chat",cache_dir="/data/share/storage/checkpoint/llm/Yi-1.5",max_workers=1)
snapshot_download(repo_id="01-ai/Yi-1.5-9B-Chat",cache_dir="/data/share/storage/checkpoint/llm/Yi-1.5",max_workers=1)

snapshot_download(repo_id="THUDM/glm-4-9b-chat",cache_dir="/data/share/storage/checkpoint/llm/GLM4",max_workers=1)

snapshot_download(repo_id="deepseek-ai/DeepSeek-V2-Lite",cache_dir="/data/share/storage/checkpoint/llm/DeepSeek",max_workers=1)
snapshot_download(repo_id="deepseek-ai/DeepSeek-V2-Lite-Chat",cache_dir="/data/share/storage/checkpoint/llm/DeepSeek",max_workers=1)