from executing.executing import non_sentinel_instructions
from omegaconf import DictConfig
from jinja2 import Environment, FileSystemLoader
from vllm import LLM, SamplingParams
import os
from transformers import AutoTokenizer

def llm_inference(cfg: DictConfig, prompts=None):

    instruct = f"""
    你现在需要按照拼音来解题。请直接输出一个答案,不要有除答案外的其他文字。
    问题为：拼音 jin tian tian qi zheng hao 组成的句子最可能是什么? 回答为：今天天气真好。
    问题为：拼音 wo men da che qu ba 组成的句子最可能是什么? 回答为：我们打车去吧。
    问题为：拼音 jin tian zhen dao mei 组成的句子最可能是什么? 回答为：今天真倒霉。
    问题为：拼音 chu qu zou zou ba 组成的句子最可能是什么? 回答为：出去走走吧。
    问题为：
    """
    if prompts is None:
        prompts = [
            "拼音 yao xia yu le 组成的句子最可能是什么? 回答为：",
            "拼音 chi pu tao bu tu pu tao pi 组成的句子最可能是什么? 回答为："
        ]

    if cfg.llm.prompt_type == 'instruct':
        messages = [{"role": "user", "content": instruct + prompt} for prompt in prompts]
    else:
        messages = [{"role": "user", "content": prompt} for prompt in prompts]

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.llm.CUDA_VISIBLE_DEVICES
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_config.model_path[cfg.llm.size], trust_remote_code=True)

    if cfg.llm_config.name == 'yi':
        chat_template = "{%- if messages[0]['role'] == 'system' -%}    {%- set system_message = messages[0]['content'] -%}    {%- set messages = messages[1:] -%}{%- else -%}    {% set system_message = '' -%}{%- endif -%}{{ bos_token + system_message }}{%- for message in messages -%}      {%- if message['role'] == 'user' -%}        {{ 'USER: ' + message['content'] + '\n' }}    {%- elif message['role'] == 'assistant' -%}        {{ 'ASSISTANT: ' + message['content'] + eos_token + '\n' }}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ 'ASSISTANT:' }} {% endif %}"

        message = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=chat_template, add_generation_prompt=True)
    else:
        message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(temperature=cfg.llm.temperature,
                                     top_p=cfg.llm.top_p,
                                     min_tokens=cfg.llm.min_tokens,
                                     max_tokens=cfg.llm.max_tokens,
                                     repetition_penalty=cfg.llm.repetition_penalty,
                                     stop_token_ids=[tokenizer.eos_token_id]
                                     )

    model = LLM(model=cfg.llm_config.model_path[cfg.llm.size],
                gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
                tensor_parallel_size=cfg.llm.tensor_parallel_size,
                trust_remote_code=True)

    outputs = model.generate(prompts=message, sampling_params=sampling_params)
    generated_text = [[output.outputs[0].text] for output in outputs]
    for i in range(len(generated_text)):
        print(f"Prompt: {prompts[i]}, Generated text: {generated_text[i][0]}")
    return generated_text