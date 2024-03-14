from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, BartForConditionalGeneration
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from loguru import logger

from config import config


tf_device, tf_tokenizer, tf_model = None, None, None
llm_chain = None


def load_tf_model():
    tf_config = AutoConfig.from_pretrained(config.tf_model_id)

    global tf_device
    tf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global tf_tokenizer
    tf_tokenizer = AutoTokenizer.from_pretrained(config.tf_model_id)
    
    global tf_model
    tf_model = BartForConditionalGeneration.from_pretrained(config.tf_model_id, config=tf_config).to(tf_device)


def get_tf_model():
    return tf_device, tf_tokenizer, tf_model


def make_summary(chunks: List[str], device: torch.device, tokenizer: AutoTokenizer, model: PreTrainedModel) -> str:
    inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask, 
        max_length=64, num_beams=5, length_penalty=1.2, use_cache=True, early_stopping=True).detach().cpu()
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "".join(summary)


def async_func(sync_func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return loop.run_in_executor(pool, sync_func, *args, **kwargs) # sync_func(*args, **kwargs)



def load_llm_chain():
    # https://huggingface.co/hyeogi/SOLAR-10.7B-v1.2/discussions/1
    # template = """### System:\n{input}\n\n### User:\n{instruction}\n### Assistant:\n"""
    template = """### System:
    보고서는 {username}의 하루를 기록한 내용이야. 다음 지침과 예시를 참고해서 회고를 작성해줘.
    1. {username}의 하루를 루나라는 친구가 평가하는 내용을 작성해줘.
    2. 친구가 {username}에게 말해주는 것처럼 친근한 말투로 써줘.
    3. 모든 문장을 과거형으로 작성해줘.

    ### 보고서:
    고등학교 친구인 예원이를 만나서 카페를 다녀왔다고 한다.
    집에 돌아와서는 너무 늦게 들어왔다는 이유로 엄마와 다퉜다고 한다.
    기분이 좋지 않았는데 대학 과제를 해야했기 때문에 열심히 다 마치고 잤다고 한다.

    ### 회고:
    {username}아! 너는 오늘 고등학교 친구 예원이랑 만나서 카페를 다녀왔었네. 재미있었어?
    {username}는 집에 늦게 들어왔다고 엄마와 다퉜었지. 속상했겠다..
    그리고 기분이 좋지 않았는데도 대학 과제를 마무리하고 잤구나. {username}는 대단한 것 같아!

    ### 보고서:
    {summary}
    ### 회고:
    """

    prompt = PromptTemplate.from_template(template)

    callback_manager = CallbackManager([StdOutCallbackHandler()])

    model_path = config.llm_model_path

    # logger.info(f"Loading model from {model_path}")

    try:
        # https://github.com/abetlen/llama-cpp-python
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            callback_manager=callback_manager,
            temperature=0.6,
            top_p=1,
            max_tokens=128,
            verbose=True,
        )
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error while loading model from {model_path}: {e}")
        llm = None

    global llm_chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)


def get_llm_chain():
    return llm_chain


