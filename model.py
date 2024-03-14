from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

from loguru import logger

from config import config



chain = None


def load_chain():
    # https://huggingface.co/hyeogi/SOLAR-10.7B-v1.2/discussions/1
    template = """### System:\n{input}\n\n### User:\n{instruction}\n### Assistant:\n"""

    prompt = PromptTemplate.from_template(template)

    callback_manager = CallbackManager([StdOutCallbackHandler()])

    model_path = config.retrospective_model_path

    logger.info(f"Loading model from {model_path}")

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

    global chain
    chain = LLMChain(prompt=prompt, llm=llm)


def get_chain():
    return chain


