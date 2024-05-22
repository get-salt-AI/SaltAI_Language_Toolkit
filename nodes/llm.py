import json
import os
from typing import Dict, Any

# Implementation of LLM folder generated dropdowns using ComfyUI.folder_paths
import folder_paths

from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_EMBEDDING_DIM, DEFAULT_TEMPERATURE
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
from llama_index.core import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from .. import MENU_NAME, SUB_MENU_NAME

HAS_LLAMA_CPP = False
try:
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import (
        messages_to_prompt,
        completion_to_prompt,
    )
    HAS_LLAMA_CPP = True
except Exception:
    pass

HAS_GEMINI = False
try:
    from llama_index.llms.gemini import Gemini
    from llama_index.multi_modal_llms.gemini import GeminiMultiModal
    from llama_index.embeddings.gemini import GeminiEmbedding
    HAS_GEMINI = True
except Exception:
    pass

HAS_MISTRAL = False
try:
    from llama_index.llms.mistralai import MistralAI
    HAS_MISTRAL = True
except Exception:
    pass

HAS_OLLAMA = False
try:
    from llama_index.llms.ollama import Ollama
    HAS_OLLAMA = True
except Exception:
    pass

HAS_GROQ = False
try:
    from llama_index.llms.groq import Groq
    HAS_GROQ = True
except Exception:
    pass

#import openai
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType


Salt_LLM_DIR = os.path.abspath(os.path.join(folder_paths.models_dir, 'llm'))
Salt_LLM_supported_file_extensions = set(['.gguf'])
folder_paths.folder_names_and_paths["llm"] = ([Salt_LLM_DIR], Salt_LLM_supported_file_extensions)


class LLMGemini:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/gemini/
    @Source: 
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "models/gemini-pro",
                    "models/gemini-1.5-pro-latest",
                    "models/gemini-pro-vision",
                    "models/gemini-1.0-pro",					
                    "models/gemini-ultra",
                ],),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "dynamicPrompts": False, 
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name:str, api_key:str) -> Dict[str, Any]:
        llm = Gemini(model_name=model_name, api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "llm_name": model_name, "embed_model": embed_model, "embed_name": "bge-small-en-v1.5"},)


class GeminiMultiModalLoader:
    """
    Load Gemini Multi-Modal Model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "models/gemini-pro",
                    "models/gemini-1.5-pro-latest",
                    "models/gemini-pro-vision",
                    "models/gemini-1.0-pro",					
                    "models/gemini-ultra",
                ],),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name: str, api_key: str) -> dict:
        gemini = GeminiMultiModal(model_name=model_name, api_key=api_key)
        embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)
        return ({"llm": gemini, "llm_name": model_name, "embed_model": embed_model, "embed_name": "embedding-001"},)


LOCAL_FILES = folder_paths.get_filename_list("llm")
class LLMLlamaCPP:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/llama_cpp/
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (LOCAL_FILES, ), 
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name: str) -> Dict[str, Any]:
        path = folder_paths.get_full_path('llm', model_name)
        llm = LlamaCPP(model_path=path)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "llm_name": model_name,"embed_model": embed_model, "embed_name": "BAAI/bge-small-en-v1.5"},)


class LLMMistralAI:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/mistralai/
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "open-mistral-7b",
                    "open-mixtral-8x7b",
                    "mistral-small-latest",
                    "mistral-medium-latest",
                    "mistral-large-latest",
                ],),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("MISTRAL_API_KEY", "")
                }),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name:str, api_key:str) -> Dict[str, Any]:
        llm = MistralAI(model_name=model_name, api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "embed_model":embed_model},)


class LLMOllama:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "llama2",
                ],), 
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name:str) -> Dict[str, Any]:
        llm = Ollama(model=model_name)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "embed_model":embed_model},)


class LLMOpenAI:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/openai/
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(ALL_AVAILABLE_MODELS),),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("OPENAI_API_KEY", "")
                }),
                "embedding_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-ada-002"},
                ),
            },
            "optional": {
                "multimodal": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_EMBED_MODEL")
    RETURN_NAMES = ("llm_model", "embed_model_only")

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model:str, embedding_model:str, api_key:str, multimodal:bool = False) -> Dict[str, Any]:
        llm = OpenAI(model=model, api_key=api_key) if not multimodal else OpenAIMultiModal(model=model, api_key=api_key)
        embed_model = OpenAIEmbedding(model_name=embedding_model, api_key=api_key,)
        return ({"llm":llm, "llm_name": model, "embed_model":embed_model, "embed_name": embedding_model}, {"embed_model": embed_model, "embed_name": embedding_model})


# GROQ


class LLMGroq:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/groq/
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],),
                "groq_api_key": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GROQ_API_KEY", "")
                }),
                "embedding_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-ada-002"},
                ),
            },
            "optional": {
                "openai_api_key": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": os.environ.get("OPENAI_API_KEY", "")
                }),
            }
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_EMBED_MODEL")
    RETURN_NAMES = ("llm_model", "embed_model_only")

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model: str, groq_api_key: str, embedding_model:str, openai_api_key:str = None) -> Dict[str, Any]:
        llm = Groq(model=model, api_key=groq_api_key)
        embed_model = OpenAIEmbedding(model_name=embedding_model, api_key=openai_api_key,)
        return ({"llm": llm, "llm_name": model, "embed_model": embed_model, "embed_name": embedding_model}, {"embed_model": embed_model, "embed_name": embedding_model})


# MODEL OPTIONS


class LLMOpenAIModelOpts:
    """
    Sets various options for the model, and embedding.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "model_temperature": ("FLOAT", {"min": 0, "max": 1.0, "default": DEFAULT_TEMPERATURE, "step": 0.001}),
                "model_max_tokens": ("INT", {"min": 8, "default": 4096}),
                "model_api_max_retries": ("INT", {"min": 1, "max": 12, "default": 3}),
                "model_api_timeout": ("INT", {"min": 8, "max": 120, "default": 60}),
                "model_reuse_anyscale_client": ("BOOLEAN", {"default": True}),
                
                "multimodal_max_new_tokens": ("INT", {"min": 8, "default": 300}),
                "multimodal_image_detail": (["low", "high", "auto"],),

                "embed_batch_size": ("INT", {"min": 8, "default": DEFAULT_EMBED_BATCH_SIZE}),
                "embed_dimensions": ("INT", {"min": 1, "default": DEFAULT_EMBEDDING_DIM}),
                "embed_api_max_retries": ("INT", {"min": 1, "max": 12, "default": 3}),
                "embed_api_timeout": ("INT", {"min": 8, "max": 120, "default": 60}),
                "embed_reuse_anyscale_client": ("BOOLEAN", {"default": True}),
                
                "model_additional_kwargs": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "placeholder": "Additional model kwargs JSON"}),
                "embed_additional_kwargs": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "placeholder": "Additional embed kwargs JSON"}),
                "model_system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "System directions, or rules to follow globally across nodes."}),
            }
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "set_options"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders/Options"

    def set_options(self, llm_model:Dict[str, Any], **kwargs) -> Dict[str, Any]:
        llm = llm_model['llm']
        embed = llm_model['embed_model']
        
        # LLM Options
        llm.temperature = kwargs.get("model_temperature", DEFAULT_TEMPERATURE)
        llm.max_retries = kwargs.get("model_api_max_retries", 3)
        llm.reuse_client = kwargs.get("model_reuse_anyscale_client", True)
        llm.additional_kwargs = json.loads(kwargs.get("model_additional_kwargs", {}).strip()) if kwargs.get("model_additional_kwargs", {}).strip() != "" else {} # Default to `None` if empty string
        llm.system_prompt = kwargs.get("model_system_prompt", None)

        # Embed Options
        embed.embed_batch_size = kwargs.get("embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)
        embed.dimensions = kwargs.get("embed_dimensions", DEFAULT_EMBEDDING_DIM) if kwargs.get("embed_dimensions", DEFAULT_EMBEDDING_DIM) > 0 else None # Default to `None` if not above 0
        embed.additional_kwargs = json.loads(kwargs.get("embed_additional_kwargs", {}).strip()) if kwargs.get("embed_additional_kwargs", "").strip() != "" else {} # Default to `None` if empty string
        embed.max_retries = kwargs.get("embed_api_max_retries", 3)
        embed.timeout = kwargs.get("embed_api_timeout", 60)
        embed.reuse_client = kwargs.get("embed_reuse_anyscale_client", True)

        if isinstance(llm, OpenAIMultiModal):
            llm.max_new_tokens = kwargs.get("multimodal_max_new_tokens", 300)
            llm.image_detail = kwargs.get("multimodal_image_detail", "low")
        else:
            llm.max_tokens = kwargs.get("model_max_tokens", 4096)

        llm_model['llm'] = llm
        llm_model['embed_model'] = embed

        return (llm_model,)


NODE_CLASS_MAPPINGS = {
    "LLMOpenAIModel": LLMOpenAI,
    "LLMOpenAIModelOpts": LLMOpenAIModelOpts,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMOpenAIModel": "∞ OpenAI Model",
    "LLMOpenAIModelOpts": "∞ OpenAI Model Options",
}

if HAS_LLAMA_CPP:
    NODE_CLASS_MAPPINGS["LLMLlamaCPPModel"] = LLMLlamaCPP
    NODE_DISPLAY_NAME_MAPPINGS["LLMLlamaCPPModel"] = "∞ LlamaCPP Model"

if HAS_MISTRAL:
    NODE_CLASS_MAPPINGS["LLMMistralAI"] = LLMMistralAI
    NODE_DISPLAY_NAME_MAPPINGS["LLMMistralAI"] = "∞ MistralAI Model"
    
if HAS_GEMINI:
    NODE_CLASS_MAPPINGS["LLMGeminiModel"] = LLMGemini
    NODE_CLASS_MAPPINGS["GeminiMultiModalLoader"] = GeminiMultiModalLoader

    NODE_DISPLAY_NAME_MAPPINGS["LLMGeminiModel"] = "∞ Gemini Model"
    NODE_DISPLAY_NAME_MAPPINGS["GeminiMultiModalLoader"] = "∞ Gemini Multimodal Model"
     
if HAS_OLLAMA:
    NODE_CLASS_MAPPINGS["LLMOllamaModel"] = LLMOllama
    NODE_DISPLAY_NAME_MAPPINGS["LLMOllamaModel"] = "∞ Ollama Model"

if HAS_GROQ:
    NODE_CLASS_MAPPINGS["LLMGroqModel"] = LLMGroq
    NODE_DISPLAY_NAME_MAPPINGS["LLMGroqModel"] = "∞ Groq Model"