"""
@NOTE:
	Classes are sorted close to alphabetically

@TODO: 

	
@REQUIREMENTS:
	llama-index
	# llama-index-llms-gemini #LLMGemini
	llama-index-llms-llama-cpp #LLMLlamaCPP
	# llama-index-llms-mistralai #LLMMistralAI
	# llama-index-llms-ollama #LLMOllama
	llama-index-llms-openai #LLMOpenAI

	# pip install -q llama-index google-generativeai !!!!! this is weird, investigate

@Source:


@BUGS: 
	Gemini is a Microsoft model and needs an API key and some OS environment stuff? - Daniel
"""

import json
import logging
import os
import re
import sys
from typing import Dict, Any

# Implementation of LLM folder generated dropdowns using ComfyUI.folder_paths
import folder_paths

Salt_LLM_DIR = os.path.abspath(os.path.join(folder_paths.models_dir, 'llm'))
Salt_LLM_supported_file_extensions = set(['.gguf'])
folder_paths.folder_names_and_paths["llm"] = ([Salt_LLM_DIR], Salt_LLM_supported_file_extensions)


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
#from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
#from llama_index.llms.mistralai import MistralAI
#from llama_index.llms.ollama import Ollama
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType


class LLMGemini:
	#@Documentation: https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/gemini.ipynb
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {},
		}

	RETURN_TYPES = ("LLM_MODEL", )
	RETURN_NAMES = ("Model", )

	FUNCTION = "load_model"
	CATEGORY = "SALT/Llama-Index/Loaders"

	def load_model(self, model: str) -> Dict[str, Any]:
		name = f"{model}_{embed_model}"
		llm = Gemini()
		embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
		return ({"llm":llm, "embed_model":embed_model},)


class LLMLlamaCPP:
	#@Documentation: https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb

	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"ModelName": (folder_paths.get_filename_list("llm"), ), 
			},
		}

	RETURN_TYPES = ("LLM_MODEL", )
	RETURN_NAMES = ("Model", )

	FUNCTION = "load_model"
	CATEGORY = "SALT/Llama-Index/Loaders"

	def load_model(self, ModelName: str) -> Dict[str, Any]:
		name = f"{ModelName}"
		path = folder_paths.get_full_path('llm',ModelName)
		llm = LlamaCPP(model_path=path)
		embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
		return ({"llm":llm, "embed_model":embed_model},)


class LLMMistralAI:
	#@Documentation: https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/mistralai.ipynb
 
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"ModelPath": (folder_paths.get_filename_list("llm"), ), 
				"api_key": ("STRING", {
					"multiline": False, 
					"dynamicPrompts": False, 
					"default": os.environ.get("MISTRAL_API_KEY", "")
				}),
			},
		}

	RETURN_TYPES = ("LLM_MODEL", )
	RETURN_NAMES = ("Model", )

	FUNCTION = "load_model"
	CATEGORY = "SALT/Llama-Index/Loaders"

	def load_model(self, ModelPath: str, api_key:str) -> Dict[str, Any]:
		name = f"{ModelPath}"
		path = folder_paths.get_full_path('llm',ModelPath)
		llm = MistralAI(model=path)
		embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
		return ({"llm":llm, "embed_model":embed_model},)


class LLMOllama:
	#@Documentation: https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/ollama.ipynb
	
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"ModelPath": (folder_paths.get_filename_list("llm"), ), 
			},
		}

	RETURN_TYPES = ("LLM_MODEL", )
	RETURN_NAMES = ("Model", )

	FUNCTION = "load_model"
	CATEGORY = "SALT/Llama-Index/Loaders"

	def load_model(self, ModelPath:str) -> Dict[str, Any]:
		name = f"{ModelPath}"
		path = folder_paths.get_full_path('llm',ModelPath)
		llm = Ollama(model=path)
		embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
		return ({"llm":llm, "embed_model":embed_model},)


class LLMOpenAI:
	#@Documentation: https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/openai.ipynb

	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"ModelName": ([
					"gpt-3.5-turbo-0125",
					"gpt-3.5-turbo",
					"gpt-3.5-turbo-1106",
					"gpt-3.5-turbo-instruct",
					"gpt-3.5-turbo-16k",
					"gpt-3.5-turbo-0613",
					"gpt-3.5-turbo-16k-0613",
					"gpt-4-0125-preview",
					"gpt-4-turbo-preview",
					"gpt-4-1106-preview",
					"gpt-4-vision-preview",
					"gpt-4-1106-vision-preview",
					"gpt-4",
					"gpt-4-0613",
					"gpt-4-32k",
					"gpt-4-32k-0613"
				],),
				"EmbedModel": (
					sorted([x.value for x in OpenAIEmbeddingModelType]),
					{"default": "text-embedding-3-small"},
				),
				"api_key": ("STRING", {
					"multiline": False, 
					"dynamicPrompts": False, 
					"default": os.environ.get("GOOGLE_API_KEY", "")
				}),
			},
		}

	RETURN_TYPES = ("LLM_MODEL", )
	RETURN_NAMES = ("Model", )

	FUNCTION = "load_model"
	CATEGORY = "SALT/Llama-Index/Loaders"

	def load_model(self, ModelName:str, EmbedModel:str, api_key:str) -> Dict[str, Any]:
		name = f"{ModelName}"
		llm = OpenAI(model=ModelName)
		llm.api_key = api_key
		embed_model = OpenAIEmbedding(model=EmbedModel)
		return ({"llm":llm, "embed_model":embed_model},)


NODE_CLASS_MAPPINGS = {
#	"LLMGeminiModel": LLMGemini,
	"LLMLlamaCPPModel": LLMLlamaCPP,
#	"LLMMistralAIModel": LLMMistralAI,
#	"LLMOllamaModel": LLMOllama,
	"LLMOpenAIModel": LLMOpenAI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
#	"LLMGeminiModel": "∞ Gemini Model",
	"LLMLlamaCPPModel": "∞ LlamaCPP Model",
#	"LLMMistralAIModel": "∞ MistralAI Model",
#	"LLMOllamaModel": "∞ Ollama Model",
	"LLMOpenAIModel": "∞ OpenAI Model",
}
