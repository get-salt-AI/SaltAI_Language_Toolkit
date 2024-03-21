
# ATTN: Classes are sorted alphabetically

"""
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

# Set default embedding model to the free one:
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


# Requirements:
# llama-index
# llama-index-llms-gemini
# pip install -q llama-index google-generativeai !!!!! this is weird, investigate


# Imports:
# from llama_index.llms.gemini import Gemini

# Documentation:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/gemini.ipynb

# Binding:
class SaltGemini:
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

"""
# Requirements:
# llama-index
# llama-index-embeddings-huggingface
# llama-index-llms-llama-cpp
"""
# Imports:
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

# Documentation:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb

# Binding:
class SaltLlamaCPP:
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


# Requirements:
# llama-index
# llama-index-llms-mistralai

# Imports:
from llama_index.llms.mistralai import MistralAI

# Documentation:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/mistralai.ipynb

# Binding:
class SaltMistralAI:
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



# Requirements:
# llama-index
# llama-index-llms-ollama

# Imports:
from llama_index.llms.ollama import Ollama

# Documentation:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/ollama.ipynb

# Binding:
class SaltOllama:
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


# Requirements:
# llama-index
# llama-index-llms-openai

# Imports:
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

# Documentation:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/openai.ipynb

# Binding:
class SaltOpenAI:
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
#	"GeminiModel": SaltGemini,
	"LlamaCPPModel": SaltLlamaCPP,
#	"MistralAIModel": SaltMistralAI,
#	"OllamaModel": SaltOllama,
	"OpenAIModel": SaltOpenAI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
#	"GeminiModel": "∞ Gemini Model",
	"LlamaCPPModel": "∞ LlamaCPP Model",
#	"MistralAIModel": "∞ MistralAI Model",
#	"OllamaModel": "∞ Ollama Model",
	"OpenAIModel": "∞ OpenAI Model",
}
