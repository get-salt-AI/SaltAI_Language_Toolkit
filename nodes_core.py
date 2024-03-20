
"""
	BUGS:
		Switched all instances of LLM_DOCUMENTS to DOCUMENT
"""

import json
import logging
import os
import re
import sys
from typing import Dict, Any, Sequence, List, cast

import folder_paths

from pprint import pprint

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

import openai
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, NodeParser
#from llama_index.core.base.embeddings.base import BaseEmbedding
#from llama_index.core.schema import Document, BaseNode, IndexGraph, LLM, BasePromptTemplate
from llama_index.core.indices.tree import TreeIndex

from llama_index.core.indices.struct_store import JSONQueryEngine as BaseJSONQueryEngine

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import Document


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def valid_url(url):
	regex = re.compile(
		r'^(?:http|ftp)s?://'
		r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
		r'localhost|'
		r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
		r'(?::\d+)?'
		r'(?:/?|[/?]\S+)$', re.IGNORECASE)
	return re.match(regex, url) is not None

# OpenAI

class SaltChatMessages:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "prompt"}),
				"role": (["SYSTEM", "USER"],),
			},
		}

	RETURN_TYPES = ("LIST", )
	RETURN_NAMES = ("llm_message", )

	FUNCTION = "prepare_messages"
	CATEGORY = "SALT/Llama-Index/Messages"

	def prepare_messages(self, prompt, role):
		messages = [
				ChatMessage(role=MessageRole.SYSTEM if role == "SYSTEM" else MessageRole.USER, content=prompt ),
		]
		return (messages,)


class SaltChatMessagesAdv:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "You are a dog, you cannot speak, only woof, and react as a dog would."}),
				"user_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "plaeholder": "What is your name?"}),
			},
		}

	RETURN_TYPES = ("LIST", )
	RETURN_NAMES = ("llm_message", )

	FUNCTION = "prepare_messages"
	CATEGORY = "SALT/Llama-Index/Messages"

	def prepare_messages(self, system_prompt, user_prompt):
		messages = [
				ChatMessage(role=MessageRole.SYSTEM, content=system_prompt ),
				ChatMessage(role=MessageRole.USER, content=user_prompt ),
		]
		return (messages,)


class SaltChatMessageConcat:
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"message_a": ("LIST", ),
				"message_b": ("LIST", ),
			},
		}

	RETURN_TYPES = ("LIST",)
	RETURN_NAMES = ("llm_message", )

	FUNCTION = "concat_messages"
	CATEGORY = "SALT/Llama-Index/Messages"

	def concat_messages(self, message_a, message_b):
		return (message_a + message_b, )


# Service Context    
class SaltServiceContextDefault:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_model": ("LLM_MODEL",),
			},
		}

	RETURN_TYPES = ("LLM_CONTEXT",)
	RETURN_NAMES = ("llm_context",)

	FUNCTION = "context"
	CATEGORY = "SALT/Llama-Index/Context"

	def context(self, llm_model: Dict[str, Any]):
		service_context = ServiceContext.from_defaults(
			llm=llm_model['llm'],
			embed_model=Settings.embed_model,
		)
		return (service_context,)

class SaltServiceContextAdv:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_model": ("LLM_MODEL",),
			},
			"optional": {
				"llm_embed_model": ("LLM_EMBED",),
				"llm_node_parser": ("LLM_NODE_PARSER",),
				"enable_chunk_overlap": ("BOOLEAN", {"default": True}),
				"chunk_overlap": ("INT", {"default": 50, "min": 0, "max": 100}),
				"enable_context_window": ("BOOLEAN", {"default": True}),
				"context_window": ("INT", {"default": 4096, "min": 2048, "max": 8192}),
				"enable_num_output": ("BOOLEAN", {"default": True}),
				"num_output": ("INT", {"default": 256, "min": 64, "max": 1024}),
				"enable_chunk_size_limit": ("BOOLEAN", {"default": True}),
				"chunk_size_limit": ("INT", {"default": 1024, "min": 512, "max": 2048}),
			},
		}

	RETURN_TYPES = ("LLM_CONTEXT",)
	RETURN_NAMES = ("llm_context",)

	FUNCTION = "context"
	CATEGORY = "SALT/Llama-Index/Context"

	def context(self, 
		llm_model:Dict[str, Any], 
		llm_embed_model="default", 
		llm_node_parser=None, 
		enable_chunk_size=True, 
		chunk_size=1024, 
		enable_chunk_overlap=True,
		chunk_overlap=50, 
		enable_context_window=True, 
		context_window=4096, 
		enable_num_output=True,
		num_output=256, 
		enable_chunk_size_limit=True,
		chunk_size_limit=1024
	):
		prompt_helper = None
		if enable_context_window and enable_num_output:
			prompt_helper = PromptHelper(
				context_window=context_window if enable_context_window else None,
				num_output=num_output if enable_num_output else None,
				chunk_overlap_ratio=(chunk_overlap / 100.0) if enable_chunk_overlap else None,
				chunk_size_limit=chunk_size_limit if enable_chunk_size_limit else None,
			)

		service_context = ServiceContext.from_defaults(
				llm=llm_model['llm'],
				prompt_helper=prompt_helper,
				embed_model=llm_embed_model if llm_embed_model != "default" else None,
				node_parser=llm_node_parser,
				chunk_size=chunk_size if enable_chunk_size else None,
				chunk_overlap=chunk_overlap if enable_chunk_overlap else None,
		)
		return (service_context,)


# Index Store
class SaltVectorStoreIndex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_documents": ("DOCUMENT",),
				"llm_context": ("LLM_CONTEXT",),
			},
		}

	RETURN_TYPES = ("LLM_INDEX",)
	RETURN_NAMES = ("llm_index",)

	FUNCTION = "index"
	CATEGORY = "SALT/Llama-Index/Indexing"

	def index(self, 
		llm_documents: Sequence[Document], 
		llm_context: StorageContext
	):
		llm_documents = cast(Sequence[Document], llm_documents)
		index = VectorStoreIndex.from_documents(llm_documents, service_context=llm_context)
		return (index,)


class SaltSummaryIndex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_documents": ("DOCUMENT",),
				"llm_context": ("LLM_CONTEXT",),
			},
		}

	RETURN_TYPES = ("LLM_INDEX",)
	RETURN_NAMES = ("llm_index",)

	FUNCTION = "index"
	CATEGORY = "SALT/Llama-Index/Indexing"

	def index(self, llm_documents, llm_context):
		index = SummaryIndex.from_documents(llm_documents, service_context=llm_context)
		return (index,)


class SaltTreeIndex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_nodes": ("LLM_NODES",),
				"service_context": ("LLM_CONTEXT",),
			},
			"optional": {
				"num_children": ("INT", {"default": 10}),
				"build_tree": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("LLM_INDEX",)
	RETURN_NAMES = ("llm_index",)

	FUNCTION = "index"
	CATEGORY = "SALT/Llama-Index/Indexing"

	def index(self, llm_nodes, service_context, num_children=10, build_tree=True):
		index = TreeIndex(
			nodes=llm_nodes,
			num_children=num_children,
			build_tree=build_tree,
			use_async=False,
			show_progress=True,
			service_context=service_context,
		)
		return (index,)


# Node Parser
class SaltSentenceSplitterNodeCreator:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_documents": ("DOCUMENT",),
			},
			"optional": {
				"chunk_size": ("INT", {"default": 1024, "min": 1}),
				"chunk_overlap": ("INT", {"default": 20, "min": 0}),
			},
		}

	RETURN_TYPES = ("LLM_NODES",)
	RETURN_NAMES = ("llm_nodes",)

	FUNCTION = "create_nodes"
	CATEGORY = "SALT/Llama-Index/Tools"

	def create_nodes(self, llm_documents, chunk_size=1024, chunk_overlap=20):
		node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
		nodes = node_parser.get_nodes_from_documents(llm_documents, show_progress=False)        
		return (nodes,)


# TODO
class SaltSemanticSplitterNodeParser:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_documents": ("DOCUMENT",),
				"llm_embed_model": ("LLM_EMBED_MODEL",),
			},
			"optional": {
				"buffer_size": ("INT", {"default": 1, "min": 1}),
				"sentence_splitter": ("LLM_SENTENCE_SPLITTER",),
				"include_metadata": ("BOOLEAN", {"default": True}),
				"include_prev_next_rel": ("BOOLEAN", {"default": True}),
			},
		}

	RETURN_TYPES = ("LLM_NODE_PARSER",)
	RETURN_NAMES = ("llm_node_parser",)

	FUNCTION = "semantic_nodes"
	CATEGORY = "SALT/Llama-Index/Parsing"

	def semantic_nodes(self, llm_documents, llm_embed_model, buffer_size=1, sentence_splitter=None, include_metadata=True, include_prev_next_rel=True):
		parser = SemanticSplitterNodeParser(
			embed_model=llm_embed_model,
			buffer_size=buffer_size,
			sentence_splitter=sentence_splitter,
			include_metadata=include_metadata,
			include_prev_next_rel=include_prev_next_rel,
		)
		return (parser.build_semantic_nodes_from_documents(llm_documents, show_progress=True), )


# Query Engine
    

class SaltJSONQueryEngine:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_model": ("LLM_MODEL",),
				"json_data": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON data here..."}),
				"json_schema": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON schema here..."}),
				"json_query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ener your JSON query / question here..."}),
				"output_mode": (["RAW", "Human Readable"],),
			}
		}

	RETURN_TYPES = ("STRING", "STRING")
	RETURN_NAMES = ("result", "json_path")

	FUNCTION = "query_engine"
	CATEGORY = "SALT/Llama-Index/Querying"

	def query_engine(self,
		llm_model:Dict[str, Any], 
		json_schema, 
		json_data, 
		json_query, 
		output_mode
	):
		try:
			schema = json.loads(json_schema)
			data = json.loads(json_data)
		except json.JSONDecodeError as e:
			error_message = f"JSON parsing error: {str(e)}. Please ensure your JSON schema and data are correctly formatted."
			print(error_message)
			return (error_message, "")

		query_engine = BaseJSONQueryEngine(
			json_value = data,
			json_schema = schema,
			llm = llm_model['llm'],
			synthesize_response = True if output_mode == "Human Readable" else False,
		)

		response = query_engine.query(json_query)

		pprint(response, indent=4)

		return (response, response.metadata["json_path_response_str"])


# Binding:
class SaltChatEngine:
		@classmethod
		def INPUT_TYPES(cls):
			return {
				"required": {
					"llm_index": ("LLM_INDEX",),
					"query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ask a question"}),
				},
			}

		RETURN_TYPES = ("STRING",)
		FUNCTION = "chat"
		CATEGORY = "SALT/Llama-Index/Messages"

		def chat(self, llm_index, query:str) -> str:
			chat_engine = llm_index.as_chat_engine()
			response = chat_engine.chat(query)
			pprint(response, indent=4)
			return (response.response,)
		

# Binding:
class SaltChat:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_model": ("LLM_MODEL", ),
				"message": ("LIST", ),
			},
		}

	RETURN_TYPES = ("STRING", )
	RETURN_NAMES = ("response", )

	FUNCTION = "chat"
	CATEGORY = "SALT/Llama-Index/Messages"

	def chat(self, llm_model:Dict[str, Any], message) -> str:
		response = llm_model['llm'].chat(message)
		pprint(response, indent=4)
		return (response.message.content, )


# Binding:
class SaltComplete:
	def __init__(self):
		pass
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_model": ("LLM_MODEL", ),
				"prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "The circumference of the Earth is"}),
			},
		}

	RETURN_TYPES = ("STRING", )
	RETURN_NAMES = ("completion", )

	FUNCTION = "complete"
	CATEGORY = "SALT/Llama-Index/Messages"

	def complete(self, llm_model:Dict[str, Any], prompt:str) -> str:
		response = llm_model['llm'].complete(prompt)
		pprint(response, indent=4)
		return (response.text, )


NODE_CLASS_MAPPINGS = {

	# Messages
	"SaltChatMessages": SaltChatMessages,
	"SaltChatMessagesAdv": SaltChatMessagesAdv,
	"SaltChatMessageConcat": SaltChatMessageConcat,

	# Service Context
	"SaltServiceContextDefault": SaltServiceContextDefault,
	"SaltServiceContextAdv": SaltServiceContextAdv,

	# Indexing
	"SaltVectorStoreIndex": SaltVectorStoreIndex,
	"SaltSummaryIndex": SaltSummaryIndex,
	"SaltTreeIndex": SaltTreeIndex,

	# Nodes
	"SaltSentenceSplitterNodeCreator": SaltSentenceSplitterNodeCreator,

	# Parser
	"SaltSemanticSplitterNodeParser": SaltSemanticSplitterNodeParser,

	# Chat
	"SaltChatEngine": SaltChatEngine,
	"SaltChat": SaltChat,
	"SaltComplete": SaltComplete,


}

NODE_DISPLAY_NAME_MAPPINGS = {

	# Messages
	"SaltChatMessages": "∞ Message",
	"SaltChatMessagesAdv": "∞ Message (Advanced)",
	"SaltChatMessageConcat": "∞ Messages Concat",

	# Service Context
	"SaltServiceContextDefault": "∞ Service Context",
	"SaltServiceContextAdv": "∞ Service Context (Advanced)",

	# Indexing
	"SaltVectorStoreIndex": "∞ Vector Store Index",
	"SaltSummaryIndex": "∞ Summary Index",
	"SaltTreeIndex": "∞ Tree Index",

	# Nodes
	"SaltSentenceSplitterNodeCreator": "∞ Setence Splitter Node Creator",

	# Parsers
	"SaltSemanticSplitterNodeParser": "∞ Semantics Splitter Node Parser",

	# Chat
	"SaltChatEngine": "∞ Chat Engine",
	"SaltChat": "∞ Chat",
	"SaltComplete": "∞ Complete",

}
