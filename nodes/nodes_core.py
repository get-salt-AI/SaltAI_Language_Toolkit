import logging
import re
import sys
from typing import Dict, Any


from pprint import pprint

#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

import openai
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import SummaryIndex, VectorStoreIndex

#from llama_index.core.retrievers import VectorIndexRetriever
#from llama_index.core.query_engine import RetrieverQueryEngine
#from llama_index.core.postprocessor import SimilarityPostprocessor


from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
#from llama_index.core.base.embeddings.base import BaseEmbedding
#from llama_index.core.schema import Document, BaseNode, IndexGraph, LLM, BasePromptTemplate
from llama_index.core.indices.tree import TreeIndex

#from llama_index.core.storage.storage_context import StorageContext
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
class LLMChatMessages:
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


class LLMChatMessagesAdv:
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


class LLMChatMessageConcat:
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
class LLMServiceContextDefault:
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
            embed_model=llm_model['embed_model'],
        )
        return (service_context,)

class LLMServiceContextAdv:
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
class LLMVectorStoreIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "document": ("DOCUMENT",),
            },
            "optional": {
                "optional_llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_model, document, optional_llm_context = None):
        
        #document = cast(Sequence[Document], document) # This could be why documents are not working correctly
        embed_model = llm_model.get("embed_model", None)
        
        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")
        
        index = VectorStoreIndex.from_documents(document, embed_model=embed_model, service_context=optional_llm_context)
        return (index,)


class LLMSummaryIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "document": ("DOCUMENT",),
            },
            "optional": {
                "optional_llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_model, document, optional_llm_context=None):
        index = SummaryIndex.from_documents(document, embed_model=llm_model['embed_model'], service_context=optional_llm_context or None)
        return (index,)


class LLMTreeIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_nodes": ("LLM_NODES",),
            },
            "optional": {
                "service_context": ("LLM_CONTEXT",),
                "num_children": ("INT", {"default": 10}),
                "build_tree": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_model, llm_nodes, service_context=None, num_children=10, build_tree=True):
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
class LLMSentenceSplitterNodeCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": ("DOCUMENT",),
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

    def create_nodes(self, document, chunk_size=1024, chunk_overlap=20):
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(document, show_progress=False)        
        return (nodes,)


# TODO
class LLMSemanticSplitterNodeParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": ("DOCUMENT",),
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

    def semantic_nodes(self, document, llm_embed_model, buffer_size=1, sentence_splitter=None, include_metadata=True, include_prev_next_rel=True):
        parser = SemanticSplitterNodeParser(
            embed_model=llm_embed_model,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
        return (parser.build_semantic_nodes_from_documents(document, show_progress=True), )


NODE_CLASS_MAPPINGS = {

    # Messages
    "LLMChatMessages": LLMChatMessages,
    "LLMChatMessagesAdv": LLMChatMessagesAdv,
    "LLMChatMessageConcat": LLMChatMessageConcat,

    # Service Context
    "LLMServiceContextDefault": LLMServiceContextDefault,
    "LLMServiceContextAdv": LLMServiceContextAdv,

    # Indexing
    "LLMVectorStoreIndex": LLMVectorStoreIndex,
    "LLMSummaryIndex": LLMSummaryIndex,
    "LLMTreeIndex": LLMTreeIndex,

    # Nodes
    "LLMSentenceSplitterNodeCreator": LLMSentenceSplitterNodeCreator,

    # Parser
    "LLMSemanticSplitterNodeParser": LLMSemanticSplitterNodeParser,

}

NODE_DISPLAY_NAME_MAPPINGS = {

    # Messages
    "LLMChatMessages": "∞ Message",
    "LLMChatMessagesAdv": "∞ Message (Advanced)",
    "LLMChatMessageConcat": "∞ Messages Concat",

    # Service Context
    "LLMServiceContextDefault": "∞ Service Context",
    "LLMServiceContextAdv": "∞ Service Context (Advanced)",

    # Indexing
    "LLMVectorStoreIndex": "∞ Vector Store Index",
    "LLMSummaryIndex": "∞ Summary Index",
    "LLMTreeIndex": "∞ Tree Index",

    # Nodes
    "LLMSentenceSplitterNodeCreator": "∞ Setence Splitter Node Creator",

    # Parsers
    "LLMSemanticSplitterNodeParser": "∞ Semantics Splitter Node Parser",

}
