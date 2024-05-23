import json
import time

from typing import Dict, Any, List

from pprint import pprint

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import ImageDocument

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate

from llama_index.core import Settings

import tiktoken

from .. import MENU_NAME, SUB_MENU_NAME, logger
from ..modules.tokenization import MockTokenizer


# Documentation:
# https://github.com/run-llama/llama_index/tree/main/docs/examples/query_engine

class LLMQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "llm_index": ("LLM_INDEX", ),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def query_engine(self, llm_model, llm_index, query=None, llm_message=None):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))
        else:
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

        if query:
            if query.strip():
                query_components.append("user: " + query)
        query_components.append("assistant:")

        query_join = "\n".join(query_components)

        query_engine = llm_index.as_query_engine(llm=llm_model.get("llm", None), embed_model=llm_model.get("embed_model", None))
        response = query_engine.query(query_join)
        return (response.response,)
        

class LLMQueryEngineAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
                "top_k": ("INT", {"default": 10}),
                "similarity_cutoff": ("FLOAT", {"default": 0.7}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def query_engine(self, llm_model, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):

        model = llm_model['llm']
        embed_model = llm_model.get('embed_model', None)

        Settings.llm = model
        Settings.embed_model = embed_model
        
        if not embed_model:
            raise AttributeError("Unable to determine embed model from provided `LLM_MODEL` input.")

        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg.content).strip():
                    query_components.append(str(msg.content))
        else:
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

        if query and query.strip():
            query_components.append("user: " + query)

        query_components.append("assistant:")

        query_join = "\n".join(query_components)

        retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k, embed_model=embed_model)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

        response = query_engine.query(query_join)

        Settings.llm = None
        Settings.embed_model = None

        return (response.response,)
     
class LLMQueryEngineAsTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "code"}),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "A function that allows you to communicate with a document. Ask a question and this function will find information in the document and generate an answer."}),
                "llm_index": ("LLM_INDEX",),
            },
        }

    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ("query_tool",)

    FUNCTION = "return_tool"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"
    
    def return_tool(self, name, description, llm_index):
        def query_engine(query: str) -> str:
            query_components = []
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

            if query:
                if query.strip():
                    query_components.append("user: " + query)
            query_components.append("assistant:")
            query_join = "\n".join(query_components)

            query_engine = llm_index.as_query_engine()
            response = query_engine.query(query_join)
            return (response.response,)
        tool = {"name": name, "description": description, "function": query_engine}
        return (tool,)
    
# Query Engine
class LLMJSONQueryEngine:
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
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

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
            logger.error(error_message)
            return ("", "")

        query_engine = JSONQueryEngine(
            json_value = data,
            json_schema = schema,
            llm = llm_model['llm'],
            synthesize_response = True if output_mode == "Human Readable" else False,
        )

        response = query_engine.query(json_query)
        logger.data(response, indent=4)

        return (response, response.metadata["json_path_response_str"])

class LLMChatEngine:
        def __init__(self):
            self.chat_engine = None
            
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "llm_index": ("LLM_INDEX",),
                    "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ask a question"}),
                },
                "optional": {
                    "reset_engine": ("BOOLEAN", {"default": False})
                }
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "chat"
        CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

        def chat(self, llm_index, query:str, reset_engine:bool = False) -> str:
            if not self.chat_engine or reset_engine:
                self.chat_engine = llm_index.as_chat_engine()
            response = self.chat_engine.chat(query)
            return (response.response,)

class LLMChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                  "llm_context": ("LLM_CONTEXT", ),
                  "llm_message": ("LIST", {}),
                  "llm_documents": ("DOCUMENT", {}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("response", )

    FUNCTION = "chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def chat(self, llm_model:Dict[str, Any], prompt:str, llm_context:Any = None, llm_message:List[ChatMessage] = None, llm_documents:List[Any] = None) -> str:

        embed_model = llm_model.get('embed_model', None)

        # Spoof documents -- Why can't we just talk to a modeL?
        if not llm_documents:
            documents = [Document(text="null", extra_info={})]
        else:
            documents = llm_documents

        index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model,
            service_context=llm_context,
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)]
        )
        
        if not llm_message:
            llm_message = [ChatMessage(MessageRole.USER, content="")]

        if not prompt.strip():
            prompt = "null"

        template = ChatPromptTemplate(message_templates=llm_message)
        query_engine = index.as_query_engine(llm=llm_model.get("llm", None), embed_model=embed_model, text_qa_template=template)
        response = query_engine.query(prompt)
        return (response.response, )
    
class LLMChatBot:
    def __init__(self):
        self.chat_history = []
        self.history = []
        self.token_map = {} 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "llm_context": ("LLM_CONTEXT", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompt": False}),
            },
            "optional": {
                "reset_engine": ("BOOLEAN", {"default": False}),
                "user_nickname": ("STRING", {"default": "User"}),
                "system_nickname": ("STRING", {"default": "Assistant"}),
                "char_per_token": ("INT", {"min": 1, "default": 4})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("chat_history", "response", "chat_token_count")

    FUNCTION = "chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def chat(self, llm_model: Dict[str, Any], llm_context:Any, prompt: str, reset_engine:bool = False, user_nickname:str = "User", system_nickname:str = "Assistant", char_per_token:int = 4) -> str:

        if reset_engine:
            self.chat_history.clear()
            self.history.clear()
            self.token_map.clear()

        max_tokens = llm_model.get("max_tokens", 4096)
        using_mock_tokenizer = False
        try:
            tokenizer = tiktoken.encoding_for_model(llm_model.get('llm_name', 'gpt-3-turbo'))
        except Exception:
            using_mock_tokenizer = True
            tokenizer = MockTokenizer(max_tokens, char_per_token=char_per_token)

        if not self.chat_history:
            system_prompt = getattr(llm_model['llm'], "system_prompt", None)
            if system_prompt not in (None, ""):
                initial_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
                self.chat_history.append(initial_msg)
                self.token_map[0] = tokenizer.encode(system_prompt)

        # Tokenize and count initial tokens
        cumulative_token_count = 0
        for index, message in enumerate(self.chat_history):
            if index not in self.token_map:
                self.token_map[index] = tokenizer.encode(message.content)
            #if not using_mock_tokenizer:
            cumulative_token_count += len(self.token_map[index])
            #else:
            #    cumulative_token_count += tokenizer.count(self.token_map[index])

        # Prune messages from the history if over max_tokens
        index = 0
        while cumulative_token_count > max_tokens and index < len(self.chat_history):
            tokens = self.token_map[index]
            token_count = len(tokens) #if not using_mock_tokenizer else tokenizer.count(tokens)
            if token_count > 1:
                tokens.pop(0)
                self.chat_history[index].content = tokenizer.decode(tokens)
                cumulative_token_count -= 1
            else:
                cumulative_token_count -= token_count
                self.chat_history.pop(index)
                self.token_map.pop(index)
                for old_index in list(self.token_map.keys()):
                    if old_index > index:
                        self.token_map[old_index - 1] = self.token_map.pop(old_index)
                continue
            index += 1
                
        history_string = ""
        reply_string = ""
        documents = []

        # Build prior history string
        for history in self.history:
            user, assistant, timestamp = history
            history_string += f"""[{user_nickname}]: {history[user]}

[{system_nickname}]: {history[assistant]}

"""
        # Spoof documents -- Why can't we just talk to a modeL?
        documents = [Document(text="null", extra_info={})]

        index = VectorStoreIndex.from_documents(
            documents, 
            service_context=llm_context,
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)]
        )
        chat_engine = index.as_chat_engine(chat_mode="best")

        response = chat_engine.chat(prompt, chat_history=self.chat_history)

        response_dict = {
            user_nickname: prompt, 
            system_nickname: response.response,
            "timestamp": str(time.time())
        }

        user_cm = ChatMessage(role=MessageRole.USER, content=prompt)
        system_cm = ChatMessage(role=MessageRole.SYSTEM, content=response.response)
        self.chat_history.append(user_cm)
        self.chat_history.append(system_cm)

        self.history.append(response_dict)

        reply_string = response.response

        history_string += f"""[{user_nickname}]: {prompt}

[{system_nickname}]: {response.response}"""
            
        return (history_string, reply_string, cumulative_token_count)


class LLMComplete:
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
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def complete(self, llm_model:Dict[str, Any], prompt:str) -> str:
        response = llm_model['llm'].complete(prompt)
        return (response.text, )


class LLMMultiModalImageEvaluation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "image_documents": ("DOCUMENT",),
                "llm_message": ("LIST",),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("response", )

    FUNCTION = "complete"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def complete(self, llm_model, image_documents, llm_message):

        model = llm_model.get("llm", None)

        if not model:
            raise ValueError("LLMMultiModalImageEvaluation unable to detect valid model")

        prompt = ""
        llm_message = sorted(llm_message, key=lambda message: message.role.value)
        for msg in llm_message:
            if isinstance(msg, ChatMessage) and msg.role == MessageRole.SYSTEM:
                if "SYSTEM:" not in prompt:
                    prompt += "SYSTEM: "
                prompt += msg.content + "\n\n"
            if isinstance(msg, ChatMessage) and msg.role == MessageRole.USER:
                if "USER:" not in prompt:
                    prompt += "USER: "
                prompt += msg.content + "\n\n"
            if isinstance(msg, str):
                prompt += msg + "\n\n"
        
        
        response = model.complete(
            prompt=prompt,
            image_documents=image_documents
        )

        return (response.text, )


NODE_CLASS_MAPPINGS = {
      
    # Query
    "LLMQueryEngine": LLMQueryEngine,
    "LLMQueryEngineAdv": LLMQueryEngineAdv,
      
    # Chat
    "LLMChatEngine": LLMChatEngine,
    "LLMChat": LLMChat,
    "LLMComplete": LLMComplete,
    "LLMChatBot": LLMChatBot,
    "LLMMultiModalImageEvaluation": LLMMultiModalImageEvaluation,
    
    # Agent
    "LLMQueryEngineAsTool": LLMQueryEngineAsTool,
    #"SaltJSONQueryEngine": SaltJSONQueryEngine,

}

NODE_DISPLAY_NAME_MAPPINGS = {
      
    # Query
    "LLMQueryEngine": "∞ Query Engine",
    "LLMQueryEngineAdv": "∞ Query Engine (Advanced)",
    #"SaltJSONQueryEngine": "JSON Query Engine",
      
    # Chat
    "LLMChatEngine": "∞ Documents Chat Engine",
    "LLMChat": "∞ Multi Query",
    "LLMComplete": "∞ Complete Query",
    "LLMChatBot": "∞ Chat Engine",
    "LLMMultiModalImageEvaluation": "∞ Image Documents Evaluation",

    # Agent
    "LLMQueryEngineAsTool": "∞ Query Engine As Tool",

}
