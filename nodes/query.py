import json
import time

from typing import Dict, Any, List

from pprint import pprint

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core import Document, VectorStoreIndex

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole

import tiktoken


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
	CATEGORY = "SALT/Llama-Index/Querying"

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

		pprint(query_components, indent=4)

		query_join = "\n".join(query_components)

		query_engine = llm_index.as_query_engine(llm=llm_model.get("llm", None), embed_model=llm_model.get("embed_model", None))
		response = query_engine.query(query_join)
		pprint(response, indent=4)
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
	CATEGORY = "SALT/Llama-Index/Querying"

	def query_engine(self, llm_model, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):
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

		pprint(query_components, indent=4)
		query_join = "\n".join(query_components)

		retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k, embed_model=llm_model.get("embed_model", None))
		query_engine = RetrieverQueryEngine(
			retriever=retriever,
			node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
		)

		response = query_engine.query(query_join)
		pprint(response, indent=4)
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
	CATEGORY = "SALT/Llama-Index/Querying"
	
	def return_tool(self, name, description, llm_index):
		def query_engine(query: str) -> str:
			query_components = []
			query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

			if query:
				if query.strip():
					query_components.append("user: " + query)
			query_components.append("assistant:")
			pprint(query_components, indent=4)
			query_join = "\n".join(query_components)

			query_engine = llm_index.as_query_engine()
			response = query_engine.query(query_join)
			pprint(response, indent=4)
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

        query_engine = JSONQueryEngine(
            json_value = data,
            json_schema = schema,
            llm = llm_model['llm'],
            synthesize_response = True if output_mode == "Human Readable" else False,
        )

        response = query_engine.query(json_query)

        pprint(response, indent=4)

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
        CATEGORY = "SALT/Llama-Index/Querying"

        def chat(self, llm_index, query:str, reset_engine:bool = False) -> str:
            if not self.chat_engine or reset_engine:
                self.chat_engine = llm_index.as_chat_engine()
            response = self.chat_engine.chat(query)
            pprint(response, indent=4)
            return (response.response,)

from llama_index.core.prompts import ChatPromptTemplate

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
    CATEGORY = "SALT/Llama-Index/Querying"

    def chat(self, llm_model:Dict[str, Any], prompt:str, llm_context:Any = None, llm_message:List[ChatMessage] = None, llm_documents:List[Any] = None) -> str:
        response = llm_model['llm'].complete(prompt)

        # Spoof documents -- Why can't we just talk to a modeL?
        if not llm_documents:
            documents = [Document(text="null", extra_info={})]
        else:
            documents = llm_documents

        index = VectorStoreIndex.from_documents(
            documents, 
            service_context=llm_context,
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)]
        )
        
        if not llm_message:
            llm_message = [ChatMessage(MessageRole.USER, content="")]

        if not prompt.strip():
            prompt = "null"

        template = ChatPromptTemplate(message_templates=llm_message)

        query_engine = index.as_query_engine(llm=llm_model.get("llm", None), embed_model=llm_model.get("embed_model", None), text_qa_template=template)

        response = query_engine.query(prompt)

        pprint(response, indent=4)
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
                "system_nickname": ("STRING", {"default": "Assistant"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("chat_history", "response", "chat_token_count")

    FUNCTION = "chat"
    CATEGORY = "SALT/Llama-Index/Querying"

    def chat(self, llm_model: Dict[str, Any], llm_context:Any, prompt: str, reset_engine:bool = False, user_nickname:str = "User", system_nickname:str = "Assistant") -> str:

        if reset_engine:
            self.chat_history.clear()
            self.history.clear()
            self.token_map.clear()

        tokenizer = tiktoken.encoding_for_model(llm_model.get('llm_name', 'gpt-3-turbo'))
        max_tokens = llm_model.get("max_tokens", 4096)

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
            cumulative_token_count += len(self.token_map[index])

        # Prune messages from the history if over max_tokens
        index = 0
        while cumulative_token_count > max_tokens and index < len(self.chat_history):
            tokens = self.token_map[index]
            if len(tokens) > 1:
                tokens.pop(0)
                self.chat_history[index].content = tokenizer.decode(tokens)
                cumulative_token_count -= 1
            else:
                cumulative_token_count -= len(tokens)
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
            
        pprint(self.chat_history, indent=4)

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
    CATEGORY = "SALT/Llama-Index/Querying"

    def complete(self, llm_model:Dict[str, Any], prompt:str) -> str:
        response = llm_model['llm'].complete(prompt)
        pprint(response, indent=4)
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

    # Agent
    "LLMQueryEngineAsTool": "∞ Query Engine As Tool",

}
