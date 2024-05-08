from autogen import ConversableAgent
from typing import Any, Callable, Optional, Sequence
from llama_index.core.base.llms.types import (
	ChatMessage,
	CompletionResponse,
	CompletionResponseGen,
	LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import PydanticProgramMode
from .utils import clone_conversable_agent


class BaseModel(CustomLLM):
	agent: Any

	def __init__(
		self,
		agent,
		max_tokens: Optional[int] = None,
		callback_manager: Optional[CallbackManager] = None,
		system_prompt: Optional[str] = None,
		messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
		completion_to_prompt: Optional[Callable[[str], str]] = None,
		pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
	) -> None:
		super().__init__(
			agent=agent,
			max_tokens=max_tokens,
			callback_manager=callback_manager,
			system_prompt=system_prompt,
			messages_to_prompt=messages_to_prompt,
			completion_to_prompt=completion_to_prompt,
			pydantic_program_mode=pydantic_program_mode,
		)

	@classmethod
	def class_name(cls) -> str:
		return "AutogenAgentLLM"

	@property
	def metadata(self) -> LLMMetadata:
		return LLMMetadata()

	@llm_completion_callback()
	def complete(
		self, prompt: str, formatted: bool = False, **kwargs: Any
	) -> CompletionResponse:
		answer = self.agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
		return CompletionResponse(text=answer)

	@llm_completion_callback()
	def stream_complete(
		self, prompt: str, formatted: bool = False, **kwargs: Any
	) -> CompletionResponseGen:

		def gen_response() -> CompletionResponseGen:
			result = self.complete(prompt, formatted=formatted, **kwargs)
			result.delta = result.text
			yield result

		return gen_response()


class ConvertAgentToLlamaindex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"agent": ("AGENT",),
			},
			"optional": {
				"optional_embed_model": ("LLM_EMBED_MODEL",)
            }
		}

	RETURN_TYPES = ("LLM_MODEL",)
	RETURN_NAMES = ("model",)

	FUNCTION = "convert_agent"
	CATEGORY = "SALT/Shakers/Agents"

	def convert_agent(self, agent, optional_embed_model=None):
		llm = {"llm": BaseModel(agent)}
		if optional_embed_model:
			llm.update(optional_embed_model)
		return (llm,)


class ConversableAgentCreator:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"multiline": False, "placeholder": "Assistant"}),
				"system_message": ("STRING", {
					"multiline": True,
					"default": "You are a helpful AI assistant. You can help with document QA. Return 'TERMINATE' when the task is done."
				}),
			},
			"optional": {
				"llm_model": ("LLM_MODEL",),
			}
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "create_agent"
	CATEGORY = "SALT/Shakers/Agents"

	def create_agent(self, name, system_message, llm_model=None):
		agent = ConversableAgent(
			name=name,
			system_message=system_message,
			llm_config={"config_list": [{"model": llm_model["llm"].model, "api_key": llm_model["llm"].api_key}]} if llm_model is not None else False,
			human_input_mode="NEVER",
		)
		return (agent,)


class ConversableAgentCreatorAdvanced:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"multiline": False, "placeholder": "Assistant"}),
				"system_message": ("STRING", {
					"multiline": True,
					"default": "You are a helpful AI assistant. You can help with document QA. Return 'TERMINATE' when the task is done."
				}),
			},
			"optional": {
				"llm_model": ("LLM_MODEL",),
				# default auto reply when no code execution or llm-based reply is generated.
				"default_auto_reply": ("STRING", {"multiline": True}),
				# a short description of the agent, this description is used by other agents.
				"description": ("STRING", {"multiline": True}),
			}
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "create_agent"
	CATEGORY = "SALT/Shakers/Agents"

	def create_agent(self, name, system_message, llm_model=None, default_auto_reply="", description=None):
		agent = ConversableAgent(
			name=name,
			system_message=system_message,
			llm_config={"config_list": [{"model": llm_model["llm"].model, "api_key": llm_model["llm"].api_key}]} if llm_model is not None else False,
			human_input_mode="NEVER",
			default_auto_reply=default_auto_reply,
			description=description if description is not None else system_message,
		)
		return (agent,)


class GroupChatManagerCreator:
	"""
	A chat manager agent that can manage a group chat of multiple agents.
	Can only be used in group chats.
	"""
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"multiline": False, "placeholder": "Manager"}),
				"system_message": ("STRING", {
					"multiline": True,
					"placeholder": "Group chat manager.",
				}),
			},
			"optional": {
				"llm_model": ("LLM_MODEL",),
				"max_consecutive_auto_reply": ("INT", {"default": 10}),
			}
		}

	RETURN_TYPES = ("GROUP_MANAGER",)
	RETURN_NAMES = ("group_manager",)

	FUNCTION = "create_agent"
	CATEGORY = "SALT/Llama-Index/Agents"

	def create_agent(self, name, system_message, llm_model=None, max_consecutive_auto_reply=None):
		group_manager = {
			"name": name,
			"system_message": system_message,
			"llm_config": {
				"config_list": [
					{
						"model": llm_model["llm"].model,
						"api_key": llm_model["llm"].api_key,
					}
				]
			} if llm_model is not None else None,
			"max_consecutive_auto_reply": max_consecutive_auto_reply,
		}
		return (group_manager,)


class ChangeSystemMessage:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"agent": ("AGENT",),
				"system_message": ("STRING", {
					"multiline": True,
					"default": "You are a helpful AI assistant. You can help with document QA. Return 'TERMINATE' when the task is done."
				}),
			},
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "update_system_prompt"
	CATEGORY = "SALT/Llama-Index/Agents"

	def update_system_prompt(self, agent, system_message, llm_model=None):
		agent = clone_conversable_agent(agent)
		agent.update_system_message(system_message)
		return (agent,)


class ClearMemory:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"agent": ("AGENT",),
			},
			"oprional": {
				"recipient": ("AGENT", {"default": None}),
			},
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "clear_memory"
	CATEGORY = "SALT/Llama-Index/Agents"

	def clear_memory(self, agent, recipient=None):
		agent = clone_conversable_agent(agent)
		agent.clear_history(recipient)
		return (agent,)
