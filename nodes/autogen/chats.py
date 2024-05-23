from autogen import GroupChatManager
from autogen import GroupChat as AutogenGroupChat
from .utils import clone_conversable_agent

from ... import MENU_NAME, SUB_MENU_NAME

class SendMessage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipient": ("AGENT",),
                "sender": ("AGENT",),
                "message": ("STRING", {
                    "multiline": True,
                    "default": "Hi"
                }),
            },
        }

    RETURN_TYPES = ("AGENT", "AGENT")
    RETURN_NAMES = ("recipient", "sender")

    FUNCTION = "add_info"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

    def add_info(self, recipient, sender, message):
        recipient = clone_conversable_agent(recipient)
        sender = clone_conversable_agent(sender)
        sender.send(message, recipient)
        return (recipient, sender)


class GenerateReply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipient": ("AGENT",),
                "message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "sender": ("AGENT", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("reply",)

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

    def start_chat(self, recipient, message, sender=None):
        recipient = clone_conversable_agent(recipient)
        if sender:
            sender = clone_conversable_agent(sender)
        message = recipient._oai_messages[sender] + [{"content": message, "role": "user"}]
        reply = recipient.generate_reply(message, sender)
        return (reply,)


class SimpleChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initiator": ("AGENT",),
                "recipient": ("AGENT",),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                # whether to clear the chat history with the agent
                "clear_history": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "max_turns": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "AGENT", "AGENT",)
    RETURN_NAMES = ("chat_history", "summary", "initiator", "recipient",)

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

    def start_chat(self, initiator, recipient, summary_method, clear_history, init_message=None, max_turns=None):
        initiator = clone_conversable_agent(initiator)
        recipient = clone_conversable_agent(recipient)

        chat_result = initiator.initiate_chat(
            recipient,
            message=init_message,
            max_turns=max_turns,
            summary_method=summary_method,
            clear_history=clear_history,
        )
        summary = chat_result.summary
        chat_history = ""
        for message in chat_result.chat_history:
            if message["content"] is not None:
                chat_history += "- " + message["content"] + "\n\n"
        return (chat_history, summary, initiator, recipient,)


class GroupChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_manager": ("GROUP_MANAGER",),
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "send_introductions": ("BOOLEAN", {"default": False},),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                "max_turns": ("INT", {"default": 10}),
                "clear_history": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "agent_1": ("AGENT",),
                "agent_2": ("AGENT",),
                "agent_3": ("AGENT",),
                "agent_4": ("AGENT",),
                "agent_5": ("AGENT",),
                "agent_6": ("AGENT",),
                "agent_7": ("AGENT",),
                "agent_8": ("AGENT",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("chat_history", "summary", )

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

    def start_chat(
        self,
        group_manager,
        init_message,
        send_introductions,
        summary_method,
        max_turns,
        clear_history,
        **kwargs,
    ):
        agents = [kwargs[i] for i in kwargs if "agent_" in i]
        assert len(agents) > 1, "At least 2 agents are needed to start a group chat session"
        # create chat
        group_chat = AutogenGroupChat(
            agents=agents,
            messages=[],
            max_round=max_turns,
            send_introductions=send_introductions,
        )
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            **group_manager,
        )
        # start chat
        chat_result = agents[0].initiate_chat(
            group_chat_manager,
            message=init_message,
            summary_method=summary_method,
            max_turns=max_turns,
            clear_history=clear_history,
        )
        # parse results
        summary = chat_result.summary
        chat_history = ""
        for message in chat_result.chat_history:
            if message["content"] is not None:
                chat_history += "- " + message["content"] + "\n\n"
        return (chat_history, summary)


class GroupChatAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_manager": ("GROUP_MANAGER",),
                "init_message": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                # select_speaker_message_template: customize the select speaker message (used in "auto" speaker selection), which appears first in the message context and generally includes the agent descriptions and list of agents. The string value will be converted to an f-string, use "{roles}" to output the agent's and their role descriptions and "{agentlist}" for a comma-separated list of agent names in square brackets.
                "select_speaker_message_template": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "You are in a role play game. The following roles are available:\n{roles}.\nRead the following conversation.\nThen select the next role from {agentlist} to play. Only return the role."
                }),
                # select_speaker_prompt_template: customize the select speaker prompt (used in "auto" speaker selection), which appears last in the message context and generally includes the list of agents and guidance for the LLM to select the next agent. The string value will be converted to an f-string, use "{agentlist}" for a comma-separated list of agent names in square brackets.
                "select_speaker_prompt_template": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
                }),
                "summary_method": ([
                    "last_msg",
                    "reflection_with_llm",
                ],),
                "max_turns": ("INT", {"default": 10}),
                # When set to True and when a message is a function call suggestion,
                # the next speaker will be chosen from an agent which contains the corresponding function name
                # in its `function_map`
                "func_call_filter": ("BOOLEAN", {"default": True},),
                # speaker_selection_method: the method for selecting the next speaker.
                # Could be any of the following (case insensitive), will raise ValueError if not recognized:
                # - "auto": the next speaker is selected automatically by LLM.
                # - "manual": the next speaker is selected manually by user input.
                # - "random": the next speaker is selected randomly.
                # - "round_robin": the next speaker is selected in a round robin fashion, i.e., iterating in the same order as provided in `agents`.
                "speaker_selection_method": ([
                    "auto",
                    # "manual",
                    "random",
                    "round_robin",
                ],),
                # whether to allow the same speaker to speak consecutively.
                "allow_repeat_speaker": ("BOOLEAN", {"default": True},),
                # send_introductions: send a round of introductions at the start of the group chat, so agents know who they can speak to
                "send_introductions": ("BOOLEAN", {"default": False},),
                # role_for_select_speaker_messages: sets the role name for speaker selection when in 'auto' mode, typically 'user' or 'system'.
                "role_for_select_speaker_messages": ([
                    "system",
                    "user",
                ],),
                "clear_history": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "agent_1": ("AGENT",),
                "agent_2": ("AGENT",),
                "agent_3": ("AGENT",),
                "agent_4": ("AGENT",),
                "agent_5": ("AGENT",),
                "agent_6": ("AGENT",),
                "agent_7": ("AGENT",),
                "agent_8": ("AGENT",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("chat_history", "summary", )

    FUNCTION = "start_chat"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

    def start_chat(
        self,
        group_manager,
        init_message,
        select_speaker_message_template,
        select_speaker_prompt_template,
        func_call_filter,
        speaker_selection_method,
        allow_repeat_speaker,
        send_introductions,
        role_for_select_speaker_messages,
        summary_method,
        max_turns,
        clear_history,
        **kwargs,
    ):
        agents = [kwargs[i] for i in kwargs if "agent_" in i]
        assert len(agents) != 1, "At least 1 agent is needed to start a group chat session"
        # create chat
        group_chat = AutogenGroupChat(
            agents=agents,
            messages=[],
            max_round=max_turns,
            func_call_filter=func_call_filter,
            select_speaker_prompt_template=select_speaker_prompt_template,
            speaker_selection_method=speaker_selection_method,
            allow_repeat_speaker=allow_repeat_speaker,
            send_introductions=send_introductions,
            role_for_select_speaker_messages=role_for_select_speaker_messages,
        )
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            **group_manager,
        )
        # start chat
        chat_result = agents[0].initiate_chat(
            group_chat_manager,
            message=init_message,
            summary_method=summary_method,
            max_turns=max_turns,
            clear_history=clear_history,
        )
        # parse results
        summary = chat_result.summary
        chat_history = ""
        for message in chat_result.chat_history:
            chat_history += "- " + str(message["content"]) + "\n\n"
        return (chat_history, summary)
