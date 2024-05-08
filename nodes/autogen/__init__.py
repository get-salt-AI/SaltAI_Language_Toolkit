from .agents import (
	ConvertAgentToLlamaindex, 
	ConversableAgentCreator,
	ConversableAgentCreatorAdvanced,
	GroupChatManagerCreator,
	ChangeSystemMessage,
	ClearMemory,
)
from .chats import (
	SendMessage,
	GenerateReply,
	SimpleChat, 
	GroupChat,
	GroupChatAdvanced,
)
from .tools import (
	AddTool,
	CreateTavilySearchTool,
	ConvertAgentAsTool,
)


NODE_CLASS_MAPPINGS = {
	"ConvertAgentToLlamaindex": ConvertAgentToLlamaindex,
	"ConversableAgentCreator": ConversableAgentCreator,
	"ConversableAgentCreatorAdvanced": ConversableAgentCreatorAdvanced,
	"GroupChatManagerCreator": GroupChatManagerCreator,
	"ChangeSystemMessage": ChangeSystemMessage,
	"ClearMemory": ClearMemory,

	"SendMessage": SendMessage,
	"GenerateReply": GenerateReply,
	"SimpleChat": SimpleChat,
	"GroupChat": GroupChat,
	"GroupChatAdvanced": GroupChatAdvanced,

	"AddTool": AddTool,
	"CreateTavilySearchTool": CreateTavilySearchTool,
	"ConvertAgentAsTool": ConvertAgentAsTool,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"ConvertAgentToLlamaindex": "∞ Convert Agent To Llamaindex",
	"ConversableAgentCreator": "∞ Conversable Agent",
	"ConversableAgentCreatorAdvanced": "∞ Conversable Agent (Adv)",
	"GroupChatManagerCreator": "∞ Group Chat Manager",
	"ChangeSystemMessage": "∞ Change System Message",
	"ClearMemory": "∞ Clear Memory",

	"SendMessage": "∞ Send Message",
	"GenerateReply": "∞ Generate Reply",
	"SimpleChat": "∞ Simple Chat",
	"GroupChat": "∞ Group Chat",
	"GroupChatAdvanced": "∞ Group Chat (Adv)",

	"AddTool": "∞ Add Tool",
	"CreateTavilySearchTool": "∞ Tavily Search Tool",
	"ConvertAgentAsTool": "∞ Agent As Tool",
}
