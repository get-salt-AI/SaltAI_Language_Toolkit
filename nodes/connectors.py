"""
@NOTE:
	Classes are sorted almost alphabetically

@REQUIREMENTS:
	llama-index
	# llama-index-readers-google

@BUGS: 
	LLMGoogleDocsReader document ID is supposed to be a List[str] of *something*?
"""

# Imports:
#from llama_index.readers.google import GoogleDocsReader
from llama_index.readers.notion import NotionPageReader

from .. import MENU_NAME, SUB_MENU_NAME

class LLMGoogleDocsReader:
	"""
	Documentation: https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/docs
	Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/docs/base.py
	"""
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				# document_ids: List[str]
				"document ID": ("STRING", { "default":""}),
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("document", )

	FUNCTION = "execute"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

	def execute(self):
		pass


class LLMNotionReader:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"notion_integration_token": ("STRING", {}),
				"page_ids": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Page ID 1, Page ID 2"}),
				"database_id": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Database ID", "optional": True}),
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("document", )

	FUNCTION = "read_notion"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

	def read_notion(self, notion_integration_token, page_ids, database_id=None):

		page_id_list = None
		if page_ids:
			page_id_list = [page_id.strip() for page_id in page_ids.split(",") if page_id.strip()] if page_ids.strip() else None

		db_id = None
		if database_id:
			db_id = database_id.strip() if database_id.strip() else None
		
		if db_id:
			documents = NotionPageReader(integration_token=notion_integration_token).load_data(database_id=db_id)
		else:
			documents = NotionPageReader(integration_token=notion_integration_token).load_data(page_ids=page_id_list)
		return (documents,)


NODE_CLASS_MAPPINGS = {
#	"LLMGoogleDocsReader": LLMGoogleDocsReader,
	#"LLMNotionReader": LLMNotionReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
#	"LLMGoogleDocsReader": "∞ GoogleDocs",
	#"LLMNotionReader": "∞ Notion",
}
