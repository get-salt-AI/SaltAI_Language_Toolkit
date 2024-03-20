
"""
@BUGS: 
	SaltGoogleDocsReader document ID is supposed to be a List[str] of *something*?

"""

import json
import logging
import os
import re
import sys

from typing import List

# Requirements:
# llama-index

# Source:
# https://github.com/run-llama/llama_index/tree/main/docs/examples/data_connectors


# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/docs

# Example:
# https://github.com/run-llama/llama_index/blob/main/docs/examples/data_connectors/GoogleDocsDemo.ipynb

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/docs/base.py

# Requirements:
# llama-index-readers-google

# Imports:
from llama_index.readers.google import GoogleDocsReader

# Binding:
class SaltGoogleDocsReader:
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
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self):
		pass




# Imports:
#from llama_index.readers.notion import NotionPageReader

class SaltNotionReader:
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
	RETURN_NAMES = ("documents", )

	FUNCTION = "read_notion"
	CATEGORY = "SALT/Llama-Index/Readers"

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
#	"GoogleDocsReader": SaltGoogleDocsReader,
#	"NotionReader": SaltNotionReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
#	"GoogleDocsReader": "∞ GoogleDocs",
#	"NotionReader": "∞ Notion",
}