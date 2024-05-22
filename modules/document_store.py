import re

from llama_index.core import Document
from typing import Dict

from .. import logger

# TODO: create add/remove nodes and implement with LLMChatBot
class DocumentStore:
    """
        Stores title => Document pairs for retrieval in inline prompts such as LLMChatBot node
    """
    def __init__(self):
        self.store = {}

    def add(self, key:str, document:Document, overwrite:bool = False):
        if isinstance(document.extra_info, dict):
            document.extra_info.update({"document_title": key})
        if overwrite:
            self.store[key] = document
        else:
            if not self.store.__contains__(key):
                self.store[key] = document
            else:
                logger.error(f"DocumentStore: document already exists by the title `{key}`")

    def remove(self, key:str, delete:bool = False):
        if self.store.__contains__(key):
            doc = self.store[key]
            if delete:
                del doc
            return self.store.pop(key)
        else:
            logger.error(f"DocumentStore: The document title `{key}` doesn't exist in the store.")
            return False

    def parse(text:str, document_store:Dict[str, Document]):
        """
        Input:
        - text (str): The string to replace inline-document strings with their respective titles
        - document_store (Dict[str, Document]): A dictionary of document title to Document object pairs
        )
        Output:
        - modified_text: All occurrences of `document:[document_name]` replaced with just their "document_name"
        - referenced_documents: Any existing documents that match the name referenced
        """
        pattern = re.compile(r'\bdocument:([^\s]+)\b')
        matches = pattern.findall(text)
        modified_text = pattern.sub(r'\1', text)
        referenced_documents = [document_store[name] for name in matches if name in document_store]
        return modified_text, referenced_documents