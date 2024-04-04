"""
@NOTE:
	Classes are sorted close to alphabetically

@TODO: 
	Need to get Embedding and Indexing done before this step
	See documentation link, it's useful

@Requirements:
	llama-index
	# llama-index-llms-openai
	# llama-index-embeddings-openai

@Documentation:
	https://github.com/run-llama/llama_index/tree/main/docs/examples/query_engine
"""

import json
import logging
import os
import re
import sys

from typing import List


class LLMQueryEngine:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"llm_index": ("LLM_INDEX",),
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

	def query_engine(self, llm_index, query=None, llm_message=None):
		query_components = []
		
		if llm_message and isinstance(llm_message, list):
			for msg in llm_message:
				if str(msg).strip():
					query_components.append(str(msg))

		if query:
			if query.strip():
				query_components.append("user: " + query)

		pprint(query_components, indent=4)

		query_join = "\n".join(query_components)

		query_engine = llm_index.as_query_engine()
		response = query_engine.query(query_join)
		pprint(response, indent=4)
		return (response.response,)
		

class LLMQueryEngineAdv:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
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

	def query_engine(self, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):
		query_components = []
		
		if llm_message and isinstance(llm_message, list):
			for msg in llm_message:
				if str(msg).strip():
					query_components.append(str(msg))

		if query and query.strip():
			query_components.append("user: " + query)

		pprint(query_components, indent=4)
		query_join = "\n".join(query_components)

		retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k)
		query_engine = RetrieverQueryEngine(
			retriever=retriever,
			node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
		)

		response = query_engine.query(query_join)
		pprint(response, indent=4)
		return (response.response,)



NODE_CLASS_MAPPINGS = {
	"LLMQueryEngine": LLMQueryEngine,
	"LLMQueryEngineAdv": LLMQueryEngineAdv,
#	"SaltJSONQueryEngine": SaltJSONQueryEngine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"LLMQueryEngine": "∞ Query Engine",
	"LLMQueryEngineAdv": "∞ Query Engine (Advanced)",
#	"SaltJSONQueryEngine": "JSON Query Engine",
}
