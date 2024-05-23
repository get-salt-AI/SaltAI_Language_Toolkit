import json
import requests
from llama_index.core import Document

from .. import logger

class ScaleSERP:
    def __init__(self, api_key, base_url='https://api.scaleserp.com/search'):
        self.api_key = api_key
        self.base_url = base_url
        self.search_type = None
        self.results = {}
        self.link_dict = {}
        self.supported_document_types = ("news", "scholar", "places", "shopping")
        self.supported_document_sections = ('scholar_results', 'related_searches', 'related_questions', 'organic_results',
                       'news_results', 'places_results', 'shopping_results', 'video_results')
        self.supported_document_mappings = {
            'related_searches': {'title': "Related Searches", 'key_map': {'query': 'search_query', 'link': 'link'}},
            'related_questions': {'title': "Related Questions", 'key_map': {'question': 'question', 'answer': 'answer', 'source': {'link': 'source', 'title': 'source_title'}}},
            'organic_results': {'title': "Search Results", 'key_map': {'title': 'title', 'snippet': 'snippet', 'link': 'url', 'displayed_link': 'display_url'}},
            'news_results': {'title': "News Results", 'key_map': {'title': 'title', 'source': 'source', 'snippet': 'snippet', 'link': 'link'}},
            'scholar_results': {'title': "Scholar Results", 'key_map': {'title': 'title', 'link': 'link', 'snippet': 'snippet'}},
            'places_results': {'title': "Places Results", 'key_map': {'title': 'title', 'address': 'address', 'phone': 'phone', 'link': 'link'}},
            'shopping_results': {'title': "Shopping Results", 'key_map': {'title': 'title', 'price': 'price', 'link': 'link', 'merchant': 'merchant', 'summary': 'summary'}},
            'video_results': {'title': "Video Results", 'key_map': {'title': 'title', 'link': 'link', 'domain': 'domain', 'displayed_link': 'displayed_link', 'date': 'date'}}
        }

    def search(self, query, **kwargs):
        params = {
            'api_key': self.api_key,
            'q': query
        }
        params.update(kwargs)

        if 'search_type' in params:
            self.search_type = params['search_type']

        response = requests.get(self.base_url, params=params)
        if response.status_code == 401:
            logger.error("Error: Unauthorized. Check if the API key is correct.")
            return []
        elif response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code} - {response.text}")
            return []

        results = response.json()

        if not results:
            logger.error("Error: Received empty results from the API")
            return []

        self.results = results
        self._update_links()

        return results

    def _update_links(self):
        if not self.results:
            raise ValueError("No results are cached to update links internally. Please run a search with `ScaleSERP.search()`")

        if self.search_type and self.search_type not in self.supported_document_types:
            logger.warning(f"The `search_type` \"{self.search_type}\" does not support link scraping currently")

        for key in self.supported_document_sections:
            if key in self.results:
                for item in self.results[key]:
                    title = item.get('title', item.get('query', 'Unknown'))
                    self.link_dict[title] = item.get('link', 'N/A')

    def results_to_link_dict(self):
        return self.link_dict

    def results_to_documents(self):
        if not self.results:
            raise ValueError("No results are cached to parse to documents. Please run a search with `ScaleSERP.search()`")

        if self.search_type and self.search_type not in self.supported_document_types:
            logger.warning(f"The `search_type` \"{self.search_type}\" does not support document parsing currently")
        
        documents = []

        for result_key, config in self.supported_document_mappings.items():
            if result_key in self.results:
                text = f"{config['title']}:\n\n"
                extra_info = {'title': config['title'], 'result_links': []}
                for item in self.results[result_key]:
                    item_text = ""
                    item_data = {}
                    for key, value in config['key_map'].items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                item_text += f"{sub_key.title()}: {item.get(key, {}).get(sub_key, 'N/A')}\n"
                                item_data[sub_value] = item.get(key, {}).get(sub_key, 'N/A')
                        else:
                            item_text += f"{key.title()}: {item.get(key, 'N/A')}\n"
                            item_data[value] = item.get(key, 'N/A')
                    text += f"{item_text}\n"
                    extra_info['result_links'].append(item_data)
                documents.append(Document(text=text, extra_info=extra_info))
        
        return documents
