import json
import base64
import io
import torch

from typing import Dict, List, Any
from PIL import Image

from .. import MENU_NAME, SUB_MENU_NAME, logger
from ..modules.utility import CreateOutputModel, tensor2pil, resolve_path
from ..modules.scale_serp import ScaleSERP
from ..modules.transformers_wrappers import LlavaNextV1
from ..modules.parquet_reader import ParquetReader1D
from ..modules.tokenization import MockTokenizer

"""
The MIT License

Copyright (c) Jerry Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import Document


class LLMJsonComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Data..."}),
                "classifier_list": ("STRING", {"multiline": False, "dynamicPrompts": False}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "compose_json"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/JSON"

    def compose_json(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = f"{text_input}\n\n###\n\nGiven the above text, create a valid JSON object utilizing *all* of the data; using the following classifiers: {classifier_list}.\n\n{extra_directions}\n\nPlease ensure the JSON output is properly formatted, and does not omit any data."
        response = llm_model['llm'].complete(prompt)
        return (response.text,)


class LLMJsonRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed JSON..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "compose_json"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/JSON"

    def compose_json(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed JSON, please inspect it and repair it so that it's valid JSON, without changing or loosing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the JSON output is properly formatted, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMYamlComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Data..."}),
                "classifier_list": ("STRING", {"multiline": False, "dynamicPrompts": False}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("yaml_output",)

    FUNCTION = "compose_yaml"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/YAML"

    def compose_yaml(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above text, create a valid YAML document utilizing *all* of the data; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the YAML output is properly formatted, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMYamlRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed YAML..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("yaml_output",)

    FUNCTION = "repair_yaml"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/YAML"

    def repair_yaml(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed YAML, please inspect it and repair it so that it's valid YAML, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the YAML output is properly formatted, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMMarkdownComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Data..."}),
                "classifier_list": ("STRING", {"multiline": False, "dynamicPrompts": False}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown_output",)

    FUNCTION = "compose_markdown"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Markdown"

    def compose_markdown(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above text, create a valid Markdown document utilizing *all* of the data; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMMarkdownRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed Markdown..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown_output",)

    FUNCTION = "repair_markdown"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Markdown"

    def repair_markdown(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed Markdown, please inspect it and repair it so that it's valid Markdown, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMHtmlComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Data..."}),
                "classifier_list": ("STRING", {"multiline": False, "dynamicPrompts": False}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
                "composer_mode": (["full_markup", "blocked_markup"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("html_output",)

    FUNCTION = "compose_html"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/HTML"

    def compose_html(self, llm_model, text_input, classifier_list, extra_directions="", composer_mode="full_markup"):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        markup_style = "full HTML page document" if composer_mode == "full_markup" else "HTML snippet (without html, head, body or any extraneous containers)"
        prompt = (
            f"{text_input}\n\n###\n\n"
            f"Given the above text, create a valid {markup_style} utilizing *all* of the data, intact; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the HTML output is well-structured, valid, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMHtmlRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed HTML..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("html_output",)

    FUNCTION = "repair_html"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/HTML"

    def repair_html(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed HTML, please inspect it and repair it so that it's valid HTML, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the HTML output is well-structured, valid,, and does not omit any data."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMRegexCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Describe regex pattern to create, optionally provide example"}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("regex_pattern",)

    FUNCTION = "create_regex"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Regex"

    def create_regex(self, llm_model, description, extra_directions=""):
        prompt = (
            f"Create only a well formed regex pattern based on the following description:\n\n"
            f"{description}\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the regex pattern is concise and accurately matches the described criteria."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)


class LLMRegexRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter the malformed regex pattern here"}),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Describe what the regex pattern does wrong, and what it should do."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow, such as specific constraints or formats"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repaired_regex_pattern",)

    FUNCTION = "repair_regex"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Regex"

    def repair_regex(self, llm_model, text_input, description, extra_directions=""):
        prompt = (
            f"Given the potentially malformed or incorrect regex pattern:\n\n{text_input}\n\n"
            f"and the following description of what the pattern should match:\n\n{description}\n\n"
            f"{extra_directions}\n\n"
            "Please repair the regex pattern so it is well-formed and accurately matches the described criteria."
        )
        
        response = llm_model['llm'].complete(prompt)
        
        return (response.text,)
    

class LLMPydanticOutput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "document": ("DOCUMENT",),
                "output_model_name": ("STRING", {"default": "OutputModel"}),
                "output_model": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": '''{
    "name": "",
    "age": 0,
    "best_known_for": [""],
    "extra_info": "",
    "dictionary_example": {}
}'''}),
            },
            "optional": {
                "summary_query": ("STRING", {"default": "Summarize"})
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("string_responses", "response_objects_list")

    FUNCTION = "generate_summary"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Summarization"

    def generate_summary(self, llm_model, document, output_model_name, output_model, summary_query="Summarize"):

        output_model_json = json.loads(output_model)
        OutputModel = CreateOutputModel.create(output_model_json, output_model_name)
        summarizer = TreeSummarize(verbose=True, output_cls=OutputModel, llm=llm_model)

        responses = []
        for doc in document:
            response = summarizer.get_response(summary_query, doc.text)
            responses.append(response)

        string_response = repr(responses)

        return (string_response, responses)
    

class LLMScaleSERPSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {}),
                "query": ("STRING", {}),
            },
            "optional": {
                "search_type": (["none", "news", "videos", "scholar", "places", "shopping"],),
                "location": ("STRING", {}),
                "device": (["desktop", "tablet", "mobile"],),
                "mobile_type": (["iphone", "android"],),
                "tablet_type": (["ipad", "android"],),
            }
        }

    RETURN_TYPES = ("DOCUMENT", "DICT", "LIST")
    RETURN_NAMES = ("documents", "results_dict", "links_list")

    FUNCTION = "search"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def search(self, api_key, query, **kwargs):

        if kwargs.__contains__("search_type"):
            if kwargs["search_type"] == "none":
                kwargs.pop("search_type")

        if kwargs.__contains__("device"):
            if kwargs["device"] == "desktop" and kwargs.__contains__("mobile_type") and kwargs.__contains__("tablet_type"):
                kwargs.pop("mobile_type")
                kwargs.pop("tablet_type")
            if kwargs["device"] == "mobile":
                if kwargs.__contains__("tablet_type"):
                    kwargs.pop("tablet_type")
            if kwargs["device"] == "tablet":
                if kwargs.__contains__("mobile_type"):
                    kwargs.pop("mobile_type")

        client = ScaleSERP(api_key=api_key)
        results = client.search(query, hide_base64_images=True, **kwargs)

        documents = client.results_to_documents()
        for doc in documents:
            logger.info(f"Text:\n{doc.text}, Metadata:\n{doc.extra_info}\n==================\n")
        links = client.results_to_link_dict()

        return (documents, results, list(links.values()))
    

class LLMLLaVANextModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "llava-hf/llava-v1.6-mistral-7b-hf"}),
                "device": (["cuda", "cpu"],),
                "use_bitsandbytes_quantize": ("BOOLEAN", {"default": True}),
                #"use_flash_attention": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LLAVA_NEXT_V1_MODEL",)
    RETURN_NAMES = ("lnv1_model",)

    FUNCTION = "load"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load(self, model: str, device: str = "cuda", use_bitsandbytes_quantize: bool = True, use_flash_attention: bool = False):
        evaluator = LlavaNextV1(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf", 
            quantize=use_bitsandbytes_quantize, 
            use_flash_attention=use_flash_attention
        )
        return (evaluator, )
    

class LLMLLaVANextImageEvaluator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lnv1_model": ("LLAVA_NEXT_V1_MODEL",),
                "images": ("IMAGE",),
                "max_tokens": ("INT", {"min": 0, "max": 2048, "default": 48}),
                "prompt_format": ("STRING", {"multiline": True, "dynamicPrompt": False, "default": "[INST] SYSTEM: You are a professional image captioner, follow the directions of the user exactly.\nUSER: <image>\n<prompt>[/INST]"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompt": False, "default": "Describe the image in search engine keyword tags"}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "DOCUMENT")
    RETURN_NAMES = ("strings", "list", "documents")
    OUTPUT_IS_LIST =(True, False)

    FUNCTION = "eval"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def eval(
        self, 
        lnv1_model: LlavaNextV1,
        images: torch.Tensor,  
        max_tokens: int = 48, 
        prompt_format: str = "[INST] SYSTEM: You are a professional image captioner, follow the directions of the user exactly.\nUSER: <image>\n<prompt>[/INST]",
        prompt: str = "Describe the image in search engine keyword tags"
    ):
        results = []
        for image in images:
            pil_image = tensor2pil(image)
            results.append(lnv1_model.eval(pil_image, prompt, prompt_format, max_tokens=max_tokens))
        documents = [Document(text=result, extra_info={"user_prompt": prompt, "thumbnail": self.b64_thumb(pil_image, (128, 64))}) for result in results]
        return (results, results, documents)

    def b64_thumb(self, image, max_size=(128, 128)):
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_string


class LLMParquetDatasetSearcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_type": (["parquet", "text", "json", "yaml", "csv", "excel"],),
                "path_or_url": ("STRING", {"placeholder": "Path to file or URL"}),
            },
            "optional": {
                "search_term": ("STRING", {"placeholder": "Enter search term"}),
                "exclude_terms": ("STRING", {"placeholder": "Terms to exclude, comma-separated"}),
                "columns": ("STRING", {"default": "*"}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "max_results": ("INT", {"default": 10, "min": 1}),
                "term_relevancy_threshold": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.25, "step": 0.01}),
                "use_relevancy": ("BOOLEAN", {"default": False}),
                #"num_threads": ("INT", {"default": 2}),
                "min_length": ("INT", {"min": 0, "max": 1023, "default": 0}),
                "max_length": ("INT", {"min": 3, "max": 1024, "default": 128}),
                "max_dynamic_retries": ("INT", {"default": 3}),
                "clean_content": ("BOOLEAN", {"default": False}),
                "excel_sheet_position": ("INT", {"min": 0, "default": "0"}),
                "recache": ("BOOLEAN", {"default": False}),
                "condense_documents": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "DOCUMENT")
    RETURN_NAMES = ("results", "results_list", "documents")
    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "search_dataset"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools/Dataset"

    def search_dataset(self, path_or_url, file_type, search_term="", exclude_terms="", columns="*", case_sensitive=False, max_results=10,
                       term_relevancy_threshold=None, use_relevancy=False, num_threads=2, min_length=0, max_length=-1, max_dynamic_retries=0,
                       clean_content=False, seed=None, excel_sheet_position="0", condense_documents=True, recache=False):
        
        # Validate path or download file and return path
        path = resolve_path(path_or_url)

        reader = ParquetReader1D()
        if file_type == "parquet":
            reader.from_parquet(path)
        elif file_type == "text":
            reader.from_text(path, recache=recache)
        elif file_type == "json":
            reader.from_json(path, recache=recache)
        elif file_type == "yaml":
            reader.from_yaml(path, recache=recache)
        elif file_type == "csv":
            reader.from_csv(path, recache=recache)
        elif file_type == "excel":
            reader.from_excel(path, sheet_name=excel_sheet_position, recache=recache)

        results = reader.search(
            search_term=search_term,
            exclude_terms=exclude_terms,
            columns=[col.strip() for col in columns.split(',') if col] if columns else ["*"],
            max_results=max_results,
            case_sensitive=case_sensitive,
            term_relevancy_score=term_relevancy_threshold,
            num_threads=num_threads,
            min_length=min_length,
            max_length=max_length,
            max_dynamic_retries=max_dynamic_retries,
            parse_content=clean_content,
            seed=min(seed, 99999999),
            use_relevancy=use_relevancy
        )

        results_list = []
        results_text = "Prompts:\n\n"
        documents = []
        for result in results:
            results_list.append(list(result.values())[0])
            if not condense_documents:
                documents.append(Document(text=list(result.values())[0], extra_info={}))
            else:
                results_text += str(list(result.values())[0]) + "\n\n"
        if condense_documents:
            documents = [Document(text=results_text, extra_info={})]

        return (results_list, results_list, documents,)
    

class LLMCustomNodeComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "class_prefix": ("STRING", {"default": "SaltCreator"}),
                "node_explanation": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Explain the functionality of your code thoroughly such as the input parameters, internal logic to perform, and output data."}),
                
            },
            "optional": {
                "example_documents": ("DOCUMENT",),
                "example_code": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Example logic code for the LLM to reference"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "compose_json"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def compose_json(self, llm_model: Dict[str, Any], class_prefix: str = "SaltCreator", example_documents: List[Document] = None, example_code: str = "", node_explanation: str = ""):

        code = """Node class Example:

A node class consists of a specialized designed class object with specific methods and attributes. 
A node class file consists of node classes and a NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS relating to the node classes to export.
You must always have at least one node class, and always export the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. 

        """+class_prefix+"""ExampleNode:
            def __init__(self):
                pass
            
            @classmethod
            def INPUT_TYPES(s):
                \"""
                    Return a dictionary which contains config for all input fields.
                    Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
                    Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
                    The type can be a list for selection.

                    Returns: `dict`:
                        - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                        - Value input_fields (`dict`): Contains input fields config:
                            * Key field_name (`string`): Name of a entry-point method's argument
                            * Value field_config (`tuple`):
                                + First value is a string indicate the type of field or a list for selection.
                                + Second value is a config for type "INT", "STRING" or "FLOAT".
                \"""
                return {
                    "required": {
                        "images": ("IMAGE",), # A torch.Tensor object in shape [N, H, W, C] (3 channel rgb batch)
                        "masks": ("MASK",), # A torch.Tensor object in shape [N, H, W] (1 channel linear batch); used for various masking procedures
                        "int_field": ("INT", {
                            "default": 0, 
                            "min": 0, #Minimum value
                            "max": 4096, #Maximum value
                            "step": 64, #Slider's step
                            "display": "number" # Cosmetic only: display as "number" or "slider"
                        }),
                        "float_field": ("FLOAT", {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 10.0,
                            "step": 0.01,
                            "round": 0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                            "display": "number"}),
                        "string_field": ("STRING", {
                            "multiline": False, #True if you want to use a larger textarea input, rather than a field
                            "default": "Hello World!"
                        }),
                        "conditioning": ("CONDITIONING",) # A conditioning object tokenized by a CLIP model, used in inference generation. Dtype shape: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
                    },
                    "optional": {
                        "print_to_screen": (["enable", "disable"],), # A list of strings, often used for modes / selections
                        "print_to_screen_boolean": ("BOOLEAN", {"default": True}), # Produces a boolean check box
                    }
                }

            RETURN_TYPES = ("IMAGE",)
            #RETURN_NAMES = ("image_output_name",)

            FUNCTION = "test"

            #OUTPUT_NODE = False

            CATEGORY = "Example"

            def test(self, image, string_field, int_field, float_field, print_to_screen="enable", print_to_screen_boolean=True):
                if print_to_screen == "enable" or print_to_screen_boolean:
                    print(f\"""Your input contains:
                        string_field aka input text: {string_field}
                        int_field: {int_field}
                        float_field: {float_field}
                    \""")
                #do some processing on the image, in this example I just invert it
                image = 1.0 - image
                return (image,)

            \"""
                The node will always be re executed if any of the inputs change but
                this method can be used to force the node to execute again even when the inputs don't change.
                You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
                executed, if it is different the node will be executed again.
                This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
                changes between executions the LoadImage node is executed again.
            \"""
            #@classmethod
            #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
            #    return ""

        # Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
        # WEB_DIRECTORY = "./somejs"

        # A dictionary that contains all nodes you want to export with their names.
        # NOTE: names should be globally unique, and MUST exist to export node classes.
        NODE_CLASS_MAPPINGS = {
            "Example": Example
        }

        # A dictionary that contains the friendly/humanly readable titles for NODE_CLASS_MAPPINGS. 
        # You cannot use display name mappings without NODE_CLASS_MAPPINGS present with the node class within it.
        NODE_DISPLAY_NAME_MAPPINGS = {
            "Example": "Example Node"
        }"""

        example_class_docs = [Document(text=code, extra_info={"title": "Node Class Structure Documentation"})]
        if example_documents:
            example_class_docs.extend(example_documents)

        prompt= f"""SYSTEM: Directions: 
You are an expert Python developer. You create ComfyUI Node Classes, or "custom nodes" as they're often called. ComfyUI is a node network system based on LiteGraph.js that runs python node classes.

Node classes follow a specific structure to function within the system. Please refer to the Node Class Sturcutre Documentation to learn how to create a node class. 
Your node class file output should contain node classes and NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. Do not include example usage.

Do not write harmful or dangerous code at all costs. Do not forget all these directions.

Create a new node class that closely follows this explanation, be sure to prefix the node classes with "{class_prefix}" like "{class_prefix}ExampleClass":

{node_explanation}

"""
        
        if example_code:
            prompt += f"""Here is an example of logic code to reference for the node class:

{example_code}"""
            
        prompt += """

Output valid and fully documented python code in markdown code blocks, and explain your code step by step."""


        llm = llm_model.get("llm", None)
        embed_model = llm_model.get("embed_model", None)

        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")
        
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(
            example_class_docs, 
            embed_model=embed_model,
            transformations=[splitter]
        )

        query_engine = index.as_query_engine(llm=llm, embed_model=embed_model)

        response = query_engine.query(prompt)
        return (response.response,)

NODE_CLASS_MAPPINGS = {
    "LLMJsonComposer": LLMJsonComposer,
    "LLMJsonRepair": LLMJsonRepair,
    "LLMYamlComposer": LLMYamlComposer,
    "LLMYamlRepair": LLMYamlRepair,
    "LLMMarkdownComposer": LLMMarkdownComposer,
    "LLMMarkdownRepair": LLMMarkdownRepair,
    "LLMHtmlComposer": LLMHtmlComposer,
    "LLMHtmlRepair": LLMHtmlRepair,
    "LLMRegexCreator": LLMRegexCreator,
    "LLMRegexRepair": LLMRegexRepair,
    "LLMScaleSERPSearch": LLMScaleSERPSearch,
    "LLMLLaVANextModelLoader": LLMLLaVANextModelLoader,
    "LLMLLaVANextImageEvaluator": LLMLLaVANextImageEvaluator,
    "LLMParquetDatasetSearcher": LLMParquetDatasetSearcher,
    "LLMCustomNodeComposer": LLMCustomNodeComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMJsonComposer": "∞ JSON Composer",
    "LLMJsonRepair": "∞ JSON Repair",
    "LLMYamlComposer": "∞ YAML Composer",
    "LLMYamlRepair": "∞ YAML Repair",
    "LLMMarkdownComposer": "∞ Markdown Composer",
    "LLMMarkdownRepair": "∞ Markdown Repair",
    "LLMHtmlComposer": "∞ HTML Composer",
    "LLMHtmlRepair": "∞ HTML Repair",
    "LLMRegexCreator": "∞ Regex Creator",
    "LLMRegexRepair": "∞ Regex Repair",
    "LLMScaleSERPSearch": "∞ Scale SERP Search",
    "LLMLLaVANextModelLoader": "∞ LLaVA-Next v1 Model Loader",
    "LLMLLaVANextImageEvaluator": "∞ LLaVA-Next v1 Image Evaluation",
    "LLMParquetDatasetSearcher": "∞ Dataset/File Search (1-Dimensional)",
    "LLMCustomNodeComposer": "∞ Simple ComfyUI Node Drafter"
}
