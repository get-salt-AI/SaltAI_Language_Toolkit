
import json
from pydantic import create_model
from ..modules.utility import CreateOutputModel
from llama_index.core.response_synthesizers import TreeSummarize


# JSON Composer Tool
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
    CATEGORY = "SALT/Llama-Index/Tools/JSON"

    def compose_json(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = f"{text_input}\n\n###\n\nGiven the above text, create a valid JSON object utilizing *all* of the data; using the following classifiers: {classifier_list}.\n\n{extra_directions}\n\nPlease ensure the JSON output is properly formatted, and does not omit any data."
        response = llm_model.complete(prompt)
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
    CATEGORY = "SALT/Llama-Index/Tools/JSON"

    def compose_json(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed JSON, please inspect it and repair it so that it's valid JSON, without changing or loosing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the JSON output is properly formatted, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/YAML"

    def compose_yaml(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above text, create a valid YAML document utilizing *all* of the data; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the YAML output is properly formatted, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/YAML"

    def repair_yaml(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed YAML, please inspect it and repair it so that it's valid YAML, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the YAML output is properly formatted, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/Markdown"

    def compose_markdown(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above text, create a valid Markdown document utilizing *all* of the data; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/Markdown"

    def repair_markdown(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed Markdown, please inspect it and repair it so that it's valid Markdown, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/HTML"

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
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/HTML"

    def repair_html(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed HTML, please inspect it and repair it so that it's valid HTML, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the HTML output is well-structured, valid,, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/Regex"

    def create_regex(self, llm_model, description, extra_directions=""):
        prompt = (
            f"Create only a well formed regex pattern based on the following description:\n\n"
            f"{description}\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the regex pattern is concise and accurately matches the described criteria."
        )
        
        response = llm_model.complete(prompt)
        
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
    CATEGORY = "SALT/Llama-Index/Tools/Regex"

    def repair_regex(self, llm_model, text_input, description, extra_directions=""):
        prompt = (
            f"Given the potentially malformed or incorrect regex pattern:\n\n{text_input}\n\n"
            f"and the following description of what the pattern should match:\n\n{description}\n\n"
            f"{extra_directions}\n\n"
            "Please repair the regex pattern so it is well-formed and accurately matches the described criteria."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)
    
    
# Formatting
class LLMPydanticOutput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_documents": ("LLM_DOCUMENTS",),
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
    CATEGORY = "SALT/Llama-Index/Summarization"

    def generate_summary(self, llm_model, llm_documents, output_model_name, output_model, summary_query="Summarize"):

        output_model_json = json.loads(output_model)
        OutputModel = CreateOutputModel.create(output_model_json, output_model_name)
        summarizer = TreeSummarize(verbose=True, output_cls=OutputModel, llm=llm_model)

        responses = []
        for doc in llm_documents:
            response = summarizer.get_response(summary_query, doc.text)
            from pprint import pprint
            pprint(response)
            responses.append(response)

        string_response = repr(responses)

        return (string_response, responses)



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
    "LLMRegexRepair": "∞ Regex Repair"
}
