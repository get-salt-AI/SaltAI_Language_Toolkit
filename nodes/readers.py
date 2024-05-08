"""
@BUGS:
    CSV OK
    DOCX OK
    EPUB exceeds n_ctx
    FLAT OK
    HTML OK
    ImageReader FAIL by making stuff up
    ImageCaption OK
    ImageTabular OK
    ImageVisionLLM FAIL - low VRAM (needs 15GB+)
    IPYNB FAIL
    Markdown OK
    Mbox no example
    PDF OK
    PagedPDF OK
    PandasPDF FAIL won't create node
    Pptx OK
    PymuPDF OK
    RTF OK
    Unstructured FAIL

            Error occurred when executing LLMUnstructuredReader:

        **********************************************************************
        Resource [93maveraged_perceptron_tagger[0m not found.
        Please use the NLTK Downloader to obtain the resource:

    VideoAudio OK
    XML OK
"""

import json
import os
import numpy as np
from PIL import Image
import re
import torch
import uuid

from pathlib import Path
from typing import List
from pprint import pprint

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
)

from llama_index.readers.file import (
    CSVReader,
    DocxReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    HWPReader,
    ImageCaptionReader,
    ImageReader,
    ImageTabularChartReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PagedCSVReader,
    PandasCSVReader,
    PDFReader,
    PptxReader,
    PyMuPDFReader,
    RTFReader,
    UnstructuredReader,
    VideoAudioReader,
    XMLReader,
)

from llama_index.readers.web import (
    RssReader,
    SimpleWebPageReader,
    TrafilaturaWebReader,
)

from tavily import TavilyClient

from SaltAI_LlamaIndex.modules.utility import WILDCARD, get_full_path
from SaltAI_LlamaIndex.modules.crawler import WebCrawler

import folder_paths

def valid_url(url):
    if not url:
        return False
    regex = r"^(?:http|https):\/\/(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9-._~:/?#[\]@!$&'()*+,;=]*)?$"
    return re.match(regex, url) is not None

def read_extra_info(input_str):
    try:
        dictionary = json.loads(input_str)
        return dictionary
    except Exception as e:
        print("Parsing error:", e)
        return None


class LLMCSVReader(CSVReader):
    """
    @NOTE: Reads CSV files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.CSVReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "concat_rows": ([False,True], {"default":True}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, concat_rows:bool, extra_info:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        self.concat_rows = concat_rows
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )

class LLMDocxReader(DocxReader):
    """
    @NOTE: Reads MS Word files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.DocxReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),

            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMEpubReader(EpubReader):
    """
    @NOTE: Reads Epub book files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/epub/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.EpubReader
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMFlatReader(FlatReader):
    """
    @NOTE: Reads 'flat' files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/flat/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.FlatReader
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMHTMLTagReader(HTMLTagReader):
    """
    @NOTE: Reads HTML tags into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/html/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.HTMLTagReader
    @Imports: from bs4 import BeautifulSoup
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "tag": ("STRING", {"default":"section"}),
                "ignore_no_id": ([False, True],),
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, tag:str="section", ignore_no_id:bool=False, extra_info:str="{}"):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        self._tag = tag
        self._ignore_no_id = ignore_no_id
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMHWPReader(HWPReader):
    """
    @NOTE: Reads HWP (Korean) files into a llama_index Document
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.HWPReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str,  extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMImageTextReader(ImageReader):
    """
    @NOTE: Not sure what this does yet
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.ImageReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
    #			"keep_image": ([False, True], {"default": False}),
                "parse_text": ([False, True], {"default": False}),
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, parse_text:bool, extra_info:str, keep_image:bool=False, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
    #	self._keep_image = keep_image
        self._parse_text = parse_text
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMImageCaptionReader(ImageCaptionReader):
    """
    @NOTE: Describes the image file as if it were captioning it into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_caption/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.ImageCaptionReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
    #			"keep_image": ([False, True], {"default": False}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, prompt:str, extra_info:str, keep_image:bool=False):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
    #	self._keep_image = keep_image
        self._prompt = prompt
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMImageTabularChartReader(ImageTabularChartReader):
    """
    @NOTE: Reads an Image chart as if it were tabular data into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_deplot/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.ImageTabularChartReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
    #			"keep_image": ([False, True], {"default": False}),
                "max_output_tokens": ("INT", {"default": 512}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, max_output_tokens:int=512, prompt:str=None, extra_info:str="{}"):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        self._max_output_tokens = max_output_tokens
        if prompt and len(prompt) > 2:
            self._prompt = prompt
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMImageVisionLLMReader(ImageVisionLLMReader):
    """
    @NOTE: Not sure what this does yet
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_vision_llm/base.py
    @Documentation:https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.ImageVisionLLMReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                #"warning": ("STRING", {"default":"WARNING: This downloads a 15GB file in two parts."}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, warning:str, extra_info:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMIPYNBReader(IPYNBReader):
    """
    @NOTE: Reads IPYNB documentation files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/ipynb/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.IPYNBReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMMarkdownReader(MarkdownReader):
    """
    @NOTE: Reads Markdown documentation files (like github readmes) into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/markdown/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.MarkdownReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMMboxReader(MboxReader):
    """
    @NOTE: Reads Email files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/mbox/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.MboxReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMPDFReader(PDFReader):
    """
    @NOTE: Reads PDF files into a llama_index Document, currently doesn't support embedded images
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PDFReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMPagedCSVReader(PagedCSVReader):
    """
    @NOTE: Reads CSV files into a llama_index Document list, with paging
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/paged_csv/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PagedCSVReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "encoding": ([
                    "utf-8"
                ],),
                "delimiter": ("STRING", { "default": ","}),
                "quotechar": ("STRING", { "default": '"'}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, encoding:str, delimiter:str, quotechar:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        self._encoding = encoding
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info, delimiter, quotechar)
        return (data, )


class LLMPandasCSVReader(PandasCSVReader):
    """
    @NOTE: Reads CSV files into a llama_index Document, with some additional joiner config
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PandasCSVReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "concat_rows": ([False, True], {"default": True}),
                "col_joiner": ("STRING", {"default":""}),
                "row_joiner": ("STRING", {"default":""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
    			#"pandas_config": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path, concat_rows, col_joiner, row_joiner, extra_info:str="{}", fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        self._concat_rows=concat_rows
        self._col_joiner=col_joiner
        self._row_joiner=row_joiner
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMPptxReader(PptxReader):
    """
    @NOTE: Reads MS Powerpoint files into a llama_index Document, not sure if images are interpreted
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/slides/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PptxReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMPyMuPDFReader(PyMuPDFReader):
    """
    @NOTE: Reads PDF files into a llama_index Document using Pymu
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/pymu_pdf/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PyMuPDFReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "metadata": ([False, True], {"default": True}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, metadata:bool, extra_info:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, metadata, extra_info)
        return (data, )


class LLMRTFReader(RTFReader):
    """
    @NOTE: Reads RTF rich text files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/rtf/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.RTFReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMUnstructuredReader(UnstructuredReader):
    """
    @NOTE: Reads unstructured (most kinds of) files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.UnstructuredReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "split_documents": ([False, True], { "default": False})
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, split_documents:bool = False):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info, split_documents)
        return (data, )


class LLMVideoAudioReader(VideoAudioReader):
    """
    @NOTE: Reads Mp3 and Mp4 files and parses audio transcript into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/video_audio/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.VideoAudioReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str, fs = None):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMXMLReader(XMLReader):
    """
    @NOTE: Reads XML files into a llama_index Document
    @Source: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/xml/base.py
    @Documentation: https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.XMLReader
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "execute"
    CATEGORY = "SALT/Llama-Index/Readers"

    def execute(self, path:str, extra_info:str):
        get_full_path(1, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file available at: {path}")
        path = Path(path)
        extra_info = read_extra_info(extra_info)
        data = self.load_data(path, extra_info)
        return (data, )


class LLMDirectoryReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {}),
            },
            "optional": {
                "optional_path_list": (WILDCARD, {}),
                "recursive": ("BOOLEAN", {"default": False}),
                "required_ext_list": ("STRING", {"default": ".json, .txt, .html"}),
                "exclude_glob_list": ("STRING", {"default": ".sqlite, .zip"}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_directory"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_directory(self, input_directory, optional_path_list=[], recursive=False, required_ext_list=None, exclude_glob_list=None):

        if required_ext_list.strip():
            required_exts = [ext.strip() for ext in required_ext_list.split(",") if ext.strip()]
        else:
            required_exts = None

        if exclude_glob_list.strip():
            exclude = [pattern.strip() for pattern in exclude_glob_list.split(",") if pattern.strip()]
        else:
            exclude = None

        print("Excluding: ", exclude)
        print("Required Extensions: ", required_exts)

        if not optional_path_list:
            full_path = get_full_path(1, input_directory.strip())
            input_dir = full_path if os.path.isdir(full_path) else None
            if not input_dir:
                raise ValueError("The provided subdirectory does not exist.")

            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                exclude_hidden=True,
                recursive=recursive,
                required_exts=required_exts,
                exclude=exclude
            )
        elif optional_path_list and isinstance(optional_path_list, (str, list)):

            if isinstance(optional_path_list, str):
                path_list = [optional_path_list]
            else:
                path_list = []
                for path in optional_path_list:
                    if os.path.isfile(path): # and path.startswith(folder_paths.get_input_directory()):
                        path_list.append(path)

            reader = SimpleDirectoryReader(
                input_files=path_list,
            )


        documents = reader.load_data()
        if not documents:
            raise ValueError("No documents found in the specified directory.")

        return (documents,)


class LLMSimpleWebPageReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web(self, url_1, url_2=None, url_3=None, url_4=None, html_to_text=True):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")

        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                print("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        print("Valided URLs:", valid_urls)

        documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(valid_urls)
        return (documents,)
    

class LLMSimpleWebPageReaderAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("LIST", {}),
            },
            "optional": {
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web(self, urls, html_to_text=True):

        if not urls:
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReaderAdv")

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                print("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        print("Valided URLs:", valid_urls)

        documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(valid_urls)
        return (documents,)


class LLMTrafilaturaWebReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web_trafilatura(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMTrafilaturaWebReader")

        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                print("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        print("Valided URLs:", valid_urls)

        documents = TrafilaturaWebReader().load_data(valid_urls)
        return (documents,)
    

class LLMTrafilaturaWebReaderAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("LIST", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web_trafilatura(self, urls):

        if not urls:
            raise ValueError("At least one URL must be provided to LLMTrafilaturaWebReaderAdv")

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                print("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        print("Valided URLs:", valid_urls)

        documents = TrafilaturaWebReader().load_data(valid_urls)
        return (documents,)



class LLMRssReaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_rss"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_rss(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")

        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())

        urls = [url for url in urls if valid_url(url)]

        print("Valided URLs:", urls)

        documents = RssReader().load_data(urls)
        return (documents,)


class LLMInputToDocuments:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": (WILDCARD,),
            },
            "optional": {
                "extra_info": ("STRING", {"default": "{}"}),
                "concat_input": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "to_documents"
    CATEGORY = "SALT/Llama-Index/Documents"

    def to_documents(self, input_data, extra_info="{}", concat_input=False):
        documents = []

        # Try to parse extra_info
        try:
            extra_info = json.loads(extra_info) if extra_info.strip() else {}
        except json.JSONDecodeError:
            print("Invalid JSON for `extra_info` supplied, defaulting to empty `extra_info` dict.")
            extra_info = {}

        if not isinstance(extra_info, dict):
            print("Failed to decode `extra_info`, defaulting to empty `extra_info` dict.")
            extra_info = {}

        # Dict to string doc
        if isinstance(input_data, dict) and concat_input:
            doc_text = "\n".join(f"{key}: {value}" for key, value in input_data.items())
            documents.append(Document(text=doc_text, metadata={"source_type": "dict", **extra_info}))

        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                doc_text = f"{key}: {value}"
                documents.append(Document(text=doc_text, metadata={"source_type": "dict", **extra_info}))

        # List to string doc
        elif isinstance(input_data, list) and concat_input:
            doc_text = "\n".join(str(item) for item in input_data)
            documents.append(Document(text=doc_text, metadata={"source_type": "list", **extra_info}))

        elif isinstance(input_data, list):
            for item in input_data:
                doc_text = str(item)
                documents.append(Document(text=doc_text, metadata={"source_type": "list", **extra_info}))

        # Primitive to string doc
        elif isinstance(input_data, (str, int, float)):
            documents.append(Document(text=str(input_data), metadata={"source_type": type(input_data).__name__, **extra_info}))

        elif isinstance(input_data, torch.Tensor):

            temp = folder_paths.get_temp_directory()
            os.makedirs(temp, exist_ok=True)
            output_path = os.path.join(temp, str(uuid.uuid4()))
            os.makedirs(output_path, exist_ok=True)

            images = []
            image_paths = []
            for img in input_data:
                if img.shape[-1] == 3:
                    images.append(self.tensor2pil(img))
                else:
                    images.append(self.mask2pil(img))

            if not images:
                raise ValueError("Invalid image tensor input provided to convert to PIL!")
            
            try:
                for index, pil_image in enumerate(images):
                    file_prefix = "llm_image_input_"
                    file_ext = ".jpeg"
                    filename = f"{file_prefix}_{index:04d}{file_ext}"
                    image_path = os.path.join(output_path, filename)
                    pil_image.save(image_path, quality=100)
                    image_paths.append(image_path)

                    if os.path.exists(image_path):
                        print(f"[SALT] Saved LLM document image to `{image_path}`")
                    else:
                        print(f"[SALT] Unable to save LLM document image to `{image_path}`")

            except Exception as e:
                raise e
            
            reader = SimpleDirectoryReader(
                input_dir=output_path,
                exclude_hidden=True,
                recursive=False
            )
            
            documents = reader.load_data()

            if not documents:
                raise ValueError("No documents found in the specified directory.")
            
        else:
            raise ValueError(f"LLMInputToDocuments does not support type `{type(input_data).__name__}`. Please provide: dict, list, str, int, float.")

        pprint(documents, indent=4)

        return (documents,)
    
    def tensor2pil(self, x):
        return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def mask2pil(self, x):
        x = 1. - x
        if x.ndim != 3:
            print(f"Expected a 3D tensor ([N, H, W]). Got {x.ndim} dimensions.")
            x = x.unsqueeze(0) 
        x_np = x.cpu().numpy()
        if x_np.ndim != 3:
            x_np = np.expand_dims(x_np, axis=0) 
        return Image.fromarray(np.clip(255. * x_np[0, :, :], 0, 255).astype(np.uint8), 'L')


class LLMDocumentListAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("DOCUMENT",),
                "append_llm_documents": ("DOCUMENT",),
            },
            "optional": {
                "extra_info": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "document_append"
    CATEGORY = "SALT/Llama-Index/Documents"

    def document_append(self, llm_documents, append_llm_documents, extra_info={}):
        extra_info = json.loads(extra_info)
        for doc in append_llm_documents:
            if isinstance(doc.metadata, dict):
                doc.metadata.update(extra_info)
            elif doc.metadata == None:
                doc.metadata = extra_info
            llm_documents.append(doc)
        return (llm_documents, )


class LLMPostProcessDocuments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": ("DOCUMENT",),
            },
            "optional": {
                "required_keywords": ("STRING", {}),
                "exclude_keywords": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "process_documents"
    CATEGORY = "SALT/Llama-Index/Tools"

    def process_documents(self, document, required_keywords=[], exclude_keywords=[]):

        if required_keywords.strip():
            required = [ext.strip() for ext in required_keywords.split(",") if ext.strip()]
        else:
            required = None

        if exclude_keywords.strip():
            excluded = [pattern.strip() for pattern in exclude_keywords.split(",") if pattern.strip()]
        else:
            excluded = None

        if required or excluded:
            document = [doc for doc in document if not set(required).isdisjoint(doc.keywords) and set(excluded).isdisjoint(doc.keywords)]

        return (document,)
    

class LLMTavilyResearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tavily_api_key": ("STRING", {"default": "tvly-*******************************"}),
                "search_query": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "search_depth": (["basic", "advanced"],),
                "max_results": ("INT", {"min": 1, "max": 20, "default": 1}),
                "include_answer": ("BOOLEAN", {"default": False},),
                "include_raw_content": ("BOOLEAN", {"default": False},),
                "include_domains": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "A list of domains to specifically include in the search results. Default is None, which includes all domains. e.g. \"google.com, twitter.com\"",
                }),
                "exclude_domains": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains. e.g. \"google.com, twitter.com\"",
                }),
                "keep_looking_limit": ("INT", {"min": 1, "max": 20, "default": 10})
            }
        }
    
    RETURN_TYPES = ("DOCUMENT", "LIST")
    RETURN_NAMES = ("documents", "urls")

    FUNCTION = "search"
    CATEGORY = "SALT/Llama-Index/Tools"

    def search(self, tavily_api_key, search_query, search_depth="basic", max_results=1, include_answer=False, include_raw_content=False, include_domains="google.com", exclude_domains=None, keep_looking_limit=10):
        
        tavily = TavilyClient(api_key=tavily_api_key)

        def tavily_search():
            return tavily.search(
                query=search_query,
                search_depth=search_depth,
                max_results=max_results,
                include_images=False,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_domains=include_domains.split(
                    ", ") if include_domains is not None and include_domains != "" else None,
                exclude_domains=exclude_domains.split(
                    ", ") if include_domains is not None and exclude_domains != "" else None,
            )
        
        print("Tavily Search Query:", search_query)

        # Increment the search results because when using `include_raw_content` 
        # results are found in order of accessibility, so first X results may not 
        # be traversible, and end up in there being no result to return. But maybe 
        # the next search result does have traversible content.
        adjusted_max_results = max_results + keep_looking_limit
        current_retry = 0
        response = {}
        while "results" not in response or not response["results"] and max_results < adjusted_max_results:
                max_results += 1
                if current_retry > 0:
                    print(f"Unable find any results. Continuing Search...\nRetry {current_retry} of {keep_looking_limit}")
                response = tavily_search()
                current_retry += 1

        pprint(response, indent=4)

        results = response.get("results", None)
        urls = []
        documents = []
        if results:
            for result in results:
                content = result.pop("content", "null")
                raw_content = result.pop("raw_content", None)
                document = Document(
                    text=(raw_content if raw_content else content),
                    extra_info=result
                )
                documents.append(document)
                if result.__contains__("url"):
                    urls.append(result["url"])
        else:
            documents.append(Document(text="No document data available", extra_info={"error": "No document data available"}))
            urls.append(None)

        return (documents, urls)


class LLMSaltWebCrawler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "url": ("STRING", {}),
                "urls": ("LIST", {}),
                "max_depth": ("INT", {"min": 1, "max": 4, "default": 2}),
                "max_links": ("INT", {"min": 1, "max": 6, "default": 2}),
                "trim_line_breaks": ("BOOLEAN", {"default": True}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "exclude_domains": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Optional exclude domains, eg: example.com, example2.com"}),
                "keywords": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Optional relevancy keywords, eg: artificial intelligence, ai"})
            }
        }
    

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "crawl"
    CATEGORY = "SALT/Llama-Index/Tools"

    def crawl(self, url:str="google.com", urls:list=None, max_depth:int=2, max_links:int=2, trim_line_breaks:bool=True, verify_ssl:bool=True, exclude_domains:str="", keywords:str="") -> list:

        search_urls = []

        print(urls)

        if not url.strip() and not urls:
            raise Exception("Please provide at lease one URL")
        
        url = url.strip()
        if url != "" and valid_url(url):
            search_urls.append(url)
        if urls:
            search_urls.extend([url for url in urls if valid_url(url)])

        print(search_urls)

        crawler = WebCrawler(search_urls, exclude_domains=exclude_domains, relevancy_keywords=keywords, max_links=max_links)

        results = crawler.parse_sites(crawl=True, max_depth=max_depth, trim_line_breaks=trim_line_breaks, verify_ssl=verify_ssl)

        from pprint import pprint
        pprint(results.to_dict(), indent=4)

        return (results.to_documents(), )


NODE_CLASS_MAPPINGS = {
    "LLMCSVReader": LLMCSVReader,
    "LLMDocxReader": LLMDocxReader,
    "LLMEpubReader": LLMEpubReader,
    "LLMFlatReader": LLMFlatReader,
    "LLMHTMLTagReader": LLMHTMLTagReader,
    "LLMHWPReader": LLMHWPReader,
    "LLMImageTextReader": LLMImageTextReader,
    "LLMImageCaptionReader": LLMImageCaptionReader,
    "LLMImageTabularChartReader": LLMImageTabularChartReader,
    "LLMImageVisionLLMReader": LLMImageVisionLLMReader,
    "LLMIPYNBReader": LLMIPYNBReader,
    "LLMMarkdownReader": LLMMarkdownReader,
    "LLMMboxReader": LLMMboxReader,
    "LLMPDFReader": LLMPDFReader,
    "LLMPagedCSVReader": LLMPagedCSVReader,
    "LLMPandasCSVReader": LLMPandasCSVReader,
    "LLMPptxReader": LLMPptxReader,
    "LLMPyMuPDFReader": LLMPyMuPDFReader,
    "LLMRTFReader": LLMRTFReader,
    "LLMUnstructuredReader": LLMUnstructuredReader,
    "LLMVideoAudioReader": LLMVideoAudioReader,
    "LLMXMLReader": LLMXMLReader,
    "LLMDirectoryReader": LLMDirectoryReader,
    "LLMSimpleWebPageReader": LLMSimpleWebPageReader,
    "LLMSimpleWebPageReaderAdv": LLMSimpleWebPageReaderAdv,
    "LLMTrafilaturaWebReader": LLMTrafilaturaWebReader,
    "LLMTrafilaturaWebReaderAdv": LLMTrafilaturaWebReaderAdv,
    "LLMRssReaderNode": LLMRssReaderNode,
    "LLMInputToDocuments": LLMInputToDocuments,
    "LLMDocumentListAppend": LLMDocumentListAppend,
    "LLMPostProcessDocuments": LLMPostProcessDocuments,
    "LLMTavilyResearch": LLMTavilyResearch,
    "LLMSaltWebCrawler": LLMSaltWebCrawler
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMCSVReader": "∞ CSV",
    "LLMDocxReader": "∞ Docx",
    "LLMEpubReader": "∞ Epub",
    "LLMFlatReader": "∞ Flat",
    "LLMHTMLTagReader": "∞ HTML Tag",
    "LLMHWPReader": "∞ HWP",
    "LLMImageTextReader": "∞ Image Text Parser",
    "LLMImageCaptionReader": "∞ Image BLIP Caption",
    "LLMImageTabularChartReader": "∞ Image Tabular Chart",
    "LLMImageVisionLLMReader": "∞ Image Vision LLM",
    "LLMIPYNBReader": "∞ IPYNB",
    "LLMMarkdownReader": "∞ Markdown",
    "LLMMboxReader": "∞ Mbox",
    "LLMPDFReader": "∞ PDF",
    "LLMPagedCSVReader": "∞ Paged CSV",
    "LLMPandasCSVReader": "∞ Pandas CSV",
    "LLMPptxReader": "∞ Pptx",
    "LLMPyMuPDFReader": "∞ PyMuPDF",
    "LLMRTFReader": "∞ RTF",
    "LLMUnstructuredReader": "∞ Unstructured File",
    "LLMVideoAudioReader": "∞ Video/Audio",
    "LLMXMLReader": "∞ XML",
    "LLMDirectoryReader": "∞ Simple Directory",
    "LLMSimpleWebPageReader": "∞ Simple Web Page",
    "LLMSimpleWebPageReaderAdv": "∞ Simple Web Page (Advanced)",
    "LLMTrafilaturaWebReader": "∞ Trafilatura Web",
    "LLMTrafilaturaWebReaderAdv": "∞ Trafilatura Web (Advanced)",
    "LLMRssReaderNode": "∞ RSS",
    "LLMInputToDocuments": "∞ Input to Documents",
    "LLMDocumentListAppend": "∞ Append to Documents List",
    "LLMPostProcessDocuments": "∞ Post Process Documents",
    "LLMTavilyResearch": "∞ Tavily Research",
    "LLMSaltWebCrawler": "∞ Web Crawler"
}