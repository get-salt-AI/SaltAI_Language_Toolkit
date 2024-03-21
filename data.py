
# ATTN: Classes are sorted alphabetically

"""
@BUGS: 
	
	RTF Reader can't be found in imports
	Notion Reader belongs in data_connectors

	Readers from nodes_core don't load from file, are weird
	Image*Readers currently only accept files, should consider allowing IMAGE
"""

import json
import logging
import os
import re
import sys

from pathlib import Path
from typing import List

# Requirements:
# llama-index
# llama-index-readers-file

# Documentation:
# As of the time of this writing there was no documentation forthcoming as it's being rebuilt
# These all extend BaseReader
# https://llamahub.ai/l/readers/llama-index-readers-file?from=readers


# Implementation of input folder generated dropdowns using ComfyUI.folder_paths
import folder_paths
def defineInputFileExtensions():

	Salt_READER_files_dir = folder_paths.get_input_directory()		

	folder_paths.folder_names_and_paths["csv"] = ([Salt_READER_files_dir], {'.csv'})
	folder_paths.folder_names_and_paths["docx"] = ([Salt_READER_files_dir], {'.docx', '.doc', '.dot', '.docm'})
	folder_paths.folder_names_and_paths["epub"] = ([Salt_READER_files_dir], {'.epub'})
	folder_paths.folder_names_and_paths["flat"] = ([Salt_READER_files_dir], {'.csv', '.tsv', '.txt'})
	folder_paths.folder_names_and_paths["html"] = ([Salt_READER_files_dir], {'.htm', '.html', '.php', '.htx'})
	folder_paths.folder_names_and_paths["hwp"] = ([Salt_READER_files_dir], {'.hwp'})
	folder_paths.folder_names_and_paths["img"] = ([Salt_READER_files_dir], {'.gif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.ppm'})
	folder_paths.folder_names_and_paths["ipynb"] = ([Salt_READER_files_dir], {'.ipynb'})
	folder_paths.folder_names_and_paths["mbox"] = ([Salt_READER_files_dir], {'.mbox'})
	folder_paths.folder_names_and_paths["md"] = ([Salt_READER_files_dir], {'.md'})
	folder_paths.folder_names_and_paths["pdf"] = ([Salt_READER_files_dir], {'.pdf'})
	folder_paths.folder_names_and_paths["pptx"] = ([Salt_READER_files_dir], {'.pptx', '.ppt'})
	folder_paths.folder_names_and_paths["rtf"] = ([Salt_READER_files_dir], {'.rtf'})
	folder_paths.folder_names_and_paths["unstructured"] = ([Salt_READER_files_dir], {'.txt', '.docx', 'pptx', '.jpg', '.png', '.eml', '.html', 'pdf'})
	folder_paths.folder_names_and_paths["videoaudio"] = ([Salt_READER_files_dir], {'.mp3', '.mp4'})
	folder_paths.folder_names_and_paths["xml"] = ([Salt_READER_files_dir], {'.svg', '.xml', '.xhtml'})

defineInputFileExtensions()





# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular/base.py

# Imports:
from llama_index.readers.file import CSVReader

class LLMCSVReader(CSVReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("csv"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py

# Requirements:
# docx2txt

# Imports:
from llama_index.readers.file import DocxReader

# Binding:
class LLMDocxReader(DocxReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("docx"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/epub/base.py

# Imports:
from llama_index.readers.file import EpubReader

class LLMEpubReader(EpubReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("epub"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/flat/base.py

# Imports:
from llama_index.readers.file import FlatReader

class LLMFlatReader(FlatReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("flat"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/html/base.py

# Imports:
from llama_index.readers.file import HTMLTagReader

class LLMHTMLTagReader(HTMLTagReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("html"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Imports:
from llama_index.readers.file import HWPReader

class LLMHWPReader(HWPReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("hwp"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image/base.py

# Imports:
from llama_index.readers.file import ImageReader

class LLMImageReader(ImageReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("img"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_caption/base.py

# Requirements:
# torch transformers sentencepiece Pillow

# Imports:
from llama_index.readers.file import ImageCaptionReader

class LLMImageCaptionReader(ImageCaptionReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("img"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_deplot
# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_deplot/base.py

# Imports:
from llama_index.readers.file import ImageTabularChartReader

class LLMImageTabularChartReader(ImageTabularChartReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("img"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		print(f"@@@@@@{path}@@@@@")
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/image_vision_llm/base.py

# Imports:
from llama_index.readers.file import ImageVisionLLMReader

class LLMImageVisionLLMReader(ImageVisionLLMReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("img"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file:str, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/ipynb/base.py

# Imports:
from llama_index.readers.file import IPYNBReader

class LLMIPYNBReader(IPYNBReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("ipynb"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/markdown/base.py

# Imports:
from llama_index.readers.file import MarkdownReader

class LLMMarkdownReader(MarkdownReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("md"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/mbox/base.py

# Imports:
from llama_index.readers.file import MboxReader

class LLMMboxReader(MboxReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("mbox"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )
		
# Source:
# This one is weird, may change
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py

# Imports:
from llama_index.readers.file import PDFReader

# Binding:
class LLMPDFReader(PDFReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("pdf"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/paged_csv

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/paged_csv/base.py

# Imports:
from llama_index.readers.file import PagedCSVReader

# Binding:
class LLMPagedCSVReader(PagedCSVReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("csv"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/tabular/base.py

# Imports:
from llama_index.readers.file import PandasCSVReader

# Binding:
class LLMPandasCSVReader(PandasCSVReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("csv"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/slides/base.py

# Imports:
from llama_index.readers.file import PptxReader

# Binding:
class LLMPptxReader(PptxReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("pptx"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/pymu_pdf

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/pymu_pdf/base.py

# Imports:
from llama_index.readers.file import PyMuPDFReader

# Binding:
class LLMPyMuPDFReader(PyMuPDFReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("pdf"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/rtf
# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/rtf/base.py

# Imports:
#from llama_index.readers.file import RTFReader

# Binding:
"""
class LLMRTFReader(RTFReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("rtf"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )
"""

# Source
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured/base.py

# Imports:
from llama_index.readers.file import UnstructuredReader

# Binding:
class LLMUnstructuredReader(UnstructuredReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": ("STRING", { "default":""}),
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/video_audio/base.py

# Imports:
from llama_index.readers.file import VideoAudioReader

# Binding:
class LLMVideoAudioReader(VideoAudioReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("videoaudio"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )


# Documentation:
# https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/xml

# Source:
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/xml/base.py

# Imports:
from llama_index.readers.file import XMLReader

# Binding:
class LLMXMLReader(XMLReader):
	def __init__(self):
		super().__init__()
	
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"file": (folder_paths.get_filename_list("xml"), ), 
				#extra_info: Optional[Dict] = None,
				#"fs": Optional[AbstractFileSystem] = None,
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "execute"
	CATEGORY = "SALT/Llama-Index/Readers"

	def execute(self, file, extra_info = None, fs = None):
		path = os.path.join(folder_paths.get_input_directory(), file)
		path = Path(path)
		data = self.load_data(path, extra_info)
		return (data, )

# Imports:
from llama_index.core import SimpleDirectoryReader
 
class LLMDirectoryReader:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"input_directory": ("STRING", {}),
			},
			"optional": {
				"recursive": ("BOOLEAN", {"default": False}),
				"required_ext_list": ("STRING", {"default": ".json, .txt, .html"}),
				"exclude_glob_list": ("STRING", {"default": ".sqlite, .zip"}),
			},
		}

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "read_directory"
	CATEGORY = "SALT/Llama-Index/Readers"

	def read_directory(self, input_directory, recursive=False, required_ext_list=None, exclude_glob_list=None):
		full_path = os.path.join(folder_paths.get_input_directory(), input_directory.strip())

		input_dir = full_path if os.path.isdir(full_path) else None
		if not input_dir:
			raise ValueError("The provided subdirectory does not exist.")
		
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
		
		reader = SimpleDirectoryReader(
			input_dir=input_dir,
			exclude_hidden=True,
			recursive=recursive,
			required_exts=required_exts,
			exclude=exclude
		)

		documents = reader.load_data()
		if not documents:
			raise ValueError("No documents found in the specified directory.")

		return (documents,)

# Imports:
from llama_index.readers.web import SimpleWebPageReader

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

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

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
		
		urls = [url for url in urls if valid_url(url)]

		print("Valided URLs:", urls)

		documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(urls)
		return (documents,)


# Imports:
from llama_index.readers.web import TrafilaturaWebReader

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

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

	FUNCTION = "read_web_trafilatura"
	CATEGORY = "SALT/Llama-Index/Readers"

	def read_web_trafilatura(self, url_1, url_2=None, url_3=None, url_4=None):
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

		documents = TrafilaturaWebReader().load_data(urls)
		return (documents,)


# Imports:
from llama_index.readers.web import RssReader
  
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

	#) -> List[Document]:
	RETURN_TYPES = ("DOCUMENT", )
	RETURN_NAMES = ("documents", )

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

# Documents
from llama_index.core import Document

class LLMInputToDocuments:

    class AnyType(str):
        def __ne__(self, __value: object) -> bool:
            return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": (cls.AnyType("*"),),
            },
            "optional": {
                "extra_info": ("STRING", {"default": "{}"}),
                "concat_input": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

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

        else:
            raise ValueError(f"LLMInputToDocuments does not support type `{type(input_data).__name__}`. Please provide: dict, list, str, int, float.")
        
        pprint(documents, indent=4)

        return (documents,)

# Processing
class LLMPostProcessDocuments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("DOCUMENT",),
            },
            "optional": {
                "required_keywords": ("STRING", {}),
                "exclude_keywords": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "process_documents"
    CATEGORY = "SALT/Llama-Index/Tools"

    def process_documents(self, llm_documents, required_keywords=[], exclude_keywords=[]):

        if required_keywords.strip():
            required = [ext.strip() for ext in required_keywords.split(",") if ext.strip()]
        else:
            required = None

        if exclude_keywords.strip():
            excluded = [pattern.strip() for pattern in exclude_keywords.split(",") if pattern.strip()]
        else:
            excluded = None

        if required or excluded:
            llm_documents = [doc for doc in llm_documents if not set(required).isdisjoint(doc.keywords) and set(excluded).isdisjoint(doc.keywords)]

        return (llm_documents,)


NODE_CLASS_MAPPINGS = {
	"LLMCSVReader": LLMCSVReader,
#	"LLMDocxReader": LLMDocxReader,
#	"LLMEpubReader": LLMEpubReader,
	"LLMFlatReader": LLMFlatReader,
#	"LLMHTMLTagReader": LLMHTMLTagReader,
#	"LLMHWPReader": LLMHWPReader,
#	"LLMImageReader": LLMImageReader,
	"LLMImageCaptionReader": LLMImageCaptionReader,
	"LLMImageTabularChartReader": LLMImageTabularChartReader,
	"LLMImageVisionLLMReader": LLMImageVisionLLMReader,
#	"LLMIPYNBReader": LLMIPYNBReader,
	"LLMMarkdownReader": LLMMarkdownReader,
#	"LLMMboxReader": LLMMboxReader,
	"LLMPDFReader": LLMPDFReader,
	"LLMPagedCSVReader": LLMPagedCSVReader,
	"LLMPandasCSVReader": LLMPandasCSVReader,
	"LLMPptxReader": LLMPptxReader,
#	"LLMPyMuPDFReader": LLMPyMuPDFReader,
#	"LLMRTFReader": LLMRTFReader,
#	"LLMUnstructuredReader": LLMUnstructuredReader,
#	"LLMVideoAudioReader": LLMVideoAudioReader,
#	"LLMXMLReader": LLMXMLReader,
	
# From nodes_core.py
	"LLMDirectoryReader": LLMDirectoryReader,
	"LLMSimpleWebPageReader": LLMSimpleWebPageReader,
	"LLMTrafilaturaWebReader": LLMTrafilaturaWebReader,
	"LLMRssReaderNode": LLMRssReaderNode,
	"LLMInputToDocuments": LLMInputToDocuments,
	"LLMPostProcessDocuments": LLMPostProcessDocuments,
}


NODE_DISPLAY_NAME_MAPPINGS = {
	"LLMCSVReader": "∞ CSV",
#	"LLMDocxReader": "∞ Docx",
#	"LLMEpubReader": "∞ Epub",
	"LLMFlatReader": "∞ Flat",
#	"LLMHTMLTagReader": "∞ HTML Tag",
#	"LLMHWPReader": "∞ HWP",
#	"LLMImageReader": "∞ Image",
	"LLMImageCaptionReader": "∞ Image Caption",
	"LLMImageTabularChartReader": "∞ Image Tabular Chart",
	"LLMImageVisionLLMReader": "∞ Image Vision LLM",
#	"LLMIPYNBReader": "∞ IPYNB",
	"LLMMarkdownReader": "∞ Markdown",
#	"LLMMboxReader": "∞ Mbox",
	"LLMPDFReader": "∞ PDF",
	"LLMPagedCSVReader": "∞ Paged CSV",
	"LLMPandasCSVReader": "∞ Pandas CSV",
	"LLMPptxReader": "∞ Pptx",
#	"LLMPyMuPDFReader": "∞ PyMuPDF",
#	"LLMRTFReader": "∞ RTF",
#	"LLMUnstructuredReader": "∞ Unstructured File",
#	"LLMVideoAudioReader": "∞ Video/Audio",
#	"LLMXMLReader": "∞ XML",
	
# From nodes_core.py
	"LLMDirectoryReader": "∞ Simple Directory",
	"LLMSimpleWebPageReader": "∞ Simple Web Page",
	"LLMTrafilaturaWebReader": "∞ Trafilatura Web",
	"LLMRssReaderNode": "∞ RSS",
	"LLMInputToDocuments": "∞ Input to Documents",
	"LLMPostProcessDocuments": "∞ Post Process Documents",
}