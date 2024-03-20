
# Create /models/llm if it's not there
import os
import folder_paths

Salt_LLM_DIR = os.path.join(folder_paths.models_dir, 'llm')
os.makedirs(Salt_LLM_DIR, exist_ok=True)

# Set up JS
WEB_DIRECTORY = "./web"

# Core
from .nodes_core import NODE_CLASS_MAPPINGS as nodes_core_classes
from .nodes_core import NODE_DISPLAY_NAME_MAPPINGS as nodes_core_display_mappings
NODE_CLASS_MAPPINGS = nodes_core_classes
NODE_DISPLAY_NAME_MAPPINGS = nodes_core_display_mappings

# Tools
from .nodes_tools import NODE_CLASS_MAPPINGS as nodes_tools_classes
from .nodes_tools import NODE_DISPLAY_NAME_MAPPINGS as nodes_tools_display_mappings
NODE_CLASS_MAPPINGS.update(nodes_tools_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(nodes_tools_display_mappings)

# LLMs
from .llm import NODE_CLASS_MAPPINGS as llm_classes
from .llm import NODE_DISPLAY_NAME_MAPPINGS as llm_display_mappings
NODE_CLASS_MAPPINGS.update(llm_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(llm_display_mappings)

# Data
from .data import NODE_CLASS_MAPPINGS as data_classes
from .data import NODE_DISPLAY_NAME_MAPPINGS as data_display_mappings
NODE_CLASS_MAPPINGS.update(data_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(data_display_mappings)

# Data Connectors
from .data_connector import NODE_CLASS_MAPPINGS as data_connector_classes
from .data_connector import NODE_DISPLAY_NAME_MAPPINGS as data__connector_display_mappings
NODE_CLASS_MAPPINGS.update(data_connector_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(data__connector_display_mappings)

# Query
from .query import NODE_CLASS_MAPPINGS as query_classes
from .query import NODE_DISPLAY_NAME_MAPPINGS as query_display_mappings
NODE_CLASS_MAPPINGS.update(query_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(query_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
