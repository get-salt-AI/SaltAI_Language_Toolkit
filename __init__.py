
# Create /models/llm if it's not there
import os

# Logger
from .modules.log import create_logger
logger = create_logger()

import folder_paths

Salt_LLM_DIR = os.path.join(folder_paths.models_dir, 'llm')
os.makedirs(Salt_LLM_DIR, exist_ok=True)

# Set up JS
WEB_DIRECTORY = "./web"

ROOT = os.path.abspath(os.path.dirname(__file__))
NAME = "Salt.AI Language Toolkit"
PACKAGE = "SaltAI_Language_Toolkit"
MENU_NAME = "SALT"
SUB_MENU_NAME = "Language Toolkit"
NODES_DIR = os.path.join(ROOT, 'nodes')
EXTENSION_WEB_DIRS = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Load modules
from .modules.node_importer import ModuleLoader

module_timings = {}
module_loader = ModuleLoader(PACKAGE)
module_loader.load_modules(NODES_DIR)

# Mappings
NODE_CLASS_MAPPINGS = module_loader.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = module_loader.NODE_DISPLAY_NAME_MAPPINGS

# Timings and such
logger.info("")
module_loader.report(NAME)
logger.info("")

# Export nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']