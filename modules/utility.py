import json
import os
import requests
import torch
import numpy as np
import uuid

from PIL import Image
from typing import List, Dict, Any, Type, Tuple
from pydantic import BaseModel, create_model
from urllib.parse import urlparse

import folder_paths

from .. import logger

# Pydantic output model mock

class CreateOutputModel:
    @staticmethod
    def create(json_str: str, model_name: str) -> Type[BaseModel]:
        input_dict = json.loads(json_str)
        model_fields = {
            key: CreateOutputModel.__get_type(key, value) for key, value in input_dict.items()
        }
        DynamicModel = create_model(model_name, **model_fields)
        return DynamicModel

    @staticmethod
    def __get_type(key: str, value: Any) -> Tuple[Type, Ellipsis]: # type: ignore
        if isinstance(value, list):
            if value and all(isinstance(elem, str) for elem in value):
                return (List[str], ...)
            else:
                return (List[Any], ...)
        elif isinstance(value, dict):
            return (Dict[str, Any], ...)
        elif isinstance(value, str):
            return (str, ...)
        elif isinstance(value, int):
            return (int, ...)
        elif isinstance(value, float):
            return (float, ...)
        else:
            return (Any, ...)
        

# ANYTYPE - WILDCARD

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
        
WILDCARD = AnyType("*")

# Comfy Tensor Conversion

def tensor2pil(x):
    return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
def pil2tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)

# Dynamic Pathing

def get_full_path(dir_type, relative_path):
    '''
    Returns the full path via a relative path within a ComfyUI IO directory.

    dir_type: 
        `0` or `temp` for temp directory
        `1` or `input` for input directory
        `2` or `output` for output directory
    relative_path: The relative path within a `dir_type` base path.
    '''
    match(dir_type):
        case 0 | 'temp':
            base_path = folder_paths.get_temp_directory()
        case 1 | 'input':
            base_path = folder_paths.get_input_directory()
        case 2 | 'output':
            base_path = folder_paths.get_output_directory()
        case _:
            raise ValueError("Invalid directory type. Please use `0` for temp, `1` for input, or `2` for output.")
        
    return os.path.join(base_path, relative_path)

def resolve_path(path: str, cache_dir: str = None):
    """Resolve a path, validating it, and if it's a URL, downloading it and providing the path to the file"""

    def is_url(path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def download(url, cache_dir):
        if not cache_dir:
            cache_dir = get_full_path(1, "downloads")

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        filename = os.path.basename(urlparse(url).path) or str(uuid.uuid4())
        file_path = os.path.join(cache_dir, filename)

        if os.path.exists(file_path):
            logger.warning(f"File already exists: {file_path}")
            return file_path

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"File downloaded: {file_path}")
        else:
            raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")

        return file_path

    def validate(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The local path does not exist: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"The path does not point to a valid file: {path}")

    if is_url(path):
        logger.info(f"Path is a URL. Downloading from: {path}")
        return download(path, cache_dir)
    else:
        logger.info(f"Path is a local file: {path}")
        validate(path)
        return path