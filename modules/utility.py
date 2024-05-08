import json
import os

from typing import List, Dict, Any, Type, Tuple
from pydantic import BaseModel, create_model

import folder_paths

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

# GET PATH

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

    