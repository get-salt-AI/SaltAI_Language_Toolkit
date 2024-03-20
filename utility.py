from typing import List, Dict, Any, Type, Tuple
from pydantic import BaseModel, create_model
import json

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
    def __get_type(key: str, value: Any) -> Tuple[Type, Ellipsis]:
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