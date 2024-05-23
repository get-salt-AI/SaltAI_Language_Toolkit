import torch
import os

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

import folder_paths

class LlavaNextV1:
    def __init__(self, model_name: str, quantize: bool = False, use_flash_attention: bool = False, device: str = "cuda"):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.processor = LlavaNextProcessor.from_pretrained(model_name)

        model_path = os.path.join(folder_paths.models_dir, "llava_next")
        os.makedirs(model_path, exist_ok=True)

        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, 
                quantization_config=quantization_config, 
                device_map="auto",
                cache_dir=model_path
            )
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attention_2=use_flash_attention,
                cache_dir=model_path
            )
            self.model.to(self.device)

    def eval(self, image: Image.Image, prompt: str, custom_format: str = None, max_tokens: int = 100) -> str:
        formatted_prompt = self._format_prompt(prompt, custom_format)
        inputs = self.processor(formatted_prompt, image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        result = self.processor.decode(output[0], skip_special_tokens=True)
        cleaned_result = self._clean_response(result, formatted_prompt)
        return cleaned_result

    def _format_prompt(self, prompt: str, custom_format: str = None) -> str:
        if custom_format:
            return custom_format.replace("<prompt>", prompt)
        else:
            # Default format for llava-v1.6-mistral-7b-hf
            # We'll have to map other models; or use custom
            return f"[INST] <image>\n{prompt} [/INST]"

    def _clean_response(self, response: str, prompt: str) -> str:
        return response.split("[/INST]")[-1].strip()