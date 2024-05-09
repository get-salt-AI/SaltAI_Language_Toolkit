# SaltAI Llama-Index

The project integrates the Retrieval Augmented Generation (RAG) tool [Llama-Index](https://www.llamaindex.ai/) and [Microsoft's AutoGen](https://microsoft.github.io/autogen/) with [ComfyUI's](https://github.com/comfyanonymous/ComfyUI) adaptable node interface, enhancing the functionality and user experience of the platform.

🔥 May 9, 2024: Added agents, more information can be found [here](./nodes/autogen/README.md).

## Installation Instructions

### Using Git and Pip

Follow these steps to set up the environment:

1. Set up a virtual environment as needed.
2. Navigate to `ComfyUI/custom_nodes`.
3. Clone the repository:
   git clone https://github.com/get-salt-AI/SaltAI_Llama-Index
4. Change to the cloned directory:
   cd SaltAI_Llama-index
5. **Install dependencies:**

	5.a **Python venv:**
   - `pip install -r requirements.txt`
  
	5.b **ComfyUI Portable:**
	- `path\to\ComfyUI\python_embeded\python.exe -m pip install -r requirements.txt`

## ComfyUI Manager

1. Have ComfyUI-Manager installed.
2. Open up Manager within ComfyUI and search for the nodepack "SaltAI_LlamaIndex"
3. Install
4. Restart the server.
5. Ctrl+F5 Hard refresh the browser.

## Troubleshooting

If you encounter issues due to package conflicts, ensure your virtual environment is configured correctly.

## Acquiring Models

You can install and use any GGUF files loaded into your `ComfyUI/custom_nodes/models/llm` folder.

Here is probably the world's largest repository of those:

[Hugging Face LLM Category](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)

## Examples

Example workflows and images can be found in the `/examples` folder

## Documentation and Contributions

_Detailed documentation and guidelines for contributing to the project will be provided soon._

You can find out existing documentation at https://docs.getsalt.ai/

## License

The project is open-source under the MIT license.
