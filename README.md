# SaltAI Language Toolkit

The project integrates the Retrieval Augmented Generation (RAG) tool [Llama-Index](https://www.llamaindex.ai/), [Microsoft's AutoGen](https://microsoft.github.io/autogen/), and [LlaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) with [ComfyUI's](https://github.com/comfyanonymous/ComfyUI) adaptable node interface, enhancing the functionality and user experience of the platform.

🔥 May 9, 2024: Added agents, more information can be found [here](./nodes/autogen/README.md).

## Installation Instructions

### Using Git and Pip

Follow these steps to set up the environment:

1. Set up a virtual environment as needed.
2. Navigate to `ComfyUI/custom_nodes`.
3. Clone the repository:
   git clone https://github.com/get-salt-AI/SaltAI_LlamaIndex
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

## Installation Note:

You may need to update your environments packaging, wheels, and setuptools for newer Transformers and LlaVA-Next models.
 - `pip install --upgrade packaging setuptools wheel`
 ***Or***
 - `path\to\ComfyUI\python_embeded\python.exe -m pip install --upgrade packaging setuptools wheel`
 
## Examples

Example workflows and images can be found in the [Examples Section](examples) folder. 

- **[Example_agents.json](examples/Example_agents.json)** - shows you how to create conversible agents, with various examples of how they could be setup.
- **[Example_groq_search.json](examples/Example_groq_search.json)** - shows you how to search with a Groq LLM model, featuring Tavily Research node.
- **[Example_SERP_search.json](examples/Example_SERP_search.json)** - shows you how to search with Scale SERP, and also demonstrates how to use different models with same setup. 
- **[Example_search_to_json.json](examples/Example_search_to_json.json)** - shows you how to take search results, and convert them to JSON output which could be fed to another system for use. 

## Troubleshooting

If you encounter issues due to package conflicts, ensure your virtual environment is configured correctly.

## Acquiring Models

You can install and use any GGUF files loaded into your `ComfyUI/custom_nodes/models/llm` folder.

Here is probably the world's largest repository of those:

[Hugging Face LLM Category](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)

## Documentation and Contributions

_Detailed documentation and guidelines for contributing to the project will be provided soon._

You can find out existing documentation at https://docs.getsalt.ai/

## License

The project is open-source under the MIT license.