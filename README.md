# Quest Graphs for Agentic Systems

An implementation of agentic systems with quest graphs.

| Generation | Model Name | Description |
| ---------- | ---------- | ----------- |
| 1          | Foundation | See paper   |

## Prerequisites

1.  Install Docker Desktop

    -   Linux, please follow [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
    -   Linux, also add your user to docker group `sudo usermod -aG docker $USER`
    -   Windows and Mac, please install [Docker Desktop](https://www.docker.com/products/docker-desktop)

2.  Accelerator support: Follow the installation guide for your machine configuration. I would recommend using Linux for the best experience.

    2.1 CUDA support
    -   Nvidia driver version 555.xx or higher (for CUDA 12.5.1+)
    -   Linux, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    -   Windows, follow [this guide](https://docs.docker.com/desktop/gpu/) to enable gpu support in docker desktop.

    2.2 ROCm support
    -   Install [ROCm-kernel (amdgpu-dkms)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)

4.  Create `secrets.env` file to install neccessary tokens (Huggingface, OpenAI, etc.) (See the running section for more details.)
    ```
    export QUEST_LM_DEPLOYMENT="cloud-api-litellm"
    export QUEST_LM_MODEL="openai/gpt-4o"
    ...
    ```

## Run experiments

-   By default, use program script `./run_manual.sh {configuration} {path to file} {optional flags}` to execute the python file with the selected configuration. (See table below.)
    -   The program **may fail** to run on the first attempt due to the failure to find package directories. If this happens, run the program again.
    -   To clear the cache and reset the experiment, use `./run_manual.sh {configuration} {path to file} --reset`.
-   For VSCode, press `F5` to run the selected configuration:
    -   Launch `pytorch current file` to run the experiment in the opening file. You will also need to choose the configuration from the dropdown list.
        -   `torch-cpu` for torch in cpu environment
        -   `torch-cuda` for torch in CUDA environment.
        -   `torch-rocm` for torch in ROCm environment.
    -   Launch `reset` to clear the cache and reset the experiment.
    -   To assist pylance, add paths to local install python packages in `.vscode/settings.json`:
        ```json
        {
            "python.analysis.extraPaths": [...]
        }
        ```
        -   We recommend using `.venv` as the default virtual environment directory.
        -   Note that when building docker, python packages required for the experiment will be installed under `artifacts/pip_modules` directory. Except for pytorch, which will be installed in the docker image. To fix pylance, either refer to local install pytorch or use virtual environment on top.
-   Running on Windows
    -   The relative path in Windows that passes to docker has invalid path separators. _Always use POSIX path separators_ when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` with the hard coded configuration you wish to run with the POSIX path separators.
-   Running plots:
    -   Plots use graphics therefore they cannot be run in the docker container.
    -   Create python virtual environment in the root directory of the project, e.g. `python -m venv .venv`.
    -   Install matplotlib and other dependencies in the virtual environment, e.g. `pip install matplotlib`.
    -   Run the plot script in the virtual environment, e.g. `python tasks/plot.py`.

| Experiment       | Description                                             | Valid configurations (pick one)         | Path to file (+flags)   |
| ---------------- | ------------------------------------------------------- | --------------------------------------- | ----------------------- |
| **Test devices** | Run the test to compare the performance of the devices. | `torch-cpu`, `torch-cuda`, `torch-rocm` | `tasks/benchmark.py`    |
| **Multihop-QA**  | Run the multi-hop QA experiment.                        | `torch-cpu`, `torch-cuda`, `torch-rocm` | `tasks/qa_multihop.py`  |
| **TextWorld**    | Run the TextWorld experiment.                           | `torch-cpu`, `torch-cuda`, `torch-rocm` | `tasks/rl_textworld.py` |
| **ALFWorld**     | Run the ALFWorld experiment.                            | `torch-cpu`, `torch-cuda`, `torch-rocm` | `tasks/rl_alfworld.py`  |
| **Reset**          | Reset the selected experiment.                          | Any                                     | Any with `--reset`      |

### Environment Variables

| Environment Variables        | Description                                                              | Values                                                                      |
| ---------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `QUEST_LM_DEPLOYMENT`        | Use language model via API vs local.                                     | "cloud-api-litellm", "cloud-api-runpod", "local-hf"                         |
| `QUEST_LM_MODEL`             | Language model name.                                                     | e.g. "openai/gpt-4", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"            |
| `QUEST_LM_API_KEY`           | Language model service provider API key. (Or use provider' own variable) | string                                                                      |
| `QUEST_EMBEDDING_DEPLOYMENT` | Use embedding model via API.                                             | "cloud-api-litellm", "local-hf"                                             |
| `QUEST_EMBEDDING_MODEL`      | Embedding model name.                                                    | e.g. "text-embedding-3-small", "ibm-granite/granite-embedding-125m-english" |
| `QUEST_EMBEDDING_API_KEY`    | Embedding service provider API key. (Or use provider' own variable)      | string                                                                      |

Apart from the above environment variables, you must also include _third-party API_ keys in the `secrets.env` file in order to use their services.

## To do

### Algorithms

-   [x] basic tree
    -   [x] manual flow
    -   [x] vector dictionary (simulate hippocampus)
-   [x] ReAct
-   [x] ReAct with dynamic hierarchy
-   [x] Transformer RL
    -   [x] Actor-critic (a policy gradient method; **works well**, because it can auto-tune exploration and exploitation.)
    -   [x] Q-learning (a value-based method; works but not as good as actor-critic, hard to tune exploration.)
-   [ ] Positional encoding
    -   [x] Add PE to state, action, and context
    -   [ ] RoPE
-   [x] Reduce VRAM usage (increase training time)
    -   [x] Disentangle gradient
    -   [x] Sample action sets
-   [x] Numerical stability
    -   [x] Min clip probability
    -   [x] Gradient clipping
-   [x] Hierarchical RL
    -   [x] Score conversion when up-hierarchy
    -   [x] Check learnining on done within the sub-problem
-   [x] Infinite loop prevention
    -   [x] Contracting sub-problem filter
-   [x] Sub-problem parallel training
-   [x] Prospect training
-   [x] The process ends under the sub-problem
    -   [x] Reassign the surrogate rush goal after sub-problem end

### Backends

-   [x] CUDA support
-   [x] ROCm support
-   [x] LiteLLM
-   [x] vLLM API
-   [x] Save weights
-   [ ] GPTQ (The model outputs NaNs. Need to wait for support. When it arrives and is tested, just add `torch-rocm-gptq` to the `tasks.json` list of input configurations.)

### Experiments

-   [x] Multi-hop Q&A
    -   [x] [Musique Dataset](https://github.com/StonyBrookNLP/musique) loader
-   [x] Non-Markovian Reinforcement learning
    -   [x] [TextWorld](https://github.com/microsoft/TextWorld) environment
    -   [x] [ALFWorld](https://github.com/alfworld/alfworld) environment
    -   [ ] [Textworld express] (https://github.com/cognitiveailab/TextWorldExpress) environment, to speed up simulation.
-   [x] Sub-problem reward structure
    -   [x] Define textworld reward structure
    -   [ ] Define ALFWorld reward structure

## Note

### Language models

These models seem to work with the given instructions:

-   openai/gpt-4o, openai/gpt-4o-mini
-   Qwen/Qwen2.5-7B-Instruct
-   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

### Embedding models

-   text-embedding-3-large
-   ibm-granite/granite-embedding-125m-english
-   intfloat/multilingual-e5-large-instruct

### RL models

Actor critic models work well when combined with the following:

-   Deeper transformer layers
-   Fully connected layers for heads.

### Hiearachy

-   Low sub-problem trial count allows agent to cut losses and learn higher level faster. (Around 10 trials.)
