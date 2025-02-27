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

3.  Create `secrets.env` file to install neccessary tokens (Huggingface, OpenAI, etc.) (See the running section for more details.)
    ```
    export VARIABLE_NAME_1="VARIABLE_VALUE"
    export VARIABLE_NAME_2="VARIABLE_VALUE"
    ...
    ```

## Run experiments

-   By default, use program script `./run_manual.sh {configuration} {path to file} {optional flags}` to execute the python file with the selected configuration. (See table below.)
-   For VSCode, press `F5` to run the selected configuration:
    -   launch `torch-cpu` for torch in cpu environment
    -   launch `torch-cuda` for torch in CUDA environment.
    -   launch `torch-rocm` for torch in ROCm environment.
-   Running on Windows
    -   The relative path in Windows that passes to docker has invalid path separators. _Always use POSIX path separators_ when passing `{path to file}` parameter when running `run_manual.sh` script. Or simply create a new configuration in `.vscode/launch.json` with the hard coded configuration you wish to run with the POSIX path separators.

| Experiment      | Task              | Description                                                  | Valid configurations (pick one) | File (--flags)         | Required env vars |
| --------------- | ----------------- | ------------------------------------------------------------ | ------------------------------- | ---------------------- | ----------------- |
| **Benchmark**   | Benchmark devices | Run the benchmark to compare the performance of the devices. | `torch-cuda`, `torch-rocm`      | `tasks/benchmark.py`   | -                 |
| **Multihop-QA** | Multi-hop QA      | Run the multi-hop QA experiment.                             | `torch-cuda`, `torch-rocm`      | `tasks/qa_multihop.py` | -                 |
| **TextWorld**   | TextWorld RL      | Run the TextWorld RL experiment.                             | `torch-cuda`, `torch-rocm`      | `tasks/textworld.py`   | -                 |

## To do

-   [ ] Algorithms
    -   [ ] self-consistency
    -   [ ] ReAct
    -   [ ] Reflexion

### Experiments

-   [ ] Multi-hop Q&A
    -   [ ] [Musique Dataset](https://github.com/StonyBrookNLP/musique) loader
-   [x] Reinforcement learning
    -   [x] [TextWorld](https://github.com/microsoft/TextWorld) environment
    -   [x] [ALFWorld](https://github.com/alfworld/alfworld) environment
