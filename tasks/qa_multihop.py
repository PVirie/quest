import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import yaml

import torch

musique_data_path = "/app/cache/musique_data"
os.makedirs(musique_data_path, exist_ok=True)
if len(os.listdir(musique_data_path)) == 0:
    subprocess.run(["git", "clone", "https://github.com/StonyBrookNLP/musique.git", f"{musique_data_path}/repo"])
    # run bash download_data.sh, the result will be downloaded to musique_data_path/repo/data
    subprocess.run(["bash", "download_data.sh"], cwd=f"{musique_data_path}/repo")

