import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import yaml

import torch

import utilities

utilities.install('alfworld')

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

alfworld_path = "/app/cache/alfworld_data"
os.environ['ALFWORLD_DATA'] = alfworld_path
os.makedirs(alfworld_path, exist_ok=True)
if len(os.listdir(alfworld_path)) == 0:
    subprocess.run(["alfworld-download"])
    # clone the Alfred data repository
    subprocess.run(["git", "clone", "https://github.com/alfworld/alfworld.git", f"{alfworld_path}/alfworld"])

# load config
with open(f"{alfworld_path}/alfworld/configs/base_config.yaml", 'r') as f:
    config = yaml.safe_load(f)
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
for i in range(5):
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]

    # step
    obs, scores, dones, infos = env.step(random_actions)
    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))