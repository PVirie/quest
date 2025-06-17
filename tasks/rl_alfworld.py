import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import argparse
import shutil
import re

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

import utilities

utilities.install('alfworld')
utilities.install('yaml')

import yaml

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

alfworld_path = f"{APP_ROOT}/cache/alfworld_data"
os.environ['ALFWORLD_DATA'] = alfworld_path
os.makedirs(alfworld_path, exist_ok=True)
if len(os.listdir(alfworld_path)) == 0:
    subprocess.run(["alfworld-download"])
    # clone the Alfred data repository
    subprocess.run(["git", "clone", "https://github.com/alfworld/alfworld.git", f"{alfworld_path}/alfworld"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from quest_interface import Quest_Graph, Action
from quest_interface.mdp_state import MDP_State
from implementations.rl_agent import agent_functions, rl_graph
from implementations.rl_agent.persona import Persona


MAX_VOCAB_SIZE = 1000
tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

def play(env, agent, nb_episodes=10, verbose=True, train=False):
    torch.manual_seed(20250301)  # For reproducibility when using action sampling.

    def flatten_batch(infos):
        return {k: v[0] for k, v in infos.items()}

    def env_step(action):
        # adapter between non-batched and batched environment
        obs, score, done, infos = env.step([action])
        return obs[0], score[0], done[0], flatten_batch(infos)
    
    def observation_difference(obs1, obs2, carry):
        return False, ""
    
    persona = Persona(env_step, agent, tokenizer, observation_difference, train=train)
    
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.
        infos = flatten_batch(infos)
        obs = infos["description"]
        score = 0
        done = False
        nb_moves = 0

        root_node = rl_graph.Quest_Node(
            quest = {
                "objective": infos["objective"],
            }, 
            start_observation = (obs, score, done, infos, None)
        )
        working_memory = Quest_Graph(root_node)

        while True:
            action, param_1, param_2 = agent_functions.basic_tree(persona, working_memory.query())
            if action == Action.ANSWER:
                working_memory.respond(param_1, param_2)
                if param_2 is None:
                    break
            elif action == Action.DISCOVER:
                working_memory.discover(param_1, param_2)
                if len(working_memory) > 100:
                    break
                nb_moves += 1
            else:
                raise ValueError("Invalid action")

        if verbose:
            print(".", end="")

        if root_node.observation is not None:
            score = root_node.observation[1]
        else:
            score = 0
            for node in reversed(root_node.get_children()):
                if isinstance(node, rl_graph.Quest_Node):
                    if node.observation is not None:
                        score = node.observation[1]
                        break
                elif isinstance(node, rl_graph.Observation_Node):
                    score = node.observation[1]
                    break

        avg_moves.append(nb_moves)
        avg_scores.append(score)

    if verbose:
        msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f}."
        print(msg.format(np.mean(avg_moves), np.mean(avg_scores)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", "-r", action="store_true")
    args = parser.parse_args()

    experiment_path = f"{APP_ROOT}/experiments/rl_alfworld"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    # load config
    with open(f"{alfworld_path}/alfworld/configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

    # setup environment
    env = get_environment(env_type)(config, train_eval='train')
    # now quest graph does not support batched version
    env = env.init_env(batch_size=1)

    from implementations.rl_algorithms.hierarchy_ac import Hierarchy_AC
    # agent = RandomAgent()
    agent = Hierarchy_AC(input_size=MAX_VOCAB_SIZE, device=device)
    play(env, agent, nb_episodes=100, verbose=True)
