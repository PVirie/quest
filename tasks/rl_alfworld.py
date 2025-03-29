import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping

import torch

import utilities

utilities.install('alfworld')
utilities.install('yaml')

import yaml

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from quest_interface import Quest_Graph, Action
from implementations.rl_torch import agent_functions, rl_graph
from implementations.rl_torch.persona import Persona


MAX_VOCAB_SIZE = 1000
tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device)

def play(env, agent, nb_episodes=10, verbose=True, train=False):
    torch.manual_seed(20250301)  # For reproducibility when using action sampling.

    if train:
        agent.train()
    else:
        agent.test()
        
    def flatten_batch(infos):
        return {k: v[0] for k, v in infos.items()}

    def env_step(action):
        # adapter between non-batched and batched environment
        obs, score, done, infos = env.step([action])
        return obs[0], score[0], done[0], flatten_batch(infos)
    
    persona = Persona(env_step, agent, tokenizer)
    
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.
        obs = obs[0]
        infos = flatten_batch(infos)
        score = 0
        done = False
        nb_moves = 0

        root_node = rl_graph.Quest_Node(None, None)
        working_memory = Quest_Graph(root_node)
        working_memory.discover(rl_graph.Observation_Node(None, (obs, score, done, infos)), root_node)

        while True:
            action, param_1, param_2 = agent_functions.basic_tree(persona, working_memory.query())
            if action == Action.ANSWER:
                working_memory.respond(param_1, param_2)
                if param_2 is None:
                    if param_1 is None:
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

        last_observation = root_node.get_last_child().observation
        score = last_observation[1]

        avg_moves.append(nb_moves)
        avg_scores.append(score)

    if verbose:
        msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f}."
        print(msg.format(np.mean(avg_moves), np.mean(avg_scores)))


if __name__ == "__main__":
    # load config
    with open(f"{alfworld_path}/alfworld/configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

    # setup environment
    env = get_environment(env_type)(config, train_eval='train')
    # now quest graph does not support batched version
    env = env.init_env(batch_size=1)

    from implementations.tw_agents.agent_neural import RandomAgent, NeuralAgent
    # agent = RandomAgent()
    agent = NeuralAgent(input_size=MAX_VOCAB_SIZE, device=device)
    play(env, agent, nb_episodes=100, verbose=True)
