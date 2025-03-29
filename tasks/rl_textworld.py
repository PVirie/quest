import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping

import torch

import utilities

utilities.install('textworld')
utilities.install('textworld.gym')

import textworld
import textworld.gym

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

textworld_path = "/app/cache/textworld_data"
os.makedirs(textworld_path, exist_ok=True)

if len(os.listdir(textworld_path)) == 0:
    # tw-make custom --world-size 5 --nb-objects 10 --quest-length 5 --seed 1234 --output tw_games/custom_game.z8
    # subprocess.run(["tw-make", "custom", "--world-size", "5", "--nb-objects", "10", "--quest-length", "5", "--seed", "1234", "--output", f"{textworld_path}/games/default/custom_game.z8"])
    # tw-make tw-simple --rewards dense  --goal detailed --seed 18 --test --silent -f --output tw_games/tw-rewardsDense_goalDetailed.z8
    subprocess.run(["tw-make", "tw-simple", "--rewards", "dense", "--goal", "detailed", "--seed", "18", "--test", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-rewardsDense_goalDetailed_18.z8"])
    subprocess.run(["tw-make", "tw-simple", "--rewards", "dense", "--goal", "detailed", "--seed", "19", "--test", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-rewardsDense_goalDetailed_19.z8"])

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
        msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
        print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))



if __name__ == "__main__":

    game_path = f"{textworld_path}/games/default/tw-rewardsDense_goalDetailed_18.z8"

    request_infos = textworld.EnvInfos(
        facts=True,  # All the facts that are currently true about the world.
        admissible_commands=True,  # All commands relevant to the current state.
        entities=True,              # List of all interactable entities found in the game.
        max_score=True,            # The maximum reachable score.
        description=True,          # The description of the current room.
        inventory=True,            # The player's inventory.
        objective=True,            # The player's objective.
        won=True,                  # Whether the player has won.
        lost=True,                 # Whether the player has lost.
    )

    env_id = textworld.gym.register_game(game_path, request_infos, max_episode_steps=100, batch_size=1)
    env = textworld.gym.make(env_id)

    from implementations.example_tw_agents.agent_neural import RandomAgent, NeuralAgent
    # agent = RandomAgent()
    agent = NeuralAgent(input_size=MAX_VOCAB_SIZE, device=device)
    play(env, agent, nb_episodes=100, verbose=True)
    # play(env, agent, nb_episodes=500, verbose=False, train=True)
    # play(env, agent, nb_episodes=100, verbose=True)
    env.close()