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

class RandomAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return self.rng.choice(infos["admissible_commands"])

MAX_VOCAB_SIZE = 1000
tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device)

def play(env, agent, nb_episodes=10, verbose=True):
    torch.manual_seed(20211021)  # For reproducibility when using action sampling.

    def prepare_tensors(obs, infos):
        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

        # Tokenize and pad the input and the commands to chose from.
        state_tensor = tokenizer([input_])
        action_list_tensor = tokenizer(infos["admissible_commands"])

        return state_tensor, action_list_tensor

    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        while not done:
            state_tensor, action_list_tensor = prepare_tensors(obs, infos)
            command = agent.act(state_tensor, action_list_tensor, score, done, infos)
            obs, score, done, infos = env.step(command)
            nb_moves += 1

        state_tensor, action_list_tensor = prepare_tensors(obs, infos)
        agent.act(state_tensor, action_list_tensor, score, done, infos)  # Let the agent know the game is done.

        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

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

    env_id = textworld.gym.register_game(game_path, request_infos, max_episode_steps=100)
    env = textworld.gym.make(env_id)

    from implementations.example_tw_agents.agent_neural import NeuralAgent

    # agent = RandomAgent()
    agent = NeuralAgent(input_size=MAX_VOCAB_SIZE, device=device)
    play(env, agent, nb_episodes=100, verbose=True)
    agent.train()
    play(env, agent, nb_episodes=500, verbose=False)
    agent.test()
    play(env, agent, nb_episodes=100, verbose=True)
    env.close()