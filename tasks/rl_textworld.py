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

import utilities

utilities.install('textworld')
utilities.install('textworld.gym')

import textworld
import textworld.gym

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
from implementations.rl import agent_functions, rl_graph
from implementations.rl.persona import Persona


MAX_VOCAB_SIZE = 1000
tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

def play(env, agent, nb_episodes=10, verbose=True, train=False):
    torch.manual_seed(20250301)  # For reproducibility when using action sampling.

    def flatten_batch(infos):
        return {k: v[0] for k, v in infos.items()}

    def env_step(objective, action, num_children):
        # adapter between non-batched and batched environment
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)

        success = False
        current_value = 0
        if done:
            if infos["won"]:
                success = True
                current_value = 100
            else:
                success = False
                current_value = -100

        # obs, score, done, infos, fulfilled, success, current_value
        # score is the main env score
        # fulfill is for sub task, success 
        return obs, score, done, infos, False, success, current_value
    
    def extract_inventory(infos):
        # find anything after "You are carrying:"" and split using "and"
        if "inventory" not in infos or "nothing" in infos["inventory"]:
            return set()
        inv_str = infos["inventory"].replace("You are carrying:", "").strip()
        return set([item.strip() for item in inv_str.split("and") if item.strip() != ""])

    def sub_env_step(objective, action, num_children):
        # adapter between non-batched and batched environment
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)

        go_to = None
        find_items = []
        use_items = []
        # find Go to {location}( and Find {item1} {item2} ...)*( and Use {item1} {item2} ...)*
        # first split objective into components
        parts = objective.split(" and ")
        for part in parts:
            if part.startswith("Go to "):
                go_to = part.replace("Go to ", "")
            elif part.startswith("Find "):
                find_items = part.replace("Find ", "").split(" ")
            elif part.startswith("Use "):
                use_items = part.replace("Use ", "").split(" ")

        # check inventory first
        task_score = 0
        current_inventory = extract_inventory(infos)
        success = True
        if len(find_items) > 0:
            for item in find_items:
                if item in current_inventory:
                    task_score += 1
                else:
                    success = False
            
        if len(use_items) > 0:
            for item in use_items:
                if item not in current_inventory:
                    task_score += 1
                else:
                    success = False

        if go_to is not None:
            # now must get all the item and go to the location to fulfill the objective
            at_the_location = f"-= {go_to} =" in obs
            if at_the_location:
                task_score += 1
            fulfilled = success and at_the_location
        else:
            fulfilled = success

        current_value = 20 if success else 0

        if done:
            # if env end before fulfilling the task, success is False
            success = False

        if num_children > 20:
            # too many children, stop the task
            return obs, score, done, infos, True, False, 0

        # obs, score, done, infos, fulfilled, success, current_value
        return obs, score, done, infos, fulfilled, success, current_value

    
    def observation_difference(from_obs, to_obs, carry):
        if carry is None:
            carry = {
                "current_location": None,
                "current_inventory": set()
            }

        # Compare the two observations and return the difference.
        # First check location: "-= Kitchen =-" - anything else = "Move to Kitchen" 
        # Second check inventory: "You are carrying: a half of a bag of chips and an old key" - "You are carrying: an old key" = "Find and Take: a half of a bag of chips"
        # If inventory size is reduce use command: "Use: a half of a bag of chips"
        to_obs, to_score, _, to_infos, _ = to_obs
        from_obs, from_score, _, from_infos, _ = from_obs
        # first find location pattern -= {location} =-
        def extract_location(obs):
            result = re.search(r"-= (.*?) =-", obs)
            location = None
            if result is not None:
                location = result.group(1)
            return location
        to_location = extract_location(to_obs)
        from_location = extract_location(from_obs)
        if from_location is not None:
            carry["current_location"] = from_location

        # next check inventory
        to_inv = extract_inventory(to_infos)
        from_inv = extract_inventory(from_infos)
        carry["current_inventory"] = from_inv

        count_diff = 0
        differences = []
        if to_location is not None and to_location != carry["current_location"]:
            differences.append(f"Go to {to_location}")
            count_diff = 1

        to_from_diff = to_inv - from_inv
        from_to_diff = from_inv - to_inv
        if len(to_from_diff) > 0:
            differences.append(f"Find {' '.join(to_from_diff)}")
            count_diff += len(to_from_diff)
        if len(from_to_diff) > 0:
            differences.append(f"Use {' '.join(from_to_diff)}")
            count_diff += len(from_to_diff)
        
        # can also check score here
        if to_score > from_score and count_diff >= 1:
            return True, " and ".join(differences), carry
        return False, "", carry
    
    
    persona = Persona(agent, tokenizer, observation_difference, sub_env_step, train=train)
    
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
            objective = infos["objective"],
            env_step = env_step,
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

        if root_node.end_observation is not None:
            score = root_node.end_observation[1]
        else:
            score = 0
            for node in reversed(root_node.get_children()):
                if isinstance(node, rl_graph.Quest_Node):
                    if node.end_observation is not None:
                        score = node.end_observation[1]
                        break
                elif isinstance(node, rl_graph.Observation_Node):
                    score = node.observation[1]
                    break

        avg_moves.append(nb_moves)
        avg_scores.append(score)

    if verbose:
        msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
        print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

        with open(os.path.join(experiment_path, "rollouts.txt"), "a") as f:
            data = persona.print_context(root_node)
            f.write(f"Episode {no_episode}\n")
            f.write(data)
            f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", "-r", action="store_true")
    args = parser.parse_args()

    experiment_path = "/app/experiments/rl_textworld"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

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

    # from implementations.tw_agents.agent_neural import Random_Agent, Neural_Agent
    # agent = RandomAgent()
    # agent = Neural_Agent(input_size=MAX_VOCAB_SIZE, device=device)
    from implementations.tw_agents.hierarchy_agent import Hierarchy_Agent
    agent = Hierarchy_Agent(input_size=MAX_VOCAB_SIZE, device=device)
    # play(env, agent, nb_episodes=100, verbose=True)
    play(env, agent, nb_episodes=500, verbose=False, train=True)
    play(env, agent, nb_episodes=100, verbose=True)
    env.close()