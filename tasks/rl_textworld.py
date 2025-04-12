import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import argparse
import shutil
import re
from itertools import combinations
import logging

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
from implementations.rl import agent_functions, rl_graph, mdp_state
from implementations.rl.persona import Persona


def extract_location(infos):
    # first find location pattern -= {location} =-
    result = re.search(r"-= (.*?) =-", infos["description"])
    location = None
    if result is not None:
        location = result.group(1)
    return location


def extract_inventory(infos):
    # find anything after "You are carrying:"" and split using "and"
    if "inventory" not in infos or "nothing" in infos["inventory"]:
        return set()
    inv_str = infos["inventory"].replace("You are carrying:", "").strip()
    return set([item.strip() for item in inv_str.split("and") if item.strip() != ""])


class Textworld_Transition(mdp_state.MDP_Transition):
    def __init__(self, delta_score, from_context_mark, to_context_mark, new_location=None, added_items=set(), removed_items=set()):
        self.delta_score = delta_score
        self.from_context_mark = from_context_mark
        self.to_context_mark = to_context_mark
        self.new_location = new_location
        self.added_items = added_items
        self.removed_items = removed_items

        count_diff = 0
        differences = []
        if self.new_location is not None:
            differences.append(f"Go to {self.new_location}")
            count_diff = 1

        if len(self.added_items) > 0:
            differences.append(f"Find {' , '.join(self.added_items)}")
            count_diff += len(self.added_items)
        if len(self.removed_items) > 0:
            differences.append(f"Use {' , '.join(self.removed_items)}")
            count_diff += len(self.removed_items)
        
        self.objective = f"({self.delta_score}) " + " and ".join(differences)
        self.count_diff = count_diff


    def __len__(self):
        return self.count_diff
    

    def __sub__(self, other):
        diff = 0
        if self.new_location != other.new_location:
            diff += 1
        diff += len(self.added_items.symmetric_difference(other.added_items))
        diff += len(self.removed_items.symmetric_difference(other.removed_items))
        return diff
    

    @staticmethod
    def parse(objective):
        # first take the score
        # find (score), identify the first )
        close_parenthesis = objective.find(")")
        if close_parenthesis == -1:
            raise ValueError("Invalid objective format")
        score = float(objective[1:close_parenthesis])

        objective = objective[close_parenthesis + 2:]
        # second split objective into components
        # find Go to {location}( and Find {item1} , {item2} ...)*( and Use {item1} , {item2} ...)*
        go_to = None
        find_items = []
        use_items = []
        parts = objective.split(" and ")
        for part in parts:
            if part.startswith("Go to "):
                go_to = part.replace("Go to ", "")
            elif part.startswith("Find "):
                find_items = part.replace("Find ", "").split(" , ")
            elif part.startswith("Use "):
                use_items = part.replace("Use ", "").split(" , ")

        return Textworld_Transition(score, -1, go_to, set(find_items), set(use_items))


class Textworld_State(mdp_state.MDP_State):
    def __init__(self, score, info, last_context_mark):
        self.location = extract_location(info)
        self.inventory = extract_inventory(info)
        self.score = score
        self.last_context_mark = last_context_mark


    def __sub__(self, other):
        return Textworld_Transition(
            self.score - other.score, 
            other.last_context_mark,
            self.last_context_mark,
            self.location if self.location != other.location else None, 
            self.inventory - other.inventory, 
            other.inventory - self.inventory
        )


MAX_VOCAB_SIZE = 1000
tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

def play(env, persona, nb_episodes=10, verbose=False, verbose_step=10):
    
    with open(os.path.join(experiment_path, "rollouts.txt"), "a") as f:
        # mark date
        f.write(f"====================================\n")
        f.write(f"Date: {utilities.get_current_time_string()}\n")

    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores = [], []
    for no_episode in range(1, nb_episodes + 1):
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
                if len(working_memory) > 120:
                    break
                nb_moves += 1
            else:
                raise ValueError("Invalid action")

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

        if verbose and no_episode % verbose_step == 0:
            msg = "\tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
            logging.info(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

            with open(os.path.join(experiment_path, "rollouts.txt"), "a") as f:
                data = persona.print_context(root_node)
                f.write(f"Episode {no_episode}\n")
                f.write(data)
                f.write("\n\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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

    agent_parameter_path = os.path.join(experiment_path, "parameters")
    os.makedirs(agent_parameter_path, exist_ok=True)

    game_path = f"{textworld_path}/games/default/tw-rewardsDense_goalDetailed_18.z8"

    torch.manual_seed(20250301)  # For reproducibility when using action sampling.

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

    def flatten_batch(infos):
        return {k: v[0] for k, v in infos.items()}

    def env_step(objective, action, start_obs, num_children):
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
    

    def sub_env_step(objective, action, start_obs, num_children):
        # adapter between non-batched and batched environment
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)

        current_state = Textworld_State(score, infos, 0)
        start_state = Textworld_State(start_obs[1], start_obs[3], num_children)
        progress = current_state - start_state

        target = Textworld_Transition.parse(objective)
        score_diff = target - progress

        if score_diff == 0:
            fulfilled = True
            success = True
            current_value = 100
            score = start_obs[1] + target.delta_score - num_children * 0.1
        else:
            fulfilled = False
            success = False
            current_value = 0
            score = start_obs[1] + target.delta_score - score_diff - num_children * 0.1

        if done:
            # if env end before fulfilling the task, success is False
            success = False

        if num_children > 25:
            # too many children, stop the task
            return obs, score, done, infos, True, False, 0

        # obs, score, done, infos, fulfilled, success, current_value
        return obs, score, done, infos, fulfilled, success, current_value

    
    def compute_folds(states):
        # states is a list of obs, score, info, last_context_mark
        # return list of delta_score, diff_str, from_context_mark, to_context_mark
        states = [Textworld_State(score, info, lcm) for _, score, info, lcm in states]
        transition_matrix = []
        for i in range(1, len(states)):
            transition_row = []
            for j in range(0, i):
                transition_row.append(states[i] - states[j])
            transition_matrix.append(transition_row)
        pivots = [0]
        for i in range(1, len(transition_matrix)):
            if len(transition_matrix[i][i-1]) > 0:
                pivots.append(i)
        # now compute all pairs of pivots
        pairs = combinations(reversed(pivots), 2)
        # gap greater than 4 steps
        selected_transitions = [(transition_matrix[i][j], j, i) for i, j in pairs if i - j >= 4]
        return [(st.delta_score, st.objective, j, i) for st, j, i in selected_transitions if st.count_diff > 0 and st.delta_score > 0]
    

    # from implementations.tw_agents.agent_neural import Random_Agent, Neural_Agent
    # agent = RandomAgent()
    # agent = Neural_Agent(input_size=MAX_VOCAB_SIZE, device=device)
    from implementations.tw_agents.hierarchy_agent import Hierarchy_Agent
    agent = Hierarchy_Agent(input_size=MAX_VOCAB_SIZE, device=device)

    persona = Persona(agent, tokenizer, compute_folds, sub_env_step)

    # play(env, persona, nb_episodes=100, verbose=True)
    
    if not persona.load(agent_parameter_path):
        logging.info("Initiate agent training ....")
        persona.set_training_mode(True)
        persona.set_allow_relegation(False)
        play(env, persona, nb_episodes=500, verbose=True)
        persona.save(agent_parameter_path)

    persona.set_training_mode(False)
    play(env, persona, nb_episodes=100, verbose=True)
    env.close()