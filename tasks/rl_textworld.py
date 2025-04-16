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
import random

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utilities

utilities.install('textworld')
utilities.install('textworld.gym')

import textworld
import textworld.gym

textworld_path = "/app/cache/textworld_data"
os.makedirs(textworld_path, exist_ok=True)

# expected envs
# https://textworld.readthedocs.io/en/latest/tw-make.html
# tw-make custom --world-size 5 --nb-objects 10 --quest-length 5 --seed 1234 --output tw_games/custom_game.z8
# tw-make tw-simple --rewards dense  --goal detailed --seed 18 --test --silent -f --output tw_games/tw-rewardsDense_goalBrief.z8
tw_envs = {
    "custom_game": ["tw-make", "custom", "--world-size", "5", "--nb-objects", "10", "--quest-length", "5", "--seed", "1234", "--output", f"{textworld_path}/games/default/custom_game.z8"],
    "tw-rewardsDense_goalBrief": ["tw-make", "tw-simple", "--rewards", "dense", "--goal", "brief", "--seed", "20250301", "--test", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-rewardsDense_goalBrief.z8"]
}
for env_name, env_args in tw_envs.items():
    env_path = env_args[-1]
    if not os.path.exists(env_path):
        subprocess.run(env_args)


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
    return set([item.strip().replace(".", "") for item in inv_str.split("and") if item.strip() != ""])


def parse_transition(objective):
    if "(Main)" in objective:
        is_main = True
        objective = objective.replace("(Main)", "").strip()
    else:
        is_main = False
    
    # find Go to {location}( and Find {item1} , {item2} ...)*( and Use {item1} , {item2} ...)*
    go_to = None
    find_items = []
    parts = objective.split(" and ")
    for part in parts:
        if part.startswith("Go to "):
            go_to = part.replace("Go to ", "")
        elif part.startswith("Find "):
            find_items = part.replace("Find ", "").split(" , ")

    return Textworld_Transition(0, -1, -1, go_to, set(find_items), is_main=is_main)


class Textworld_Transition(mdp_state.MDP_Transition):
    def __init__(self, delta_score, from_context_mark, to_context_mark, new_location=None, added_items=set(), is_main=False):
        self.delta_score = round(delta_score)
        self.from_context_mark = from_context_mark
        self.to_context_mark = to_context_mark
        self.new_location = new_location
        self.added_items = set(added_items)
        self.is_main = is_main

        count_diff = 0
        differences = []
        if self.new_location is not None:
            differences.append(f"Go to {self.new_location}")
            count_diff = 1

        if len(self.added_items) > 0:
            differences.append(f"Find {' , '.join(self.added_items)}")
            count_diff += len(self.added_items)
        
        self.objective = " and ".join(differences)
        self.count_diff = count_diff


    def __len__(self):
        return self.count_diff
    

    def __sub__(self, other):
        diff = 0
        if self.new_location is not None and self.new_location != other.new_location:
            diff += 1
        diff += len(self.added_items - other.added_items)
        return diff
    

    def __lt__(self, other):
        # test of stictly less than
        # item change < location change < None
        if other is None or other.is_main:
            return True
        elif self.new_location is None and other.new_location is not None:
            return True
        elif self.new_location is not None:
            if other.new_location is None:
                return False
            if self.new_location != other.new_location:
                return False

        if self.added_items == other.added_items:
            items_compare = 0
        elif len(self.added_items - other.added_items) == 0:
            if len(other.added_items - self.added_items) == 0:
                items_compare = 0
            else:
                items_compare = -1
        else:
            items_compare = 1

        return items_compare < 0


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
            self.inventory - other.inventory
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

        objective = "(Main) Go to Kitchen and Find a note , a carrot"
        objective_transition = parse_transition(objective)
        root_node = rl_graph.Quest_Node(
            objective = objective,
            eval_func = goal_pursuit_eval,
            start_observation = (obs, score, done, infos)
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
                if len(working_memory) > 500:
                    break
                nb_moves += 1
            else:
                raise ValueError("Invalid action")

        score, _, _, _, _ = goal_pursuit_eval(root_node, root_node.end_observation)

        avg_moves.append(nb_moves)
        avg_scores.append(score)

        if verbose and no_episode % verbose_step == 0:
            msg = "\tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
            logging.info(msg.format(np.mean(avg_moves), np.mean(avg_scores), len(objective_transition)))
            avg_moves, avg_scores = [], []

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

    game_path = f"{textworld_path}/games/default/tw-rewardsDense_goalBrief.z8"

    random.seed(20250301)  # For reproducibility when using the game.
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

    def env_step(action):
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)
        return obs, score, done, infos

    def env_eval(node, obs):
        _, env_score, done, infos = obs
        mdp_score = env_score - node.size() * 0.05

        if infos["won"]:
            terminated = True
            truncated = False
            result = "Success"
            next_value = 100
        elif infos["lost"]:
            terminated = True
            truncated = False
            result = "Failed"
            next_value = -10
        elif done:
            terminated = False
            truncated = True
            result = None
            next_value = None
        else:
            terminated = False
            truncated = False
            result = None
            next_value = None
        # mdp_score, terminated, truncated, result, finish_value
        # mdp_score is the main env score
        # fulfill is for sub task, success 
        return mdp_score, terminated, truncated, result, next_value

    def goal_pursuit_eval(node, obs):
        _, _, done, infos = obs
        objective = node.objective
        target_transition = parse_transition(objective)
        progress_transition = Textworld_Transition(0, -1, -1, extract_location(infos), extract_inventory(infos))
        score_diff = target_transition - progress_transition
        expected_score = len(target_transition) - node.size() * 0.05
        mdp_score = expected_score - score_diff 

        if score_diff == 0:
            terminated = True
            truncated = False
            result = "Success"
            next_value = 50
        elif (expected_score < 0 and not target_transition.is_main) or done:
            # too many children, stop the task
            terminated = False
            truncated = True
            result = None
            next_value = None
        else:
            terminated = False
            truncated = False
            result = None
            next_value = None

        # mdp_score, terminated, truncated, result, finish_value
        return mdp_score, terminated, truncated, result, next_value

    
    def compute_folds(objective, states):
        # states is a list of obs, score, info, last_context_mark
        # return list of end value, diff_str, comparable_transition, from_context_mark, to_context_mark
        objective_transition = parse_transition(objective)
        states = [Textworld_State(score, info, lcm) for _, score, info, lcm in states]
        transition_matrix = [] # the first row is at index 1 but the first column is at index 0
        for i in range(1, len(states)):
            transition_row = []
            for j in range(0, i):
                transition_row.append(states[i] - states[j])
            transition_matrix.append(transition_row)
        pivots = [0]
        for i in range(0, len(transition_matrix)):
            if len(transition_matrix[i][i]) > 0:
                pivots.append(i+1)
        # now compute all pairs of pivots
        pairs = combinations(reversed(pivots), 2)
        # gap greater or equal 2 steps
        selected_transitions = [(transition_matrix[i - 1][j], j, i) for i, j in pairs if i - j >= 2]
        # return fixed end state value of 100 for first training
        return [(10, -1, st.objective, st, j, i) for st, j, i in selected_transitions if st.count_diff >= 1]
    

    # from implementations.tw_agents.agent_neural import Random_Agent, Neural_Agent
    # agent = RandomAgent()
    # agent = Neural_Agent(input_size=MAX_VOCAB_SIZE, device=device)
    from implementations.tw_agents.hierarchy_agent import Hierarchy_Agent
    agent = Hierarchy_Agent(input_size=MAX_VOCAB_SIZE, device=device)

    persona = Persona(agent, tokenizer, compute_folds, env_step, goal_pursuit_eval=goal_pursuit_eval, action_parser=parse_transition)

    # play(env, persona, nb_episodes=100, verbose=True)
    
    if not persona.load(agent_parameter_path):
        logging.info("Initiate agent training ....")
        persona.set_training_mode(True)
        persona.set_allow_relegation(True)
        play(env, persona, nb_episodes=1000, verbose=True)
        persona.save(agent_parameter_path)

    persona.set_training_mode(False)
    play(env, persona, nb_episodes=100, verbose=True, verbose_step=20)
    env.close()