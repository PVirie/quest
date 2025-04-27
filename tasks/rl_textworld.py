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
    "tw-simple": ["tw-make", "tw-simple", "--rewards", "balanced", "--goal", "brief", "--seed", "20250401", "--test", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-simple.z8"]
}
for env_name, env_args in tw_envs.items():
    env_path = env_args[-1]
    if not os.path.exists(env_path):
        subprocess.run(env_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from quest_interface import Quest_Graph, Action
from implementations.rl_agent import agent_functions, rl_graph, mdp_state
from implementations.rl_agent.persona import Persona


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
    inv_str = infos["inventory"].replace("You are carrying:", "").replace("and", ",").strip()
    return set([item.replace(".", "").strip() for item in inv_str.split(",") if item.replace(".", "").strip() != ""])


def parse_transition(objective):
    if "(Main)" in objective:
        is_main = True
        objective = objective.replace("(Main) ", "")
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
            find_items = [item.strip() for item in part.replace("Find ", "").split(",") if item.strip() != ""]

    return Textworld_Transition(go_to, set(find_items), is_main=is_main)


class Textworld_Transition(mdp_state.MDP_Transition):
    def __init__(self, new_location=None, added_items=set(), is_main=False):
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
    

    def __eq__(self, other):
        return self.delta(other) == 0
    

    def delta(self, other):
        diff = 0
        if self.new_location is not None:
            if self.new_location != other.new_location:
                diff += 1
        diff += len(self.added_items.symmetric_difference(other.added_items))
        return diff
    

    def __str__(self):
        if self.is_main:
            return f"(Main) {self.objective}"
        else:
            return f"{self.objective}"
    
    
    def __lt__(self, other):
        # test of stictly less than
        # item change < location change < None
        if other.is_main:
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
    

    def applicable_from(self, state):
        diff = 0
        if self.new_location is not None and self.new_location != state.location:
            diff += 1
        diff += len(self.added_items - state.inventory)
        return diff == self.count_diff


class Textworld_State(mdp_state.MDP_State):
    def __init__(self, obs, score, done, info):
        self.location = extract_location(info)
        self.inventory = extract_inventory(info)
        self.obs = obs
        self.score = score
        self.done = done
        self.info = info


    def __sub__(self, other):
        return Textworld_Transition(
            self.location if self.location != other.location else None, 
            self.inventory - other.inventory
        )
    

    def get_available_actions(self):
        return self.info["admissible_commands"]
    

    def get_context(self):
        return self.obs


def env_eval(node, obs):
    cl_len = node.context_length()
    mdp_score = obs.score - cl_len * 0.02
    done = obs.done
    infos = obs.info

    if done and not node.last_child_succeeded():
        # if the last child is not success, do not account the score
        terminated = False
        truncated = True
        result = None
    elif infos["won"]:
        terminated = True
        truncated = False
        result = "Success"
        mdp_score = mdp_score + 100
    elif infos["lost"]:
        terminated = True
        truncated = False
        result = "Failed"
        mdp_score = mdp_score - 1
    elif done:
        terminated = False
        truncated = True
        result = None
    else:
        terminated = False
        truncated = False
        result = None
    # mdp_score, terminated, truncated, result
    # mdp_score is the main env score
    # fulfill is for sub task, success 
    return mdp_score, terminated, truncated, result


def goal_pursuit_eval(node, obs):
    done = obs.done
    objective = node.objective
    target_transition = parse_transition(objective)
    progress_transition = obs - node.start_observation
    score_diff = target_transition.delta(progress_transition)
    cl_len = node.context_length()
    max_score = len(target_transition)
    mdp_score = max_score - score_diff  - cl_len * 0.02

    if done and not node.last_child_succeeded():
        # if the last child is not success, do not account the score
        terminated = False
        truncated = True
        result = None
    elif target_transition == progress_transition:
        terminated = True
        truncated = False
        result = "Success"
        mdp_score = mdp_score + 100
    elif done:
        terminated = False
        truncated = True
        result = None
    else:
        terminated = False
        truncated = False
        result = None

    # mdp_score, terminated, truncated, result
    return mdp_score, terminated, truncated, result


def compute_folds(objective, state_scores):
    # states is a list of obs, score, info, last_context_mark
    # return list of end value, diff_str, comparable_transition, from_context_mark, to_context_mark
    objective_transition = parse_transition(objective)
    states = [state for state, mdp_score in state_scores]
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
    # check fit gap size, and also whether all the changes are the same
    selected_transitions = [(transition_matrix[i - 1][j], j, i) for i, j in pairs if i - j >= 3 and i - j <= 10 and transition_matrix[i - 1][j] == transition_matrix[i - 1][i - 1]]
    # return fixed end state value of 100 for first training
    return [(st.objective, st, j, i) for st, j, i in selected_transitions if st.count_diff == 1 and st < objective_transition]


def play(env, persona, nb_episodes=10, allow_relegation=True, verbose=False, verbose_step=10):
    
    with open(os.path.join(experiment_path, "rollouts.txt"), "a") as f:
        # mark date
        f.write(f"========================================================================\n")
        f.write(f"Date: {utilities.get_current_time_string()}\n")
        f.write(f"------------------------------------------------------------------------\n")

    # Collect some statistics: nb_steps, final reward.
    stat_n_moves = []
    stat_scores = []
    stat_mean_context_length = []
    stat_max_context_length = []

    for no_episode in range(1, nb_episodes + 1):
        obs, infos = env.reset()  # Start new episode.
        infos = flatten_batch(infos)
        obs = infos["description"]
        obs = re.sub(r"\n+", "\n", obs)
        score = 0
        done = False

        # objective = "(Main) Go to Kitchen and Find a carrot"
        # objective_transition = parse_transition(objective)
        # max_score = len(objective_transition)
        # eval_func = goal_pursuit_eval

        objective = "(Main) " + infos["objective"]
        max_score = infos["max_score"]
        eval_func = env_eval
        root_node = rl_graph.Quest_Node(
            objective = objective,
            eval_func = eval_func,
            start_observation = Textworld_State(obs, score, done, infos),
            allow_relegation=persona.compute_should_allow_relegation(allow_relegation),
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
                if len(working_memory) > 400:
                    break
            else:
                raise ValueError("Invalid action")

        if root_node.end_observation is None:
            # error skip
            continue
        score, _, _, _ = eval_func(root_node, root_node.end_observation)

        num_children, num_quest_node = root_node.total_context_length()
        stat_n_moves.append(num_children)
        stat_scores.append(score)
        stat_mean_context_length.append(num_children / num_quest_node if num_quest_node > 0 else 0)
        stat_max_context_length.append(root_node.max_context_length())

        if verbose and no_episode % verbose_step == 0:
            # cl means context length
            msg = "episode: {}/{} steps: {:5.1f}; score: {:4.1f}/{:4.1f}; cl: {:4.1f}; max cl: {:4.1f}"
            report = msg.format(
                no_episode,
                nb_episodes,
                np.mean(stat_n_moves[-verbose_step:]), 
                np.mean(stat_scores[-verbose_step:]), max_score,
                np.mean(stat_mean_context_length[-verbose_step:]),
                np.mean(stat_max_context_length[-verbose_step:]))
            logging.info(report)
            with open(os.path.join(experiment_path, "rollouts.txt"), "a") as f:
                data = persona.print_context(root_node)
                f.write(f"Episode {no_episode}\n")
                f.write("[Report]\t" + report + "\n")
                f.write(data)
                f.write("\n")
                f.write(f"------------------------------------------------------------------------\n")


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

    game_path = f"{textworld_path}/games/default/tw-simple.z8"

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

    MAX_VOCAB_SIZE = 1000
    tokenizer = utilities.Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

    def flatten_batch(infos):
        return {k: v[0] for k, v in infos.items()}

    def env_step(action):
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        # use regex to suppress multiple \n\n to one
        obs = re.sub(r"\n+", "\n", obs)
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)
        return Textworld_State(obs, score, done, infos)

    # from implementations.rl_algorithms.hierarchy_q import Hierarchy_Q as Model
    from implementations.rl_algorithms.hierarchy_ac import Hierarchy_AC as Model
    rl_core = Model(input_size=MAX_VOCAB_SIZE, device=device)

    persona = Persona(
        rl_core,
        tokenizer,
        compute_folds,
        env_step,
        goal_pursuit_eval=goal_pursuit_eval,
        action_parser=parse_transition,
        training_relegation_probability=0.4
    )

    if not persona.load(agent_parameter_path):
        logging.info("Initiate agent training ....")
        persona.set_training_mode(True)
        play(env, persona, nb_episodes=10000, allow_relegation=True, verbose=True)
        persona.save(agent_parameter_path)

    persona.set_training_mode(False)
    play(env, persona, nb_episodes=100, allow_relegation=True, verbose=True, verbose_step=20)
    env.close()
