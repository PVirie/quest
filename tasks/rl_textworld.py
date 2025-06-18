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

APP_ROOT = os.getenv("APP_ROOT", "/app")

import utilities
from utilities.tokenizer import *

utilities.install('textworld')
utilities.install('textworld.gym')

import textworld
import textworld.gym

textworld_path = f"{APP_ROOT}/cache/textworld_data"
os.makedirs(textworld_path, exist_ok=True)

# expected envs
# https://textworld.readthedocs.io/en/latest/tw-make.html
# tw-make custom --world-size 5 --nb-objects 10 --quest-length 5 --seed 1234 --output tw_games/custom_game.z8
# tw-make tw-simple --rewards dense  --goal detailed --seed 18 --test --silent -f --output tw_games/tw-rewardsDense_goalBrief.z8
tw_envs = {
    "tw-simple": ["tw-make", "tw-simple", "--rewards", "balanced", "--goal", "brief", "--seed", "20250401", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-simple.z8"],
    "tw-simple-2": ["tw-make", "tw-simple", "--rewards", "balanced", "--goal", "brief", "--seed", "20250402", "--silent", "-f", "--output", f"{textworld_path}/games/default/tw-simple-2.z8"],
    "custom-game": ["tw-make", "custom", "--world-size", "5", "--nb-objects", "10", "--quest-length", "5", "--seed", "20250401", "--output", f"{textworld_path}/games/default/custom-game.z8"],
    "custom-game-2": ["tw-make", "custom", "--world-size", "5", "--nb-objects", "10", "--quest-length", "10", "--seed", "20250402", "--output", f"{textworld_path}/games/default/custom-game-2.z8"],
    "treasure_hunter": ["tw-make", "tw-treasure_hunter", "--level", "15", "--seed", "20250401", "--output", f"{textworld_path}/games/default/treasure_hunter.z8"],
    "coin_collector": ["tw-make", "tw-coin_collector", "--level", "100", "--seed", "20250401", "--output", f"{textworld_path}/games/default/coin_collector.z8"],
    "cooking": ["tw-make", "tw-cooking", "--recipe", "4", "--take", "4", "--go", "6", "--open", "--cut", "--cook", "--recipe-seed", "20250401", "--output", f"{textworld_path}/games/default/cooking.z8"],
    "cooking-2": ["tw-make", "tw-cooking", "--recipe", "3", "--take", "3", "--go", "6", "--cook", "--recipe-seed", "20250402", "--output", f"{textworld_path}/games/default/cooking-2.z8"],
}
for env_name, env_args in tw_envs.items():
    env_path = env_args[-1]
    if not os.path.exists(env_path):
        subprocess.run(env_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from quest_interface import Quest_Graph, Action
from quest_interface.mdp_state import MDP_State, MDP_Transition
from implementations.rl_agent import agent_functions, rl_graph
from implementations.rl_agent.persona import Persona


def flatten_batch(infos):
    return {k: v[0] for k, v in infos.items()}


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


class Textworld_Main_Goal(MDP_Transition):
    def __init__(self, objective, max_score):
        self.objective = objective
        self.max_score = max_score
        self.is_main = True


    def __str__(self):
        return f"{self.objective}"
    

    def __len__(self):
        return self.max_score

    
    def __lt__(self, other):
        return False


    def applicable_from(self, state):
        return False
    

    def eval(self, node, obs):
        n_action_node, _, n_quest_node, n_succeeded_node = node.count_context_type()
        mdp_score = obs.score - (n_action_node + n_quest_node) * 0.02 - (n_quest_node - n_succeeded_node) * 0.4
        done = obs.done
        infos = obs.info
        override_objective = None
        if done:
            if infos["won"]:
                terminated = True
                truncated = False
                succeeded = True
                if n_succeeded_node >= 2:
                    mdp_score = mdp_score + 20
                else:
                    mdp_score = mdp_score + 10
            elif infos["lost"]:
                terminated = True
                truncated = False
                succeeded = False
                mdp_score = mdp_score - 1
            else:
                terminated = False
                truncated = True
                succeeded = False
        else:
            terminated = False
            truncated = False
            succeeded = None
        
        return mdp_score, terminated, truncated, succeeded, override_objective


class Textworld_Transition(MDP_Transition):
    def __init__(self, new_location=None, added_items=set(), is_main=False, is_rush_goal=False):
        self.new_location = new_location
        self.added_items = set(added_items)
        self.is_main = is_main
        self.is_rush_goal = is_rush_goal

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


    @staticmethod
    def from_string(objective, is_main=False):
        if "Rush goal" in objective:
            is_rush_goal = True
        else:
            is_rush_goal = False
        
        # find Go to {location}( and Find {item1} , {item2} ...)*( and Use {item1} , {item2} ...)*
        go_to = None
        find_items = []
        parts = objective.split(" and ")
        for part in parts:
            if part.startswith("Go to "):
                go_to = part.replace("Go to ", "")
            elif part.startswith("Find "):
                find_items = [item.strip() for item in part.replace("Find ", "").split(",") if item.strip() != ""]

        return Textworld_Transition(go_to, set(find_items), is_main=is_main, is_rush_goal=is_rush_goal)


    def __str__(self):
        return f"{self.objective}"
    

    def __len__(self):
        return self.count_diff
    

    def __eq__(self, other):
        if self.new_location is not None:
            if not self.new_location == other.new_location:
                return False
        if len(self.added_items) > 0:
            if len(self.added_items - other.added_items) > 0:
                return False
        return True


    def score(self, other):
        score = 0
        if self.new_location is not None:
            if self.new_location == other.new_location:
                score += 1
    
        score += len(self.added_items.intersection(other.added_items))
        return score
    
    
    def __lt__(self, other):
        # test of stictly less than
        # location change < item change < main
        # rush goal < main
        if other.is_main:
            return True
        elif self.is_rush_goal or other.is_rush_goal:
            # rush goal allow no sub task, nor be a sub task of anyone except main
            return False

        if len(self.added_items.symmetric_difference(other.added_items)) == 0 and self.new_location == other.new_location:
            # everything is the same
            return False

        s_i = len(self.added_items - other.added_items)
        o_i = len(other.added_items - self.added_items)

        if s_i > 0:
            return False

        if o_i > 0:
            if other.new_location is not None and self.new_location != other.new_location:
                return False
        elif other.new_location is not None:
            return False

        return True
    

    def applicable_from(self, state):
        if self.is_rush_goal:
            return True
        diff = 0
        if self.new_location is not None and self.new_location != state.location:
            diff += 1
        diff += len(self.added_items - state.inventory)
        return diff == self.count_diff


    def eval(self, node, obs):
        done = obs.done
        infos = obs.info
        progress_transition = obs - node.start_observation
        score = self.score(progress_transition)
        n_action_node, _, n_quest_node, n_succeeded_node = node.count_context_type()
        mdp_score = score - (n_action_node + n_quest_node) * 0.02 - (n_quest_node - n_succeeded_node) * 0.4
        override_objective = None
        if self == progress_transition:
            terminated = True
            truncated = False
            succeeded = True if n_action_node + n_quest_node >= 2 else False # if sub task can be done in one step, discourage it
            if n_succeeded_node >= 2:
                mdp_score = mdp_score + 20
            else:
                mdp_score = mdp_score + 10
        elif done:
            if infos["won"]:
                terminated = True
                truncated = False
                succeeded = False
                mdp_score = mdp_score - 1
            elif infos["lost"]:
                terminated = True
                truncated = False
                succeeded = False
                mdp_score = mdp_score - 1
            else:
                terminated = False
                truncated = True
                succeeded = False
        elif not self.is_main and n_action_node + n_quest_node > 10:
            terminated = False
            truncated = True
            succeeded = False
        else:
            terminated = False
            truncated = False
            succeeded = None

        return mdp_score, terminated, truncated, succeeded, override_objective


class Textworld_State(MDP_State):
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


def compute_folds(objective_transition, state_scores):
    # states is a list of obs, score, info, last_context_mark
    # return list of end value, diff_str, comparable_transition, from_context_mark, to_context_mark
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
    # pairs = combinations(reversed(pivots), 2)
    pairs = [(pivots[i], pivots[i-1] + 1) for i in range(1, len(pivots))]

    # check fit gap size, and also whether all the changes are the same
    selected_transitions = [(transition_matrix[i - 1][j], j, i) for i, j in pairs if i - j >= 3 and i - j <= 10]
    # return fixed end state value of 100 for first training
    return [(st, j, i) for st, j, i in selected_transitions if st.count_diff == 1 and st < objective_transition]


def play(env, available_objectives, persona, rollout_file_path, nb_episodes=10, verbose=False, verbose_step=10, verbose_prefix=""):
    
    with open(rollout_file_path, "a", encoding="utf-8") as f:
        # mark date
        f.write(f"========================================================================\n")
        f.write(f"Date: {utilities.get_current_time_string()}\n")
        f.write(f"Allow relegation: {persona.allow_relegation}\n")
        f.write(f"Relegation probability: {persona.training_relegation_probability}\n")
        f.write(f"Allow sub training: {persona.allow_sub_training}\n")
        f.write(f"Allow prospect training: {persona.allow_prospect_training}\n")
        f.write(f"------------------------------------------------------------------------\n")

    # Collect some statistics: nb_steps, final reward.
    stat_n_moves = []
    stat_scores = []
    stat_mean_context_length = []
    stat_max_context_length = []
    stat_max_score = []
    stat_count_succeeded = []

    for no_episode in range(1, nb_episodes + 1):
        obs, infos = env.reset()  # Start new episode.
        infos = flatten_batch(infos)
        obs = infos["description"]
        obs = re.sub(r"\n+", "\n", obs)
        score = 0
        done = False

        objective_transition = available_objectives[no_episode % len(available_objectives)]   
        root_node = rl_graph.Quest_Node(
            objective = objective_transition,
            start_observation = Textworld_State(obs, score, done, infos),
            allow_relegation=persona.compute_allow_relegation()
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

        if root_node.observation is None:
            # error skip
            continue
        score, _, _, succeeded, _ = root_node.eval(root_node.observation)

        num_children, num_quest_node, max_context, min_context = root_node.compute_statistics()
        stat_n_moves.append(num_children)
        stat_scores.append(score)
        stat_mean_context_length.append(num_children / num_quest_node if num_quest_node > 0 else 0)
        stat_max_context_length.append(max_context)
        stat_max_score.append(len(objective_transition))
        stat_count_succeeded.append(1.0 if succeeded else 0.0)

        if verbose and no_episode % verbose_step == 0:
            # cl means context length
            msg = "{} episode: {}/{}; steps: {:5.1f}; succeeded: {:4.1f}; score: {:4.1f}/{:4.1f}; cl: {:4.1f}; max cl: {:4.1f}"
            report = msg.format(
                verbose_prefix,
                no_episode,
                nb_episodes,
                np.mean(stat_n_moves[-verbose_step:]),
                np.mean(stat_count_succeeded[-verbose_step:]),
                np.mean(stat_scores[-verbose_step:]), np.mean(stat_max_score[-verbose_step:]),
                np.mean(stat_mean_context_length[-verbose_step:]),
                np.mean(stat_max_context_length[-verbose_step:]))
            logging.info(report)
            with open(rollout_file_path, "a") as f:
                data = persona.print_context(root_node)
                f.write("Episode " + str(no_episode) + "\n")
                f.write("[Report]\t" + str(report) + "\n")
                f.write(data)
                f.write("\n")
                f.write("------------------------------------------------------------------------\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", "-r", action="store_true")
    parser.add_argument("--record_file",            "-o",   type=str, default="rollouts.txt",   help="The file to record the rollouts. Default is 'rollouts.txt'.")
    parser.add_argument("--run_count",              "-rc",  type=int, default=1,                help="The number of runs to perform. Default is 1.")
    parser.add_argument("--no_relegation",          "-nre", action="store_true",                help="Disable relegation during training.")
    parser.add_argument("--rel_prob",               "-rp",  type=float, default=1.0,            help="The probability of relegation during training. Default is 1.0.")
    parser.add_argument("--no-sub-training",        "-nst", action="store_true",                help="Disable sub training during training.")
    parser.add_argument("--no-prospect-training",   "-npt", action="store_true",                help="Disable prospect training during training.")
    args = parser.parse_args()

    experiment_path = f"{APP_ROOT}/experiments/rl_textworld"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    rollout_file_path = os.path.join(experiment_path, "rollouts.txt")
    if args.record_file:
        rollout_file_path = os.path.join(experiment_path, args.record_file)

    agent_parameter_path = os.path.join(experiment_path, "parameters")
    os.makedirs(agent_parameter_path, exist_ok=True)

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

    game_path = tw_envs["tw-simple-2"][-1]
    env_id = textworld.gym.register_game(game_path, request_infos, max_episode_steps=100, batch_size=1)
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()
    infos = flatten_batch(infos)

    available_objectives = [
        Textworld_Transition.from_string("Find a carrot", is_main=True),
        Textworld_Transition.from_string("Find a soap bar", is_main=True),
        Textworld_Transition.from_string("Find a note", is_main=True),
        Textworld_Transition.from_string("Find a bell pepper", is_main=True),
        Textworld_Transition.from_string("Find a toothbrush", is_main=True),
        Textworld_Transition.from_string("Find a shovel", is_main=True),
        Textworld_Transition.from_string("Find an apple", is_main=True),
        Textworld_Main_Goal(infos["objective"], infos["max_score"])
    ]

    MAX_VOCAB_SIZE = 1000
    tokenizer = Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

    def env_step(action):
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        # use regex to suppress multiple \n\n to one
        obs = re.sub(r"\n+", "\n", obs)
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)
        return Textworld_State(obs, score, done, infos)

    # from implementations.rl_algorithms.hierarchy_q import Hierarchy_Q as Model, Network_Scale_Preset
    # rl_core = Model(input_size=MAX_VOCAB_SIZE, network_preset=Network_Scale_Preset.medium, device=device, discount_factor=0.97, learning_rate=0.00001, train_temperature=0.05)

    from implementations.rl_algorithms.hierarchy_ac import Hierarchy_AC as Model, Network_Scale_Preset
    rl_core = Model(input_size=MAX_VOCAB_SIZE, network_preset=Network_Scale_Preset.medium, device=device, discount_factor=0.97, learning_rate=0.000002, entropy_weight=0.1, train_temperature=1.0)

    persona = Persona(
        rl_core,
        tokenizer,
        compute_folds,
        env_step,
        training_relegation_probability=args.rel_prob,
    )

    persona.set_allow_relegation(not args.no_relegation)
    persona.set_allow_sub_training(not args.no_sub_training)
    persona.set_allow_prospect_training(not args.no_prospect_training)

    # if not persona.load(agent_parameter_path):
    logging.info(f"Initiate agent training with following parameters:")
    logging.info(f"  - Allow relegation: {not args.no_relegation}")
    logging.info(f"  - Relegation probability: {args.rel_prob}")
    logging.info(f"  - Allow sub training: {not args.no_sub_training}")
    logging.info(f"  - Allow prospect training: {not args.no_prospect_training}")
    
    for i in range(int(args.run_count)):
        rl_core.reset()

        persona.set_training_mode(True)
        play(env, available_objectives, persona, 
            rollout_file_path=rollout_file_path, 
            nb_episodes=10000, verbose=True, verbose_step=100, verbose_prefix=f"[Run {i+1}/{args.run_count}]")
        # persona.save(agent_parameter_path)

        persona.set_training_mode(False)
        play(env, available_objectives, persona, 
            rollout_file_path=rollout_file_path, 
            nb_episodes=100, verbose=True, verbose_step=20)
        
    env.close()
