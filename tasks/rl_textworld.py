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


def parse_transition(objective):
    if "Welcome" in objective:
        return None
    # find Go to {location}( and Find {item1} , {item2} ...)*( and Use {item1} , {item2} ...)*
    go_to = None
    find_items = []
    parts = objective.split(" and ")
    for part in parts:
        if part.startswith("Go to "):
            go_to = part.replace("Go to ", "")
        elif part.startswith("Find "):
            find_items = part.replace("Find ", "").split(" , ")

    return Textworld_Transition(0, -1, -1, go_to, set(find_items))


class Textworld_Transition(mdp_state.MDP_Transition):
    def __init__(self, delta_score, from_context_mark, to_context_mark, new_location=None, added_items=set()):
        self.delta_score = round(delta_score)
        self.from_context_mark = from_context_mark
        self.to_context_mark = to_context_mark
        self.new_location = new_location
        self.added_items = added_items

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
        if self.new_location != other.new_location:
            diff += 1
        diff += len(self.added_items.symmetric_difference(other.added_items))
        return diff
    

    def __lt__(self, other):
        # test of stictly less than
        # item change < location change < None
        if other is None:
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

        root_node = rl_graph.Quest_Node(
            objective = infos["objective"],
            eval_func = env_eval,
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

    env_id = textworld.gym.register_game(game_path, request_infos, max_episode_steps=200, batch_size=1)
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

    def env_eval(node, env_score, infos):
        if infos["won"]:
            fulfilled = True
            success = True
            next_value = 100
        elif infos["lost"]:
            fulfilled = True
            success = False
            next_value = -100
        else:
            fulfilled = False
            success = False
            next_value = 0
        # mdp_score, fulfilled, success, finish_value
        # mdp_score is the main env score
        # fulfill is for sub task, success 
        return env_score, fulfilled, success, next_value

    def goal_pursuit_eval(node, env_score, infos):
        num_children = len(node.get_children())
        objective = node.objective
        parent = node.get_parent()
        parent_transition = parse_transition(parent.objective) if parent is not None else None
        target_transition = parse_transition(objective)

        if parent_transition is not None and not target_transition < parent_transition:
            return 0, True, False, -100

        _, _, _, start_info = node.start_observation
        current_state = Textworld_State(0, infos, 0)
        start_state = Textworld_State(0, start_info, num_children)
        progress_transition = current_state - start_state

        score_diff = target_transition - progress_transition
        mdp_score = len(target_transition) - score_diff - num_children * 0.02

        if num_children > 25:
            # too many children, stop the task
            fulfilled = True
            success = False
            next_value = 0
        elif score_diff == 0:
            fulfilled = True
            success = True
            next_value = 50
        else:
            fulfilled = False
            success = False
            next_value = 0

        # mdp_score, fulfilled, success, finish_value
        return mdp_score, fulfilled, success, next_value

    
    def compute_folds(objective, states):
        # states is a list of obs, score, info, last_context_mark
        # return list of end value, diff_str, from_context_mark, to_context_mark
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
        # gap greater than 4 steps
        selected_transitions = [(transition_matrix[i - 1][j], j, i) for i, j in pairs if i - j >= 4]
        # return fixed end state value of 100 for first training
        return [(100, st.objective, j, i) for st, j, i in selected_transitions if st.count_diff >= 1 and st.delta_score >= 1 and st < objective_transition]
    

    # from implementations.tw_agents.agent_neural import Random_Agent, Neural_Agent
    # agent = RandomAgent()
    # agent = Neural_Agent(input_size=MAX_VOCAB_SIZE, device=device)
    from implementations.tw_agents.hierarchy_agent import Hierarchy_Agent
    agent = Hierarchy_Agent(input_size=MAX_VOCAB_SIZE, device=device)

    persona = Persona(agent, tokenizer, compute_folds, env_step, goal_pursuit_eval=goal_pursuit_eval)

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