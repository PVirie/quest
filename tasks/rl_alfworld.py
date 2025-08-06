import os
import sys
import subprocess
import numpy as np
import argparse
import shutil
import re
import logging
import random

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

import utilities
from utilities.tokenizer import *

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
from quest_interface.mdp_state import MDP_State, MDP_Transition
from implementations.rl_agent import rl_graph
from implementations.rl_agent.rl_graph import Trainable, Observation_Node, Quest_Node, Thought_Node
from implementations.rl_agent.persona import Persona


def flatten_batch(infos):
    return {k: v[0] for k, v in infos.items()}


class Alfworld_Main_Goal(MDP_Transition):
    def __init__(self, obs):
        self.objective = obs
        self.is_main = True


    def __str__(self):
        return f"{self.objective}"
    

    def __len__(self):
        return 1

    
    def __lt__(self, other):
        return False


    def applicable_from(self, state):
        return False
    

    def eval(self, node, obs):
        n_action_node, _, n_quest_node, n_succeeded_node = node.count_context_type()
        mdp_score = 0 - (n_action_node + n_quest_node) * 0.1 - (n_quest_node - n_succeeded_node) * 1.0
        done = obs.done
        infos = obs.info
        override_objective = None
        if done:
            if infos["won"]:
                terminated = True
                truncated = False
                succeeded = True
                mdp_score = mdp_score + 50 + 25 * n_succeeded_node
            else:
                terminated = False
                truncated = True
                succeeded = False
        else:
            terminated = False
            truncated = False
            succeeded = None
        
        return mdp_score, terminated, truncated, succeeded, override_objective


class Alfworld_Transition(MDP_Transition):
    def __init__(self, new_location=None, new_item=None):
        self.new_location = new_location
        self.new_item = new_item
        self.is_main = False

        if new_location is not None:
            self.objective = f"Go to {new_location}"
        elif new_item is not None:
            self.objective = f"Find {new_item}"
        else:
            self.objective = ""


    @staticmethod
    def from_string(objective):
        if objective.startswith("Go to "):
            return Alfworld_Transition(new_location=objective[7:])
        elif objective.startswith("Find "):
            return Alfworld_Transition(new_item=objective[5:])
        else:
            return Alfworld_Transition()


    def __str__(self):
        return f"{self.objective}"
    

    def __len__(self):
        return 1 if self.new_location is not None or self.new_item is not None else 0
    

    def __eq__(self, state):
        if self.new_location is not None:
            if not self.new_location == state.new_location:
                return False
        if self.new_item is not None:
            if not self.new_item == state.new_item:
                return False
        return True
    
    
    def __lt__(self, other):
        # test of stictly less than
        # location change < find item < main
        if other.is_main:
            return True
        
        if other.new_location is not None:
            return False

        if other.new_item is not None:
            return self.new_location is not None
        
        return False
    

    def applicable_from(self, state):
        return True


    def eval(self, node, obs):
        done = obs.done
        infos = obs.info
        n_action_node, _, n_quest_node, n_succeeded_node = node.count_context_type()
        mdp_score = 0 - (n_action_node + n_quest_node) * 0.1 - (n_quest_node - n_succeeded_node) * 1.0
        override_objective = None
        if self == obs:
            terminated = True
            truncated = False
            succeeded = True if n_action_node + n_quest_node >= 2 else False # if sub task can be done in one step, discourage it
            mdp_score = mdp_score + 50 + 25 * n_succeeded_node
        elif done:
            if infos["won"]:
                terminated = True
                truncated = False
                succeeded = False
                mdp_score = mdp_score - 5
            else:
                terminated = False
                truncated = True
                succeeded = False
        elif not self.is_main and n_action_node + n_quest_node > 20:
            terminated = False
            truncated = True
            succeeded = False
        else:
            terminated = False
            truncated = False
            succeeded = None

        return mdp_score, terminated, truncated, succeeded, override_objective


class Alfworld_State(MDP_State):
    def __init__(self, obs, score, done, info):
        self.obs = obs
        self.score = score
        self.done = done
        self.info = info

        # for new_location: detect "You arrive at <location>." in obs
        if "You arrive at " in obs:
            location = obs.split("You arrive at ")[1].split(".")[0]
            self.new_location = location.strip()
        else:
            self.new_location = None

        # for new_item: detect "You pick up the {obj id} from the..." in obs
        if "You pick up the " in obs:
            item = obs.split("You pick up the ")[1].split(" from the")[0]
            self.new_item = item.strip()
        else:
            self.new_item = None


    def get_available_actions(self):
        return self.info["admissible_commands"]
    

    def get_context(self):
        return self.obs
    

    def __len__(self):
        size = 0
        if self.new_location is not None:
            size += 1
        if self.new_item is not None:
            size += 1
        return size


def compute_folds(objective_transition, state_tuples):
    # states is a list of index, train_ref(score), MDP_State, sub_objective
    # return list of end value, diff_str, comparable_transition, from_context_mark, to_context_mark
    states = [state for _, train_ref, state, sub_objective  in state_tuples]
    pivots = []
    for i in range(len(states)):
        if len(states[i]) > 0:
            pivots.append(i)
    pairs = [(pivots[i-1] + 1, pivots[i]) for i in range(1, len(pivots))]

    selected_transitions = []
    for i, j in pairs:
        if not (j - i >= 2 and j - i <= 10):
            continue
        st = Alfworld_Transition(states[j].new_location, states[j].new_item)
        if not st < objective_transition:
            continue
        if i > 0 and not st.applicable_from(states[i - 1]):
            continue
        valid_sub_state = True
        for k in range(i, j):
            # check whether the sub_objective in between is valid
            sub_objective = state_tuples[k][3]
            if sub_objective is not None and not (sub_objective < st):
                valid_sub_state = False
                break
        if not valid_sub_state:
            continue
        selected_transitions.append((st, i, j))
    
    # if all conditions are met, return the transition
    return selected_transitions


def play(env, env_step, persona, rollout_file_path, epoch=10, verbose=False, verbose_step=10, verbose_prefix=""):

    num_environments = len(env.gamefiles)
    logging.info(f"Found {num_environments} environments")

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

    no_episode = 1
    for no_epoch in range(epoch):
        for no_env in range(num_environments):
            obs, infos = env.reset()  # Start new episode.
            obs = obs[0]
            infos = flatten_batch(infos)
            obs = re.sub(r"\n+", "\n", obs)
            obs_parts = obs.split("\n")
            obs = obs_parts[1]
            objective = obs_parts[2]
            score = 0
            done = False

            objective_transition = Alfworld_Main_Goal(objective)
            root_node = rl_graph.Quest_Node(
                objective = objective_transition,
                start_observation = Alfworld_State(obs, score, done, infos),
                allow_relegation=persona.compute_allow_relegation()
            )

            working_memory = Quest_Graph(root_node)
            step = 0
            while True:
                focus_node = working_memory.get_focus_node()
                last_child = focus_node.last_child()

                ################# Evaluate current situation #################

                if isinstance(last_child, Trainable):
                    last_observation = last_child.observation
                    mdp_score, terminated, truncated, succeeded, new_objective = focus_node.eval(last_observation)
                    last_child.train_ref.mdp_score = mdp_score
                else:
                    children = focus_node.get_children()
                    # search for trainable last child
                    found = False
                    for child in reversed(children):
                        if isinstance(child, Trainable):
                            last_child = child
                            found = True
                            break
                    if not found:
                        last_observation = focus_node.start_observation
                    else:
                        last_observation = last_child.observation
                    mdp_score, terminated, truncated, succeeded, new_objective = focus_node.eval(last_observation)
                    if found:
                        last_child.train_ref.mdp_score = mdp_score

                if persona.training_mode:
                    step += 1
                    if terminated or truncated or step % persona.TRAIN_STEPS == 0:
                        persona.train(focus_node, train_last_node=terminated)

                if terminated or truncated:
                    if terminated:
                        return_node = Quest_Node(succeeded=succeeded, observation=last_observation)
                    elif truncated:
                        return_node = Quest_Node(truncated=True, succeeded=succeeded, observation=last_observation)
                    parent = focus_node.get_parent()
                    working_memory.respond(return_node, parent)
                    step = 0
                    if parent is None:
                        break
                    continue

                subact, new_node = persona.think(focus_node)
                if isinstance(new_node, Quest_Node):
                    working_memory.discover(new_node, focus_node)
                    working_memory.respond(None, new_node)
                    step = 0
                elif isinstance(new_node, Observation_Node):
                    new_node.observation = env_step(new_node.action)
                    working_memory.discover(new_node, focus_node)
                else:
                    working_memory.discover(new_node, focus_node)

                if len(working_memory) > 400:
                    break


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
                    num_environments * epoch,
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
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--record-file",            "-o",   type=str, default="rollouts.txt",   help="The file to record the rollouts. Default is 'rollouts.txt'.")
    parser.add_argument("--q-learning",             "-q",   action="store_true",                help="Use Q-learning instead of Actor-Critic. Default is False.")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--no-relegation",          "-nre", action="store_true",                help="Disable relegation during training.")
    parser.add_argument("--rel-prob",               "-rp",  type=float, default=1.0,            help="The probability of relegation during training. Default is 1.0.")
    parser.add_argument("--no-sub-training",        "-nst", action="store_true",                help="Disable sub training during training.")
    parser.add_argument("--no-prospect-training",   "-npt", action="store_true",                help="Disable prospect training during training.")
    args = parser.parse_args()

    experiment_path = f"{APP_ROOT}/experiments/rl_alfworld"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    rollout_file_path = os.path.join(experiment_path, f"rollouts.txt")
    if args.record_file:
        rollout_file_path = os.path.join(experiment_path, args.record_file)

    agent_parameter_path = os.path.join(experiment_path, "parameters")
    os.makedirs(agent_parameter_path, exist_ok=True)

    # For reproducibility (https://docs.pytorch.org/docs/stable/notes/randomness.html)
    random.seed(20250701)  
    torch.manual_seed(20250701)
    np.random.seed(20250701)
    torch.use_deterministic_algorithms(True)

    # load config
    with open(f"{alfworld_path}/alfworld/configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    env_type = "AlfredTWEnv"

    MAX_VOCAB_SIZE = 1000
    tokenizer = Text_Tokenizer(MAX_VOCAB_SIZE, device=device)

    if args.q_learning:
        from implementations.rl_algorithms.hierarchy_q import Hierarchy_Q as Model, Network_Scale_Preset
        rl_core = Model(input_size=MAX_VOCAB_SIZE, network_preset=Network_Scale_Preset(args.scale), device=device, discount_factor=0.95, learning_rate=0.00001, epsilon_greedy=1.0, train_temperature=0.05)
    else:
        from implementations.rl_algorithms.hierarchy_ac import Hierarchy_AC as Model, Network_Scale_Preset
        rl_core = Model(input_size=MAX_VOCAB_SIZE, network_preset=Network_Scale_Preset(args.scale), device=device, discount_factor=0.95, learning_rate=0.00001, entropy_weight=1.0, train_temperature=1.0)

    rl_core.reset()
    persona = Persona(
        rl_core,
        tokenizer,
        compute_folds,
        training_relegation_probability=args.rel_prob,
    )

    logging.info(f"Selected environment: ALFWorld")
    logging.info(f"Initiate agent training with following parameters:")
    logging.info(f"  - Algorithm: {'Q-learning' if args.q_learning else 'Actor-Critic'}")
    logging.info(f"  - Network scale: {str(Network_Scale_Preset(args.scale).value)}")
    logging.info(f"  - Allow relegation: {not args.no_relegation}")
    logging.info(f"  - Relegation probability: {args.rel_prob}")
    logging.info(f"  - Allow sub training: {not args.no_sub_training}")
    logging.info(f"  - Allow prospect training: {not args.no_prospect_training}")

    persona.set_allow_relegation(not args.no_relegation)
    persona.set_allow_sub_training(not args.no_sub_training)
    persona.set_allow_prospect_training(not args.no_prospect_training)

    # setup train
    persona.set_training_mode(True)
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    def env_step(action):
        obs, score, done, infos = env.step([action])
        obs = obs[0]
        # use regex to suppress multiple \n\n to one
        obs = re.sub(r"\n+", "\n", obs)
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)
        return Alfworld_State(obs, score, done, infos)

    play(env, env_step, persona, rollout_file_path=rollout_file_path, epoch=1, verbose=True, verbose_step=100, verbose_prefix=f"")

    # # interact
    # obs, info = env.reset()
    # # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    # admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    # random_actions = [np.random.choice(admissible_commands[0])]

    # # step
    # obs, scores, dones, infos = env.step(random_actions)
    # logging.info("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
    
    # setup_test
    persona.set_training_mode(False)
    test_env = get_environment(env_type)(config, train_eval='valid_unseen')
    test_env = env.init_env(batch_size=1)

    def test_env_step(action):
        obs, score, done, infos = test_env.step([action])
        obs = obs[0]
        # use regex to suppress multiple \n\n to one
        obs = re.sub(r"\n+", "\n", obs)
        score = score[0]
        done = done[0]
        infos = flatten_batch(infos)
        return Alfworld_State(obs, score, done, infos)

    play(env, test_env_step, persona, rollout_file_path=rollout_file_path, epoch=1, verbose=True, verbose_step=100, verbose_prefix=f"")

