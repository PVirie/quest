from enum import Enum
from utilities.language_models import Language_Model
from .rl_graph import Trainable, Quest_Node, Observation_Node, Thought_Node
import random
import os

"""
Non-Batched version of the Persona class.
"""

class Sub_Action_Type(Enum):
    Fulfill = 1
    Relegate = 2
    Thought = 3
    Act = 4
    Done = 5


def extract_command_and_detail(text):
    parts = text.split(":")
    command = parts[0].strip()
    detail = ":".join(parts[1:]).strip().replace("#", "")
    # remove space and '#'
    return command, detail


class Persona:
    TRAIN_STEP=10
    PRINT_STEP=1000

    def __init__(self, rl_core, tokenizer, compute_folds, env_step, goal_pursuit_eval, action_parser, training_relegation_probability=0.25, train_prompt=None):
        self.rl_core = rl_core
        self.tokenizer = tokenizer
        self.compute_folds = compute_folds
        self.env_step = env_step
        self.goal_pursuit_eval = goal_pursuit_eval
        self.action_parser = action_parser
        self.training_relegation_probability = training_relegation_probability
        self.action_set = set()
        self.extra_actions = {}

        self.training_mode = False
        self.allow_relegation = True

        self.use_lm = False
        self.long_lm = None
        self.prompt = None
        if train_prompt is not None:
            self.use_lm = True
            self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
            self.prompt = train_prompt

        self.step = 0


    def set_training_mode(self, flag):
        self.training_mode = flag


    def set_allow_relegation(self, flag):
        self.allow_relegation = flag


    def compute_allow_relegation(self):
        return self.allow_relegation and (random.random() < self.training_relegation_probability or not self.training_mode)


    def save(self, path):
        # save the rl_core and the extra actions
        with open(os.path.join(path, "extra_actions.txt"), "w", encoding="utf-8") as f:
            for action in self.extra_actions.keys():
                f.write(action[10:] + "\n")

        tokenizer_path = os.path.join(path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save(tokenizer_path)

        rl_core_path = os.path.join(path, "rl_core")
        os.makedirs(rl_core_path, exist_ok=True)
        self.rl_core.save(rl_core_path)


    def load(self, path):
        extra_actions_path = os.path.join(path, "extra_actions.txt")
        if not os.path.exists(extra_actions_path):
            return False
        with open(extra_actions_path, "r", encoding="utf-8") as f:
            action_keys = set([line.strip() for line in f.readlines()])
            self.extra_actions = {f"Sub Task: {action}": self.action_parser(action) for action in action_keys}
        # load the rl_core and the extra actions
        succeeded = self.tokenizer.load(os.path.join(path, "tokenizer"))
        if not succeeded:
            return False
        succeeded = self.rl_core.load(os.path.join(path, "rl_core"))
        if not succeeded:
            return False
        return succeeded


    def print_context(self, quest_node, prefix=""):
        children = quest_node.get_children()
        objective_context, start_obs_context = quest_node.get_start_contexts()
        contexts = [
            f"{prefix}Relegation: {'enabled' if quest_node.allow_relegation else 'disabled'}",
            f"{prefix}{objective_context}", 
            f"{prefix}{start_obs_context.replace("\n", f"\n{prefix}")}"
        ]
        last_score = 0
        for i, node in enumerate(children):
            node_contexts = node.get_context()
            if isinstance(node, Quest_Node):
                sub_task_context = self.print_context(node, prefix=prefix + "\t")
                node_contexts.insert(1, sub_task_context)
                score = node.train_ref.mdp_score
                rank = node.train_ref.selected_action_rank
            elif isinstance(node, Observation_Node):
                score = node.train_ref.mdp_score
                rank = node.train_ref.selected_action_rank
            else:
                score = last_score
                rank = -1
            node_contexts.insert(0, f"{i} ----- reward: {(score - last_score):.2f} rank: {rank:d}")
            node_contexts = [f"{prefix}" + c.replace("\n", f"\n{prefix}") for c in node_contexts]
            contexts.extend(node_contexts)
            last_score = score
        if len(prefix) == 0:
            contexts.append(f"Extra Actions: {";".join(self.extra_actions.keys())}")
        return f"\n".join(contexts)


    def train(self, quest_node, train_last_node: bool = False):
        # finish_value only used when training the last node
        last_observation = quest_node.start_observation
        objective_context, start_obs_context = quest_node.get_start_contexts()
        objective_contexts = [objective_context]
        rl_contexts = [start_obs_context] 
        last_context_mark = 0
        pivots = []
        train_data = []
        selected_nodes = []
        last_score = 0
        supports = quest_node.get_children()
        i = 0
        while i < len(supports):
            node = supports[i]
            if isinstance(node, Trainable):
                rl_contexts.extend(node.get_context())
                last_observation = node.observation
            else:
                continue

            train_ref = node.train_ref
            score = train_ref.mdp_score
            selected_nodes.append((last_observation, score))
            pivots.append((score - last_score, last_context_mark, list(train_ref.available_actions)))
            if i >= len(supports) - self.TRAIN_STEP:
                if i < len(supports) - 1 or train_last_node:
                    train_data.append((train_ref.selected_action, len(pivots) - 1, len(pivots)))
                    train_ref.release()

            last_context_mark = len(rl_contexts) - 1
            last_score = score
            i += 1

        folds = self.compute_folds(quest_node.objective, selected_nodes)
        for diff_str, obj, from_transition_index, to_transition_index in folds:
            fold_action = f"Sub Task: {diff_str}"
            self.extra_actions[fold_action] = obj
            pivots[from_transition_index][2].append(fold_action)
            # train_data.append((fold_action, from_transition_index, to_transition_index))

        # add extra actions
        all_action_list = list(self.action_set.union(self.extra_actions.keys()))

        if len(train_data) > 0:
            objective_tensor = self.tokenizer(objective_contexts, stack=True)
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.rl_core.train(train_last_node, pivots, train_data, objective_tensor, state_tensor, action_list_tensor, all_action_list)

            if not self.allow_relegation:
                return

            for diff_str, _, from_transition_index, to_transition_index in folds:
                sub_quest_node = Quest_Node(
                    objective=diff_str,
                    start_observation=supports[from_transition_index-1].observation if from_transition_index > 0 else quest_node.start_observation
                )
                sub_quest_node.parent = quest_node.parent
                sub_objective_context, sub_start_obs_context = sub_quest_node.get_start_contexts()
                sub_objective_tensor = self.tokenizer([sub_objective_context], stack=True)
                last_sub_score = 0
                sub_pivots = []
                sub_train_data = []
                start_context_mark = pivots[from_transition_index][1] # in my rl_contexts, the start context mark is the start observation
                end_context_mark = pivots[to_transition_index][1]
                for i in range(from_transition_index, to_transition_index + 1):
                    sub_quest_node.children.append(supports[i])
                    sub_mdp_score, _, _, _, _ = self.goal_pursuit_eval(sub_quest_node, supports[i].observation)
                    sub_pivots.append((sub_mdp_score - last_sub_score, pivots[i][1] - start_context_mark, pivots[i][2]))
                    sub_train_data.append((supports[i].train_ref.selected_action, len(sub_pivots) - 1, len(sub_pivots)))
                    last_sub_score = sub_mdp_score
                self.rl_core.train(True, sub_pivots, sub_train_data, sub_objective_tensor, state_tensor[start_context_mark:(end_context_mark + 1), :], action_list_tensor, all_action_list)


    def think(self, quest_node):
        # supports is a list of nodes
        supports = quest_node.get_children()
        last_observation = quest_node.start_observation
        objective_context, start_obs_context = quest_node.get_start_contexts()
        objective_contexts = [objective_context]
        contexts = [start_obs_context]
        should_eval = False
        for node in supports:
            if isinstance(node, Trainable):
                contexts.extend(node.get_context())
                last_observation = node.observation
                should_eval = True
            elif isinstance(node, Thought_Node):
                contexts.extend(node.get_context())
                should_eval = False
                continue
        
        ################# Evaluate current situation #################
        if should_eval:
            mdp_score, terminated, truncated, succeeded, new_objective = quest_node.eval(last_observation)
            if len(supports) > 0:
                # Because training has to update weight anyway, which violate the functional programming paradigm
                # I'll just update the last child's mdp_score
                last_node = supports[-1]
                last_node.train_ref.mdp_score = mdp_score
                if new_objective is not None:
                    # increase end goal discover speed
                    new_action = f"Sub Task: {new_objective}"
                    last_node.objective = new_objective
                    last_node.train_ref.selected_action = new_action
                    last_node.train_ref.selected_action_rank = -1
                    last_node.train_ref.available_actions.add(new_action)
                    self.extra_actions[new_action] = self.action_parser(new_objective)

            train_last_node = False
            if terminated:
                train_last_node = True
                return_sub_action = Sub_Action_Type.Fulfill
                return_node = Quest_Node(succeeded=succeeded, observation=last_observation)
            elif truncated:
                return_sub_action = Sub_Action_Type.Done
                return_node = Quest_Node(observation=last_observation, truncated=True)

            if self.training_mode:
                self.step += 1
                if terminated or truncated or self.step % self.TRAIN_STEP == 0:
                    self.train(quest_node, train_last_node=train_last_node)

                # if self.step % self.PRINT_STEP == 0:
                #     self.rl_core.print(self.step)

            if terminated or truncated:
                return return_sub_action, return_node

        ################# ACTION SELECTION #################
        selectible_action_set = set([f"Action: {ac}" for ac in last_observation.get_available_actions()])
        self.action_set.update(selectible_action_set)
        valid_extra_actions = set()
        current_objective = self.action_parser(quest_node.objective)
        for key, obj in self.extra_actions.items():
            if obj < current_objective and obj.applicable_from(last_observation):
                # must check less than and diff to prevent infinite loop
                valid_extra_actions.add(key)
        available_actions = selectible_action_set.union(valid_extra_actions)
        if quest_node.allow_relegation:
            selectible_action_set.update(valid_extra_actions)

        lm_response = ""
        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(action_list=",".join(selectible_action_set), contexts="\n".join(objective_contexts + contexts)))
            # get the first part before newline
            lm_response = text_response.split("\n")[0]
            lm_command, lm_detail = extract_command_and_detail(lm_response)
            if lm_command.startswith("Thought"):
                return Sub_Action_Type.Thought, Thought_Node(lm_detail)
            elif not lm_command.startswith("Action") and not lm_command.startswith("Sub Task"):
                # if the response is not an action, sub task or final respond, ignore it
                lm_response = ""
            else:
                available_actions.add(lm_response)

        rl_response = ""
        # remove thoughts from the context for RL
        rl_contexts = [c for c in contexts if not c.startswith("Thought")]
        objective_tensor = self.tokenizer(objective_contexts, stack=True)
        state_tensor = self.tokenizer(rl_contexts, stack=True)
        action_list_tensor = self.tokenizer(selectible_action_set, stack=True)
        train_ref = self.rl_core.act(objective_tensor, state_tensor, action_list_tensor, list(selectible_action_set), sample_action=True)
        train_ref.available_actions = available_actions
        rl_response = train_ref.selected_action

        if not self.training_mode and train_ref is not None:
            train_ref.release()

        if len(lm_response) > 0 and random.random() < 0.1:
            # override RL response with the index of LM response
            train_ref.selected_action = lm_response
            response = lm_response
        else:
            response = rl_response

        ################# PERFORM ACTION #################

        command, detail = extract_command_and_detail(response)

        if command.startswith("Sub Task"):
            sub_objective = detail
            return_sub_action = Sub_Action_Type.Relegate
            return_node = Quest_Node(
                objective = sub_objective,
                eval_func = self.goal_pursuit_eval,
                start_observation = last_observation,
                train_ref = train_ref,
                allow_relegation = self.compute_allow_relegation()
            )
            return return_sub_action, return_node
        elif command.startswith("Action"):
            action = detail
            last_observation = self.env_step(action)
            return_sub_action = Sub_Action_Type.Act
            return_node = Observation_Node(
                action = action, 
                observation = last_observation, 
                train_ref = train_ref
            )
            return return_sub_action, return_node


