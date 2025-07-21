from enum import Enum
from utilities.language_models import Language_Model
from .rl_graph import Trainable, Quest_Node, Observation_Node, Thought_Node
import random
import os
import pickle

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
    TRAIN_STEPS=20

    def __init__(self, rl_core, tokenizer, compute_folds, training_relegation_probability=0.25, train_prompt=None):
        self.rl_core = rl_core
        self.tokenizer = tokenizer
        self.compute_folds = compute_folds
        self.training_relegation_probability = training_relegation_probability
        self.action_set = set()
        self.extra_actions = {}

        self.training_mode = False
        self.allow_relegation = True
        self.allow_sub_training = True
        self.allow_prospect_training = True

        self.use_lm = False
        self.long_lm = None
        self.prompt = None
        if train_prompt is not None:
            self.use_lm = True
            self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
            self.prompt = train_prompt


    def set_training_mode(self, flag):
        self.training_mode = flag


    def set_allow_relegation(self, flag):
        self.allow_relegation = flag


    def compute_allow_relegation(self):
        return self.allow_relegation and (random.random() < self.training_relegation_probability or not self.training_mode)


    def set_allow_sub_training(self, flag):
        self.allow_sub_training = flag


    def set_allow_prospect_training(self, flag):
        self.allow_prospect_training = flag


    def save(self, path):
        # save the rl_core and the extra actions
        with open(os.path.join(path, "extra_actions.pickle"), "wb") as f:
            pickle.dump(list(self.extra_actions.values()), f)

        tokenizer_path = os.path.join(path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save(tokenizer_path)

        rl_core_path = os.path.join(path, "rl_core")
        os.makedirs(rl_core_path, exist_ok=True)
        self.rl_core.save(rl_core_path)


    def load(self, path):
        extra_actions_path = os.path.join(path, "extra_actions.pickle")
        if not os.path.exists(extra_actions_path):
            return False
        with open(extra_actions_path, "rb") as f:
            actions = pickle.load(f)
            self.extra_actions = {f"Sub Task: {str(action)}": action for action in actions}
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
        str_levelled_prefix = "\n" + prefix
        formatted_start_obs_context = start_obs_context.replace("\n", str_levelled_prefix)
        contexts = [
            f"{prefix}Relegation: {'enabled' if quest_node.allow_relegation else 'disabled'}",
            f"{prefix}{objective_context}", 
            f"{prefix}{formatted_start_obs_context}"
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
            node_contexts = [f"{prefix}" + c.replace("\n", str_levelled_prefix) for c in node_contexts]
            contexts.extend(node_contexts)
            last_score = score
        if len(prefix) == 0:
            contexts.append(f"Extra Actions: {';'.join(self.extra_actions.keys())}")
        return f"\n".join(contexts)


    def train(self, quest_node, train_last_node: bool = False):
        # finish_value only used when training the last node
        objective_context, start_obs_context = quest_node.get_start_contexts()
        objective_contexts = [objective_context]
        rl_contexts = [start_obs_context]
        pivots = []
        train_data = []
        selected_nodes = []
        supports = quest_node.get_children()
        last_observation = quest_node.start_observation
        last_context_mark = 0
        last_score = 0
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
            pivots.append((last_context_mark, train_ref.available_actions))
            selected_nodes.append((i, train_ref, last_observation, node.objective if isinstance(node, Quest_Node) else None))
            if i >= len(supports) - self.TRAIN_STEPS:
                if i < len(supports) - 1 or train_last_node:
                    train_data.append((score - last_score, train_ref.selected_action, len(pivots) - 1, len(pivots)))
                    train_ref.release()

            last_context_mark = len(rl_contexts) - 1
            last_score = score
            i += 1

        folds = self.compute_folds(quest_node.objective, selected_nodes)
        for sub_objective, from_transition_index, to_transition_index in folds:
            fold_action = f"Sub Task: {str(sub_objective)}"
            self.extra_actions[fold_action] = sub_objective
            pivots[from_transition_index][1].add(fold_action)

        # add extra actions
        all_action_list = list(self.action_set.union(self.extra_actions.keys()))

        if len(train_data) > 0:
            objective_tensor = self.tokenizer(objective_contexts, stack=True)
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.rl_core.train(train_last_node, pivots, train_data, objective_tensor, state_tensor, action_list_tensor, all_action_list)

            if self.allow_prospect_training and len(folds) > 0:
                i = max(len(selected_nodes) - self.TRAIN_STEPS, 0)
                prospect_node = quest_node[i]
                if i == 0:
                    _, start_obs_context = prospect_node.get_start_contexts()
                    prospect_rl_contexts = [start_obs_context]
                    prospect_pivots = []

                    last_prospect_observation = prospect_node.start_observation
                    last_prospect_score = 0
                else:
                    _, train_ref, observation, _ = selected_nodes[i - 1]
                    prospect_rl_contexts = rl_contexts[:pivots[i][0] + 1].copy()
                    prospect_pivots = pivots[:i - 1].copy()

                    last_prospect_observation = observation
                    last_prospect_score = train_ref.mdp_score

                prospect_train_data = []
                last_prospect_context_mark = len(prospect_rl_contexts) - 1
                
                def include_node(pivot_index, sub_prospect_node, action, observation):
                    prospect_node.children.append(sub_prospect_node)
                    prospect_rl_contexts.extend(sub_prospect_node.get_context())
                    prospect_mdp_score, _, _, _, _ = prospect_node.eval(observation)
                    prospect_pivots.append((last_prospect_context_mark, pivots[pivot_index][1]))
                    prospect_train_data.append((prospect_mdp_score - last_prospect_score, action, len(prospect_pivots) - 1, len(prospect_pivots)))
                    return prospect_mdp_score, len(prospect_rl_contexts) - 1

                # sort folds by the second element (from_transition_index)
                sorted_folds = sorted(folds, key=lambda x: x[1])
                for sub_objective, from_transition_index, to_transition_index in sorted_folds:
                    if from_transition_index < i:
                        continue

                    while i < from_transition_index:
                        node_index, train_ref, observation, _ = selected_nodes[i]
                        sub_prospect_node = supports[node_index]
                        action = train_ref.selected_action
                        last_prospect_score, last_prospect_context_mark = include_node(i, sub_prospect_node, action, observation)
                        last_prospect_observation = observation
                        i += 1
                    
                    _, _, observation, _ = selected_nodes[to_transition_index]
                    sub_prospect_node = Quest_Node(
                                objective=sub_objective,
                                start_observation=last_prospect_observation,
                                allow_relegation=False,
                                succeeded=True,
                                truncated=False,
                                train_ref=None,
                                observation=observation
                            )
                    action = f"Sub Task: {str(sub_objective)}"
                    last_prospect_score, last_prospect_context_mark = include_node(i, sub_prospect_node, action, observation)
                    last_prospect_observation = observation
                    i = to_transition_index + 1

                while i < (len(selected_nodes) if train_last_node else len(selected_nodes) - 1):
                    node_index, train_ref, observation, _ = selected_nodes[i]
                    sub_prospect_node = supports[node_index]
                    action = train_ref.selected_action
                    last_prospect_score, last_prospect_context_mark = include_node(i, sub_prospect_node, action, observation)
                    last_prospect_observation = observation
                    i += 1

                prospect_state_tensor = self.tokenizer(prospect_rl_contexts, stack=True)
                self.rl_core.train(train_last_node, prospect_pivots, prospect_train_data, objective_tensor, prospect_state_tensor, action_list_tensor, all_action_list)


            if self.allow_sub_training:
                for sub_objective, from_transition_index, to_transition_index in folds:
                    # start_pivot_index = len(supports) - self.TRAIN_STEPS
                    # if from_transition_index <= start_pivot_index:
                    #     continue

                    if from_transition_index > 0:
                        _, _, observation, _ = selected_nodes[from_transition_index]
                    else:
                        observation = quest_node.start_observation
                        
                    sub_quest_node = Quest_Node(
                        objective=sub_objective,
                        start_observation=observation
                    )
                    sub_quest_node.parent = quest_node.parent
                    sub_objective_context, _ = sub_quest_node.get_start_contexts()
                    sub_objective_tensor = self.tokenizer([sub_objective_context], stack=True)
                    last_sub_score = 0
                    sub_pivots = []
                    sub_train_data = []
                    start_context_mark = pivots[from_transition_index][0]
                    end_context_mark = pivots[to_transition_index][0]
                    for i in range(from_transition_index, to_transition_index + 1):
                        node_index, train_ref, observation, _ = selected_nodes[i]
                        sub_quest_node.children.append(supports[node_index])
                        sub_mdp_score, _, _, _, _ = sub_quest_node.eval(observation)
                        sub_pivots.append((pivots[i][0] - start_context_mark, pivots[i][1]))
                        sub_train_data.append((sub_mdp_score - last_sub_score, train_ref.selected_action, len(sub_pivots) - 1, len(sub_pivots)))
                        last_sub_score = sub_mdp_score
                    sub_state_tensor = state_tensor[start_context_mark:(end_context_mark + 1), :]
                    self.rl_core.train(True, sub_pivots, sub_train_data, sub_objective_tensor, sub_state_tensor, action_list_tensor, all_action_list)

            self.rl_core.update_sheduler()


    def think(self, quest_node):
        # supports is a list of nodes
        supports = quest_node.get_children()
        last_observation = quest_node.start_observation
        objective_context, start_obs_context = quest_node.get_start_contexts()
        objective_contexts = [objective_context]
        contexts = [start_obs_context]
        for node in supports:
            if isinstance(node, Trainable):
                contexts.extend(node.get_context())
                last_observation = node.observation
            elif isinstance(node, Thought_Node):
                contexts.extend(node.get_context())
                continue
        
        ################# ACTION SELECTION #################
        selectible_action_set = set([f"Action: {ac}" for ac in last_observation.get_available_actions()])
        self.action_set.update(selectible_action_set)
        valid_extra_actions = set()
        current_objective = quest_node.objective
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
        train_ref = self.rl_core.act(objective_tensor, state_tensor, action_list_tensor, list(selectible_action_set), sample_action=self.training_mode)
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
            sub_objective = self.extra_actions[response]
            return_sub_action = Sub_Action_Type.Relegate
            return_node = Quest_Node(
                objective = sub_objective,
                start_observation = last_observation,
                train_ref = train_ref,
                allow_relegation = self.compute_allow_relegation()
            )
            return return_sub_action, return_node
        elif command.startswith("Action"):
            action = detail
            return_sub_action = Sub_Action_Type.Act
            return_node = Observation_Node(
                action = action,
                train_ref = train_ref
            )
            return return_sub_action, return_node


