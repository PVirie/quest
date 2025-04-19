from enum import Enum
from utilities.language_models import Language_Model, Chat, Chat_Message
from .rl_graph import Quest_Node, Observation_Node, Thought_Node
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

    def __init__(self, rl_core, tokenizer, compute_folds, env_step, goal_pursuit_eval, action_parser, compute_action, allow_relegation=True, train_prompt=None):
        self.rl_core = rl_core
        self.tokenizer = tokenizer
        self.compute_folds = compute_folds
        self.env_step = env_step
        self.goal_pursuit_eval = goal_pursuit_eval
        self.action_parser = action_parser
        self.compute_action = compute_action
        self.allow_relegation = allow_relegation
        self.action_set = set()
        self.extra_actions = {}

        self.training_mode = False

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


    def save(self, path):
        # save the rl_core and the extra actions
        rl_core_path = os.path.join(path, "rl_core")
        os.makedirs(rl_core_path, exist_ok=True)
        self.rl_core.save(rl_core_path)
        with open(os.path.join(path, "extra_actions.txt"), "w", encoding="utf-8") as f:
            for action in self.extra_actions.keys():
                f.write(action[10:] + "\n")


    def load(self, path):
        if not os.path.exists(path):
            return False
        # load the rl_core and the extra actions
        success = self.rl_core.load(os.path.join(path, "rl_core"))
        if success:
            with open(os.path.join(path, "extra_actions.txt"), "r", encoding="utf-8") as f:
                action_keys = set([line.strip() for line in f.readlines()])
                self.extra_actions = {f"Sub Task: {action}": self.action_parser(action) for action in action_keys}
        return success


    def print_context(self, quest_node, prefix=""):
        children = quest_node.get_children()
        objective_context, start_obs_context = quest_node.get_start_contexts()
        contexts = [f"{prefix}{objective_context}", f"{prefix}{start_obs_context.replace("\n\n", "\n").replace("\n", f"\n{prefix}")}"]
        for i, node in enumerate(children):
            node_contexts = node.get_context()
            node_contexts.insert(0, f"{i} -----")
            node_contexts = [f"{prefix}" + c.replace("\n\n", "\n").replace("\n", f"\n{prefix}") for c in node_contexts]
            if isinstance(node, Quest_Node):
                sub_task_context = self.print_context(node, prefix=prefix + "\t")
                node_contexts.insert(2, sub_task_context)
            contexts.extend(node_contexts)
        if len(prefix) == 0:
            contexts.append(f"{prefix}Extra Actions: {";".join(self.extra_actions.keys())}")
        return f"\n".join(contexts)


    def train(self, quest_node, train_last_node: bool = False):
        # finish_value only used when training the last node
        obs, _, _, info = quest_node.start_observation
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
            if isinstance(node, Quest_Node):
                rl_contexts.extend(node.get_context())
                obs, _, _, info = node.end_observation
            elif isinstance(node, Observation_Node):
                rl_contexts.extend(node.get_context())
                obs, _, _, info = node.observation
            else:
                continue

            train_ref = node.train_ref
            score = train_ref.mdp_score
            selected_nodes.append((obs, score, info, last_context_mark))
            pivots.append((score - last_score, last_context_mark, list(train_ref.available_actions)))
            if i >= len(supports) - self.TRAIN_STEP:
                if i < len(supports) - 1 or train_last_node:
                    train_data.append((train_ref.selected_action, len(pivots) - 1, len(pivots)))
                    train_ref.release()

            last_context_mark = len(rl_contexts) - 1
            last_score = score
            i += 1
        
        folds = self.compute_folds(quest_node.objective, selected_nodes)
        for _, _, diff_str, obj, from_transition_index, to_transition_index in folds:
            fold_action = f"Sub Task: {diff_str}"
            self.extra_actions[fold_action] = obj
            # train_data.append((fold_action, from_transition_index, to_transition_index))

        # add extra actions
        all_action_list = list(self.action_set.union(self.extra_actions.keys()))

        if len(train_data) > 0:
            objective_tensor = self.tokenizer(objective_contexts, stack=True)
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.rl_core.train(train_last_node, pivots, train_data, objective_tensor, state_tensor, action_list_tensor, all_action_list)

            for value, step_cost, diff_str, _, from_transition_index, to_transition_index in folds:
                sub_objective_tensor = self.tokenizer([diff_str], stack=True)
                sub_pivots = []
                sub_train_data = []
                start_context_mark = pivots[from_transition_index][1]
                end_context_mark = pivots[to_transition_index][1]
                for i in range(from_transition_index, to_transition_index + 1):
                    sub_pivots.append((step_cost if i < to_transition_index else value, pivots[i][1] - start_context_mark, pivots[i][2]))
                    sub_train_data.append((supports[i].train_ref.selected_action, len(sub_pivots) - 1, len(sub_pivots)))
                self.rl_core.train(True, sub_pivots, sub_train_data, sub_objective_tensor, state_tensor[start_context_mark:(end_context_mark + 1), :], action_list_tensor, all_action_list)


    def think(self, quest_node, supports):
        # supports is a list of nodes
        last_observation = quest_node.start_observation
        objective_context, start_obs_context = quest_node.get_start_contexts()
        objective_contexts = [objective_context]
        contexts = [start_obs_context]
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.extend(node.get_context())
                last_observation = node.end_observation
            elif isinstance(node, Observation_Node):
                contexts.extend(node.get_context())
                last_observation  = node.observation
            elif isinstance(node, Thought_Node):
                contexts.extend(node.get_context())
                continue

        mdp_score, terminated, truncated, result = quest_node.eval(last_observation)
        if len(supports) > 0:
            # Because training has to update weight anyway, which violate the functional programming paradigm
            # I'll just update the last child's mdp_score
            last_node = supports[-1]
            last_node.train_ref.mdp_score = mdp_score
            # # now compute the correct Sub Task (sometimes, sub task does not follow the original objective)
            # if isinstance(last_node, Quest_Node):
            #     diff_str, action_obj = self.compute_action(last_node.start_observation, last_node.end_observation)
            #     if len(action_obj) > 0:
            #         action_str = f"Sub Task: {diff_str}"
            #         last_node.train_ref.selected_action = action_str
            #         self.extra_actions[action_str] = action_obj

        train_last_node = False
        if terminated:
            train_last_node = True
            return_sub_action = Sub_Action_Type.Fulfill
            return_node = Quest_Node(result=result, end_observation=last_observation)
        elif truncated:
            return_sub_action = Sub_Action_Type.Done
            return_node = Quest_Node(end_observation=last_observation, truncated=True)

        if self.training_mode:
            self.step += 1
            if terminated or truncated or self.step % self.TRAIN_STEP == 0:
                self.train(quest_node, train_last_node=train_last_node)

            if self.step % self.PRINT_STEP == 0:
                self.rl_core.print(self.step)

        if terminated or truncated:
            return return_sub_action, return_node

        ################# ACTION SELECTION #################
        _, _, _, infos = last_observation
        selectible_action_set = set([f"Action: {ac}" for ac in infos["admissible_commands"]])
        self.action_set.update(selectible_action_set)
        if self.allow_relegation:
            current_objective = self.action_parser(quest_node.objective)
            for key, obj in self.extra_actions.items():
                if obj < current_objective and obj.difference(last_observation) == len(obj):
                    # must check less than and diff to prevent infinite loop
                    selectible_action_set.add(key)

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
                selectible_action_set.add(lm_response)

        rl_response = ""
        # remove thoughts from the context for RL
        rl_contexts = [c for c in contexts if not c.startswith("Thought")]
        objective_tensor = self.tokenizer(objective_contexts, stack=True)
        state_tensor = self.tokenizer(rl_contexts, stack=True)
        action_list_tensor = self.tokenizer(selectible_action_set, stack=True)
        train_ref = self.rl_core.act(objective_tensor, state_tensor, action_list_tensor, list(selectible_action_set), sample_action=self.training_mode)
        train_ref.available_actions = selectible_action_set
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
                objective=sub_objective,
                eval_func=self.goal_pursuit_eval,
                start_observation=last_observation,
                train_ref=train_ref
            )
            return return_sub_action, return_node
        elif command.startswith("Action"):
            action = detail
            obs, env_score, done, infos = self.env_step(action)
            return_sub_action = Sub_Action_Type.Act
            return_node = Observation_Node(
                action=action, 
                observation=(obs, env_score, done, infos), 
                train_ref=train_ref
            )
            return return_sub_action, return_node


