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

    def __init__(self, agent, tokenizer, compute_folds, env_step, goal_pursuit_eval, action_parser, allow_relegation=True, train_prompt=None):
        self.agent = agent
        self.tokenizer = tokenizer
        self.compute_folds = compute_folds
        self.env_step = env_step
        self.goal_pursuit_eval = goal_pursuit_eval
        self.allow_relegation = allow_relegation
        self.action_parser = action_parser
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
        # save the agent and the extra actions
        agent_path = os.path.join(path, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.agent.save(agent_path)
        with open(os.path.join(path, "extra_actions.txt"), "w", encoding="utf-8") as f:
            for action in self.extra_actions.keys():
                f.write(action[10:] + "\n")


    def load(self, path):
        if not os.path.exists(path):
            return False
        # load the agent and the extra actions
        success = self.agent.load(os.path.join(path, "agent"))
        if success:
            with open(os.path.join(path, "extra_actions.txt"), "r", encoding="utf-8") as f:
                action_keys = set([line.strip() for line in f.readlines()])
                self.extra_actions = {f"Sub Task: {action}": self.action_parser(action) for action in action_keys}
        return success


    def print_context(self, quest_node, prefix=""):
        children = quest_node.get_children()
        obs, _, _, _ = quest_node.start_observation
        obs = obs.replace("\n\n", "\n").replace("\n", f"\n{prefix}")
        contexts = [f"{prefix}Objective: {quest_node.objective}", f"{prefix}Start Obs: {obs}"]
        for i, node in enumerate(children):
            if isinstance(node, Quest_Node):
                contexts.append(f"{prefix}{i} Sub Task: {node.objective}")
                sub_task_context = self.print_context(node, prefix=prefix + "\t")
                contexts.append(sub_task_context)
                contexts.append(f"{prefix}Result: {node.result}")
                obs = "None"
                if node.end_observation is not None:
                    obs, _, _, _ = node.end_observation
                    obs = obs.replace("\n\n", "\n").replace("\n", f"\n{prefix}")
                contexts.append(f"{prefix}Observation: {obs}")
            elif isinstance(node, Thought_Node):
                contexts.append(f"{prefix}{i} Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"{prefix}{i} Action: {node.action}")
                obs, _, _, _ = node.observation
                obs = obs.replace("\n\n", "\n").replace("\n", f"\n{prefix}")
                contexts.append(f"{prefix}Observation: {obs}")
        if len(prefix) == 0:
            contexts.append(f"{prefix}Extra Actions: {", ".join(self.extra_actions.keys())}")
        return f"\n".join(contexts)


    def train(self, quest_node, train_last_node: bool = False, finish_value=None):
        # finish_value only used when training the last node
        obs, _, _, info = quest_node.start_observation
        all_action_set = set([f"Action: {ac}" for ac in info["admissible_commands"]])
        objective_contexts = [f"Objective: {quest_node.objective}"]
        rl_contexts = [f"Observation: {obs}"] 
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
                rl_contexts.append(f"Sub Task: {node.objective}")
                rl_contexts.append(f"Result: {node.result}")
                obs, _, _, info = node.end_observation
                rl_contexts.append(f"Observation: {obs}")
            elif isinstance(node, Observation_Node):
                rl_contexts.append(f"Action: {node.action}")
                obs, _, _, info = node.observation
                rl_contexts.append(f"Observation: {obs}")
            else:
                continue
            
            all_action_set = all_action_set.union(set([f"Action: {ac}" for ac in info["admissible_commands"]]))
            train_ref = node.train_ref
            score = train_ref.mdp_score
            last_state_value = train_ref.state_value
            if not train_ref.has_released:
                if i < len(supports) - 1 or train_last_node:
                    selected_nodes.append((obs, score, info, last_context_mark))
                    pivots.append((score - last_score, last_context_mark))
                    train_data.append((train_ref.selected_action, len(pivots) - 1, len(pivots)))
                    train_ref.release()

            last_context_mark = len(rl_contexts) - 1
            last_score = score
            i += 1
        
        if train_last_node:
            last_state_value = finish_value

        folds = self.compute_folds(quest_node.objective, selected_nodes)
        for _, diff_str, obj, from_transition_index, to_transition_index in folds:
            fold_action = f"Sub Task: {diff_str}"
            self.extra_actions[fold_action] = obj
            # train_data.append((fold_action, from_transition_index, to_transition_index))

        # add extra actions
        all_action_list = list(all_action_set) + list(self.extra_actions.keys())

        if len(train_data) > 0:
            objective_tensor = self.tokenizer(objective_contexts, stack=True)
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.agent.train(last_state_value, pivots, train_data, objective_tensor, state_tensor, action_list_tensor, all_action_list)

            for value, diff_str, _, from_transition_index, to_transition_index in folds:
                sub_objective_tensor = self.tokenizer([diff_str], stack=True)
                sub_pivots = []
                sub_train_data = []
                start_context_mark = pivots[from_transition_index][1]
                end_context_mark = pivots[to_transition_index][1]
                for i in range(from_transition_index, to_transition_index + 1):
                    sub_pivots.append((0, pivots[i][1] - start_context_mark))
                    sub_train_data.append((train_data[i][0], len(sub_pivots) - 1, len(sub_pivots)))
                self.agent.train(value, sub_pivots, sub_train_data, sub_objective_tensor, state_tensor[start_context_mark:(end_context_mark + 1), :], action_list_tensor, all_action_list)


    def think(self, quest_node, supports):
        # supports is a list of nodes
        obs, env_score, done, infos = quest_node.start_observation
        objective_contexts = [f"Objective: {quest_node.objective}"]
        contexts = [f"Observation: {obs}"]
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.objective}")
                contexts.append(f"Result: {node.result}")
                # va must be taken from the last tensor from the previous node
                # score, done, infos must be taken from the sub task
                obs, env_score, done, infos = node.end_observation
                contexts.append(f"Observation: {obs}")
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, env_score, done, infos  = node.observation
                contexts.append(f"Observation: {obs}")
        last_observation = (obs, env_score, done, infos)

        mdp_score, fulfilled, success, finish_value = quest_node.eval(env_score, infos)
        if len(supports) > 0:
            # Because training has to update weight anyway, which violate the functional programming paradigm
            # I'll just update the last child's mdp_score
            supports[-1].train_ref.mdp_score = mdp_score

        train_last_node = False
        if fulfilled:
            return_sub_action = Sub_Action_Type.Fulfill
            if success:
                train_last_node = True
                return_node = Quest_Node(result = "Success", end_observation=last_observation)
            else:
                return_node = Quest_Node(result = "Failed", end_observation=last_observation)
        elif done:
            # train_last_node = True
            return_sub_action = Sub_Action_Type.Done
            return_node = Quest_Node(result = "Terminated", end_observation=last_observation)

        if self.training_mode:
            self.step += 1
            if done or fulfilled or self.step % self.TRAIN_STEP == 0:
                self.train(quest_node, train_last_node=train_last_node, finish_value=finish_value)

            if self.step % self.PRINT_STEP == 0:
                self.agent.print(self.step)

        if done or fulfilled:
            return return_sub_action, return_node

        ################# ACTION SELECTION #################

        action_list = [f"Action: {ac}" for ac in infos["admissible_commands"]]
        if self.allow_relegation:
            parent_objective = self.action_parser(quest_node.objective)
            for key, obj in self.extra_actions.items():
                if obj < parent_objective:
                    action_list.append(key)

        lm_response = ""
        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(action_list=",".join(action_list), contexts="\n".join(objective_contexts + contexts)))
            # get the first part before newline
            lm_response = text_response.split("\n")[0]
            lm_command, lm_detail = extract_command_and_detail(lm_response)
            if lm_command.startswith("Thought"):
                return Sub_Action_Type.Thought, Thought_Node(lm_detail)
            elif not lm_command.startswith("Action") and not lm_command.startswith("Sub Task"):
                # if the response is not an action, sub task or final respond, ignore it
                lm_response = ""
            else:
                if lm_response not in action_list:
                    action_list.append(lm_response)

        rl_response = ""
        # remove thoughts from the context for RL
        rl_contexts = [c for c in contexts if not c.startswith("Thought")]
        objective_tensor = self.tokenizer(objective_contexts, stack=True)
        state_tensor = self.tokenizer(rl_contexts, stack=True)
        action_list_tensor = self.tokenizer(action_list, stack=True)
        va = self.agent.act(objective_tensor, state_tensor, action_list_tensor, action_list, sample_action=self.training_mode)
        rl_response = va.selected_action

        if not self.training_mode and va is not None:
            va.release()

        if len(lm_response) > 0 and random.random() < 0.1:
            # override RL response with the index of LM response
            va.selected_action = lm_response
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
                train_ref=va
            )
            return return_sub_action, return_node
        elif command.startswith("Action"):
            action = detail
            obs, env_score, done, infos = self.env_step(action)
            return_sub_action = Sub_Action_Type.Act
            return_node = Observation_Node(
                action=action, 
                observation=(obs, env_score, done, infos), 
                train_ref=va
            )
            return return_sub_action, return_node


