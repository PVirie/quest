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

    def __init__(self, agent, tokenizer, compute_folds, sub_eval_step_func, objective_less_than, allow_relegation=True, train_prompt=None):
        self.agent = agent
        self.tokenizer = tokenizer
        self.compute_folds = compute_folds
        self.sub_eval_step_func = sub_eval_step_func
        self.objective_less_than = objective_less_than
        self.allow_relegation = allow_relegation
        self.extra_actions = set()

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
            for action in self.extra_actions:
                f.write(action + "\n")


    def load(self, path):
        if not os.path.exists(path):
            return False
        # load the agent and the extra actions
        success = self.agent.load(os.path.join(path, "agent"))
        if success:
            with open(os.path.join(path, "extra_actions.txt"), "r", encoding="utf-8") as f:
                self.extra_actions = set([line.strip() for line in f.readlines()])
        return success


    def print_context(self, quest_node, prefix="", s = 0):
        children = quest_node.get_children()
        obs, _, _, _, _ = quest_node.start_observation
        contexts = [f"Objective: {quest_node.objective}", f"Observation: {obs}"]
        for i, node in enumerate(children):
            if isinstance(node, Quest_Node):
                contexts.append(f"{i + s} Sub Task: {node.objective}")
                sub_task_context = self.print_context(node, prefix=prefix + "  ", s = i + s + 1)
                contexts.append(sub_task_context)
                contexts.append(f"\tResult: {node.result}")
                # score, done, infos are the last score from the sub task
                obs = "None"
                if node.end_observation is not None:
                    obs, _, _, _, _ = node.end_observation
                contexts.append(f"\tObservation: {obs}")
            elif isinstance(node, Thought_Node):
                contexts.append(f"{i + s} Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"{i + s} Action: {node.action}")
                obs, _, _, _, _ = node.observation
                contexts.append(f"\tObservation: {obs}")
        if s == 0:
            contexts.append(f"Extra Actions: {", ".join(self.extra_actions)}")
        return f"{prefix}\n".join(contexts)


    def train(self, objective, start_observation, supports, end_observation, value, force_train_last: bool = False):
        obs, score, _, info, _ = start_observation
        all_action_set = set([f"Action: {ac}" for ac in info["admissible_commands"]])
        objective_contexts = [f"Objective: {objective}"]
        rl_contexts = [f"Observation: {obs}"] 
        last_context_mark = 0
        pivots = []
        train_data = []
        selected_nodes = []
        last_score = score
        i = 0
        while i < len(supports):
            node = supports[i]
            if isinstance(node, Quest_Node):
                rl_contexts.append(f"Sub Task: {node.objective}")
                rl_contexts.append(f"Result: {node.result}")
                # va must be taken from the last tensor from the previous node
                _, _, _, _, va = node.start_observation
                # score must be taken from the sub task
                obs, score, _, info, _ = node.end_observation
                rl_contexts.append(f"Observation: {obs}")
            elif isinstance(node, Observation_Node):
                rl_contexts.append(f"Action: {node.action}")
                obs, score, _, info, va = node.observation
                rl_contexts.append(f"Observation: {obs}")
            else:
                continue
            
            all_action_set = all_action_set.union(set([f"Action: {ac}" for ac in info["admissible_commands"]]))
            if va is not None and not va.has_released:
                pivots.append((score - last_score, last_context_mark))
                train_data.append((va.selected_action, len(pivots) - 1, len(pivots)))
                va.release()
                selected_nodes.append((obs, score, info, last_context_mark))
            last_context_mark = len(rl_contexts) - 1
            last_score = score

            i += 1
        
        folds = self.compute_folds(selected_nodes)
        for delta_score, diff_str, from_transition_index, to_transition_index in folds:
            fold_action = f"Sub Task: {diff_str}"
            self.extra_actions.add(fold_action)
            train_data.append((fold_action, from_transition_index, to_transition_index))

        _, score, _, info, va = end_observation
        all_action_set = all_action_set.union(set([f"Action: {ac}" for ac in info["admissible_commands"]]))
        # add extra actions
        all_action_list = list(all_action_set.union(self.extra_actions))

        if force_train_last: 
            if va is not None and not va.has_released:
                pivots.append((score - last_score, last_context_mark))
                train_data.append((va.selected_action, len(pivots) - 1, len(pivots)))
                va.release()

        if len(train_data) > 0:
            objective_tensor = self.tokenizer(objective_contexts, stack=True)
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.agent.train(value, pivots, train_data, objective_tensor, state_tensor, action_list_tensor, all_action_list)

            if self.allow_relegation:
                for _, diff_str, from_transition_index, to_transition_index in folds:
                    sub_objective_tensor = self.tokenizer([diff_str], stack=True)
                    sub_pivots = []
                    sub_train_data = []
                    start_context_mark = pivots[from_transition_index][1]
                    end_context_mark = pivots[to_transition_index][1]
                    for i in range(from_transition_index, to_transition_index + 1):
                        sub_pivots.append((0, pivots[i][1] - start_context_mark))
                        sub_train_data.append((train_data[i][0], len(sub_pivots) - 1, len(sub_pivots)))
                    self.agent.train(20, sub_pivots, sub_train_data, sub_objective_tensor, state_tensor[start_context_mark:(end_context_mark + 1), :], action_list_tensor, all_action_list)


    def think(self, quest_node, supports):
        # supports is a list of nodes
        count_non_thought_steps = 0
        obs, score, done, infos, _ = quest_node.start_observation
        objective_contexts = [f"Objective: {quest_node.objective}"]
        contexts = [f"Observation: {obs}"]
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.objective}")
                contexts.append(f"Result: {node.result}")
                # va must be taken from the last tensor from the previous node
                # _, _, _, _, _ = node.start_observation
                # score, done, infos must be taken from the sub task
                obs, score, done, infos, _ = node.end_observation
                contexts.append(f"Observation: {obs}")
                count_non_thought_steps += 1
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, score, done, infos, _ = node.observation
                contexts.append(f"Observation: {obs}")
                count_non_thought_steps += 1

        action_list = [f"Action: {ac}" for ac in infos["admissible_commands"]]
        if self.allow_relegation:
            action_list = action_list + list(self.extra_actions)

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
        va = self.agent.act(objective_tensor, state_tensor, action_list_tensor, action_list, sample_action=True)
        rl_response = va.selected_action

        if not self.training_mode and va is not None:
            va.release()

        if len(lm_response) > 0 and random.random() < 0.1:
            # override RL response with the index of LM response
            va.selected_action = lm_response
            response = lm_response
        else:
            response = rl_response

        command, detail = extract_command_and_detail(response)

        force_train_last = False
        current_value = va.state_value
        return_sub_action = None
        return_node = None
        if command.startswith("Sub Task"):
            sub_objective = detail
            last_observation = (obs, score, done, infos, va)
            if self.objective_less_than(sub_objective, quest_node.objective):
                return_sub_action = Sub_Action_Type.Relegate 
                return_node = Quest_Node(
                    objective=sub_objective,
                    env_step=self.sub_eval_step_func,
                    start_observation=last_observation
                )
            else:
                # duplicate sub task, penalize
                current_value = -100
                force_train_last = True
                return_sub_action = Sub_Action_Type.Relegate 
                return_node = Quest_Node(
                    objective=sub_objective,
                    result="Failed to breakdown",
                    start_observation=last_observation,
                    end_observation=(obs, score, done, infos, None)
                )
        elif command.startswith("Action"):
            action = detail
            obs, score, done, infos, fulfilled, success, current_value = quest_node.step(action)
            last_observation = (obs, score, done, infos, va)
            if done or fulfilled:
                force_train_last = True
                return_sub_action = Sub_Action_Type.Done if done else Sub_Action_Type.Fulfill
                if success:
                    return_node = Quest_Node(result = "Success", end_observation=last_observation)
                else:
                    return_node = Quest_Node(result = "Failed", end_observation=last_observation)
            else:
                return_sub_action = Sub_Action_Type.Act
                return_node = Observation_Node(action, last_observation)

        if self.training_mode:
            self.step += 1
            if force_train_last or self.step % self.TRAIN_STEP == 0:
                self.train(quest_node.objective, quest_node.start_observation, supports, last_observation, current_value, force_train_last=force_train_last)

            if self.step % self.PRINT_STEP == 0:
                self.agent.print(self.step)

        return return_sub_action, return_node

