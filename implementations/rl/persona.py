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

    def __init__(self, env_step, agent, tokenizer, observation_difference, train_prompt=None, train=False):
        self.env_step = env_step
        self.agent = agent
        self.tokenizer = tokenizer
        self.observation_difference = observation_difference
        self.extra_actions = set()

        self.training_mode = train

        self.use_lm = False
        self.long_lm = None
        self.prompt = None
        if train_prompt is not None:
            self.use_lm = True
            self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
            self.prompt = train_prompt

        self.step = 0


    def save(self, path):
        # save the agent and the extra actions
        self.agent.save(os.path.join(path, "agent"))
        with open(os.path.join(path, "extra_actions.txt"), "w") as f:
            for action in self.extra_actions:
                f.write(action + "\n")


    def load(self, path):
        # load the agent and the extra actions
        self.agent.load(os.path.join(path, "agent"))
        with open(os.path.join(path, "extra_actions.txt"), "r") as f:
            self.extra_actions = set([line.strip() for line in f.readlines()])


    def print_context(self, quest_node):
        children = quest_node.get_children()
        obs, score, done, infos, _ = quest_node.start_observation
        contexts = [f"Task: {quest_node.quest["objective"]}", f"Observation: {obs}"]
        for node in children:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.quest["objective"]}")
                if node.is_fulfilled():
                    contexts.append(f"Result: {node.result}")
                else:
                    contexts.append("Result: Failed")
                # score, done, infos are the last score from the sub task
                obs, score, done, infos, _ = node.end_observation
                contexts.append(f"Observation: {obs}")
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, score, done, infos, _ = node.observation
                contexts.append(f"Observation: {obs}")
        contexts.append(f"Admissible Commands: {",".join(infos['admissible_commands'])}")
        contexts.append(f"Extra Actions: {",".join(self.extra_actions)}")
        return "\n".join(contexts)


    def train(self, objective, start_observation, supports, end_observation, value, force_train_last: bool = False, goal_pursue: bool = False):

        _, score, _, info, _ = start_observation
        _, _, carry = self.observation_differnce(start_observation, end_observation, None)
        rl_contexts = [objective]
        last_context_mark = 0
        transitions = []
        last_score = score
        all_action_set = set([f"Action: {ac}" for ac in info["admissible_commands"]])
        fold_exit = False
        for i, node in enumerate(supports):
            if isinstance(node, Quest_Node):
                rl_contexts.append(f"Sub Task: {node.quest["objective"]}")
                rl_contexts.append(f"Result: {node.result}")
                # va must be taken from the last tensor from the previous node
                _, _, _, _, va = node.start_observation
                # score must be taken from the sub task
                obs, score, _, info, _ = node.end_observation
                rl_contexts.append(f"Observation: {obs}")
            elif isinstance(node, Observation_Node):
                last_node = supports[-1]
                last_node_observation = last_node.observation if isinstance(last_node, Observation_Node) else last_node.end_observation
                has_diff, diff_str, carry = self.observation_difference(node.observation, last_node_observation, carry)
                if has_diff:
                    fold_action = f"Sub Task: {diff_str}"
                    self.extra_actions.add(fold_action)
                    rl_contexts.append(f"Sub Task: {diff_str}")
                    rl_contexts.append(f"Result: Success")
                    _, _, _, _, va = node.observation
                    va.selected_action = fold_action
                    obs, score, _, info, _ = last_node_observation
                    rl_contexts.append(f"Observation: {obs}")
                    self.train(diff_str, node.observation, [n for n in supports[i:]], last_node_observation, 20, force_train_last=True, goal_pursue=True)
                    fold_exit = True
                else:
                    rl_contexts.append(f"Action: {node.action}")
                    obs, score, _, info, va = node.observation
                    rl_contexts.append(f"Observation: {obs}")
            else:
                continue
            
            all_action_set = all_action_set.union(set([f"Action: {ac}" for ac in info["admissible_commands"]]))
            if va is not None and not va.has_released:
                if goal_pursue:
                    # if goal pursue, every step is penalized
                    score = last_score - 1
                transitions.append((score - last_score, va, last_context_mark))
            last_context_mark = len(rl_contexts) - 1
            last_score = score

            if fold_exit:
                break
        
        _, score, _, info, va = end_observation
        all_action_set = all_action_set.union(set([f"Action: {ac}" for ac in info["admissible_commands"]]))
        # add extra actions
        all_action_list = list(all_action_set.union(self.extra_actions))

        if force_train_last: 
            if va is not None and not va.has_released:
                if goal_pursue:
                    score = last_score - 1
                transitions.append((score - last_score, va, last_context_mark))

        if len(transitions) > 0:
            state_tensor = self.tokenizer(rl_contexts, stack=True)
            action_list_tensor = self.tokenizer(all_action_list, stack=True)
            self.agent.train(value, transitions, state_tensor, action_list_tensor, all_action_list)


    def think(self, quest_node, supports):
        # supports is a list of nodes
        objective = quest_node.quest["objective"]
        max_steps = quest_node.quest["max_steps"] if "max_steps" in quest_node.quest else None
        count_non_thought_steps = 0
        obs, score, done, infos, _ = quest_node.start_observation
        contexts = [f"Observation: {obs}"]
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.quest["objective"]}")
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

        action_list = [f"Action: {ac}" for ac in infos["admissible_commands"]] + list(self.extra_actions)

        lm_response = ""
        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(quest=objective, action_list=",".join(action_list), contexts="\n".join(contexts)))
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
        state_tensor = self.tokenizer([objective] + rl_contexts, stack=True)
        action_list_tensor = self.tokenizer(action_list, stack=True)
        va = self.agent.act(state_tensor, action_list_tensor, action_list)
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
            if sub_objective == objective:
                # duplicate sub task, penalize
                current_value = -100
                force_train_last = True
                return_sub_action = Sub_Action_Type.Relegate 
                return_node = Quest_Node(
                    quest = {
                        "objective": sub_objective,
                        "max_steps": self.TRAIN_STEP
                    },
                    result="Failed duplicate work",
                    start_observation=last_observation,
                    end_observation=(obs, score, done, infos, None)
                )
            else:
                return_sub_action = Sub_Action_Type.Relegate 
                return_node = Quest_Node(
                    quest = {
                        "objective": sub_objective,
                        "max_steps": self.TRAIN_STEP
                    },
                    start_observation=last_observation
                )
        elif command.startswith("Action"):
            action = detail
            obs, score, done, infos = self.env_step(action)
            last_observation = (obs, score, done, infos, va)

            if done:
                force_train_last = True
                return_sub_action = Sub_Action_Type.Done
                if infos["won"]:
                    current_value = 100
                    return_node = Quest_Node(result = "Success", end_observation=last_observation)
                elif infos["lost"]:
                    current_value = -100
                    return_node = Quest_Node(result = "Failed", end_observation=last_observation)
                else:
                    return_node = Quest_Node(result = "Failed", end_observation=last_observation)
            elif obs == objective:
                force_train_last = True
                current_value = 10
                return_sub_action = Sub_Action_Type.Fulfill
                return_node = Quest_Node(result = "Success", end_observation=last_observation)
            elif max_steps is not None and count_non_thought_steps > max_steps:
                force_train_last = True
                current_value = -10
                return_sub_action = Sub_Action_Type.Fulfill
                return_node = Quest_Node(result = "Failed", end_observation=last_observation)
            else:
                return_sub_action = Sub_Action_Type.Act
                return_node = Observation_Node(action, last_observation)


        if self.training_mode:
            self.step += 1
            if force_train_last or self.step % self.TRAIN_STEP == 0:
                self.train(objective, quest_node.start_observation, supports, last_observation, current_value, force_train_last=force_train_last)

            if self.step % self.PRINT_STEP == 0:
                self.agent.print(self.step)

        return return_sub_action, return_node

