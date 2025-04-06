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


def extract_detail(text):
    parts = text.split(":")
    detail = ":".join(parts[1:]).strip()
    # remove space and '#'
    return detail.strip().replace("#", "")


class Persona:
    TRAIN_STEP=10
    PRINT_STEP=1000

    def __init__(self, env_step, agent, tokenizer, observation_differnce, train_prompt=None, train=False):
        self.env_step = env_step
        self.agent = agent
        self.tokenizer = tokenizer
        self.observation_differnce = observation_differnce
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


    def train(self, current_value, quest_node, supports, last_observation, force_train: bool = False):
        self.step += 1
        if not force_train and self.step % self.TRAIN_STEP != 0:
            return

        obs, score, done, infos, _ = quest_node.start_observation
        transitions = []
        last_score = score
        for node in supports:
            if isinstance(node, Quest_Node):
                obs, score, done, infos, tf = node.end_observation
                if not tf.has_released:
                    transitions.append((score - last_score, tf))
                last_score = score
            elif isinstance(node, Observation_Node):
                obs, score, done, infos, tf = node.observation
                if not tf.has_released:
                    transitions.append((score - last_score, tf))
                last_score = score

                # compute diff with the last_observation
                has_diff, diff_str = self.observation_differnce(node.observation, last_observation)
                if has_diff:
                    fold_action = f"Sub Task: {diff_str}"
                    self.extra_actions.add(fold_action)
                    self.agent.append_action(tf, self.tokenizer([fold_action], stack=True))

        self.agent.train(current_value, transitions)

        if self.step % self.PRINT_STEP == 0:
            self.agent.print(self.step)


    def think(self, quest_node, supports):
        # supports is a list of nodes
        objective = quest_node.quest["objective"]
        max_steps = quest_node.quest["max_steps"] if "max_steps" in quest_node.quest else None
        count_non_thought_steps = 0
        obs, score, done, infos, _ = quest_node.start_observation
        last_observation = quest_node.start_observation
        contexts = [f"Observation: {obs}"]
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.quest["objective"]}")
                if node.is_fulfilled():
                    contexts.append(f"Result: {node.result}")
                else:
                    contexts.append("Result: Failed")
                # score, done, infos are the last score from the sub task
                obs, score, done, infos, tf = node.end_observation
                last_observation = node.end_observation
                contexts.append(f"Observation: {obs}")
                count_non_thought_steps += 1
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, score, done, infos, tf = node.observation
                last_observation = node.observation
                contexts.append(f"Observation: {obs}")
                count_non_thought_steps += 1

        action_list = [f"Action: {ac}" for ac in infos["admissible_commands"]] + list(self.extra_actions)

        lm_response = ""
        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(quest=objective, action_list=",".join(action_list), contexts="\n".join(contexts)))
            # get the first part before newline
            lm_response = text_response.split("\n")[0]
            if lm_response.startswith("Thought"):
                return Sub_Action_Type.Thought, Thought_Node(extract_detail(response))
            elif not lm_response.startswith("Action") and not lm_response.startswith("Sub Task"):
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
        tf = self.agent.act(state_tensor, action_list_tensor)
        rl_response = action_list[tf.indexes]

        if len(lm_response) > 0 and random.random() < 0.1:
            # override RL response with the index of LM response
            tf.override_selected_action(action_list.index(lm_response))
            response = lm_response
        else:
            response = rl_response

        should_train = False
        current_value = tf.values.item()
        return_sub_action = None
        return_node = None
        if response.startswith("Sub Task"):
            return_sub_action = Sub_Action_Type.Relegate 
            return_node = Quest_Node(
                quest = {
                    "objective": extract_detail(response),
                    "max_steps": self.TRAIN_STEP
                },
                start_observation=last_observation
            )
        elif response.startswith("Action"):
            action = extract_detail(response)
            obs, score, done, infos = self.env_step(action)
            observation = (obs, score, done, infos, tf)

            if done:
                should_train = True
                return_sub_action = Sub_Action_Type.Done
                if infos["won"]:
                    current_value = 100
                    return_node = Quest_Node(result = "Success", end_observation=observation)
                elif infos["lost"]:
                    current_value = -100
                    return_node = Quest_Node(result = "Failed", end_observation=observation)
                else:
                    return_node = Quest_Node(result = "Failed", end_observation=observation)
            elif obs == objective:
                should_train = True
                current_value = 10
                return_sub_action = Sub_Action_Type.Fulfill
                return_node = Quest_Node(result = "Success", end_observation=observation)
            elif max_steps is not None and count_non_thought_steps > max_steps:
                should_train = True
                current_value = -10
                return_sub_action = Sub_Action_Type.Fulfill
                return_node = Quest_Node(result = "Failed", end_observation=observation)
            else:
                return_sub_action = Sub_Action_Type.Act
                return_node = Observation_Node(action, observation)


        if self.training_mode:
            self.train(current_value, quest_node, supports, last_observation, force_train=should_train)
        else:
            tf.release()


        return return_sub_action, return_node

