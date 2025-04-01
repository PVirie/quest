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
    Thought = 3
    Act = 4
    Relegate = 5
    Done = 6


def get_last_observation(focus_node):
    child_nodes = focus_node.get_children()
    # reverse iterate
    for node in reversed(child_nodes):
        if isinstance(node, Observation_Node):
            return node.observation
        elif isinstance(node, Quest_Node):
            return node.end_observation
    return None


def prepare_tensors(tokenizer, obs_list, action_list):
    # Build agent's observation: feedback + look + inventory.
    # input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

    # Tokenize and pad the input and the commands to chose from.
    state_tensors = tokenizer(obs_list, stack=True)
    action_list_tensor = tokenizer(action_list, stack=True)

    return state_tensors, action_list_tensor


def extract_detail(text):
    detail = text.split(":")[1].strip()
    # remove space and '#'
    return detail.strip().replace("#", "")


class Persona:
    TRAIN_STEP=10
    PRINT_STEP=1000

    def __init__(self, env_step, agent, tokenizer, train_prompt=None, train=False):
        self.env_step = env_step
        self.agent = agent
        self.tokenizer = tokenizer
        self.extra_actions = []

        self.train = train
        # training use LM to randomly generate action with RL

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
            self.extra_actions = [line.strip() for line in f.readlines()]


    def think(self, quest_node, supports):
        # supports is a list of nodes
        quest = quest_node.quest
        obs, score, done, infos, _ = quest_node.start_observation
        contexts = [f"Observation: {obs}"]
        transitions = []
        last_score = score
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub Task: {node.quest}")
                if node.is_fulfilled():
                    contexts.append(f"Result: {node.result}")
                else:
                    contexts.append("Result: Failed")
                # score, done, infos are the last score from the sub task
                obs, score, done, infos, tf = node.end_observation
                contexts.append(f"Observation: {obs}")
                if not tf.has_released:
                    transitions.append((score - last_score, tf))
                last_score = score
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, score, done, infos, tf = node.observation
                contexts.append(f"Observation: {obs}")
                if not tf.has_released:
                    transitions.append((score - last_score, tf))
                last_score = score

        action_list = [f"Action: {ac}" for ac in infos["admissible_commands"]] + self.extra_actions

        lm_response = ""
        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(quest=quest, action_list=",".join(action_list), contexts="\n".join(contexts)))
            # get the first part before newline
            lm_response = text_response.split("\n")[0]
            if lm_response.startswith("Thought"):
                return Sub_Action_Type.Thought, Thought_Node(extract_detail(response))
            elif not lm_response.startswith("Action") and not lm_response.startswith("Sub Task") and not lm_response.startswith("Final Respond"):
                # if the response is not an action, sub task or final respond, ignore it
                lm_response = ""
            else:
                if lm_response not in action_list:
                    action_list.append(lm_response)

        rl_response = ""
        # remove thoughts from the context for RL
        rl_contexts = [c for c in contexts if not c.startswith("Thought")]
        state_tensor, action_list_tensor = prepare_tensors(self.tokenizer, [quest] + rl_contexts, action_list)
        tf = self.agent.act(state_tensor, action_list_tensor)
        rl_response = action_list[tf.indexes]

        if len(lm_response) > 0 and (random.random() < 0.1 or lm_response.startswith("Final Respond")):
            # if the teacher says it's a final respond, we should override the RL response
            # override RL response with the index of LM response
            tf.override(action_list.index(lm_response))
            response = lm_response
        else:
            response = rl_response

        if self.train:
            self.step += 1
            if len(transitions) >= self.TRAIN_STEP:
                self.agent.train(tf.values, transitions)

            if self.step % self.PRINT_STEP == 0:
                self.agent.print(self.step)
        else:
            tf.release()

        observation = get_last_observation(quest_node)

        if response.startswith("Final Respond"):
            result = extract_detail(response)
            if self.train:
                if result == "Success":
                    final_value = 100
                else:
                    final_value = -100
                self.agent.train(final_value, transitions)
            return Sub_Action_Type.Fulfill, Quest_Node(None, result, None, observation)
        elif response.startswith("Sub Task"):
            return Sub_Action_Type.Relegate, Quest_Node(extract_detail(response), None, observation, None)
        elif response.startswith("Action"):
            action = extract_detail(response)
            obs, score, done, infos = self.env_step(action)
            observation = (obs, score, done, infos, tf)
            if done:
                if self.train:
                    if infos["won"]:
                        final_value = 100
                    elif infos["lost"]:
                        final_value = -100
                    else:
                        final_value = tf.values.item()
                    self.agent.train(final_value, transitions)
                return Sub_Action_Type.Done, Quest_Node(None, "End", None, observation)
            else:
                return Sub_Action_Type.Act, Observation_Node(action, observation)

        return None, ""

