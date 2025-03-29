from enum import Enum
from utilities.language_models import Language_Model, Chat, Chat_Message
from .rl_graph import Quest_Node, Observation_Node, Thought_Node
import random

"""
Non-Batched version of the Persona class.
"""

class Sub_Action_Type(Enum):
    Fulfill = 1
    Thought = 3
    Act = 4
    Relegate = 5


def prepare_tensors(tokenizer, obs_list, infos):
    # Build agent's observation: feedback + look + inventory.
    # input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])

    # Tokenize and pad the input and the commands to chose from.
    state_tensor = tokenizer(obs_list)
    action_list_tensor = tokenizer(infos["admissible_commands"])

    return state_tensor, action_list_tensor


def extract_detail(text):
    detail = text.split(":")[1].strip()
    # remove space and '#'
    return detail.strip().replace("#", "")


class Persona:
    def __init__(self, env_step, agent, tokenizer, prompt=None, use_rl=True):
        self.env_step = env_step
        self.agent = agent
        self.tokenizer = tokenizer

        self.use_rl = use_rl
        self.use_lm = False

        self.long_lm = None
        self.prompt = None
        if prompt is not None:
            self.use_lm = True
            self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
            self.prompt = prompt


    def think(self, quest_node, supports):
        # supports is a list of nodes
        quest = quest_node.quest
        contexts = []
        score = None
        done = None
        infos = None
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub-Task: {node.quest}")
                if node.is_fulfilled():
                    contexts.append(f"Result: {node.result}")
                else:
                    contexts.append("Result: Failed to fulfill")
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                obs, score, done, infos = node.observation
                contexts.append(f"Observation: {obs}")

        if self.use_lm:
            text_response = self.long_lm.complete_text(self.prompt.format(quest=quest, contexts="\n".join(contexts)))
            text_response = text_response.split("\n")[0]
        else:
            text_response = f"Action: {random.choice(infos["admissible_commands"])}"

        # get the first part before newline
        if text_response.startswith("Final Respond"):
            return Sub_Action_Type.Fulfill, extract_detail(text_response)
        elif text_response.startswith("Thought"):
            return Sub_Action_Type.Thought, extract_detail(text_response)
        elif text_response.startswith("Sub-Task"):
            return Sub_Action_Type.Relegate, extract_detail(text_response)
        elif text_response.startswith("Action"):
            if self.use_rl:
                state_tensor, action_list_tensor = prepare_tensors(self.tokenizer, contexts, infos)
                action = self.agent.act(state_tensor, action_list_tensor, score, done, infos)
                return Sub_Action_Type.Act, action
            else:
                return Sub_Action_Type.Act, extract_detail(text_response)
        
        return None, ""


    def act(self, action):
        # forward to the environment and get observation
        # batch version, action is a list of size 1
        obs, score, done, infos = self.env_step(action)
        return done, (obs, score, done, infos)
