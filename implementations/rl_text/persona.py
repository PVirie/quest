from enum import Enum
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

from utilities.language_models import Language_Model, Chat, Chat_Message
from utilities.contextual_memory import Hippocampus
from .rl_graph import Quest_Node, Status_Node, Thought_Node, Observation_Node


class Sub_Action_Type(Enum):
    Fulfill = 1
    Check_Status = 2
    Thought = 3
    Act = 4
    Relegate = 5


class Persona:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)

        with open(os.path.join(dir_path, "prompt.txt"), "r") as file:
            self.prompt = file.read()


    def think(self, quest, supports):
        # supports is a list of sub (quest, answer)
        # return sub_act, detail
        # if return Fulfill, just use all the information to compute the answer
        # if return Recall, just compute the search term
        # if return Thought, just contemplate
        # if return Act, just compute the next action
        # if return Relegate, just compute the next sub quest

        contexts = []
        for node in supports:
            if isinstance(node, Quest_Node):
                contexts.append(f"Sub-Quest: {node.quest}")
                if node.is_answered():
                    contexts.append(f"Result: {node.result}")
                else:
                    contexts.append("Result: Failed to fulfill")
            elif isinstance(node, Status_Node):
                contexts.append(f"Recall: {node.status}")
                contexts.append(f"Result: {node.result}")
            elif isinstance(node, Thought_Node):
                contexts.append(f"Thought: {node.thought}")
            elif isinstance(node, Observation_Node):
                contexts.append(f"Action: {node.action}")
                contexts.append(f"Observation: {node.observation}")

        text_response = self.long_lm.complete_text(self.prompt.format(quest=quest, contexts="\n".join(contexts)))
        # get the first part before newline
        text_response = text_response.split("\n")[0]
        if text_response.startswith("Final Respond:"):
            return Sub_Action_Type.Fulfill, text_response[14:]
        elif text_response.startswith("Check status:"):
            return Sub_Action_Type.Check_Status, text_response[13:]
        elif text_response.startswith("Thought:"):
            return Sub_Action_Type.Thought, text_response[8:]
        elif text_response.startswith("Act:"):
            return Sub_Action_Type.Act, text_response[4:]
        elif text_response.startswith("Sub-Quest:"):
            return Sub_Action_Type.Relegate, text_response[9:]
        
        return None, None


    def check_status(self, search_term):
        # get environment status; such as current location, health, inventory, etc.
        pass


    def act(self, action):
        # forward to the environment and get observation
        pass
