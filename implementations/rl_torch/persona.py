from utilities.language_models import Language_Model, Chat, Chat_Message
from .rl_graph import Quest_Node, Observation_Node

"""
Non-Batched version of the Persona class.
"""

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
    def __init__(self, env_step, agent, tokenizer, prompt=None):
        self.agent = agent
        self.env_step = env_step
        self.tokenizer = tokenizer
        self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
        self.prompt = prompt


    def think(self, quest_node, supports):
        # supports is a list of nodes
        # return is_compound_action, request_action
        obs, score, done, infos = supports[-1].observation
        # get all node observations
        obs_list = [node.observation[0] for node in supports]

        state_tensor, action_list_tensor = prepare_tensors(self.tokenizer, obs_list, infos)
        action = self.agent.act(state_tensor, action_list_tensor, score, done, infos)

        return False, action


    def summarize(self, quest_node, supports):
        if self.prompt is None:
            return False, None

        quest = quest_node.quest

        transcripts = []
        for node in supports:
            if isinstance(node, Quest_Node):
                transcripts.append(f"Sub-Task: {node.quest}")
                transcripts.append(f"Result: {node.result}")
            elif isinstance(node, Observation_Node):
                transcripts.append(f"Action: {node.action}")
                transcripts.append(f"Observation: {node.observation}")

        # use LM
        text_response = self.long_lm.complete_text(self.prompt.format(question=quest, transcripts="\n".join(transcripts)))
        # get the first part before newline
        text_response = text_response.split("\n")[0]
        if text_response.startswith("Final Answer"):
            return True, extract_detail(text_response)
        else:
            return False, None


    def act(self, action):
        # forward to the environment and get observation
        # batch version, action is a list of size 1
        obs, score, done, infos = self.env_step(action)
        return done, (obs, score, done, infos)
