

"""
Batched version of the Persona class.
Now size = 1
"""

def prepare_tensors(tokenizer, obs, infos):
    # Build agent's observation: feedback + look + inventory.
    # input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
    if isinstance(obs, str):
        input_ = obs
        commands = infos["admissible_commands"]
    elif isinstance(obs, tuple):
        input_ = obs[0]
        commands = infos["admissible_commands"][0]

    # Tokenize and pad the input and the commands to chose from.
    state_tensor = tokenizer([input_])
    action_list_tensor = tokenizer(commands)

    return state_tensor, action_list_tensor


class Persona:
    def __init__(self, env, agent, tokenizer):
        self.agent = agent
        self.env = env
        self.tokenizer = tokenizer

    def think(self, quest, supports):
        # supports is a list of sub (quest, answer)
        # return is_compound_action, request_action
        obs, score, done, infos = supports[-1].observation
        state_tensor, action_list_tensor = prepare_tensors(self.tokenizer, obs, infos)
        action = self.agent.act(state_tensor, action_list_tensor, score, done, infos)

        return False, action


    def summarize(self, quest, supports):
        # use LM
        return False, None


    def act(self, action):
        # forward to the environment and get observation
        # batch version, action is a list of size 1
        obs, score, done, infos = self.env.step([action])
        return done, (obs, score, done, infos)


    def close(self, observation):
        obs, score, done, infos = observation
        state_tensor, action_list_tensor = prepare_tensors(self.tokenizer, obs, infos)
        self.agent.act(state_tensor, action_list_tensor, score, done, infos)  # Let the agent know the game is done.