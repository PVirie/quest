

class Persona:
    def __init__(self):
        pass


    def pursue(self, quest, contexts):
        # supports is a list of sub (quest, response) in chronological order from earliest to latest
        # get the latest state
        # get the latest possible action
        # first decide whether the current state has fulfilled the goal
        # if fulfilled, return True, and carry on state in Response format
        # now decide whether you want to think more or just act
        # if you want to think more, return False, detail to think about in Quest format
        # if you want to act, invoke evironment, return False and the detail of the next state and possible actions in Quest format


        # return goal_achieved, detail
        return True, "hello"
    

    def learn(self, state_action_sequence, reward):
        # learn from the experience, or batch
        # return trainer
        pass