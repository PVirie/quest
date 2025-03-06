

class Persona:
    def __init__(self, lm_func):
        self.lm_func = lm_func



    def compute_answer(self, question, supports):
        # supports is a list of sub (question, answer)
        # return True, answer if all sub questions are sufficient in Answer format
        # return False, next sub question if not sufficient in Question format
        # self.lm_func(question)
    
        return True, "hello"