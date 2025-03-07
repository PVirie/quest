from utilities.language_models import complete_chat, complete_text, Language_Model, Chat, Chat_Message

class Answer:
    def __init__(self, text, synonyms, support_paragraph_indices):
        self.text = text
        self.synonyms = synonyms
        self.support_paragraph_indices = support_paragraph_indices

    # check synonym
    def __eq__(self, other):
        other_text = other.text.lower().strip()
        for synonym in self.synonyms:
            if synonym.lower().strip() == other_text:
                return True
        return False


class Persona:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.lm = Language_Model()


    def compute_answer(self, question, supports):
        # supports is a list of sub (question, answer)

        # first it has to consider whether the question can be found in one of the paragraphs
        # if it can, return the answer with support paragraph index
        # if not it then has to check whether the supports are sufficient to answer the question
        # if not, it has to break another next sub question
        # if yes, it has to return the answer


        prompts = ""
        prompts += f"In order to answer this question: {question}\n"
        prompts += "We should break it down into first layer sub questions as follows:\n"
        prompts += "(In the following format: 1. sub question 1\n2. sub question 2\n...\n -----------)\n"


        # return True, answer if all sub questions are sufficient in Answer format
        # return False, next sub question if not sufficient in Question format
        # use lm_func(user_prompt, system_prompt) to facilitate the computation of the answer
    
        return True, "hello"