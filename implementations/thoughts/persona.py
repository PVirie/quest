

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
    def __init__(self, lm_func, paragraphs):
        self.lm_func = lm_func
        self.paragraphs = paragraphs


    def compute_answer(self, question, supports):
        # supports is a list of sub (question, answer)
        # return True, answer if all sub questions are sufficient in Answer format
        # return False, next sub question if not sufficient in Question format
        # use self.lm_func(user_prompt, system_prompt) to facilitate the computation of the answer
    
        return True, "hello"