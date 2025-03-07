from utilities.language_models import complete_chat, complete_text, Language_Model, Chat, Chat_Message
from utilities.vector_dictionary import Vector_Text_Dictionary

class Answer:
    def __init__(self, text, synonyms, support_paragraph_indices):
        self.text = text
        self.synonyms = synonyms
        self.support_paragraph_indices = support_paragraph_indices

    # check synonym
    def __eq__(self, other):
        other_text = other.text.lower().strip()
        if self.text.lower().strip() == other_text:
            return True
        for synonym in self.synonyms:
            if synonym.lower().strip() == other_text:
                return True
        return False


def try_get_answer(lm, question, paragraph):
    prompts = ""
    prompts += f"In order to answer this question: {question}\n"
    prompts += "We should break it down into first layer sub questions as follows:\n"
    prompts += "(In the following format: 1. sub question 1\n2. sub question 2\n...\n -----------)\n"
    return True, ""


def discover_sub_question(lm, question, supports):
    return ""


class Persona:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.lm = Language_Model()
        self.hippocampus = Vector_Text_Dictionary([p.paragraph_text for p in paragraphs], metadata=[p.idx for p in paragraphs])


    def compute_answer(self, question, supports):
        # supports is a list of sub (question, answer)

        if len(supports) == 0:
            # first it has to consider whether the question can be found in one of the paragraphs
            # if it can, return the answer with support paragraph index
            best_paragraph_index = self.hippocampus.match(question, k=1)
            selected_paragraph = self.hippocampus.get_paragraph(best_paragraph_index)
            success, answer = try_get_answer(self.lm, question, selected_paragraph)
            if success:
                best_paragraph_id = self.hippocampus.metadata[best_paragraph_index]
                return True, Answer(answer, [], [best_paragraph_id])
        else:
            # if not
            # it then has to check whether the supports are sufficient to answer the question
            success, answer =  try_get_answer(self.lm, question, [f"{s[0]}: {s[1].text}" for s in supports])
            if success:
                # merge all support paragraphs
                support_paragraph_ids = []
                for s in supports:
                    if s[1].support_paragraph_indices is not None:
                        support_paragraph_ids.extend(s[1].support_paragraph_indices)
                return True, Answer(answer, [], support_paragraph_ids)

        # if not, it has to break another next sub question
        sub_question = discover_sub_question(self.lm, question, [f"{s[0]}: {s[1].text}" for s in supports])
        return False, sub_question