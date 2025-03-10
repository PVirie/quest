from utilities.language_models import Language_Model, Chat, Chat_Message
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

    prompt = f"""
    You are an AI that determines if there is enough information to answer a given question and answer it.
    Based on the provided information ONLY, decide whether the information is enough to answer. Do not use your own knowledge.
    
    If the information is not enough, respond with exactly: "no".
    If the information is enough, respond the answer the question using the provided information only. 
    
    Question:
    {question}

    Information:
    {str(paragraph)}
    """

    chat = Chat()
    chat.append(Chat_Message(role="system", content="""You determine if the provided information is enough to answer a specific question.
                                        If so, answer it using the provided information only; otherwise, respond with 'no'."""))
    chat.append(Chat_Message(role="user", content=prompt))
    
    text_response = lm.complete_chat(chat)
    if text_response[:2].lower().strip() == "no":
        return False, None

    return True, text_response


def discover_sub_question(lm, question, supports):
    
    if len(supports) == 0:
        supports = "What is the first sub-question to ask?"
    else:
        supports = f"""Pre-existing Sub-Questions:
        {supports}
        """

    prompt = f"""
    You are an NLP AI that breaks down a question into sub-questions.
    Based on the pre-existing sub-questions, determine the next sub-question to ask in order to answer the original question.

    Question:
    {question}
    
    {str(supports)}
    """

    chat = Chat()
    chat.append(Chat_Message(role="system", content="""You determine if the provided information is a complete.
                                        If it is not, response the next sub-question only."""))
    chat.append(Chat_Message(role="user", content=prompt))
    
    text_response = lm.complete_chat(chat)

    return text_response


class Persona:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.short_lm = Language_Model(max_length=200, top_p=1, temperature=0)
        self.long_lm = Language_Model(max_length=1024, top_p=1, temperature=0)
        self.hippocampus = Vector_Text_Dictionary([p.paragraph_text for p in paragraphs], metadata=[p.idx for p in paragraphs], chunk_size=64)


    def compute_answer(self, question, supports):
        # supports is a list of sub (question, answer)

        if len(supports) == 0:
            # first it has to consider whether the question can be found in one of the paragraphs
            # if it can, return the answer with support paragraph index
            best_paragraph_index = self.hippocampus.match(question, k=1)
            selected_paragraph = self.hippocampus.get_paragraph(best_paragraph_index)
            success, answer = try_get_answer(self.short_lm, question, selected_paragraph)
            if success:
                best_paragraph_id = self.hippocampus.metadata[best_paragraph_index]
                return True, Answer(answer, [], [best_paragraph_id])
        else:
            # if not
            # it then has to check whether the supports are sufficient to answer the question
            success, answer =  try_get_answer(self.short_lm, question, "\n".join([f"{s[0]}: {s[1].text}" for s in supports]))
            if success:
                # merge all support paragraphs
                support_paragraph_ids = []
                for s in supports:
                    if s[1].support_paragraph_indices is not None:
                        support_paragraph_ids.extend(s[1].support_paragraph_indices)
                return True, Answer(answer, [], support_paragraph_ids)

        # if not, it has to break another next sub question
        sub_question = discover_sub_question(self.long_lm, question, "\n".join([f"{s[0]}: {s[1].text}" for s in supports]))
        return False, sub_question