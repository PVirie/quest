from utilities.language_models import Language_Model, Chat, Chat_Message
from utilities.vector_dictionary import Vector_Text_Dictionary

class Answer:
    def __init__(self, text, support_paragraph_indices):
        self.text = text
        self.support_paragraph_indices = support_paragraph_indices


def try_get_answer(lm, question, paragraph):

    prompt = f"""
    You are an AI that determines if there is enough information to answer a given question and answer it.
    Based on the provided information ONLY, decide whether the information is enough to answer. Do not use your own knowledge.
    
    If the information is not enough, respond with exactly: "no".
    If the information is enough, respond the answer the question using the provided information only.
    Keep your response short. Do not repeat the question.
    
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
        supports = f"""Pre-existing sub-questions:
        {supports}

        What is the next sub-question to ask?
        """

    prompt = f"""
    You are an NLP expert that determines whether the given text question can be further broken down into sub-questions?
    Based on the question and optionally pre-existing sub-questions, 
    determine whether any more sub-question can be asked in order to find the step-by-step answer to the original question.

    If yes, please provide the sub-question. Keep your response short. Do not repeat the question.
    If no, respond with exactly: "no".

    Question:
    {question}
    
    {str(supports)}
    """

    chat = Chat()
    chat.append(Chat_Message(role="system", content="""You determine if the provided question can be broken down into sub-questions.
                                        If so, response the next sub-question only; otherwise, respond with 'no'."""))
    chat.append(Chat_Message(role="user", content=prompt))
    
    text_response = lm.complete_chat(chat)
    if text_response[:2].lower().strip() == "no":
        return False, None

    return True, text_response


class Persona:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.short_lm = Language_Model(max_length=200, top_p=1, temperature=0)
        self.long_lm = Language_Model(max_length=1024, top_p=1, temperature=0)
        self.hippocampus = Vector_Text_Dictionary([p.paragraph_text for p in paragraphs], metadata=[p.idx for p in paragraphs])


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
                return True, Answer(answer, [best_paragraph_id])
        else:
            # if not
            # it then has to check whether the supports are sufficient to answer the question
            success, answer =  try_get_answer(self.short_lm, question, "\n".join([f"Q:{s[0]} A:{s[1].text}" for s in supports]))
            if success:
                # merge all support paragraphs
                support_paragraph_ids = []
                for s in supports:
                    if s[1].support_paragraph_indices is not None:
                        support_paragraph_ids.extend(s[1].support_paragraph_indices)
                return True, Answer(answer, support_paragraph_ids)

        # if not, it has to break another next sub question
        success, sub_question = discover_sub_question(self.long_lm, question, "\n".join([f"Q:{s[0]} A:{s[1].text}" for s in supports]))
        if success:
            return False, sub_question
        else:
            # cannot break sub question
            return True, Answer("Cannot find answer", [])