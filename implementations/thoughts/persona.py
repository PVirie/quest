from enum import Enum
from utilities.language_models import Language_Model, Chat, Chat_Message
from utilities.contextual_memory import Vector_Text_Dictionary
from .text_graph import Question_Node, Search_Node, Thought_Node


class Sub_Action_Type(Enum):
    Answer = 1
    Search = 2
    Thought = 3
    Sub_Question = 5


class Persona:
    def __init__(self, paragraphs, prompt):
        self.paragraphs = paragraphs
        self.long_lm = Language_Model(max_length=256, top_p=1, temperature=0)
        self.hippocampus = Vector_Text_Dictionary([p.paragraph_text for p in paragraphs], metadata=[p.idx for p in paragraphs])
        self.prompt = prompt


    def think(self, question, supports):
        # supports is a list of sub (question, answer)
        # return sub_act, detail
        # if return Answer, just use all the information to compute the answer
        # if return Search, just compute the search term
        # if return Thought, just contemplate
        # if return Sub_Question, just compute the next sub question

        transcripts = []
        for node in supports:
            if isinstance(node, Question_Node):
                transcripts.append(f"Sub-Question: {node.question}")
                if node.is_answered():
                    transcripts.append(f"Result: {node.answer}")
                else:
                    transcripts.append("Result: Failed to answer")
            elif isinstance(node, Search_Node):
                transcripts.append(f"Search: {node.search_query}")
                transcripts.append(f"Result: {node.search_result}")
            elif isinstance(node, Thought_Node):
                transcripts.append(f"Thought: {node.thought}")


        text_response = self.long_lm.complete_text(self.prompt.format(question=question, transcripts="\n".join(transcripts)))
        # get the first part before newline
        text_response = text_response.split("\n")[0]
        if text_response.startswith("Final Answer:"):
            return Sub_Action_Type.Answer, text_response[13:]
        elif text_response.startswith("Search:"):
            return Sub_Action_Type.Search, text_response[7:]
        elif text_response.startswith("Thought:"):
            return Sub_Action_Type.Thought, text_response[8:]
        elif text_response.startswith("Sub-Question:"):
            return Sub_Action_Type.Sub_Question, text_response[13:]
        
        return None, None


    def search(self, search_term):
        best_paragraph_index = self.hippocampus.match(search_term, k=1)
        selected_paragraph = self.hippocampus.get_paragraph(best_paragraph_index)
        best_paragraph_id = self.hippocampus.metadata[best_paragraph_index]
        return selected_paragraph, best_paragraph_id