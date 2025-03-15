from typing import List

class Serializable:
    def to_json(self):
        # recursively serialize all attributes
        return {k: v.to_json() if isinstance(v, Serializable) else v for k, v in vars(self).items()}


class Paragraph(Serializable):
    def __init__(self, idx: int, title: str, paragraph_text: str, is_supporting: bool = None):
        self.idx = idx
        self.title = title
        self.paragraph_text = paragraph_text
        if is_supporting is not None:
            self.is_supporting = is_supporting
        else:
            self.is_supporting = None


class Support(Serializable):
    def __init__(self, id: int, question: str, answer: str, paragraph_support_idx: int):
        self.id = id
        self.question = question
        self.answer = answer
        self.paragraph_support_idx = paragraph_support_idx


class Record(Serializable):
    def __init__(self, id: str):
        self.id = id


# {"id": "2hop__640262_122868", "question": "...", "paragraphs": [{"idx": 19, "title": "Chevrolet Camaro", "paragraph_text": "The Chevrolet..."}], "question_decomposition": [{"id": 640262, "question": "1967: The Last Good Year >> author", "answer": "Pierre Berton", "paragraph_support_idx": 0}, {"id": 122868, "question": "What is the university where #1 went?", "answer": "University of British Columbia", "paragraph_support_idx": 2}], "answer": "University of British Columbia", "answer_aliases": ["The University of British Columbia"], "answerable": true}
class Question_Record(Record):
    def __init__(self, id: str, question: str, paragraphs: List[dict], answer: str = None, answer_aliases: List[str] = None, answerable: bool = None, question_decomposition: List[dict] = None):
        super().__init__(id)
        self.paragraphs = [Paragraph(**paragraph) for paragraph in paragraphs]
        self.question = question

        if answer is not None:
            self.question_decomposition = [Support(**support) for support in question_decomposition]
            self.answer = answer
            self.answer_aliases = answer_aliases
            self.answerable = answerable
        else:
            self.answer = None
            self.question_decomposition = None
            self.answer_aliases = None
            self.answerable = None



# {"id": "2hop__635187_861533", "predicted_answer": "Cabo Delgado Province", "predicted_support_idxs": [16, 0], "predicted_answerable": false}
class Answer_Record(Record):
    def __init__(self, id: str, predicted_answer: str, predicted_support_idxs: List[int], predicted_answerable: bool):
        self.id = id
        self.predicted_answer = predicted_answer
        self.predicted_support_idxs = predicted_support_idxs
        self.predicted_answerable = predicted_answerable