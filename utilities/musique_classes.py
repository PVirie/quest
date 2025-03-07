from typing import List

class Serializable:
    def to_json(self):
        # recursively serialize all attributes
        return {k: v.to_json() if isinstance(v, Serializable) else v for k, v in vars(self).items()}


class Paragraph(Serializable):
    def __init__(self, json_obj):
        self.idx = json_obj["idx"]
        self.title = json_obj["title"]
        self.paragraph_text = json_obj["paragraph_text"]
        if "is_supporting" in json_obj:
            self.is_supporting = json_obj["is_supporting"]
        else:
            self.is_supporting = None


class Support(Serializable):
    def __init__(self, json_obj):
        self.id = json_obj["id"]
        self.question = json_obj["question"]
        self.answer = json_obj["answer"]
        self.paragraph_support_idx = json_obj["paragraph_support_idx"]


class Record(Serializable):
    def __init__(self, id):
        self.id = id


# {"id": "2hop__640262_122868", "question": "...", "paragraphs": [{"idx": 19, "title": "Chevrolet Camaro", "paragraph_text": "The Chevrolet..."}], "question_decomposition": [{"id": 640262, "question": "1967: The Last Good Year >> author", "answer": "Pierre Berton", "paragraph_support_idx": 0}, {"id": 122868, "question": "What is the university where #1 went?", "answer": "University of British Columbia", "paragraph_support_idx": 2}], "answer": "University of British Columbia", "answer_aliases": ["The University of British Columbia"], "answerable": true}
class Question_Record(Record):
    def __init__(self, json_obj):
        super().__init__(json_obj["id"])
        self.paragraphs = [Paragraph(paragraph) for paragraph in json_obj["paragraphs"]]
        self.question = json_obj["question"]

        if "answer" in json_obj:
            self.question_decomposition = [Support(support) for support in json_obj["question_decomposition"]]
            self.answer = json_obj["answer"]
            self.answer_aliases = json_obj["answer_aliases"]
            self.answerable = json_obj["answerable"]
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