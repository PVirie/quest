from quest_interface import Node, Node_List, Direction, Direction_List

from enum import Enum


class Text_Node_Type(Enum):
    Round_Node = 0
    Question_Node = 1


class Text_Node_List(Node_List, Direction_List):
    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]
    
    def get(self):
        return [(node.get_question(), node.get_answer()) for node in self.nodes]


class Text_Node(Node, Direction):
    def __init__(self, type, question, answer):
        self.type = type
        self.question = question
        self.answer = answer
        self.parents = None
        self.children = []

    def get_neighbor(self, direction):
        return direction

    def get_neighbors(self, directions = None):
        if directions is None:
            return Text_Node_List([self] + self.parents + self.children)
        else:
            return directions
        
    def set(self, other):
        self.question = other.question
        self.answer = other.answer

    def attach_to(self, others):
        for other in others:
            other.children.append(self)
        self.parents = others

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def is_answered(self):
        return self.answer is not None

    def get_question(self):
        return self.question
    
    def get_answer(self):
        return self.answer

