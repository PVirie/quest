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
    
    def __len__(self):
        return len(self.nodes)

    def get(self):
        return [(node.get_question(), node.get_answer()) for node in self.nodes]


class Text_Node(Node, Direction):
    def __init__(self, type, question, answer):
        self.type = type
        self.question = question
        self.answer = answer
        self.parent = None
        self.children = []

    def get_neighbor(self, direction):
        return direction

    def get_neighbors(self, directions = None):
        if directions is None:
            return Text_Node_List([self] + ([self.parent] if self.parent is not None else []) + self.children)
        else:
            return directions
        
    def set(self, other):
        self.question = other.question
        self.answer = other.answer

    def attach_to(self, others):
        for other in others:
            other.children.append(self)
        self.parent = others[0]

    def get_parent(self):
        return self.parent

    def get_children(self):
        return Text_Node_List(self.children)

    def is_answered(self):
        return self.answer is not None

    def get_question(self):
        return self.question
    
    def get_answer(self):
        return self.answer

    def print(self, tab=0):
        tabs = ""
        for i in range(tab):
            tabs += "\t"
        children_text = ("\n" + tabs).join([child.print(tab=tab+1) for child in self.children])
        node_text = f"{tabs}Q: {self.question}, A: {self.answer}\n{tabs}{children_text}"
        return node_text