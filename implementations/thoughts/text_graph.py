from quest_interface import Node, Node_List, Direction, Direction_List


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
        return [node for node in self.nodes]


class Text_Node(Node, Direction):
    def __init__(self):
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
        pass

    def attach_to(self, others):
        for other in others:
            other.children.append(self)
        self.parent = others[0]

    def get_parent(self):
        return self.parent

    def get_children(self):
        return Text_Node_List(self.children)
    

class Question_Node(Text_Node):
    def __init__(self, question=None, answer=None):
        super().__init__()
        self.question = question
        self.answer = answer

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.question is not None:
            self.question = another.question
        if another.answer is not None:
            self.answer = another.answer

    def is_answered(self):
        return self.answer is not None
    
    def gather_support_ids(self):
        support_ids = set()
        for child in self.children:
            if isinstance(child, Search_Node):
                support_ids.add(child.support_index)
            elif isinstance(child, Question_Node):
                if child.is_answered():
                    support_ids.union(child.gather_support_ids())
        return support_ids


class Search_Node(Text_Node):
    def __init__(self, search_query=None, search_result=None, support_index=None):
        super().__init__()
        self.search_query = search_query
        self.search_result = search_result
        self.support_index = support_index

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.search_query is not None:
            self.search_query = another.search_query
        if another.search_result is not None:
            self.search_result = another.search_result
        if another.support_index is not None:
            self.support_index = another.support_index


class Thought_Node(Text_Node):
    def __init__(self, thought=None):
        super().__init__()
        self.thought = thought

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.thought is not None:
            self.thought = another.thought
