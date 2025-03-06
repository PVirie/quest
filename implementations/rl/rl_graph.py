from quest_interface import Node, Node_List, Direction, Direction_List


class RL_Node_List(Node_List, Direction_List):
    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]
    
    def __len__(self):
        return len(self.nodes)

    def get(self):
        return [(node.get_quest(), node.get_response()) for node in self.nodes]
    


class RL_Node(Node, Direction):
    def __init__(self, is_fulfilled, quest, response):
        self.is_fulfilled = is_fulfilled
        self.quest = quest
        self.response = response
        self.parent = None
        self.children = []

    def get_neighbor(self, direction):
        return direction

    def get_neighbors(self, directions = None):
        if directions is None:
            return RL_Node_List([self] + ([self.parent] if self.parent is not None else []) + self.children)
        else:
            return directions
        
    def set(self, other):
        self.is_fulfilled = other.is_fulfilled
        self.quest = other.quest
        self.response = other.response

    def attach_to(self, others):
        for other in others:
            other.children.append(self)
        self.parent = others[0]

    def get_parent(self):
        return self.parent

    def get_children(self):
        return RL_Node_List(self.children)

    def is_fulfilled(self):
        return self.is_fulfilled
    
    def get_quest(self):
        return self.quest
    
    def get_response(self):
        return self.response


