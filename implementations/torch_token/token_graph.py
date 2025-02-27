from uuid import uuid4
from quest_interface import Quest_Graph, Node, Node_List, Direction, Direction_List



class Token_Node(Node, Direction):
    def __init__(self, tokens):
        self.id = uuid4()
        self.tokens = tokens
        self.neighbors = [self]

    def __eq__(self, other):
        return self.id == other.id

    def get_neighbor(self, direction: Direction):
        return direction

    def get_neighbors(self, directions: Direction_List = None) -> Node_List:
        if directions is None:
            # directions None = get all nodes including self [self, neighbor1, neighbor2, ...]
            return self.neighbors
        else:
            return Token_Node_List(directions)
        
    def set(self, other):
        self.tokens = other.tokens

    def attach_to(self, others: Node_List):
        neighbor_ids = [neighbor.id for neighbor in self.neighbors]
        for other in others:
            if other.id not in neighbor_ids:
                self.neighbors.append(other)


class Token_Node_List(Node_List, Direction_List):
    
    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]