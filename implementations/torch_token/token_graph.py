from quest_interface import Quest_Graph, Node, Node_List, Direction, Direction_List



class Token_Node(Node, Direction):
    def __init__(self, tokens):
        self.tokens = tokens
        self.neighbors = [self]

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
        # does not have to check for duplicates
        # because this only be called from the discover method
        for other in others:
            self.neighbors.append(other)


class Token_Node_List(Node_List, Direction_List):
    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]
    
    def serialize(self):
        # flatten for LM
        pass