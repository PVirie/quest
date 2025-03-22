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
        return [node for node in self.nodes]


class RL_Node(Node, Direction):
    def __init__(self):
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
        pass

    def attach_to(self, others):
        for other in others:
            other.children.append(self)
        self.parent = others[0]

    def get_parent(self):
        return self.parent

    def get_children(self):
        return RL_Node_List(self.children)
    

class Quest_Node(RL_Node):
    def __init__(self, quest=None, result=None):
        super().__init__()
        self.quest = quest
        self.result = result

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.quest is not None:
            self.quest = another.quest
        if another.result is not None:
            self.result = another.result

    def is_fulfilled(self):
        return self.result is not None
    

class Status_Node(RL_Node):
    def __init__(self, status=None, result=None):
        super().__init__()
        self.status = status
        self.result = result

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.status is not None:
            self.status = another.status
        if another.result is not None:
            self.result = another.result


class Thought_Node(RL_Node):
    def __init__(self, thought=None):
        super().__init__()
        self.thought = thought

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.thought is not None:
            self.thought = another.thought


class Observation_Node(RL_Node):
    def __init__(self, action=None, observation=None):
        super().__init__()
        self.action = action
        self.observation = observation

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.action is not None:
            self.action = another.action
        if another.observation is not None:
            self.observation = another.observation