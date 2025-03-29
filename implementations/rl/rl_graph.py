from quest_interface import Node, Node_List, Direction, Direction_List


class RL_Node(Node, Direction, Node_List, Direction_List):
    def __init__(self):
        self.parent = None
        self.children = []

    def get_neighbor(self, direction):
        return direction

    def get_neighbors(self, directions = None):
        if directions is None:
            return self
        else:
            return directions
        
    def set(self, other):
        pass

    def attach_to(self, others):
        others.children.append(self)
        self.parent = others

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children
    
    def get_last_child(self):
        if len(self.children) == 0:
            return None
        return self.children[-1]
    

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