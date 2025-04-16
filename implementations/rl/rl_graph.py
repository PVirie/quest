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
    
    def get_context(self):
        raise NotImplementedError("get_context() not implemented")
    

class Quest_Node(RL_Node):
    def __init__(self, objective=None, eval_func=None, start_observation=None, result=None, end_observation=None, truncated=False, train_ref=None):
        super().__init__()
        self.objective = objective
        self.eval_func = eval_func
        self.start_observation = start_observation
        self.result = result
        self.end_observation = end_observation
        self.truncated = truncated
        self.train_ref = train_ref

    def get_start_contexts(self):
        obs, _, _, _ = self.start_observation
        return f"Objective: {self.objective}", f"Observation: {obs}"

    def get_context(self):
        contexts = []
        contexts.append(f"Sub Task: {self.objective}")
        if self.result is not None:
            contexts.append(f"Result: {self.result}")
        elif self.truncated:
            contexts.append(f"Result: Truncated")
        obs, _, _, _ = self.end_observation
        contexts.append(f"Observation: {obs}")
        return contexts

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.result is not None:
            self.result = another.result
        if another.end_observation is not None:
            self.end_observation = another.end_observation
        if another.truncated is not None:
            self.truncated = another.truncated
            
    def is_completed(self):
        return self.result is not None or self.truncated
    
    def eval(self, obs):
        truncated = False
        if len(self.children) > 0:
            last_child = self.get_last_child()
            if isinstance(last_child, self.__class__):
                if last_child.truncated:
                    truncated = True
        return self.eval_func(self, obs, truncated)
    
    def size(self):
        num_children = len(self.children)
        for child in self.children:
            if isinstance(child, self.__class__):
                num_children += len(child)
        return num_children
    

class Thought_Node(RL_Node):
    def __init__(self, thought=None):
        super().__init__()
        self.thought = thought

    def get_context(self):
        contexts = [f"Thought: {self.thought}"]
        return contexts
    
    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.thought is not None:
            self.thought = another.thought


class Observation_Node(RL_Node):
    def __init__(self, action=None, observation=None, train_ref=None):
        super().__init__()
        self.action = action
        self.observation = observation
        self.train_ref = train_ref

    def get_context(self):
        contexts = []
        contexts.append(f"Action: {self.action}")
        obs, _, _, _ = self.observation
        contexts.append(f"Observation: {obs}")
        return contexts

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.action is not None:
            self.action = another.action
        if another.observation is not None:
            self.observation = another.observation