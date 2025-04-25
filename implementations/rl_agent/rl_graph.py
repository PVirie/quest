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
        return f"Objective: {self.objective}", f"Observation: {self.start_observation.get_context()}"

    def get_context(self):
        contexts = []
        contexts.append(f"Sub Task: {self.objective}")
        if self.result is not None:
            contexts.append(f"Result: {"Succeeded" if self.result else "Failed"}")
        elif self.truncated:
            contexts.append(f"Result: Truncated")
        contexts.append(f"Observation: {self.end_observation.get_context()}")
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
        return self.eval_func(self, obs)
    
    def context_length(self):
        # does not count Quest_Node type
        num_children = len(self.children)
        return num_children
    
    def total_context_length(self):
        num_children = len(self.children)
        num_quest_node = 1 if num_children > 0 else 0
        for child in self.children:
            if isinstance(child, self.__class__):
                cc, cn = child.total_context_length()
                num_children += cc
                num_quest_node += cn
        return num_children, num_quest_node
    
    def max_context_length(self):
        max_children = len(self.children)
        for child in self.children:
            if isinstance(child, self.__class__):
                cc = child.max_context_length()
                if cc > max_children:
                    max_children = cc
        return max_children
    

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
        contexts.append(f"Observation: {self.observation.get_context()}")
        return contexts

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.action is not None:
            self.action = another.action
        if another.observation is not None:
            self.observation = another.observation