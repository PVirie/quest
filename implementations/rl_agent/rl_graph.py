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
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # if key is a slice, return the children in that range
            start, stop, step = key.start, key.stop, key.step
            return self.children[start:stop:step]
        else:
            # if key is an integer, return the child at that index
            return self.children[key]

    def get_context(self):
        raise NotImplementedError("get_context() not implemented")
    
    def is_success(self):
        return True
    

class Trainable:
    def __init__(self, train_ref=None, observation=None):
        self.train_ref = train_ref
        self.observation = observation


class Quest_Node(RL_Node):
    def __init__(self, objective=None, eval_func=None, start_observation=None, result=None, observation=None, truncated=False, train_ref=None, allow_relegation=True):
        super().__init__()
        self.objective = objective
        self.eval_func = eval_func
        self.start_observation = start_observation
        self.allow_relegation = allow_relegation
        self.result = result
        self.truncated = truncated
        super().__init__(train_ref=train_ref, observation=observation)

    def get_start_contexts(self):
        return f"Objective: {self.objective}", f"Observation: {self.start_observation.get_context()}"

    def get_context(self):
        contexts = []
        contexts.append(f"Sub Task: {self.objective}")
        if self.result is not None:
            contexts.append(f"Result: {"Succeeded" if self.result else "Failed"}")
        elif self.truncated:
            contexts.append(f"Result: Truncated")
        contexts.append(f"Observation: {self.observation.get_context()}")
        return contexts

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.result is not None:
            self.result = another.result
        if another.observation is not None:
            self.observation = another.observation
        if another.truncated is not None:
            self.truncated = another.truncated
            
    def is_completed(self):
        return self.result is not None or self.truncated
    
    def is_success(self):
        return self.result is not None and self.result

    def eval(self):
        return self.eval_func(self)

    def context_length(self):
        num_children = len(self.children)
        return num_children
    
    def compute_statistics(self):
        num_children = len(self.children)
        num_quest_node = 1 if num_children > 0 else 0
        max_children = num_children
        min_children = num_children
        for child in self.children:
            if isinstance(child, self.__class__):
                cc, cn, cm, cn = child.compute_statistics()
                num_children += cc
                num_quest_node += cn
                if cm > max_children:
                    max_children = cm
                if cn < min_children:
                    min_children = cn
        if num_quest_node == 0:
            min_children = 0
        return num_children, num_quest_node, max_children, min_children
    

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


class Observation_Node(RL_Node, Trainable):
    def __init__(self, action=None, observation=None, train_ref=None):
        super().__init__()
        self.action = action
        super().__init__(train_ref=train_ref, observation=observation)

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