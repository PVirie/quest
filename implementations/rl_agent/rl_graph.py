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
            return self.children[key] if len(self.children) > 0 else None

    def get_context(self):
        raise NotImplementedError("get_context() not implemented")
    

class Trainable:
    def __init__(self, train_ref=None, observation=None):
        self.train_ref = train_ref
        self.observation = observation


class Quest_Node(RL_Node, Trainable):
    def __init__(self, objective=None, start_observation=None, succeeded=None, observation=None, truncated=False, train_ref=None, allow_relegation=True):
        self.objective = objective
        self.start_observation = start_observation
        self.allow_relegation = allow_relegation
        self.succeeded = succeeded
        self.truncated = truncated
        Trainable.__init__(self, train_ref=train_ref, observation=observation)
        RL_Node.__init__(self)

    def get_start_contexts(self):
        return f"Objective: {str(self.objective)}", f"Observation: {self.start_observation.get_context()}"

    def get_context(self):
        contexts = []
        contexts.append(f"Sub Task: {str(self.objective)}")
        if self.succeeded is not None:
            contexts.append(f"Result: {"Succeeded" if self.succeeded else "Failed"}")
        elif self.truncated:
            contexts.append(f"Result: Truncated")
        contexts.append(f"Observation: {self.observation.get_context()}")
        return contexts

    def set(self, another):
        # check same class
        if not isinstance(another, self.__class__):
            return
        if another.succeeded is not None:
            self.succeeded = another.succeeded
        if another.observation is not None:
            self.observation = another.observation
        if another.truncated is not None:
            self.truncated = another.truncated
            
    def is_completed(self):
        return self.succeeded is not None
    
    def is_succeeded(self):
        return self.succeeded is not None and self.succeeded
    
    def last_child_succeeded(self):
        if len(self.children) == 0:
            return True
        last_child = self.children[-1]
        if isinstance(last_child, self.__class__):
            return last_child.is_succeeded()
        else:
            # other type of node, assume succeed
            return True
    
    def eval(self, observation):
        return self.objective.eval(self, observation)

    def count_context_type(self):
        num_observation_node = 0
        num_thought_node = 0
        num_quest_node = 0
        num_succeeded_quest_node = 0
        for child in self.children:
            if isinstance(child, Observation_Node):
                num_observation_node += 1
            elif isinstance(child, Thought_Node):
                num_thought_node += 1
            elif isinstance(child, Quest_Node):
                num_quest_node += 1
                if child.is_succeeded():
                    num_succeeded_quest_node += 1
        
        return num_observation_node, num_thought_node, num_quest_node, num_succeeded_quest_node
    
    def compute_statistics(self):
        num_children = len(self.children)
        num_quest_nodes = 1
        max_children = num_children
        min_children = num_children
        for child in self.children:
            if isinstance(child, self.__class__):
                cc, cq, cm, cn = child.compute_statistics()
                num_children += (cc - 1)
                num_quest_nodes += cq
                if cm > max_children:
                    max_children = cm
                if cn < min_children:
                    min_children = cn
        if num_quest_nodes == 0:
            min_children = 0
        return num_children, num_quest_nodes, max_children, min_children
    

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
        self.action = action
        Trainable.__init__(self, train_ref=train_ref, observation=observation)
        RL_Node.__init__(self)

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