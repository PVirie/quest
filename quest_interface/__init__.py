from typing import Union, Tuple


class Direction:
    pass


class Direction_List:
    pass


class Node_List:
    
    def __iter__(self):
        pass

    def __getitem__(self, index):
        pass


class Node:

    def __eq__(self, other):
        pass

    def get_neighbor(self, direction: Direction):
        pass

    def get_neighbors(self, directions: Direction_List = None) -> Node_List:
        # directions None = get all nodes including self [self, neighbor1, neighbor2, ...]
        pass

    def set(self, other):
        pass

    def attach_to(self, others: Node_List):
        pass



class Quest_Graph:
    def __init__(self, start_node: Node = None, start_graph = None):
        if start_node is None and start_graph is None:
            raise ValueError("Either start_node or start_graph must be provided")
        elif start_node is not None and start_graph is not None:
            raise ValueError("Only one of start_node or start_graph must be provided")
        elif start_node is not None:
            self.root = start_node
        else:
            self.root = start_graph.root

        self.focus = {
            0: self.root
        }


    def set_focus_node(self, node: Node, head_index=0):
        # check whether node is in the graph
        # if not, raise an error
        stack = []
        stack.append(self.root)
        while stack:
            current_node = stack.pop()
            if current_node == node:
                break
            else:
                stack.extend(list(current_node.get_neighbors()))
        else:
            raise ValueError("Node not found in the graph")

        self.focus[head_index] = node


    def query(self, head_index=0) -> Node_List:
        current_focus_node = self.focus[head_index]
        neighbors = current_focus_node.get_neighbors()
        return neighbors

    
    # action 0
    def discover(self, node: Node, directions: Direction_List, head_index=0):
        current_focus_node = self.focus[head_index]
        neighbors = current_focus_node.get_neighbors(directions)
        node.attach_to(neighbors)
    
    # action 1
    def respond(self, node: Node, direction: Direction, head_index=0):
        current_focus_node = self.focus[head_index]
        current_focus_node.set(node)
        neighbor = current_focus_node.get_neighbor(direction)
        self.focus[head_index] = neighbor



def agent_function(nodes: Node_List) -> Tuple[int, Node, Union[Direction, Direction_List]]:
    pass



