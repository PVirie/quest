from quest_interface import *
from .rl_graph import Quest_Node, Observation_Node


def basic_tree(persona, focus_node):
    child_nodes = focus_node.get_children()
    last_child = focus_node.get_last_child()
    parent_node = focus_node.get_parent()

    if isinstance(last_child, Quest_Node) and not last_child.is_fulfilled():
        return Action.ANSWER, None, last_child
    
    is_fulfilled, response = persona.summarize(focus_node, child_nodes)
    if is_fulfilled:
        return Action.ANSWER, Quest_Node(None, response), parent_node

    is_compound_action, request_action = persona.think(focus_node, child_nodes)
    if is_compound_action:
        return Action.DISCOVER, Quest_Node(request_action, None), focus_node
    else:
        is_env_done, observation = persona.act(request_action)
        if is_env_done:
            # Let the agent know the game is done.
            is_compound_action, request_action = persona.think(focus_node, child_nodes + [Observation_Node(None, observation)])
            return Action.ANSWER, None, parent_node
        else:
            return Action.DISCOVER, Observation_Node(request_action, observation), focus_node
