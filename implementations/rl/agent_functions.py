from quest_interface import *
from .rl_graph import Quest_Node, Observation_Node, Thought_Node
from .persona import Sub_Action_Type


        

def basic_tree(persona, focus_node):
    child_nodes = focus_node.get_children()
    last_child = focus_node.get_last_child()
    parent_node = focus_node.get_parent()

    if isinstance(last_child, Quest_Node) and not last_child.is_fulfilled():
        return Action.ANSWER, None, last_child
    
    subact, node = persona.think(focus_node, child_nodes)
    if subact == Sub_Action_Type.Fulfill:
        return Action.ANSWER, node, parent_node
    elif subact == Sub_Action_Type.Thought:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Relegate:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Act:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Done:
        return Action.ANSWER, node, None
    
    return Action.ANSWER, None, None