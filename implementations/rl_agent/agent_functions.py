from quest_interface import *
from .rl_graph import Quest_Node, Observation_Node, Thought_Node
from .persona import Sub_Action_Type


        

def basic_tree(persona, focus_node):
    last_child = focus_node[-1]
    parent_node = focus_node.get_parent()

    if isinstance(last_child, Quest_Node) and not last_child.is_completed():
        return Action.ANSWER, None, last_child
    
    subact, node = persona.think(focus_node)
    if subact == Sub_Action_Type.Fulfill:
        return Action.ANSWER, node, parent_node
    elif subact == Sub_Action_Type.Thought:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Relegate:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Act:
        return Action.DISCOVER, node, focus_node
    elif subact == Sub_Action_Type.Done:
        return Action.ANSWER, node, parent_node
    
    return Action.ANSWER, None, None