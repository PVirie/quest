from quest_interface import *
from .rl_graph import Quest_Node, Observation_Node, Thought_Node
from .persona import Sub_Action_Type


def basic_tree(persona, focus_node):
    child_nodes = focus_node.get_children()
    last_child = focus_node.get_last_child()
    parent_node = focus_node.get_parent()
    observation = "End of the game."

    if isinstance(last_child, Quest_Node) and not last_child.is_fulfilled():
        return Action.ANSWER, None, last_child
    
    subact, details = persona.think(focus_node, child_nodes)
    if subact == Sub_Action_Type.Fulfill:
        return Action.ANSWER, Quest_Node(None, details), parent_node
    elif subact == Sub_Action_Type.Thought:
        return Action.DISCOVER, Thought_Node(details), focus_node
    elif subact == Sub_Action_Type.Relegate:
        return Action.DISCOVER, Quest_Node(details, None), focus_node
    elif subact == Sub_Action_Type.Act:
        is_env_done, observation = persona.act(details)
        if not is_env_done:
            return Action.DISCOVER, Observation_Node(details, observation), focus_node
        
    # Let the agent know the game is done.
    persona.think(focus_node, child_nodes + [Observation_Node(details, observation)])
    return Action.ANSWER, None, parent_node