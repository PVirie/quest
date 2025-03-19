from quest_interface import *
from .rl_graph import RL_Node_List, Quest_Node, Status_Node, Thought_Node, Observation_Node
from .persona import Sub_Action_Type


def basic_tree(persona, nodes: Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    focus_node = nodes[0]
    child_nodes = focus_node.get_children()
    for child in child_nodes:
        if isinstance(child, Quest_Node) and not child.is_fulfilled():
            return Action.ANSWER, None, child
        
    subact, detail = persona.think(focus_node.question, child_nodes.get())
    if len(child_nodes) >= 20:
        return Action.ANSWER, None, focus_node.get_parent()
    elif subact == Sub_Action_Type.Fulfill:
        return Action.ANSWER, Quest_Node(None, detail), focus_node.get_parent()
    elif subact == Sub_Action_Type.Check_Status:
        result = persona.check_status(detail)
        return Action.DISCOVER, Status_Node(detail, result), RL_Node_List([focus_node])
    elif subact == Sub_Action_Type.Thought:
        return Action.DISCOVER, Thought_Node(detail), RL_Node_List([focus_node])
    elif subact == Sub_Action_Type.Act:
        observation = persona.act(detail)
        return Action.DISCOVER, Observation_Node(detail, observation), RL_Node_List([focus_node])
    elif subact == Sub_Action_Type.Relegate:
        return Action.DISCOVER, Quest_Node(detail, None), RL_Node_List([focus_node])
    else:
        return Action.ANSWER, None, focus_node.get_parent()
