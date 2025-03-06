from quest_interface import *
from .rl_graph import RL_Node, RL_Node_List, RL_Node_Type


def agent_function(persona, nodes: RL_Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    quest_node = nodes[0]
    children = quest_node.get_children()
    for child in children:
        if not child.is_fulfilled():
            # move to child
            return Action.ANSWER, RL_Node(None, None, None), child
    goal_achieved, detail = persona.pursue(quest_node.get_quest(), children.get())
    if goal_achieved:
        # fulfill with detail, and move up
        # detail is the current state, 
        # you have done bla bla (describe action), 
        # and now you are at bla bla (describe the current state)
        # possible next actions are bla bla (list of action)
        parent = quest_node.get_parent()
        return Action.ANSWER, RL_Node(True, None, detail), parent
    else:
        # discover subquest
        # detail is as follow: 
        # now you are at bla bla (describe the current state), 
        # with the following available actions bla bla (list of action), 
        # and you want to bla bla (describe the goal)
        return Action.DISCOVER, RL_Node(False, detail, None), RL_Node_List([quest_node])
    