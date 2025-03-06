from quest_interface import *
from .text_graph import Text_Node, Text_Node_List, Text_Node_Type
from enum import Enum


def agent_function(persona, nodes: Text_Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    focus_node = nodes[0]
    if focus_node.type == Text_Node_Type.Question_Node:
        round_nodes = focus_node.get_children()
        for round_node in round_nodes:
            if not round_node.is_answered():
                return Action.ANSWER, focus_node, round_node
        if len(round_nodes) >= 2:
            # check consistency of the last two
            node_1 = round_nodes[-1]
            node_2 = round_nodes[-2]
            if node_1.get_answer() == node_2.get_answer():
                parents = focus_node.get_parents()
                # if parent is none, it is the root node
                return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, node_1.get_answer()), parents[0] if len(parents) > 0 else None
        return Action.DISCOVER, Text_Node(Text_Node_Type.Round_Node, len(round_nodes), None), Text_Node_List([focus_node])
    else:
        question_nodes = focus_node.get_children()
        for question_node in question_nodes:
            if not question_node.is_answered():
                return Action.ANSWER, focus_node, question_node
        
        # check if all questions are sufficient, if not, return next sub question, if yes return answer
        is_sufficient, detail = persona.compute_answer(focus_node.get_question(), question_nodes.get()) 
        if is_sufficient:
            return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, detail), focus_node.get_parents()[0]
        else:
            return Action.DISCOVER, Text_Node(Text_Node_Type.Question_Node, detail, None), Text_Node_List([focus_node])


