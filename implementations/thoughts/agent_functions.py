from quest_interface import *
from .text_graph import Text_Node, Text_Node_List, Text_Node_Type
from enum import Enum
from .persona import Answer


def basic_tree(persona, nodes: Text_Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    focus_node = nodes[0]
    question_nodes = focus_node.get_children()
    for question_node in question_nodes:
        if not question_node.is_answered():
            return Action.ANSWER, focus_node, question_node
    # check if all questions are sufficient, if not, return next sub question, if yes return answer
    is_sufficient, detail = persona.compute_answer(focus_node.get_question(), question_nodes.get()) 
    if is_sufficient:
        return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, detail), focus_node.get_parent()
    elif len(question_nodes) <= 5:
        return Action.DISCOVER, Text_Node(Text_Node_Type.Question_Node, detail, None), Text_Node_List([focus_node])
    else:
        return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, Answer("Cannot find answer.", [])), focus_node.get_parent()

