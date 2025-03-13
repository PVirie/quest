from quest_interface import *
from .text_graph import Text_Node_List, Question_Node, Search_Node, Thought_Node
from .persona import Sub_Action_Type


class Answer:
    def __init__(self, text, support_paragraph_indices):
        self.text = text
        self.support_paragraph_indices = support_paragraph_indices


def basic_tree(persona, nodes: Text_Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    focus_node = nodes[0]
    child_nodes = focus_node.get_children()
    for child in child_nodes:
        if isinstance(child, Question_Node) and not child.is_answered():
            return Action.ANSWER, None, child
        
    subact, detail = persona.act(focus_node.question, child_nodes.get())
    if len(child_nodes) >= 10:
        return Action.ANSWER, None, focus_node.get_parent()
    elif subact == Sub_Action_Type.Answer:
        return Action.ANSWER, Question_Node(None, detail), focus_node.get_parent()
    elif subact == Sub_Action_Type.Search:
        search_result, paragraph_id = persona.search(detail)
        return Action.DISCOVER, Search_Node(detail, search_result, paragraph_id), Text_Node_List([focus_node])
    elif subact == Sub_Action_Type.Thought:
        return Action.DISCOVER, Thought_Node(detail), Text_Node_List([focus_node])
    elif subact == Sub_Action_Type.Sub_Question:
        return Action.DISCOVER, Question_Node(detail, None), Text_Node_List([focus_node])
    else:
        return Action.ANSWER, None, focus_node.get_parent()
