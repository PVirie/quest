from quest_interface import *
from .text_graph import Text_Node, Text_Node_List, Text_Node_Type
from enum import Enum


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
    else:
        return Action.DISCOVER, Text_Node(Text_Node_Type.Question_Node, detail, None), Text_Node_List([focus_node])


def consistent_tree(persona, nodes: Text_Node_List) -> Tuple[Action, Node, Union[Direction, Direction_List]]:
    focus_node = nodes[0]
    if focus_node.type == Text_Node_Type.Question_Node:
        round_nodes = focus_node.get_children()
        for round_node in round_nodes:
            if not round_node.is_answered():
                return Action.ANSWER, focus_node, round_node
        if len(round_nodes) >= 3:
            # check consistency of the last two
            node_1 = round_nodes[-1]
            node_2 = round_nodes[-2]
            if node_1.get_answer() == node_2.get_answer():
                parent = focus_node.get_parent()
                # if parent is none, it is the root node
                return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, node_1.get_answer()), parent
            elif len(round_nodes) >= 5:
                # too long, use voting
                votes = {}
                for round_node in round_nodes:
                    answer = round_node.get_answer()
                    if answer in votes:
                        votes[answer] += 1
                    else:
                        votes[answer] = 1
                max_vote = max(votes, key=votes.get)
                parent = focus_node.get_parent()
                return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, max_vote), parent
        return Action.DISCOVER, Text_Node(Text_Node_Type.Round_Node, len(round_nodes), None), Text_Node_List([focus_node])
    else:
        question_nodes = focus_node.get_children()
        for question_node in question_nodes:
            if not question_node.is_answered():
                return Action.ANSWER, focus_node, question_node
        
        parent = focus_node.get_parent()
        # check if all questions are sufficient, if not, return next sub question, if yes return answer
        is_sufficient, detail = persona.compute_answer(parent.get_question(), question_nodes.get()) 
        if is_sufficient:
            return Action.ANSWER, Text_Node(Text_Node_Type.Question_Node, None, detail), parent
        else:
            return Action.DISCOVER, Text_Node(Text_Node_Type.Question_Node, detail, None), Text_Node_List([focus_node])


