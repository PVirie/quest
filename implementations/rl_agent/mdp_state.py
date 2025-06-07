

class MDP_Transition:
    """
    Represents a change in the state of an MDP.
    """

    def eval(self, state):
        """
        Evaluates the transition from a given state.
        return mdp_score, terminated, truncated, succeeded, override_objective
        """
        pass


    def __str__(self):
        """
        Returns a string representation of the transition.
        """
        pass
    

    def __lt__(self, other):
        """
        Compares this transition with another one.
        """
        pass


    def applicable_from(self, state):
        """
        Checks if the transition is applicable from a given state.
        """
        pass



class MDP_State:

    def __sub__(self, other):
        """
        Subtracts the state of another MDP_State from this one.
        """
        pass


    def get_available_actions(self):
        """
        Returns the available actions in this state.
        """
        pass


    def get_context(self):
        """
        Returns the context of this state.
        """
        pass