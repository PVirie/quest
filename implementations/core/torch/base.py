

class Q_Table:

    def forward(self, observations, actions, **kwargs):
        raise NotImplementedError

    def evaluate_state(self, observations, **kwargs):
        raise NotImplementedError
    
    def evaluate_actions(self, actions, **kwargs):
        raise NotImplementedError