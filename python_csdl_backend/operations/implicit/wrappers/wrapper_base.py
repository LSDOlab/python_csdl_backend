

class ImplicitWrapperBase():

    def __init__(self, op, ins, outs):
        pass

    def run():
        """
        evaluates residuals
        """
        pass

    def get_state():
        pass

    def set_input(state_name, val):
        """
        set state to a value
        """
        pass

    def set_state(self, state_name, val):
        """
        set state to a value
        """
        pass

    def get_residual(self, res_name):
        """
        get residual value
        """
        pass

    def get_state(self, state_name):
        """
        get state value (or exposed out)
        """
        pass

    def compute_totals(self):
        """
        computes totals of residuals/exposed to inputs/states
        """
        pass



