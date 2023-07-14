class Algorithm():

    def __init__(self, **kwargs) -> None:
        self.create_plots = 0
        self.profile = 0

    def set_create_plots(self, cp):
        self.create_plots = cp