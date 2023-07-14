import numpy as np
class StateManager():

    def __init__(
            self,
            full_rank_owner_mapping: dict(),
            rank_owner_mapping: dict(),
            comm,
        ):

        """
        arguments:
        ----------
            full_rank_owner_mapping: dict()
            -   A dictionary mapping a variable to a set of ranks that contain the variable
            comm:
            -   The MPI communicator

        This object manages all states of a graph and does not allow unnecessary allocation of variables.
        """

        self.state_values = {}
        self.state_shape_mapping = {}
        self.rank_owner_mapping_full = full_rank_owner_mapping
        self.rank_owner_mapping = rank_owner_mapping

        self.comm = comm

    def reserve_state(self, id, shape):
        self.state_values[id] = None
        self.state_shape_mapping[id] = shape

    def __getitem__(self, id):
        if self.comm is None:
            return self.state_values[id]

        if id not in self.rank_owner_mapping:
            return self.state_values[id]

        if self.comm.size == 1:
            return self.state_values[id]
        else:
            owner_rank = self.rank_owner_mapping[id]
            var = self.comm.bcast(self.state_values[id], root = owner_rank)
            return var

    def __setitem__(self, id, val):
        self.check_id(id)
        if self.comm is None:
            self.state_values[id] = val
            return

        if id not in self.rank_owner_mapping:
            self.state_values[id] = val
            return

        if self.comm.rank in self.rank_owner_mapping_full[id]:
            self.state_values[id] = val
            return
 
    # def empty_id(self, id):
    #     self.check_id(id)
    #     self.state_values[id] = False

    def check_id(self, id):
        if id not in self.state_values:
            raise ValueError(f'Variable {id} not found in state manager.')
