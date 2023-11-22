import numpy as np

def get_comm_node(in_str, rank, system_graph):
    # print(in_str)
    if ('SEND_/' in in_str):
        split_string = in_str.split("/")
        var_id = split_string[1]
        tag_int = name_to_int(var_id)
        target_rank = split_string[3]
        return SendCall(
            origin_rank=rank,
            from_rank=rank,
            to_rank=target_rank,
            var=system_graph.unique_to_node[var_id],
            tag=tag_int,
        )
    elif ('GET_/' in in_str):
        split_string = in_str.split("/")
        var_id= split_string[1]
        tag_int = name_to_int(var_id)
        source_rank = split_string[3]
        return RecvCall(
            origin_rank=rank,
            from_rank=source_rank,
            to_rank=rank,
            var=system_graph.unique_to_node[var_id],
            tag=tag_int,
        )
    else:
        return None
    # elif ('WAIT' in in_str) or ('IRECV' in in_str)  or ('SsENDONLY' in in_str) or ('irecvwait' in in_str) or ('W/F' in in_str):
    #     pass
    
def name_to_int(node_name):
    node_name = node_name.split("_")[0]
    return int(node_name[1:])

class PointToPointCall(object):

    def __init__(
            self,
            origin_rank: int, 
            from_rank: int, 
            to_rank:int ,
            var: str,
            tag: int,
        ):
        """
        arguments:
        ----------
            origin_rank: int
                The rank of the process to write instructions for
            from_rank: int
                The rank of the process that is sending the datasource_rank
            to_rank: int
                The rank of the process that is receiving the data
            var: str
                The name of the variable that is being sent
            tag: int
                The tag of the message
        """
        self.origin_rank = origin_rank
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.var = var
        self.var_id = var.id
        self.tag = tag
    
    def get_block(self, code_block, vars):
        raise NotImplementedError('get_block not implemented')

    def get_adjoint_call(self, code_block, vars, adjoint_path_name, adjoint_shape, adjoint_type):
        raise NotImplementedError('get_adjoint_call not implemented')

class SendCall(PointToPointCall):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.origin_rank != self.from_rank:
            raise ValueError('origin_rank must be the same as from_rank for MPI Send')

    def get_block(self, code_block, vars):
        code_block.write(f'comm.Send({self.var_id}.astype("float64") , dest = {self.to_rank}, tag = {self.tag})')
        
        # Uncomment to print send value
        # code_block.write(f'print(comm.rank,"SEND" ,"{self.var_id}",{self.var_id}, {self.var_id}.dtype)')

    def get_adjoint_call(self, code_block, vars, adjoint_path_name, adjoint_shape, adjoint_type):
        # vars[adjoint_path_name] = np.ones(adjoint_shape)
        # code_block.write(f'comm.Recv({adjoint_path_name}, source = {self.to_rank}, tag = {self.tag})')

        # TODO: this is a hack to send sparse arrays.
        # vars[adjoint_path_name] = np.ones(adjoint_shape)
        code_block.write(f'{adjoint_path_name} = comm.recv(source = {self.to_rank}, tag = {self.tag})')
        # code_block.write(f'print(comm.rank, type({adjoint_path_name}))')

class RecvCall(PointToPointCall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.origin_rank != self.to_rank:
            raise ValueError('origin_rank must be the same as to_rank for MPI Send')
        
    def get_block(self, code_block, vars):
        # vars[self.var_id] = np.ones(self.var.var.shape)
        code_block.write(f'{self.var_id} = np.ones({self.var.var.shape})')
        code_block.write(f'comm.Recv({self.var_id}, source = {self.from_rank}, tag = {self.tag})')
        
        # Uncomment to print recv value
        # code_block.write(f'print(comm.rank,"RECV" ,"{self.var_id}",{self.var_id}, ({self.var_id}.dtype))')

    def get_adjoint_call(self, code_block, vars, adjoint_path_name, adjoint_shape, adjoint_type):
        # code_block.write(f'comm.Send({adjoint_path_name}, dest = {self.from_rank}, tag = {self.tag})')


        # TODO: this is a hack to send sparse arrays.
        code_block.write(f'comm.send({adjoint_path_name}, dest = {self.from_rank}, tag = {self.tag})')
