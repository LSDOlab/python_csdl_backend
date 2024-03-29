# from networkx import nx
import networkx as nx
import numpy as np
import scipy.sparse as sp

from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.general_utils import format_print_number, get_deriv_name, to_unique_list, get_path_name, increment_id, lineup_string, get_reverse_seed, get_path_name_vjp
from python_csdl_backend.core.operation_map import (
    get_backend_op,
    get_backend_implicit_op,
    get_backend_custom_explicit_op,
    get_backend_custom_implicit_op,
)
from python_csdl_backend.core.accumulation_operations import diag_mult, std_mult
from python_csdl_backend.core.state_manager import StateManager 
from python_csdl_backend.operations.parallel.point_to_point import PointToPointCall, SendCall, RecvCall
from python_csdl_backend.core.instructions import SingleInstruction, MultiInstructions

from csdl import Operation, StandardOperation, ImplicitOperation, CustomExplicitOperation, Variable, Output, BracketedSearchOperation, CustomImplicitOperation, DeclaredVariable
from csdl.operations.solve_linear import SolveLinear
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.variable import Variable
from csdl.lang.concatenation import Concatenation


class SystemGraph(object):
    """

    The graph object contains a networkx graph to work with.
    Given a graph, return codeblock objects for model evaluation and derivatives.
    When iterating through each node, if the node is another graph object, recursively call the function

    """
    # @profile
    def __init__(self,
                 rep,
                 mode,
                 sparsity_type='auto',
                 dvs=None,
                 objective=None,
                 constraints=None,
                 opt_bool=False,
                 name = ''):

        # representation object processed
        self.rep = rep
        self.eval_graph = self.rep.flat_graph
        self.dvs = dvs
        self.objective = objective
        self.constraints = constraints
        self.opt_bool = opt_bool
        self.num_ops = 0
        self.all_implicit_operations = set()  # set of all implicit operations
        self.all_state_ids_to_guess = {}  # maps state ids to the names of the initial guess
        self.permanently_allocated_vars = set() # Set of VariableNodes that should always be kept in memory. (snapshots, contraints, objectives, visualization, etc.)
        self.save_all_outputs = False # ONly applies to checkpointing. If save_all_outputs, all registered outputs will be allocated permanetely.
        self.name = name
        self.process_rep()

        # rev or fwd
        self.mode = mode

        # sparsity type
        # if name == 'auto' ,   automatically use dense or sparse jacobians
        # if name == 'sparse' , use sparse jacobians
        # if name == 'dense' ,  use dense jacobians
        self.sparsity_type = sparsity_type

        # self.name = 'default'

        if mode == 'rev':
            self.reverse_graph()

    def update_permanent_vars(self, var_nodes):
        """
        Updates the set of permanently allocated variables.
        """
        var_node_list = to_unique_list(var_nodes)
        for var_node in var_node_list:
            self.permanently_allocated_vars.add(var_node)

    def process_rep(self):
        """
        Perform some extra post-processing on the representation object.
        """
        # nomenclature:
        # AV = All Variables (merged)
        # UV = User Variables (merged)

        # Initialize unique mappings for all variables.
        # maps: unique_name (# AL) --> variable node
        self.unique_to_node = {}
        unique_id_num = 'v0000000'

        # Initialize promoted name mappings for user-named variables.
        # maps: promoted name (# UV) --> unique name
        self.promoted_to_unique = {}

        # Initializeu promoted name mappings for user-named variables.
        # maps: unpromoted name (# UV) --> unique name
        self.unpromoted_to_unique = {}

        # pull promoted to unpromoted from rep
        self.promoted_to_unpromoted = self.rep.promoted_to_unpromoted
        self.unpromoted_to_promoted = self.rep.unpromoted_to_promoted

        # UNCOMMENT for potential linear preaccumulation tests 
        # combinable = 0
        # nn = 0
        # nscalars = 0
        for node in self.eval_graph.nodes:

            if isinstance(node, VariableNode):
                # increment id
                # unique_id_num = increment_id(unique_id_num)

                id_to_use = node.var.unique_id_num
                for other_node in node.connected_to:
                    if id_to_use < other_node.var.unique_id_num:
                        id_to_use = other_node.var.unique_id_num
                for other_node in node.declared_to:
                    if id_to_use < other_node.var.unique_id_num:
                        id_to_use = other_node.var.unique_id_num
                unique_id_num = 'v'+str(id_to_use)

                # set unique id
                # this is the variable name used in the generated script
                # unique_id = unique_id_num+'_'+node.name
                unique_id = (unique_id_num+'_'+node.name).replace(',','_').replace(' ','')

                node.id = unique_id
                node.ids = {unique_id}
            
                # UNCOMMENT for potential linear preaccumulation tests 
                # nn += 1                
                # all_linear_after = False
                # s_sizes = []
                # for pred in self.eval_graph.predecessors(node):
                #     if pred.op.properties['linear']:
                #         all_linear_after = True

                #         if self.eval_graph.out_degree(pred) == 0:
                #             all_linear_after = False
                #             break

                #         for successors in self.eval_graph.successors(node):
                #             s_sizes.append(str(successors.op.properties['elementwise'])[0])
                #             for ss in self.eval_graph.successors(successors):
                #                 s_sizes.append(np.prod(ss.var.shape))
                #             if isinstance(successors, OperationNode):
                #                 if not successors.op.properties['linear']:
                #                     all_linear_after = False
                #                     break
                
                # if all_linear_after:
                #     combinable+=1
                #     nscalars += node.var.size
                #     print(f'LINEAR PREACCUMULATION: {combinable}/{nn} ', nscalars, s_sizes, unique_id)
                # print(all_linear_after)

                other_ids = []
                for var in node.connected_to:
                    # if hasattr(var, 'id'):
                    #     if unique_id != var.id:
                    #         print('mismatch')
                    #         raise ValueError('mismatch')
                    var.id = unique_id
                    node.ids.add(unique_id)
                    other_ids.append(var.var.unique_id_num)
                for var in node.declared_to:
                    # if hasattr(var, 'id'):
                    #     if unique_id != var.id:
                    #         print('mismatch')
                    #         raise ValueError('mismatch')
                    var.id = unique_id
                    node.ids.add(unique_id)
                    other_ids.append(var.var.unique_id_num)

                
                # UNCOMMENT TO DEBUG PARALLELIZATION 
                # from mpi4py import MPI
                # comm = MPI.COMM_WORLD
                # rank = comm.rank
                # print(self.name, rank, unique_id_num, node.name, other_ids)

                if unique_id in self.unique_to_node:
                    raise ValueError(f'Unique id {unique_id} already exists in graph.')
                self.unique_to_node[unique_id] = node

                # get promoted name if possible
                promoted_id = prepend_namespace(
                    node.namespace,
                    node.name,
                )
                node.promoted_id = promoted_id

                # if not auto-variable
                if promoted_id in self.rep.promoted_to_node:
                    # Set mapping for promoted to unique
                    self.promoted_to_unique[promoted_id] = unique_id

                    # Set mapping for unpromoted to unique
                    unpromoted_id_set = self.promoted_to_unpromoted[promoted_id]

                    for unpromoted_id in unpromoted_id_set:
                        self.unpromoted_to_unique[unpromoted_id] = unique_id

                # add all declared target names in mappings
                # if node.name == 'geometry_control_points':
                #     for tgt_node in node.declared_to:
                #         print(tgt_node.name)
                for tgt_node in (node.declared_to | node.connected_to):
                    # get declared var promoted name if possible
                    tgt_promoted_id = prepend_namespace(
                        tgt_node.namespace,
                        tgt_node.name,
                    )

                    if tgt_promoted_id in self.rep.promoted_to_unpromoted:
                        # Set mapping for tgt promoted to unique
                        self.promoted_to_unique[tgt_promoted_id] = unique_id
                        # Set mapping for tgt unpromoted to unique
                        tgt_unpromoted_id_set = self.promoted_to_unpromoted[tgt_promoted_id]

                        for unpromoted_id in tgt_unpromoted_id_set:
                            self.unpromoted_to_unique[unpromoted_id] = unique_id

                # check dvs, constraints, objectives
                if self.opt_bool:
                    if promoted_id in self.dvs:
                        self.dvs[promoted_id]['node'] = node
                    if promoted_id == self.objective['name']:
                        self.objective['node'] = node
                    if promoted_id in self.constraints:
                        self.constraints[promoted_id]['node'] = node

            elif isinstance(node, OperationNode):
                self.num_ops += 1
                op = node.op

                if len(op.dependencies) != len(list(self.eval_graph.predecessors(node))):

                    error_str = f'This backend does not currently support operations with multiple inputs of the same variable. Operation {op} in namespace \'{node.namespace}\' has inputs: '
                    for same_input in self.eval_graph.predecessors(node):
                        error_str += f'\'{same_input.name}\', '

                    error_str += '.\nThis can be avoided by multiplying the repeated input by 1. i.e operation(x,x) --> operation(x,x*1)'
                    raise ValueError(error_str)

        # Checks:
        # if len(self.promoted_to_unpromoted) != len(self.promoted_to_unique):
        #     raise NotImplementedError(f'dev error: size mismatch. {len(self.promoted_to_unpromoted)} != {len(self.promoted_to_unique)}')
        # if len(self.unpromoted_to_promoted) != len(self.unpromoted_to_unique):
        #     raise NotImplementedError(f'dev error: size mismatch. {len(self.unpromoted_to_promoted)} != {len(self.unpromoted_to_unique)}')

    def reverse_graph(self):
        # if 'reverse mode', need reverse graph
        self.rev_graph = self.eval_graph.reverse(copy=False)

    def get_analytics(self, name):
        '''
        get and print information about the graph
        '''

        import os
        name_prepend = ''
        if name != '':
            name_prepend = f'_{name}'
        directory_name = f'MODEL_SUMMARY{name_prepend}'
        graph_file_name = f'{directory_name}/graph.txt'
        vars_file_name = f'{directory_name}/variables.txt'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        # if name == '':
        #     filename_full = 'SUMMARY_GRAPH.txt'
        #     filename_all_vars = 'SUMMARY_GRAPH_VARS.txt'
        # else:
        #     filename_full = f'SUMMARY_GRAPH_{name}.txt'
        #     filename_all_vars = f'SUMMARY_GRAPH_VARS_{name}.txt'

        # initialize return dict
        operation_analytics = {}
        operation_analytics['total'] = {}
        operation_analytics['total']['count'] = 0
        operation_analytics['elementwise'] = {}
        operation_analytics['elementwise']['count'] = 0
        total_var_size = 0
        maximum_var_size_shape = (0,(0,),'')
        num_vars = 0

        # Write to text file
        with open(graph_file_name, 'w') as f:
            f.write(f'node name, \t node object\n')
        with open(vars_file_name, 'w') as f:
            f.write(f'\n')
        # collect information from operations
        for node in self.eval_graph:

            # write to filename a summary of each node
            with open(graph_file_name, 'a') as f:

                # if isinstance(node, OperationNode):
                #     continue
                # if node.promoted_id not in self.promoted_to_unique:
                #     continue

                if isinstance(node, OperationNode):
                    f.write(f'\n{node.name}, {node.op}\n')
                else:
                    f.write(f'\n{node.name}, {node.var}, {node.unpromoted_namespace}.{node.name}\n')
                    if node.connected_to:
                        connected_to_bool = True
                    else:
                        connected_to_bool = False
                    if node.declared_to:
                        declared_to_bool = True
                    else:
                        declared_to_bool = False

                    if node.promoted_id in self.rep.promoted_to_node:
                        if not isinstance(node.var, (Output, Input)):
                            if np.array_equal(node.var.val, np.ones(node.var.shape)):
                                f.write(f'\tWARNING: this declared variable is not a promotion or connection target with a value being set.\n')
                            else:
                                f.write(f'\tWARNING: this declared variable is not a promotion or connection. \n')
                    f.write(f'\tCONNECTED TO: {connected_to_bool}\n')
                    for connected_to_node in node.connected_to:
                        f.write(f'\t\t{connected_to_node.name}, {connected_to_node.unpromoted_namespace}.{connected_to_node.name}\n')
                    f.write(f'\tPROMOTED TO: {declared_to_bool}\n')
                    for connected_to_node in node.declared_to:
                        f.write(f'\t\t{connected_to_node.name}, {connected_to_node.unpromoted_namespace}.{connected_to_node.name}\n')

                # Write predecessors
                f.write(f'\tPREDECESSORS\n')
                for dep in self.eval_graph.predecessors(node):

                    if isinstance(dep, OperationNode):
                        f.write(f'\t\t{dep.name}\n')
                    else:
                        f.write(f'\t\t{dep.name},{dep.var.shape} \n')

                # Write tSUCCESSORS
                f.write(f'\tSUCCESSORS\n')
                for dep in self.eval_graph.successors(node):

                    if isinstance(dep, OperationNode):
                        f.write(f'\t\t{dep.name}\n')
                    else:
                        f.write(f'\t\t{dep.name},{dep.var.shape} \n')


            # Write to text file
            with open(vars_file_name, 'a') as f:
                if isinstance(node, VariableNode):

                    is_auto_var = False

                    if len(node.name) == 1:
                        is_auto_var = False
                    elif (node.name[0] == '_') and (not node.name[1].isalpha()):
                        # We know if its a CSDL auto var if its named _ABC
                        # where A is an integer (it seems)
                        is_auto_var = True
                    if not is_auto_var:
                        if hasattr(node.var, 'val'):
                            avg_val = f'{np.mean(node.var.val)}'
                        else:
                            avg_val = "None"

                        if self.eval_graph.in_degree(node) > 0:
                            over = ' (overwritten):'
                        else:
                            over = ':              '

                        f.write(f'\n\n{node.id}')
                        f.write(f'\n\tname:                    {node.name}')
                        f.write(f'\n\tunpromoted name:         {prepend_namespace(node.unpromoted_namespace,node.name)}')
                        f.write(f'\n\tpromoted name:           {prepend_namespace(node.namespace,node.name)}')
                        f.write(f'\n\tshape:                   {node.var.shape}')
                        f.write(f'\n\tavg val{over}   {avg_val}')
                        f.write(f'\n\tgraph info:              {self.eval_graph.in_degree(node)} in op / {self.eval_graph.out_degree(node)} out op(s)')
                        f.write(f'\n\tconnected to:')
                        for connected_to_node in node.connected_to:
                            f.write(f'\n\t                         {prepend_namespace(connected_to_node.unpromoted_namespace,connected_to_node.name)}')
                        f.write(f'\n\tpromoted to:')
                        for connected_to_node in node.declared_to:
                            f.write(f'\n\t                         {prepend_namespace(connected_to_node.unpromoted_namespace,connected_to_node.name)}')
                        # f.write(f'\n{node.name}, {node.var}, {node.unpromoted_namespace}.{node.name}\n')
                        # if node.connected_to:
                        #     connected_to_bool = True
                        # else:
                        #     connected_to_bool = False
                        # if node.declared_to:
                        #     declared_to_bool = True
                        # else:
                        #     declared_to_bool = False

                        # if node.promoted_id in self.rep.promoted_to_node:
                        #     if not isinstance(node.var, (Output, Input)):
                        #         if np.array_equal(node.var.val, np.ones(node.var.shape)):
                        #             f.write(f'\tWARNING: this declared variable is not a promotion or connection target with a value being set.\n')
                        #         else:
                        #             f.write(f'\tWARNING: this declared variable is not a promotion or connection. \n')
                        # f.write(f'\tCONNECTED TO: {connected_to_bool}\n')
                        # for connected_to_node in node.connected_to:
                        #     f.write(f'\t\t{connected_to_node.name}, {connected_to_node.unpromoted_namespace}.{connected_to_node.name}\n')
                        # f.write(f'\tPROMOTED TO: {declared_to_bool}\n')
                        # for connected_to_node in node.declared_to:
                        #     f.write(f'\t\t{connected_to_node.name}, {connected_to_node.unpromoted_namespace}.{connected_to_node.name}\n')


            # keep a count of every type of node in the graph for printing
            if isinstance(node, OperationNode):
                csdl_node = node.op
                operation_analytics['total']['count'] += 1
                if isinstance(csdl_node, StandardOperation):
                    if csdl_node.properties['elementwise']:
                        operation_analytics['elementwise']['count'] += 1
            else:
                csdl_node = node.var
                size = np.prod(csdl_node.shape)
                total_var_size += np.prod(size)
                num_vars += 1
                if size > maximum_var_size_shape[0]:
                    maximum_var_size_shape = (size, csdl_node.shape, node.id)

            if type(csdl_node) not in operation_analytics:
                operation_analytics[type(csdl_node)] = {}
                operation_analytics[type(csdl_node)]['count'] = 0
            operation_analytics[type(csdl_node)]['count'] += 1

        average_var_size = total_var_size/num_vars
        extra_data = []
        extra_data.append(f'average variable size:   {format_print_number(average_var_size)}')
        extra_data.append(f'maximum variable size:   {format_print_number(maximum_var_size_shape[0])} {maximum_var_size_shape[1]} ({maximum_var_size_shape[2]})')
        extra_data.append(f'number of variables  :   {format_print_number(num_vars)}')
        extra_data.append(f'number of scalars    :   {format_print_number(total_var_size)}')

        return operation_analytics, extra_data
    
    # @profile
    def generate_evaluation(self):
        '''
        generate the evaluation block.

        Loop through all nodes of the forward graph. For each operation,
        initialize the backend operation object and write to evaluation block.

        Returns:
        --------
            eval_block: CodeBlock
                the evaluation code to evaluate system model.
            preeval_vars: Dict
                dictionary containing all pre computed variables for evaluation.
            state_vals: StateManager
                the initial value for every state value.
            variable_info: Dict
                split variables into outputs/inputs
        ''' 
        preeval_vars = {}  # initialize precomputed variables
        state_vals = StateManager(self.variable_owner_map_full, self.variable_owner_map, self.comm)  # initialize initial state values
        variable_info = {  # initialize variable info dict
            'inputs': {},
            'outputs': {},
            'leaf_start': {},
        }

        # for snapshot in self.checkpoint_data:
        #     print(snapshot)

        # Loop through sorted graph
        total_num_vars = 0
        total_keep_vars = 0
        total_saved_by_csdl_vars = 0
        total_linear_intermediate_vars = 0

        total_num_scalars = 0
        total_linear_intermediate_scalars = 0
        total_keep_scalars = 0
        total_saved_by_csdl_scalars = 0

        linear_intermediate_vars = set() # contains variables with all linear successors, and is an intermediate variable we can throw away
        
        for node in nx.topological_sort(self.eval_graph):

            if not isinstance(node, OperationNode):
                # If variable, set state initial value given by user
                csdl_node = node.var

                total_num_vars += 1
                total_num_scalars += np.prod(csdl_node.shape)

                # OLD
                # if isinstance(csdl_node.val, np.ndarray):
                #     csdl_node.val = csdl_node.val.astype('float64')
                #     if isinstance(csdl_node.val, np.matrix):
                #         csdl_node.val = np.asarray(csdl_node.val).astype('float64')
                # # state_vals[node.id] = csdl_node.val.copy()
                # # print(node.id)
                # # print(self.variable_owner_map)
                # state_vals.reserve_state(node.id, csdl_node.shape)
                # state_vals[node.id] = csdl_node.val.copy()

                # NEWISH
                # if hasattr(csdl_node, "val"):
                #     if isinstance(csdl_node.val, np.ndarray):
                #         csdl_node.val = csdl_node.val.astype('float64')
                #         if isinstance(csdl_node.val, np.matrix):
                #             csdl_node.val = np.asarray(csdl_node.val).astype('float64')
                #     # state_vals[node.id] = csdl_node.val.copy()
                #     # print(node.id)
                #     # print(self.variable_owner_map)
                #     state_vals.reserve_state(node.id, csdl_node.shape)
                #     state_vals[node.id] = csdl_node.val.copy()
                #     num_vars_allocated += 1
                # else:
                #     num_vars_deallocated += 1
                #     state_vals.reserve_state(node.id, csdl_node.shape)


                promoted_id = node.promoted_id
                keep_this_var = False
                del_csdl_val = True
                saved_by_csdl = False
                if isinstance(csdl_node, (Output, DeclaredVariable)):
                    is_auto_var = False

                    if len(node.name) == 1:
                        is_auto_var = False
                    # elif (node.name[0] == '_') and (not node.name[1].isalpha()):
                    elif (node.name == node.id):
                        # We know if its a CSDL auto var if its named _ABC
                        # where A is an integer (it seems)
                        is_auto_var = True

                    # if node.name[0] == '_':
                    #     is_auto_var = True

                    if not is_auto_var:
                        if promoted_id in self.promoted_to_unique:
                            if isinstance(csdl_node, Output):
                                variable_info['outputs'][promoted_id] = {}
                                variable_info['outputs'][promoted_id]['shape'] = csdl_node.shape
                                variable_info['outputs'][promoted_id]['size'] = np.prod(csdl_node.shape)

                            if self.checkpoints_bool:
                                if self.save_all_outputs:
                                    keep_this_var = True
                                else:
                                    keep_this_var = False
                            else:
                                keep_this_var = True

                if isinstance(csdl_node, Input):
                    variable_info['inputs'][promoted_id] = {}
                    variable_info['inputs'][promoted_id]['shape'] = csdl_node.shape
                    variable_info['inputs'][promoted_id]['size'] = np.prod(csdl_node.shape)
                    keep_this_var = True

                if len(list(self.eval_graph.predecessors(node))) == 0:
                    variable_info['leaf_start'][promoted_id] = {}
                    variable_info['leaf_start'][promoted_id]['shape'] = csdl_node.shape
                    variable_info['leaf_start'][promoted_id]['size'] = np.prod(csdl_node.shape)
                    keep_this_var = True
                    del_csdl_val = False
                         
                if self.eval_graph.out_degree(node) == 0:
                    keep_this_var = True

                # if self.eval_graph.in_degree(node) != 0:
                #     if isinstance(list(self.eval_graph.predecessors(node))[0], Concatenation):
                # keep_this_var = True
                # del_csdl_val = False
                if isinstance(csdl_node, Concatenation):
                    del_csdl_val = False
                    keep_this_var = True

                for predecessor in self.eval_graph.predecessors(node):
                    if isinstance(predecessor.op, (ImplicitOperation, CustomImplicitOperation, SolveLinear)):
                        del_csdl_val = False
                        keep_this_var = True
                
                # del_csdl_val = False

                # keep_this_var = True
                # del_csdl_val = False
                # NEW
                if keep_this_var:
                    if hasattr(csdl_node, "val"):
                        saved_by_csdl = True
                        if isinstance(csdl_node.val, np.ndarray):
                            csdl_node.val = csdl_node.val.astype('float64')
                            if isinstance(csdl_node.val, np.matrix):
                                csdl_node.val = np.asarray(csdl_node.val).astype('float64')
                    else:
                        # print(csdl_node.name, csdl_node)
                        csdl_node.val = np.ones(csdl_node.shape)
                    # state_vals[node.id] = csdl_node.val.copy()
                    # print(node.id)
                    # print(self.variable_owner_map)
                    state_vals.reserve_state(node.id, csdl_node.shape)
                    self.permanently_allocated_vars.add(node)
                    state_vals[node.id] = csdl_node.val.copy()
                else:
                    if hasattr(csdl_node, "val"):
                        saved_by_csdl = True
                        if isinstance(csdl_node.val, np.ndarray):
                            csdl_node.val = csdl_node.val.astype('float64')
                            if isinstance(csdl_node.val, np.matrix):
                                csdl_node.val = np.asarray(csdl_node.val).astype('float64')
                    state_vals.reserve_state(node.id, csdl_node.shape)


                # Any variables that are not intermediate and linear, we can deallocate
                if node not in self.permanently_allocated_vars:
                    if self.eval_graph.out_degree(node) == 0 or self.eval_graph.in_degree(node) == 0:
                        linear_intermediate = False
                    else:
                        linear_intermediate = True
                        for successor in self.eval_graph.successors(node):
                            if not successor.op.properties['linear']:
                                linear_intermediate = False
                                break
                    
                    if linear_intermediate:
                        linear_intermediate_vars.add(node)
                        total_linear_intermediate_vars += 1
                        total_linear_intermediate_scalars += np.prod(csdl_node.shape)

                if saved_by_csdl:
                    total_saved_by_csdl_vars += 1
                    total_saved_by_csdl_scalars += np.prod(csdl_node.shape)
                # print(np.prod(csdl_node.shape)*4)
                # del_csdl_val = False
                if del_csdl_val:
                    csdl_node.val = None
                    del csdl_node.val
        
        # import gc
        # gc.collect()
        # total_size = 0
        # for promoted_id in variable_info['outputs']:
        #     total_size += variable_info['outputs'][promoted_id]['size']
        # print('num outputs:', len(variable_info['outputs']) ,'num scalars:',  total_size)
        num_coms = 0
        num_ops = 0

        if self.comm is not None:
            eval_multi_instrunctions = MultiInstructions(f'multiinstr_{self.comm.rank}')
        else:
            eval_multi_instrunctions = MultiInstructions(f'multiinstr')       

        if self.checkpoints_bool:
            self.schedules = [snapshot['rank schedule'] for snapshot in self.checkpoint_data]
            keep_vars = self.permanently_allocated_vars.union(*[snapshot['rank snapshot'] for snapshot in self.checkpoint_data])

            for var in keep_vars:
                total_keep_vars += 1
                total_keep_scalars += np.prod(var.var.shape)

            # keep_vars = set().union(*[snapshot['rank snapshot'] for snapshot in self.checkpoint_data])
        else:
            self.schedules = [self.rep.schedule]
        
        for snap_num, current_schedule in enumerate(self.schedules):
            del_vars = set()
            del_vars_during_deriv = set()
            if self.checkpoints_bool:
                eval_block = CodeBlock(f'system evaluation block checkpoint {snap_num}')  # initialize evaluation block
                # keep_vars = self.checkpoint_data[snap_num]['rank snapshot']
            else:
                eval_block = CodeBlock('system evaluation block')  # initialize evaluation block

            instr_name = f'RUN_MODEL_{self.name}'
            if self.comm is not None:
                instr_name += f'_rank{self.comm.rank}'
            if self.checkpoints_bool:
                instr_name += f'_snap{snap_num}'

            eval_single_instructions = SingleInstruction(instr_name)

            for node in current_schedule:
                all_op_classes = (StandardOperation, CustomExplicitOperation, ImplicitOperation, BracketedSearchOperation, CustomImplicitOperation)
                if isinstance(node, PointToPointCall):
                    num_coms += 1
                    vars = {}
                    node.get_block(eval_block, vars)
                    preeval_vars.update(vars)

                    for successor in self.eval_graph.successors(node.var):
                        if self.checkpoints_bool:
                            if node.var not in keep_vars:
                                del_vars.add(node.var.id)
                            if successor in linear_intermediate_vars:
                                del_vars_during_deriv.add(successor.id)
                        else:
                            if successor in linear_intermediate_vars:
                                del_vars.add(successor.id)

                elif isinstance(node, OperationNode):
                    num_ops += 1
                    csdl_node = node.op

                    # input to operation_lite object
                    # give successors and predecessors
                    predecessors = self.eval_graph.predecessors(node)
                    nx_inputs = {}
                    for predecessor in predecessors:
                        nx_inputs[predecessor.id] = predecessor

                    successors = self.eval_graph.successors(node)
                    nx_outputs = {}
                    for successor in successors:
                        nx_outputs[successor.id] = successor
                        if self.checkpoints_bool:
                            if successor not in keep_vars:
                                del_vars.add(successor.id)
                            if successor in linear_intermediate_vars:
                                del_vars_during_deriv.add(successor.id)
                        else:
                            if successor in linear_intermediate_vars:
                                del_vars.add(successor.id)

                    # Create the backend operation
                    if isinstance(csdl_node, StandardOperation):
                        back_operation = get_backend_op(csdl_node)(csdl_node, nx_inputs, nx_outputs, node.name)
                    elif isinstance(csdl_node, CustomExplicitOperation):
                        back_operation = get_backend_custom_explicit_op(csdl_node)(csdl_node, nx_inputs, nx_outputs, node.name)
                    elif isinstance(csdl_node, (ImplicitOperation, BracketedSearchOperation)):
                        back_operation = get_backend_implicit_op(csdl_node)(csdl_node, nx_inputs, nx_outputs, node.name)
                        back_operation.set_initial_state_guess(state_vals)
                        self.all_implicit_operations.add(back_operation)
                        self.all_state_ids_to_guess.update(back_operation.state_outid_to_initial_guess)
                    elif isinstance(csdl_node, CustomImplicitOperation):
                        back_operation = get_backend_custom_implicit_op(csdl_node)(csdl_node, nx_inputs, nx_outputs, node.name)
                        back_operation.set_initial_state_guess(state_vals)
                        self.all_implicit_operations.add(back_operation)
                        self.all_state_ids_to_guess.update(back_operation.state_outid_to_initial_guess)
                    else:
                        raise NotImplementedError(f'{csdl_node} operation not found')
                    node.back_operation = back_operation

                    # Get the evaluation procedure for current operation
                    eval_block_temp = CodeBlock(back_operation.name, newline=False, add_name=False)  # script evaluation block
                    vars = {}  # precomputed variables
                    back_operation.get_evaluation(eval_block_temp, vars)

                    #       UNCOMMENT TO PRINT ALL VARIABLES
                    # eval_block.write(f'print()')
                    # eval_block.write(f'print("OPERATION")')
                    # for key in nx_inputs:
                    #     eval_block.write(f'try:')
                    #     eval_block.write(f'\tprint(comm.rank,"{key}=", {key})')
                    #     eval_block.write(f'except:')
                    #     eval_block.write(f'\tprint("{key}=", {key})')
                    #       UNCOMMENT TO PRINT ALL VARIABLES

                    # +=+=+=+=+=+=+=+=+=+=+=+=+= write the evaluation procedure +=+=+=+=+=+=+=+=+=+=+=+=+=
                    # update precomputed variables. hopefully all keys are unique.
                    preeval_vars.update(vars)

                    # write a summary of the operation
                    eval_block.write(back_operation.op_summary_block)

                    # if the input shapes are not matching, reshape (hopefully not needed)
                    if back_operation.needs_input_reshape:
                        eval_block.write(back_operation.reshape_block)

                    # the main evaluation script is written here.
                    eval_block.write(eval_block_temp)

                    # if the input shapes are not matching, reshape it back to original shape (hopefully not needed)
                    if back_operation.needs_input_reshape:
                        eval_block.write(back_operation.unreshape_block)
                    # +=+=+=+=+=+=+=+=+=+=+=+=+==+= end evaluation procedure +=+=+=+=+=+=+=+=+=+=+=+=+=+=
                    #       UNCOMMENT TO PRINT ALL VARIABLES
                    # eval_block.write(f'print("OUTPUT")')
                    # for key in nx_outputs:
                    #     eval_block.write(f'try:')
                    #     eval_block.write(f'\tprint(comm.rank,"{key}-->", {key})')
                    #     eval_block.write(f'except:')
                    #     eval_block.write(f'\tprint("{key}-->", {key})')
                    #       UNCOMMENT TO PRINT ALL VARIABLES

            if self.checkpoints_bool:
                pass

            if self.comm:
                # self.comm.barrier()
                if snap_num == len(self.schedules) - 1:
                    print(f'{self.comm.rank} ({num_coms} COMMS/{num_ops} OPS)')
                    eval_block.write('comm.barrier()')
                # exit()
            eval_single_instructions.script.write(eval_block)
            eval_single_instructions.compile()

            eval_multi_instrunctions.add_single_instruction(eval_single_instructions, del_vars)
            
            # Remember which variables we have to destroy for adjoint
            if self.checkpoints_bool:
                self.checkpoint_data[snap_num]['single instructions'] = eval_single_instructions
                self.checkpoint_data[snap_num]['del vars'] = del_vars
                self.checkpoint_data[snap_num]['del vars during deriv'] = del_vars_during_deriv

        if self.checkpoints_bool:
            print(f'{total_keep_vars}/{total_num_vars} variables permanately allocated ({total_keep_scalars:,}/{total_num_scalars:,} scalars)')
        print(f'{total_saved_by_csdl_vars}/{total_num_vars} variables allocated by CSDL ({total_saved_by_csdl_scalars:,}/{total_num_scalars:,} scalars)')
        print(f'{total_linear_intermediate_vars}/{total_num_vars} linear intermediate vars discarded ({total_linear_intermediate_scalars:,}/{total_num_scalars:,} scalars)')

        # eval_multi_instrunctions = MultiInstructions('Multi_Instructions')
        # eval_multi_instrunctions.add_single_instruction(eval_single_instructions, set())
        return eval_multi_instrunctions, preeval_vars, state_vals, variable_info

        # return eval_instructions, preeval_vars, state_vals, variable_info

    # @profile
    def generate_reverse(self, output_ids, input_ids, vjp):
        '''
        generate the reverse mode derivative evaluation script.
        generally, loop through all variables in reverse order using
        a modified breadth-first-search and generate the AD script.

        Parameters:
        ----------
            output_ids: list or string
                list or string of UNIQUE output id(s) to take derivative of
            input_ids: list or string
                list or string of UNIQUE input id(s) to take derivative wrt
        '''
        # list of output ids and input ids to get derivatives of.
        output_ids = to_unique_list(output_ids)
        input_ids = to_unique_list(input_ids)

        # dictionary of ALL predecessor nodes of inputs
        # input_ancestors: input id --> set(nodes)
        input_ancestors = {}
        all_input_ancestors = set()
        for input_id in input_ids:
            # input_ancestors[input_id] = set(nx.ancestors(self.rev_graph, self.unique_to_node[input_id]))
            all_input_ancestors = all_input_ancestors.union(set(nx.ancestors(self.rev_graph, self.unique_to_node[input_id])))
            all_input_ancestors.add(self.unique_to_node[input_id])
        # getting print statements ready
        max_outstr_len = 0
        for out_id_temp in output_ids:
            output_node = self.unique_to_node[out_id_temp]
            output_lang_name_temp = output_node.name

            if len(f'{output_lang_name_temp}-->') > max_outstr_len:
                max_outstr_len = len(f'{output_lang_name_temp}-->')

        # initialize instructions
        # name of instructions

        if not vjp:
            get_path_name_rev = get_path_name
        else:
            get_path_name_rev = get_path_name_vjp

        # Static variables
        prerev_vars = {}

        # accumulation operations
        prerev_vars['DIAG_MULT'] = diag_mult
        prerev_vars['STD_MULT'] = std_mult

        # Keep track of which partials have already been computed.
        # We only need to compute partial jacobians once no matter how many
        # derivatives we are computing
        computed_partial_jacs = set()

        # - Do not search down a node until ALL* edges leading into it has been search as well
        # - Do not search down an implicit output untill ALL* implicit outputs have been searched
        # - * All edges dependent on the output
        fully_visited = set()  # set containing nodes that were fully visited

        # For print statements
        current_op_num = 0
        # out_str = lineup_string(f'-->', 10)
        stride = round(self.num_ops/40)
        if stride == 0:
            stride = 1

        # Data that contains information for each output
        output_info_dict = {}
        totals_names = {}
        do_not_delete_paths = set()
        for out_id in output_ids:
            # Get output info
            output_node = self.unique_to_node[out_id]
            output_lang_name = output_node.name
            output_shape = output_node.var.shape
            output_size = np.prod(output_shape)

            # Get all ancestors of output for checking.
            output_descendants = set(nx.descendants(self.rev_graph, output_node))
            output_descendants = output_descendants.intersection(all_input_ancestors)
            output_descendants.add(output_node)

            output_info_dict[out_id] = {
                'output_node': output_node,
                'lang_name': output_lang_name,
                'shape': output_shape,
                'size': output_size,
                'id': out_id,
                'output_descendants': output_descendants,
                'initialized_paths': set()
            }

            for input_id in input_ids:
                input_path_name = get_path_name_rev(input_id, out_id=out_id)
                totals_names[out_id, input_id] = input_path_name
                do_not_delete_paths.add(input_path_name)

            do_not_delete_paths.add(get_path_name_rev(out_id, out_id=out_id))

        # Go through reversed operation list
        # We guarantee is that reversed schedule is in reverse topological order, so no need for search

        if self.comm is not None:
            rev_multi_instructions = MultiInstructions(f'rev_multiinstr_{self.comm.rank}')
        else:
            rev_multi_instructions = MultiInstructions(f'rev_multiinstr')       
        
        for inverse_snap_num, current_schedule in enumerate(reversed(self.schedules)):
            
            snap_num = len(self.schedules) - inverse_snap_num - 1
            # Set up code gen
            if vjp:
                instruction_name = 'REV_VJP'
            else:
                instruction_name = 'REV_'

            if self.comm is None:
                instruction_name += ': '
            else:
                instruction_name += f'{self.comm.rank}: '

            # instruction_name = 'REV_'
            for output_name in sorted(output_ids):
                output_lang_name = output_info_dict[output_name]['lang_name']
                instruction_name += f'{output_lang_name},'
            instruction_name += '-->'
            for input_name in input_ids:
                instruction_name += f'{input_name},'
            rev_block = CodeBlock(instruction_name)

            # Set up code gen
            # instruction_name = 'REV_'
            if self.comm is not None:
                instruction_name += f'_rank{self.comm.rank}'
            if self.checkpoints_bool:
                instruction_name += f'_snap{snap_num}'
            rev_single_instructions= SingleInstruction(instruction_name)


            # Main loop
            for current_node in reversed(list(current_schedule)):
                # search downstream of node 'current'
                # Node 'current' is an operation
                # queue should never contain variables that have already appeared queue before
                middle_operation = current_node

                # print(self.comm.rank, middle_operation)
                if not isinstance(middle_operation, (OperationNode, PointToPointCall)):
                    # dev error. This error should have been caught way earlier. if triggered, something is wrong with compiler.
                    raise ValueError(f'{middle_operation} must be an operation/MPI call!')

                # print(current_node.name)
                # # Make sure successors are visited?
                # for successor_var in self.rev_graph.successors(middle_operation):
                #     print(successor_var.name)
                #     fully_visited.add(successor_var)

                # We only pass through operations once. Therefore, if we already computed it,
                # do not pass through again!
                # **should NEVER trigger**
                # if middle_operation in processed_operations:
                #     continue

                # check if we need to propagate derivatives through this operation
                # If this operation is independent of all outputs and inputs, skip it
                needs_propagation = False
                if isinstance(middle_operation, (OperationNode)):
                    for out_id in output_info_dict.keys():
                        output_descendants = output_info_dict[out_id]['output_descendants']
                        if middle_operation in output_descendants:
                            needs_propagation = True
                elif isinstance(middle_operation, (PointToPointCall)):
                    for out_id in output_info_dict.keys():
                        output_descendants = output_info_dict[out_id]['output_descendants']
                        if middle_operation.var in output_descendants:
                            needs_propagation = True

                    # if needs_propagation:
                    #     print('YES!!')
                    # else:
                    #     print('NO!!')
                # for input_id in input_ancestors:
                #     if middle_operation in input_ancestors:
                #         needs_propagation = True
                # If this operation is independent of all outputs and inputs, skip it
                if not needs_propagation:
                    continue

                # if program reaches here, middle_operation has been fully visited so we now process it
                # cool print statements:
                current_op_num += 1
                # if current_op_num == 1:
                #     print_loading(
                #         out_str,
                #         current_op_num,
                #         self.num_ops,
                #         False)

                # if (current_op_num) % stride == 0:
                #     print_loading(
                #         out_str,
                #         current_op_num,
                #         self.num_ops,
                #         False)

                # :::::GENERATE CODE FOR MIDDLE_OPERATION:::::
                # If operation:
                    # Two things to do:
                    # 1) Compute partials of the operation.
                    # 2) Add partials to path.
                    # Note that 1) is only done once per operation; we do not need
                    # to compute partials of an operation more than once.

                    # 1) Compute partials:
                    # there are three ways to compute partials:
                    # 1a) Compute the Jacobian matrix independent of paths_out: called once per script
                    # --- Standard Explicit Operations
                    # --- CustomExplicitOperations: Compute Derivatives
                    # 1b) Compute the mat-mat / mat-vec product as a function of paths_out: called once per output
                    # --- ImplicitOperations: Adjoint method
                    # --- CustomExplicitOperations: Compute Jacvec Product
                    # --- CustomImplicitOperations: Adjoint method (?)

                # if MPI communication:
                    # 1) get path out if it has been computed or initialize it
                    # 2) Write adjoint of MPI call of path out 



                # For each output:
                # print('current operation: ', middle_operation.name)
                # Deallocate variables after this operation is processed
                variable_names_to_delete = set()
                for out_id in output_ids:
                    # Get output shape
                    output_node = output_info_dict[out_id]['output_node']
                    output_lang_name = output_info_dict[out_id]['lang_name']
                    output_shape = output_info_dict[out_id]['shape']
                    output_size = output_info_dict[out_id]['size']
                    initialized_paths = output_info_dict[out_id]['initialized_paths']
                    output_descendants = output_info_dict[out_id]['output_descendants']


                    # if MPI communication
                    if isinstance(middle_operation, (PointToPointCall)):
                        if middle_operation.var not in output_descendants:
                            continue

                        # Path to output
                        mpi_var_id = middle_operation.var_id
                        mpi_var = middle_operation.var
                        path_out_name = get_path_name_rev(mpi_var_id, out_id=out_id)
                        mpi_var_size = np.prod(mpi_var.var.shape)

                        if isinstance(middle_operation, SendCall):
                            if path_out_name not in initialized_paths:
                                if mpi_var_id == out_id:
                                    if output_size > 100:
                                        # rev_block.write(f'{path_out_name} = {get_reverse_seed(out_id)}, format = \'csr\')')
                                        rev_block.write(f'{path_out_name} = {get_reverse_seed(out_id)}')
                                    else:
                                        rev_block.write(f'{path_out_name} = {get_reverse_seed(out_id)}')
                                else:
                                    if output_size*mpi_var_size > 5000:
                                        # rev_block.write(f'{path_out_name} = sp.csr_array(({output_size}, {mpi_var_size}))')
                                        rev_block.write(f'{path_out_name} = sp.csr_array(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')

                                    else:
                                        # rev_block.write(f'{path_out_name} = np.zeros(({output_size}, {mpi_var_size}))')
                                        rev_block.write(f'{path_out_name} = np.zeros(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')
                                    # rev_block.write(f'{path_out_name} = sp.csr_array(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')

                                initialized_paths.add(path_out_name)

                            # Write MPI adjoint
                            path_out_name_mpi = path_out_name+'_mpi'
                            vars = {}
                            middle_operation.get_adjoint_call(
                                rev_block,
                                vars,
                                path_out_name_mpi,
                                (output_size, mpi_var_size),
                                None,
                            )
                            rev_block.write(f'{path_out_name} += {path_out_name_mpi}')
                        elif isinstance(middle_operation, RecvCall):
                            if path_out_name not in initialized_paths:
                                # if mpi_var_id == out_id:
                                #     if output_size > 100:
                                #         rev_block.write(f'{path_out_name} = sp.eye({output_size}, format = \'csr\')')
                                #     else:
                                #         rev_block.write(f'{path_out_name} = np.eye({output_size})')
                                # else:
                                if output_size*mpi_var_size > 5000:
                                    rev_block.write(f'{path_out_name} = sp.csr_array(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')
                                else:
                                    rev_block.write(f'{path_out_name} = np.zeros(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')
                                # rev_block.write(f'{path_out_name} = sp.csr_array(({get_reverse_seed(out_id)}.shape[0], {mpi_var_size}))')

                                initialized_paths.add(path_out_name)
                            # Write MPI adjoint
                            path_out_name_mpi = path_out_name
                            vars = {}
                            middle_operation.get_adjoint_call(
                                rev_block,
                                vars,
                                path_out_name_mpi,
                                (output_size, mpi_var_size),
                                None,
                            )
                        prerev_vars.update(vars)
                        continue

                    # If code reach here, middle_operation is an operation node, not an MPI node.
                    backend_op = middle_operation.back_operation
                    if middle_operation not in output_descendants:
                        continue
                    # check 1a) or 1b)
                    if not backend_op.jac_is_function:  # 1a)

                        if not middle_operation in computed_partial_jacs:

                            # Add to make sure we do not compute this partial Jacobian again within this script.
                            computed_partial_jacs.add(middle_operation)

                            partials_dict = {}

                            # compute partials for each input and output of middle_operation
                            for predecessor in self.rev_graph.predecessors(middle_operation):
                                for successor in self.rev_graph.successors(middle_operation):
                                    partials_name = get_deriv_name(predecessor.id, successor.id)

                                    partials_dict[predecessor, successor] = {}
                                    partials_dict[predecessor, successor]['name'] = partials_name

                                    # Deallocate partials once they are processed

                                    if self.lazy:
                                        variable_names_to_delete.add(partials_name)
                                    else:
                                        if not backend_op.linear:
                                            variable_names_to_delete.add(partials_name)

                                    # if not backend_op.linear:
                                        # variable_names_to_delete.add(partials_name)

                            partials_block = CodeBlock(backend_op.name + f'_{partials_name}')
                            vars = {}

                            # compute_partials for each input and output
                            is_sparse_jac = get_operation_sparsity(backend_op, self.sparsity_type)

                            # PRINT:
                            # pred_size = 0
                            # for predecessor in self.rev_graph.predecessors(middle_operation):
                            #     if np.prod(predecessor.var.shape) > pred_size:
                            #         pred_size = np.prod(predecessor.var.shape)
                            # succ_size = 0
                            # for successor in self.rev_graph.successors(middle_operation):
                            #     if np.prod(successor.var.shape) > succ_size:
                            #         succ_size = np.prod(successor.var.shape)
                            # if (pred_size > 100 and succ_size > 100) and not is_sparse_jac:
                            #     print(is_sparse_jac, f'({succ_size} x {pred_size})', middle_operation.op)
                            # elif (pred_size < 100 and succ_size < 100) and is_sparse_jac:
                            #     print(is_sparse_jac, f'({succ_size} x {pred_size})', middle_operation.op)

                            lazy = self.lazy

                            backend_op.get_partials(
                                partials_dict = partials_dict,
                                partials_block = partials_block,
                                vars = vars,
                                is_sparse_jac = is_sparse_jac,
                                lazy = lazy,
                            )
                            # backend_op.get_partials(
                            #     partials_dict = partials_dict,
                            #     partials_block = partials_block,
                            #     vars = vars,
                            #     is_sparse_jac = is_sparse_jac,
                            # )
                            # # write to script
                            # rev_block.write(partials_block)
                            prerev_vars.update(vars)

                            # write a summary of the operation
                            rev_block.write(backend_op.op_summary_block)
                            # if the input shapes are not matching, reshape (hopefully not needed)
                            if backend_op.needs_input_reshape:
                                if not backend_op.linear:
                                    rev_block.write(backend_op.reshape_block)
                            # the main evaluation script is written here.
                            rev_block.write(partials_block)
                            # if the input shapes are not matching, reshape it back to original shape (hopefully not needed)
                            if backend_op.needs_input_reshape:
                                if not backend_op.linear:
                                    rev_block.write(backend_op.unreshape_block)
                                    # rev_block.write(f'{backend_op.linear}')


                    else:  # 1b)

                        # compute partials as a function of output paths
                        partials_block = CodeBlock(backend_op.name + f'_{out_id}_{middle_operation.name}')
                        vars = {}

                        # path_in_names are paths from output passing through the jac_function operation
                        path_in_names = []
                        for jac_function_input in backend_op.ordered_in_names:
                            jac_function_input_id = backend_op.get_input_id(jac_function_input)
                            path_in_name = f'{middle_operation.name}_{out_id}_{jac_function_input_id}'
                            path_in_names.append(path_in_name)
                        # need for edge case with bracketed search with csdl variables.
                        if isinstance(backend_op.operation, BracketedSearchOperation):
                            for jac_function_input in backend_op.ordered_in_brackets:
                                jac_function_input_id = backend_op.get_input_id(jac_function_input)
                                path_in_name = f'{middle_operation.name}_{out_id}_{jac_function_input_id}'
                                brack_var = self.unique_to_node[jac_function_input_id]
                                rev_block.write(f'{path_in_name} = np.zeros(({get_reverse_seed(out_id)}.shape[0], {np.prod(brack_var.var.shape)}))')

                        # path_out_names are paths from output to the outputs of the jac_function operation
                        # THESE PATHS SHOULD BE FULLY COMPUTED BY NOW (unless the jac_function output is independent of output or is the output)
                        path_out_names = []

                        for jac_function_output_lang in backend_op.ordered_out_names:

                            jac_function_output_id = backend_op.get_output_id(jac_function_output_lang)
                            path_out_name = get_path_name_rev(jac_function_output_id, out_id=out_id)

                            # This part is complicated. We feed in paths to the jac_function method.
                            # These paths SHOULD be computed by now. However, they may not be in two cases:
                            # 1) jac_function output is independent of output -> path is zero
                            # 2) jac_function output IS output -> path is identity

                            if path_out_name not in initialized_paths:
                                if self.unique_to_node[jac_function_output_id] in output_descendants:
                                    if self.unique_to_node[jac_function_output_id] != output_node:
                                        raise ValueError(f'path output {jac_function_output_id} of jac_function operation {middle_operation.name} should have been computed but is not.')
                                if jac_function_output_id == out_id:
                                    rev_block.write(f'{path_out_name} = {get_reverse_seed(out_id)}@np.eye({output_size})')
                                    # rev_block.write(f'{path_out_name} = sp.eye({output_size}, format = \'csr\')')
                                else:
                                    implicit_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                                    rev_block.write(f'{path_out_name} = np.zeros(({get_reverse_seed(out_id)}.shape[0], {implicit_out_size}))')
                                    # rev_block.write(f'{path_out_name} = np.zeros(({output_size}, {implicit_out_size}))')
                                    # rev_block.write(f'{path_out_name} = sp.csc_matrix(({output_size}, {implicit_out_size}))')

                            path_out_names.append(path_out_name)
                            variable_names_to_delete.add(path_out_name)

                        # compute and write to script
                        backend_op.get_accumulation_function(path_in_names, path_out_names, partials_block, vars)
                        rev_block.write(partials_block)
                        prerev_vars.update(vars)

                    # rev_block.write('start = time.time()')
                    # 2) Adding partials to a path:
                    # again, if we have jacobian as a matrix, right multiply to a path and add.
                    # on the other hand, if we have jacobian as a function just add as it is already right-multiplied.
                    for successor in self.rev_graph.successors(middle_operation):
                        # successor should be VariableNode instance and is the input of middle operation

                        # If successor does not depend on any input we want to take derivative of, no need to do anything.
                        ignore_successor = True
                        for input_id in input_ids:
                            if input_id == successor.id:
                                ignore_successor = False
                            # elif successor in input_ancestors[input_id]:
                            #     # elif nx.has_path(self.rev_graph, successor, self.unique_to_node[input_id]):
                            #     ignore_successor = False
                        if successor in all_input_ancestors:
                            ignore_successor = False
                        if ignore_successor:
                            continue

                        if not backend_op.jac_is_function:  # if we have the jacobian matrix, right multiply

                            # Iterate through each individual jac if matrix
                            for predecessor in self.rev_graph.predecessors(middle_operation):

                                # predecessor is a VariablNode instance and are the outputs of middle_operation

                                # If predecessor is independent of output, there will be no path so ignore.
                                compute_path_pred = False
                                if output_node == predecessor:
                                    compute_path_pred = True
                                elif predecessor in output_descendants:
                                    compute_path_pred = True
                                if not compute_path_pred:
                                    continue

                                # Right multiplication of path
                                path_successor = get_path_name_rev(successor.id, out_id=out_id)
                                path_current = get_path_name_rev(predecessor.id, out_id=out_id)
                                partials_name = get_deriv_name(predecessor.id, successor.id)

                                # We shouldn't need path_current after this operation is fully processed. As path_current is unique (?) to this operation, we can delete it.
                                variable_names_to_delete.add(path_current)


                                # If this is the first iteration in BFS, we need to set seed for output
                                if predecessor == output_node:
                                    # The line below had issues with pointers.
                                    # initialized_path_string = get_init_path_string(partials_name, backend_op, self.sparsity_type)
                                    initialized_path_string = get_successor_path_string(get_reverse_seed(out_id), partials_name, backend_op)

                                    rev_block.write(f'{path_successor} = {initialized_path_string}')
                                    initialized_paths.add(path_successor)
                                    continue

                                if path_current not in initialized_paths:
                                    # path_current should be calculated already. if not, :(
                                    raise ValueError(f'path {path_current} has not yet been computed.')

                                # Now we write the path accumulation
                                # successor_string = f'{path_current}@{partials_name}'

                                # is either
                                # successor_string = 'path_current@partials_name'
                                # successor_string = 'DIAG_MULT(path_current, partials_name)'
                                successor_string = get_successor_path_string(path_current, partials_name, backend_op)

                                if path_successor not in initialized_paths:
                                    # if path_successor not yet initialized, set it.
                                    rev_block.write(f'{path_successor} = {successor_string}')
                                    initialized_paths.add(path_successor)
                                else:
                                    # if path_successor has been initialized, add the path.
                                    rev_block.write(f'{path_successor} += {successor_string}')
                        else:  # jac is function

                            # Right multiplication of path
                            path_successor = get_path_name_rev(successor.id, out_id=out_id)

                            # Now we write the path accumulation
                            # If jac is function, partials_name already includes the path_current, so we do not multiply!
                            successor_string = f'{middle_operation.name}_{out_id}_{successor.id}'

                            if path_successor not in initialized_paths:
                                # if path_successor not yet initialized, set it.
                                rev_block.write(f'{path_successor} = {successor_string}')
                                initialized_paths.add(path_successor)
                            else:
                                # if path_successor has been initialized, add the path.
                                rev_block.write(f'{path_successor} += {successor_string}')

                # Delete all variables that are no longer needed
                for deallocate_var in variable_names_to_delete:
                    # continue
                    if deallocate_var not in do_not_delete_paths:
                        # rev_block.write(f'{deallocate_var} = None')
                        rev_block.write(f'del {deallocate_var}')

            if snap_num == 0:
                for out_id in output_ids:
                    output_node = output_info_dict[out_id]['output_node']
                    output_lang_name = output_info_dict[out_id]['lang_name']
                    output_shape = output_info_dict[out_id]['shape']
                    output_size = output_info_dict[out_id]['size']
                    initialized_paths = output_info_dict[out_id]['initialized_paths']
                    output_descendants = output_info_dict[out_id]['output_descendants']
                    # Finally, set the derivatives
                    # There are three possible cases for setting the derivatives:
                    # 1) If the input is found in initialized_paths, the derivative is the path
                    # 2) If the input is NOT found in initialized_paths AND output == input, the derivative is identity
                    # 3) If the input is NOT found in initialized_paths AND output != input (aka all other possibilities), the derivative is zero
                    for input_id in input_ids:
                        totals_name = get_deriv_name(out_id, input_id, partials=False)
                        # input_path_name = get_path_name_rev(input_id, out_id=out_id)
                        input_path_name = totals_names[out_id, input_id]
                        input_size = np.prod(self.unique_to_node[input_id].var.shape)

                        if input_size*output_size > 5000:
                            use_sparse = True
                        else:
                            use_sparse = False

                        if input_path_name in initialized_paths:  # case 1:
                            rev_block.write(f'{totals_name} = {input_path_name}.copy()')
                        elif input_id == out_id:  # case 2:
                            rev_block.comment(f'{totals_name} = identity')
                            if use_sparse:
                                # prerev_vars[totals_name] = sp.eye(output_size, format = 'csr')
                                rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}')
                            else:
                                # prerev_vars[totals_name] = np.eye(output_size)
                                rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}')
                        else:  # case 3:
                            rev_block.comment(f'{totals_name} = zero')
                            if use_sparse:
                                # prerev_vars[totals_name] = sp.csr_array((output_size, input_size))
                                # rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}*0.0')
                                rev_block.write(f'{totals_name} = sp.csr_array(({get_reverse_seed(out_id)}.shape[0], {input_size}))')
                            else:
                                # prerev_vars[totals_name] = np.zeros((output_size, input_size))
                                # rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}*0.0')
                                rev_block.write(f'{totals_name} = np.zeros(({get_reverse_seed(out_id)}.shape[0],{input_size}))')

            # Compile single instruction
            rev_single_instructions.script.write(rev_block)
            rev_single_instructions.compile()

            if self.checkpoints_bool:
                current_eval_instruction = self.checkpoint_data[snap_num]['single instructions']
                current_del_vars = self.checkpoint_data[snap_num]['del vars']
                current_del_vars_deriv = self.checkpoint_data[snap_num]['del vars during deriv']
                rev_multi_instructions.add_single_instruction(current_eval_instruction, current_del_vars_deriv)
                # rev_multi_instructions.add_single_instruction(rev_single_instructions, set())
                rev_multi_instructions.add_single_instruction(rev_single_instructions, current_del_vars)
            else:
                rev_multi_instructions.add_single_instruction(rev_single_instructions, set())

        # print statement
        # print_loading(
        #     out_str,
        #     current_op_num,
        #     self.num_ops,
        #     True)

        # exit('EXIT')
        # for key in prerev_vars:
        #     print(f'{key=}')
        return rev_multi_instructions, prerev_vars
        # return rev_block, prerev_vars
    
    # @profile
    def generate_reverse_vjp(self, output_ids, input_ids, num_vectors):
        '''
        generate the reverse mode derivative evaluation script.
        generally, loop through all variables in reverse order using
        a modified breadth-first-search and generate the AD script.

        Parameters:
        ----------
            output_ids: list or string
                list or string of UNIQUE output id(s) to take derivative of
            input_ids: list or string
                list or string of UNIQUE input id(s) to take derivative wrt
            vjp: bool
                literally does nothing
            m: integer
                number of vectors used in the VJP
        '''
        # list of output ids and input ids to get derivatives of.
        output_ids = to_unique_list(output_ids)
        input_ids = to_unique_list(input_ids)

        # dictionary of ALL predecessor nodes of inputs
        # input_ancestors: input id --> set(nodes)
        input_ancestors = {}
        all_input_ancestors = set()
        for input_id in input_ids:
            # input_ancestors[input_id] = set(nx.ancestors(self.rev_graph, self.unique_to_node[input_id]))
            all_input_ancestors = all_input_ancestors.union(set(nx.ancestors(self.rev_graph, self.unique_to_node[input_id])))
            all_input_ancestors.add(self.unique_to_node[input_id])
        # getting print statements ready
        max_outstr_len = 0
        for out_id_temp in output_ids:
            output_node = self.unique_to_node[out_id_temp]
            output_lang_name_temp = output_node.name

            if len(f'{output_lang_name_temp}-->') > max_outstr_len:
                max_outstr_len = len(f'{output_lang_name_temp}-->')

        # initialize instructions
        # name of instructions

        # get_path_name_rev = get_path_name
        get_path_name_rev = get_path_name_vjp
        # Static variables
        prerev_vars = {}

        # accumulation operations
        prerev_vars['DIAG_MULT'] = diag_mult
        prerev_vars['STD_MULT'] = std_mult

        # Keep track of which partials have already been computed.
        # We only need to compute partial jacobians once no matter how many
        # derivatives we are computing
        computed_partial_jacs = set()

        # - Do not search down a node until ALL* edges leading into it has been search as well
        # - Do not search down an implicit output untill ALL* implicit outputs have been searched
        # - * All edges dependent on the output
        fully_visited = set()  # set containing nodes that were fully visited

        # For print statements
        current_op_num = 0
        # out_str = lineup_string(f'-->', 10)
        stride = round(self.num_ops/40)
        if stride == 0:
            stride = 1

        # Data that contains information for each output
        output_info_dict = {}
        totals_names = {}
        do_not_delete_paths = set()
        all_descendants = set()
        output_ids_set = set(output_ids)
        output_nodes_set = set()
        for out_id in output_ids:
            # Get output info
            output_node = self.unique_to_node[out_id]
            output_lang_name = output_node.name
            output_shape = output_node.var.shape
            output_size = np.prod(output_shape)

            # Get all ancestors of output for checking.
            output_descendants = set(nx.descendants(self.rev_graph, output_node))
            output_descendants = output_descendants.intersection(all_input_ancestors)
            output_descendants.add(output_node)

            # Update all descendants
            all_descendants = all_descendants.union(output_descendants)

            output_info_dict[out_id] = {
                'output_node': output_node,
                'lang_name': output_lang_name,
                'shape': output_shape,
                'size': output_size,
                'id': out_id,
                'output_descendants': output_descendants,
                'initialized_paths': set()
            }

            for input_id in input_ids:
                input_path_name = get_path_name_rev(input_id, out_id=out_id)
                totals_names[out_id, input_id] = input_path_name
                do_not_delete_paths.add(input_path_name)

            do_not_delete_paths.add(get_path_name_rev(out_id, out_id=out_id))

            output_nodes_set.add(output_node)

        # Go through reversed operation list
        # We guarantee is that reversed schedule is in reverse topological order, so no need for search
        initialized_paths = set()

        if self.comm is not None:
            rev_multi_instructions = MultiInstructions(f'rev_multiinstr_{self.comm.rank}')
        else:
            rev_multi_instructions = MultiInstructions(f'rev_multiinstr')       
        
        for inverse_snap_num, current_schedule in enumerate(reversed(self.schedules)):
            

            snap_num = len(self.schedules) - inverse_snap_num - 1
            # Set up code gen
            instruction_name = 'REV_VJP'

            if self.comm is None:
                instruction_name += ': '
            else:
                instruction_name += f'{self.comm.rank}: '

            # instruction_name = 'REV_'
            for output_name in sorted(output_ids):
                output_lang_name = output_info_dict[output_name]['lang_name']
                instruction_name += f'{output_lang_name},'
            instruction_name += '-->'
            for input_name in input_ids:
                instruction_name += f'{input_name},'
            rev_block = CodeBlock(instruction_name)

            # Set up code gen
            # instruction_name = 'REV_'
            if self.comm is not None:
                instruction_name += f'_rank{self.comm.rank}'
            if self.checkpoints_bool:
                instruction_name += f'_snap{snap_num}'
            rev_single_instructions= SingleInstruction(instruction_name)

            # Initialize seeds
            if inverse_snap_num == 0:
                for out_id in output_ids:
                    if self.comm is not None:
                        if self.variable_owner_map[out_id] == self.comm.rank:
                            rev_block.write(f'{get_path_name_rev(out_id)} = {get_reverse_seed(out_id)}')
                        else:
                            rev_block.write(f'{get_path_name_rev(out_id)} = np.zeros(({num_vectors}, {output_info_dict[out_id]["size"]}))')
                    else:
                        rev_block.write(f'{get_path_name_rev(out_id)} = {get_reverse_seed(out_id)}')
                    initialized_paths.add(get_path_name_rev(out_id))

            # Main loop
            for current_node in reversed(list(current_schedule)):
                # search downstream of node 'current'
                # Node 'current' is an operation
                # queue should never contain variables that have already appeared queue before
                middle_operation = current_node

                # print(self.comm.rank, middle_operation)
                if not isinstance(middle_operation, (OperationNode, PointToPointCall)):
                    # dev error. This error should have been caught way earlier. if triggered, something is wrong with compiler.
                    raise ValueError(f'{middle_operation} must be an operation/MPI call!')

                # print(current_node.name)
                # # Make sure successors are visited?
                # for successor_var in self.rev_graph.successors(middle_operation):
                #     print(successor_var.name)
                #     fully_visited.add(successor_var)

                # We only pass through operations once. Therefore, if we already computed it,
                # do not pass through again!
                # **should NEVER trigger**
                # if middle_operation in processed_operations:
                #     continue

                # check if we need to propagate derivatives through this operation
                # If this operation is independent of all outputs and inputs, skip it
                needs_propagation = False
                if isinstance(middle_operation, (OperationNode)):
                    for out_id in output_info_dict.keys():
                        output_descendants = output_info_dict[out_id]['output_descendants']
                        if middle_operation in output_descendants:
                            needs_propagation = True
                elif isinstance(middle_operation, (PointToPointCall)):
                    for out_id in output_info_dict.keys():
                        output_descendants = output_info_dict[out_id]['output_descendants']
                        if middle_operation.var in output_descendants:
                            needs_propagation = True

                    # if needs_propagation:
                    #     print('YES!!')
                    # else:
                    #     print('NO!!')
                # for input_id in input_ancestors:
                #     if middle_operation in input_ancestors:
                #         needs_propagation = True
                # If this operation is independent of all outputs and inputs, skip it
                if not needs_propagation:
                    continue

                # if program reaches here, middle_operation has been fully visited so we now process it
                # cool print statements:
                current_op_num += 1

                # if MPI communication
                if isinstance(middle_operation, (PointToPointCall)):
                    if middle_operation.var not in all_descendants:
                        continue

                    # Path to output
                    mpi_var_id = middle_operation.var_id
                    mpi_var = middle_operation.var
                    path_out_name = get_path_name_rev(mpi_var_id)
                    mpi_var_size = np.prod(mpi_var.var.shape)

                    if isinstance(middle_operation, SendCall):
                        if path_out_name not in initialized_paths:
                            if num_vectors*mpi_var_size > 5000:
                                rev_block.write(f'{path_out_name} = sp.csr_array(({num_vectors}, {mpi_var_size}))')

                            else:
                                rev_block.write(f'{path_out_name} = np.zeros(({num_vectors}, {mpi_var_size}))')

                            initialized_paths.add(path_out_name)

                        # Write MPI adjoint
                        path_out_name_mpi = path_out_name+'_mpi'
                        vars = {}
                        middle_operation.get_adjoint_call(
                            rev_block,
                            vars,
                            path_out_name_mpi,
                            (num_vectors, mpi_var_size),
                            None,
                        )
                        rev_block.write(f'{path_out_name} += {path_out_name_mpi}')
                    elif isinstance(middle_operation, RecvCall):
                        if path_out_name not in initialized_paths:

                            if num_vectors*mpi_var_size > 5000:
                                rev_block.write(f'{path_out_name} = sp.csr_array(({num_vectors}, {mpi_var_size}))')
                            else:
                                rev_block.write(f'{path_out_name} = np.zeros(({num_vectors}, {mpi_var_size}))')

                            initialized_paths.add(path_out_name)
                        # Write MPI adjoint
                        path_out_name_mpi = path_out_name
                        vars = {}
                        middle_operation.get_adjoint_call(
                            rev_block,
                            vars,
                            path_out_name_mpi,
                            (num_vectors, mpi_var_size),
                            None,
                        )
                    prerev_vars.update(vars)
                    continue


                # :::::GENERATE CODE FOR MIDDLE_OPERATION:::::
                # If operation:
                    # Two things to do:
                    # 1) Compute partials of the operation.
                    # 2) Add partials to path.
                    # Note that 1) is only done once per operation; we do not need
                    # to compute partials of an operation more than once.

                    # 1) Compute partials:
                    # there are three ways to compute partials:
                    # 1a) Compute the Jacobian matrix independent of paths_out: called once per script
                    # --- Standard Explicit Operations
                    # --- CustomExplicitOperations: Compute Derivatives
                    # 1b) Compute the mat-mat / mat-vec product as a function of paths_out: called once per output
                    # --- ImplicitOperations: Adjoint method
                    # --- CustomExplicitOperations: Compute Jacvec Product
                    # --- CustomImplicitOperations: Adjoint method (?)

                # if MPI communication:
                    # 1) get path out if it has been computed or initialize it
                    # 2) Write adjoint of MPI call of path out 

                # For each output:
                # print('current operation: ', middle_operation.name)
                # Deallocate variables after this operation is processed
                variable_names_to_delete = set()

                # If code reach here, middle_operation is an operation node, not an MPI node.
                backend_op = middle_operation.back_operation

                jac_is_function = backend_op.is_jac_function(vjp = True)
                if not jac_is_function:  # 1a)

                    if not middle_operation in computed_partial_jacs:

                        # Add to make sure we do not compute this partial Jacobian again within this script.
                        computed_partial_jacs.add(middle_operation)

                        partials_dict = {}

                        # compute partials for each input and output of middle_operation
                        for predecessor in self.rev_graph.predecessors(middle_operation):
                            for successor in self.rev_graph.successors(middle_operation):
                                partials_name = get_deriv_name(predecessor.id, successor.id)

                                partials_dict[predecessor, successor] = {}
                                partials_dict[predecessor, successor]['name'] = partials_name

                                # Deallocate partials once they are processed

                                if self.lazy:
                                    variable_names_to_delete.add(partials_name)
                                else:
                                    if not backend_op.linear:
                                        variable_names_to_delete.add(partials_name)

                                # if not backend_op.linear:
                                    # variable_names_to_delete.add(partials_name)

                        partials_block = CodeBlock(backend_op.name + f'_{partials_name}')
                        vars = {}

                        # compute_partials for each input and output
                        is_sparse_jac = get_operation_sparsity(backend_op, self.sparsity_type)

                        # PRINT:
                        # pred_size = 0
                        # for predecessor in self.rev_graph.predecessors(middle_operation):
                        #     if np.prod(predecessor.var.shape) > pred_size:
                        #         pred_size = np.prod(predecessor.var.shape)
                        # succ_size = 0
                        # for successor in self.rev_graph.successors(middle_operation):
                        #     if np.prod(successor.var.shape) > succ_size:
                        #         succ_size = np.prod(successor.var.shape)
                        # if (pred_size > 100 and succ_size > 100) and not is_sparse_jac:
                        #     print(is_sparse_jac, f'({succ_size} x {pred_size})', middle_operation.op)
                        # elif (pred_size < 100 and succ_size < 100) and is_sparse_jac:
                        #     print(is_sparse_jac, f'({succ_size} x {pred_size})', middle_operation.op)

                        lazy = self.lazy

                        backend_op.get_partials(
                            partials_dict = partials_dict,
                            partials_block = partials_block,
                            vars = vars,
                            is_sparse_jac = is_sparse_jac,
                            lazy = lazy,
                        )
                        # backend_op.get_partials(
                        #     partials_dict = partials_dict,
                        #     partials_block = partials_block,
                        #     vars = vars,
                        #     is_sparse_jac = is_sparse_jac,
                        # )
                        # # write to script
                        # rev_block.write(partials_block)
                        prerev_vars.update(vars)

                        # write a summary of the operation
                        rev_block.write(backend_op.op_summary_block)
                        # if the input shapes are not matching, reshape (hopefully not needed)
                        if backend_op.needs_input_reshape:
                            if not backend_op.linear:
                                rev_block.write(backend_op.reshape_block)
                        # the main evaluation script is written here.
                        rev_block.write(partials_block)
                        # if the input shapes are not matching, reshape it back to original shape (hopefully not needed)
                        if backend_op.needs_input_reshape:
                            if not backend_op.linear:
                                rev_block.write(backend_op.unreshape_block)
                                # rev_block.write(f'{backend_op.linear}')


                else:  # 1b)

                    # compute partials as a function of output paths
                    partials_block = CodeBlock(backend_op.name + f'_{middle_operation.name}')
                    vars = {}

                    # path_in_names are paths from output passing through the jac_function operation
                    path_in_names = []
                    for jac_function_input in backend_op.ordered_in_names:
                        jac_function_input_id = backend_op.get_input_id(jac_function_input)
                        path_in_name = f'{middle_operation.name}_{jac_function_input_id}'
                        path_in_names.append(path_in_name)
                    # need for edge case with bracketed search with csdl variables.
                    if isinstance(backend_op.operation, BracketedSearchOperation):
                        for jac_function_input in backend_op.ordered_in_brackets:
                            jac_function_input_id = backend_op.get_input_id(jac_function_input)
                            path_in_name = f'{middle_operation.name}_{jac_function_input_id}'
                            brack_var = self.unique_to_node[jac_function_input_id]
                            rev_block.write(f'{path_in_name} = np.zeros(({num_vectors}, {np.prod(brack_var.var.shape)}))')

                    # path_out_names are paths from output to the outputs of the jac_function operation
                    # THESE PATHS SHOULD BE FULLY COMPUTED BY NOW (unless the jac_function output is independent of output or is the output)
                    path_out_names = []

                    for jac_function_output_lang in backend_op.ordered_out_names:

                        jac_function_output_id = backend_op.get_output_id(jac_function_output_lang)
                        path_out_name = get_path_name_rev(jac_function_output_id)

                        # This part is complicated. We feed in paths to the jac_function method.
                        # These paths SHOULD be computed by now. However, they may not be in two cases:
                        # 1) jac_function output is independent of output -> path is zero
                        # 2) jac_function output IS output -> path is identity

                        if path_out_name not in initialized_paths:
                            if self.unique_to_node[jac_function_output_id] in output_descendants:
                                if self.unique_to_node[jac_function_output_id] != output_node:
                                    raise ValueError(f'path output {jac_function_output_id} of jac_function operation {middle_operation.name} should have been computed but is not.')
                            if not (jac_function_output_id in output_ids_set):
                                implicit_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                                rev_block.write(f'{path_out_name} = np.zeros(({num_vectors}, {implicit_out_size}))')
                                # rev_block.write(f'{path_out_name} = np.zeros(({output_size}, {implicit_out_size}))')
                                # rev_block.write(f'{path_out_name} = sp.csc_matrix(({output_size}, {implicit_out_size}))')


                                # rev_block.write(f'{path_out_name} = sp.eye({output_size}, format = \'csr\')')
                        
                        #     if jac_function_output_id in output_ids_set:
                        #         c_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                        #         rev_block.write(f'{path_out_name} = {get_reverse_seed(jac_function_output_id)}@np.eye({c_out_size})')
                        #         # rev_block.write(f'{path_out_name} = sp.eye({output_size}, format = \'csr\')')
                        #     else:
                        #         implicit_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                        #         rev_block.write(f'{path_out_name} = np.zeros(({num_vectors}, {implicit_out_size}))')
                        #         # rev_block.write(f'{path_out_name} = np.zeros(({output_size}, {implicit_out_size}))')
                        #         # rev_block.write(f'{path_out_name} = sp.csc_matrix(({output_size}, {implicit_out_size}))')
                        # else:
                        #     if jac_function_output_id in output_ids_set:
                        #         c_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                        #         rev_block.write(f'{path_out_name} += {get_reverse_seed(jac_function_output_id)}@np.eye({c_out_size})')
                        #         # rev_block.write(f'{path_out_name} = sp.eye({output_size}, format = \'csr\')')

                        path_out_names.append(path_out_name)
                        variable_names_to_delete.add(path_out_name)

                    # compute and write to script
                    backend_op.get_accumulation_function(path_in_names, path_out_names, partials_block, vars)
                    rev_block.write(partials_block)
                    prerev_vars.update(vars)
                
                # 2) Adding partials to a path:
                # again, if we have jacobian as a matrix, right multiply to a path and add.
                # on the other hand, if we have jacobian as a function just add as it is already right-multiplied.
                for successor in self.rev_graph.successors(middle_operation):
                    # successor should be VariableNode instance and is the input of middle operation

                    # If successor does not depend on any input we want to take derivative of, no need to do anything.
                    ignore_successor = True
                    for input_id in input_ids:
                        if input_id == successor.id:
                            ignore_successor = False
                        # elif successor in input_ancestors[input_id]:
                        #     # elif nx.has_path(self.rev_graph, successor, self.unique_to_node[input_id]):
                        #     ignore_successor = False
                    if successor in all_input_ancestors:
                        ignore_successor = False
                    if ignore_successor:
                        continue

                    # Accumulate matmats
                    if not jac_is_function:  # if we have the jacobian matrix, right multiply

                        # Iterate through each individual jac if matrix
                        for predecessor in self.rev_graph.predecessors(middle_operation):

                            # predecessor is a VariablNode instance and are the outputs of middle_operation

                            # If predecessor is independent of output, there will be no path so ignore.
                            compute_path_pred = False
                            # if output_node == predecessor:
                            if predecessor in output_nodes_set:
                                compute_path_pred = True
                            elif predecessor in all_descendants:
                                compute_path_pred = True
                            if not compute_path_pred:
                                continue
                            
                            # Right multiplication of path
                            path_successor = get_path_name_rev(successor.id)
                            path_current = get_path_name_rev(predecessor.id)
                            partials_name = get_deriv_name(predecessor.id, successor.id)

                            # We shouldn't need path_current after this operation is fully processed. As path_current is unique (?) to this operation, we can delete it.
                            variable_names_to_delete.add(path_current)


                            # # If this is the first iteration in BFS, we need to set seed for output
                            # if path_successor not in initialized_paths:
                            #     if predecessor in output_nodes_set:
                            #         # The line below had issues with pointers.
                            #         # initialized_path_string = get_init_path_string(partials_name, backend_op, self.sparsity_type)
                            #         if path_current in initialized_paths:
                            #             rev_block.write(f'{path_current} += {get_reverse_seed(predecessor.id)}')
                            #             initialized_path_string = get_successor_path_string(path_current, partials_name, backend_op)
                            #         else:
                            #             initialized_path_string = get_successor_path_string(get_reverse_seed(predecessor.id), partials_name, backend_op)
                            #         rev_block.write(f'{path_successor} = {initialized_path_string}')
                            #         initialized_paths.add(path_successor)
                            #         continue
                            # else:
                            #     if predecessor in output_nodes_set:
                            #         initialized_path_string = get_successor_path_string(get_reverse_seed(predecessor.id), partials_name, backend_op)
                            #         rev_block.write(f'{path_successor} += {initialized_path_string}')
                            #     # initialized_paths.add(path_successor)
                            #     # continue

                            if path_current not in initialized_paths:
                                # path_current should be calculated already. if not, :(
                                raise ValueError(f'path {path_current} has not yet been computed.')

                            # Now we write the path accumulation
                            # successor_string = f'{path_current}@{partials_name}'

                            # is either
                            # successor_string = 'path_current@partials_name'
                            # successor_string = 'DIAG_MULT(path_current, partials_name)'
                            successor_string = get_successor_path_string(path_current, partials_name, backend_op)

                            if path_successor not in initialized_paths:
                                # if path_successor not yet initialized, set it.
                                rev_block.write(f'{path_successor} = {successor_string}')
                                initialized_paths.add(path_successor)
                            else:
                                # if path_successor has been initialized, add the path.
                                rev_block.write(f'{path_successor} += {successor_string}')
                    else:  # jac is function

                        # Right multiplication of path
                        path_successor = get_path_name_rev(successor.id)

                        # Now we write the path accumulation
                        # If jac is function, partials_name already includes the path_current, so we do not multiply!
                        successor_string = f'{middle_operation.name}_{successor.id}'

                        if path_successor not in initialized_paths:
                            # if path_successor not yet initialized, set it.
                            rev_block.write(f'{path_successor} = {successor_string}')
                            initialized_paths.add(path_successor)
                        else:
                            # if path_successor has been initialized, add the path.
                            rev_block.write(f'{path_successor} += {successor_string}')
                # Delete all variables that are no longer needed
                for deallocate_var in variable_names_to_delete:
                    # continue
                    if deallocate_var not in do_not_delete_paths:
                        # rev_block.write(f'{deallocate_var} = None')
                        rev_block.write(f'del {deallocate_var}')

            if snap_num == 0:
                for input_id in input_ids:
                    input_path_name = get_path_name_rev(input_id)
                    # input_path_name = totals_names[out_id, input_id]
                    input_size = np.prod(self.unique_to_node[input_id].var.shape)

                    if input_size*num_vectors > 5000:
                        use_sparse = True
                    else:
                        use_sparse = False

                    if input_path_name in initialized_paths:  # case 1:
                        rev_block.write(f'{input_path_name} = {input_path_name}.copy()')
                    elif input_id in output_ids_set:  # case 2:
                        rev_block.comment(f'{input_id} = identity')
                        rev_block.write(f'{input_path_name} = {get_reverse_seed(input_id)}')
                    else:  # case 3:
                        rev_block.comment(f'{input_path_name} = zero')
                        if use_sparse:
                            # prerev_vars[totals_name] = sp.csr_array((output_size, input_size))
                            # rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}*0.0')
                            rev_block.write(f'{input_path_name} = sp.csr_array(({num_vectors}, {input_size}))')
                        else:
                            # prerev_vars[totals_name] = np.zeros((output_size, input_size))
                            # rev_block.write(f'{totals_name} = {get_reverse_seed(out_id)}*0.0')
                            rev_block.write(f'{input_path_name} = np.zeros(({num_vectors},{input_size}))')

            # Compile single instruction
            rev_single_instructions.script.write(rev_block)
            rev_single_instructions.compile()

            if self.checkpoints_bool:
                current_eval_instruction = self.checkpoint_data[snap_num]['single instructions']
                current_del_vars = self.checkpoint_data[snap_num]['del vars']
                current_del_vars_deriv = self.checkpoint_data[snap_num]['del vars during deriv']
                rev_multi_instructions.add_single_instruction(current_eval_instruction, current_del_vars_deriv)
                # rev_multi_instructions.add_single_instruction(rev_single_instructions, set())
                rev_multi_instructions.add_single_instruction(rev_single_instructions, current_del_vars)
            else:
                rev_multi_instructions.add_single_instruction(rev_single_instructions, set())

        # print statement
        # print_loading(
        #     out_str,
        #     current_op_num,
        #     self.num_ops,
        #     True)

        # exit('EXIT')
        # for key in prerev_vars:
        #     print(f'{key=}')
        return rev_multi_instructions, prerev_vars
        # return rev_block, prerev_vars


    def replace_brackets_lang_var_to_rep_var(self, bracket_node):
        # iterate through brackets map to see if there are any matching variables.
        vars_to_find = []

        # if node
        bracket_op = bracket_node.op
        for state_name in bracket_op.brackets:
            l, u = bracket_op.brackets[state_name]

            if isinstance(l, Variable):
                print(l)

            if isinstance(u, Variable):
                print(u)


def print_loading(
        output_str,
        current_op_num,
        total_op_num,
        start=False):
    dots = '.'*round(40*(current_op_num/total_op_num))
    dot_str = lineup_string(dots, 40)

    ratio_str = lineup_string(
        f'({current_op_num}/{total_op_num})',
        len(f'({total_op_num}/{total_op_num})'))
    if start:
        print(f'{output_str} {ratio_str} |{dot_str}|')
    else:
        print(f'{output_str} {ratio_str} |{dot_str}|', end="\r")


def get_successor_path_string(
        path_current,
        partials_name,
        backend_op):
    """
    returns matrix multiplication or diagonal multiplication depending on elementwise operations or not
    """

    if backend_op.elementwise:
        # specialized diagonal multiplication
        string =  f'DIAG_MULT({path_current},{partials_name})'
    else:
        # standard multiplication
        string =  f'STD_MULT({path_current},{partials_name})'

    # string += f'\nprint(\'{path_current}\',\'{partials_name}\')'
    return string


def get_init_path_string(partials_name, backend_op, sparsity_type):
    """
    returns the string to set initial 'seed' derivative of output
    """
    # print(backend_op.elementwise, backend_op)
    if not backend_op.elementwise:
        # return f'{partials_name}.copy()'
        return f'sp.csc_matrix({partials_name})'
    else:
        is_sparse = get_operation_sparsity(backend_op, sparsity_type)

        if is_sparse:
            # return f'sp.diags({partials_name}, format = \'csc\')'
            return f'sp.csc_matrix(np.diagflat({partials_name}))'

        else:
            return f'np.diagflat({partials_name})'

# def initialize_seed_path(output_seed, partials_name):
    
    
def get_operation_sparsity(backend_op, sparsity_type):
    """
    returns True or False on whether operation partials should be sparse or dense
    """
    if sparsity_type == 'auto':
        is_sparse_jac = backend_op.determine_sparse()
        if is_sparse_jac is None:
            raise ValueError(f'dev error. is_sparse_jac is None for {backend_op}')
    elif sparsity_type == 'sparse':
        is_sparse_jac = True
    elif sparsity_type == 'dense':
        is_sparse_jac = False
    else:
        raise ValueError('is_sparse_jac is not one of auto, sparse, dense')

    return is_sparse_jac
