# from networkx import nx
import networkx as nx
import numpy as np
import scipy.sparse as sp

from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.general_utils import get_deriv_name, to_list, get_path_name, increment_id, lineup_string
from python_csdl_backend.core.operation_map import (
    get_backend_op,
    get_backend_implicit_op,
    get_backend_custom_explicit_op,
    get_backend_custom_implicit_op,
)
from python_csdl_backend.core.accumulation_operations import diag_mult

from csdl import Operation, StandardOperation, ImplicitOperation, CustomExplicitOperation, Variable, Output, BracketedSearchOperation, CustomImplicitOperation
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.variable import Variable


class SystemGraph(object):
    """

    The graph object contains a networkx graph to work with.
    Given a graph, return codeblock objects for model evaluation and derivatives.
    When iterating through each node, if the node is another graph object, recursively call the function

    """

    def __init__(self,
                 rep,
                 mode,
                 sparsity_type='auto',
                 dvs=None,
                 objective=None,
                 constraints=None,
                 opt_bool=False):

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

        self.process_rep()

        # rev or fwd
        self.mode = mode

        # sparsity type
        # if name == 'auto' ,   automatically use dense or sparse jacobians
        # if name == 'sparse' , use sparse jacobians
        # if name == 'dense' ,  use dense jacobians
        self.sparsity_type = sparsity_type

        self.name = 'default'

        if mode == 'rev':
            self.reverse_graph()

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

        for node in self.eval_graph.nodes:

            if isinstance(node, VariableNode):
                # increment id
                unique_id_num = increment_id(unique_id_num)

                # set unique id
                # this is the variable name used in the generated script
                unique_id = unique_id_num+'_'+node.name
                node.id = unique_id
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

    def get_analytics(self, filename):
        '''
        get and print information about the graph
        '''

        # initialize return dict
        operation_analytics = {}
        operation_analytics['elementwise'] = {}
        operation_analytics['elementwise']['count'] = 0

        # Write to text file
        with open(filename, 'w') as f:
            f.write(f'node name, \t node object\n')

        # collect information from operations
        for node in self.eval_graph:

            # write to filename a summary of each node
            with open(filename, 'a') as f:

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

            # keep a count of every type of node in the graph for printing
            if isinstance(node, OperationNode):
                csdl_node = node.op

                if isinstance(csdl_node, StandardOperation):
                    if csdl_node.properties['elementwise']:
                        operation_analytics['elementwise']['count'] += 1
            else:
                csdl_node = node.var

            if type(csdl_node) not in operation_analytics:
                operation_analytics[type(csdl_node)] = {}
                operation_analytics[type(csdl_node)]['count'] = 0
            operation_analytics[type(csdl_node)]['count'] += 1

        return operation_analytics

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
            state_vals: Dict
                the initial value for every state value.
            variable_info: Dict
                split variables into outputs/inputs
        '''

        eval_block = CodeBlock('system evaluation block')  # initialize evaluation block
        preeval_vars = {}  # initialize precomputed variables
        state_vals = {}  # initialize initial state values
        variable_info = {  # initialize variable info dict
            'inputs': {},
            'outputs': {},
            'leaf_start': {},
        }

        # Loop through sorted graph
        for node in nx.topological_sort(self.eval_graph):

            if not isinstance(node, OperationNode):
                # If variable, set state initial value given by user
                csdl_node = node.var

                if isinstance(csdl_node.val, np.ndarray):
                    csdl_node.val = csdl_node.val.astype('float64')
                    if isinstance(csdl_node.val, np.matrix):
                        csdl_node.val = np.asarray(csdl_node.val).astype('float64')
                state_vals[node.id] = csdl_node.val

                promoted_id = node.promoted_id

                if isinstance(csdl_node, Output):
                    if node.name[0] != '_':
                        if promoted_id in self.promoted_to_unique:
                            variable_info['outputs'][promoted_id] = {}
                            variable_info['outputs'][promoted_id]['shape'] = csdl_node.shape
                            variable_info['outputs'][promoted_id]['size'] = np.prod(csdl_node.shape)

                if isinstance(csdl_node, Input):
                    variable_info['inputs'][promoted_id] = {}
                    variable_info['inputs'][promoted_id]['shape'] = csdl_node.shape
                    variable_info['inputs'][promoted_id]['size'] = np.prod(csdl_node.shape)

                if len(list(self.eval_graph.predecessors(node))) == 0:
                    variable_info['leaf_start'][promoted_id] = {}
                    variable_info['leaf_start'][promoted_id]['shape'] = csdl_node.shape
                    variable_info['leaf_start'][promoted_id]['size'] = np.prod(csdl_node.shape)

                # only write stuff for operations.
                continue

            csdl_node = node.op
            all_op_classes = (StandardOperation, CustomExplicitOperation, ImplicitOperation, BracketedSearchOperation, CustomImplicitOperation)
            if isinstance(csdl_node, all_op_classes):

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
                    raise NotImplementedError(f'{csdl_nodes} operation not found')
                node.back_operation = back_operation

                # Get the evaluation procedure for current operation
                eval_block_temp = CodeBlock(back_operation.name, newline=False, add_name=False)  # script evaluation block
                vars = {}  # precomputed variables
                back_operation.get_evaluation(eval_block_temp, vars)

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

        return eval_block, preeval_vars, state_vals, variable_info

    # @profile
    def generate_reverse(self, output_ids, input_ids):
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
        output_ids = to_list(output_ids)
        input_ids = to_list(input_ids)

        # dictionary of ALL predecessor nodes of inputs
        # input_ancestors: input id --> set(nodes)
        input_ancestors = {}
        for input_id in input_ids:
            input_ancestors[input_id] = set(nx.ancestors(self.rev_graph, self.unique_to_node[input_id]))

        # getting print statements ready
        max_outstr_len = 0
        for out_id_temp in output_ids:
            output_node = self.unique_to_node[out_id_temp]
            output_lang_name_temp = output_node.name

            if len(f'{output_lang_name_temp}-->') > max_outstr_len:
                max_outstr_len = len(f'{output_lang_name_temp}-->')

        # initialize instructions
        # name of instructions
        instruction_name = 'REV:'
        for output_name in output_ids:
            instruction_name += f'{output_name},'
        instruction_name += '-->'
        for input_name in input_ids:
            instruction_name += f'{input_name},'
        rev_block = CodeBlock(instruction_name)

        # Static variables
        prerev_vars = {}

        # accumulation operations
        prerev_vars['DIAG_MULT'] = diag_mult

        # Keep track of which partials have already been computed.
        # We only need to compute partial jacobians once no matter how many
        # derivatives we are computing
        computed_partial_jacs = set()

        # Perform modified breadth-first-search
        # For each output:
        for out_id in output_ids:

            # write for reference
            rev_block.newline()
            rev_name = f':::::::{out_id}-->'
            for input_name in input_ids:
                rev_name += f'{input_name},'
            rev_name += ':::::::'
            rev_block.comment(rev_name)

            # Get output shape
            output_node = self.unique_to_node[out_id]
            output_lang_name = output_node.name
            output_shape = output_node.var.shape
            output_size = np.prod(output_shape)

            # list of initialized paths.
            initialized_paths = set()

            # We pass through each operation at most one time.
            # Keep track of this here.
            processed_operations = set()

            # Get all ancestors of output for checking.
            output_descendants = set(nx.descendants(self.rev_graph, output_node))

            # prepare string:
            out_str = lineup_string(f'{output_lang_name}-->', max_outstr_len)
            stride = round(self.num_ops/40)
            if stride == 0:
                stride = 1

            # --------- perform BFS --------- :
            # Implement *modified* Breadth First Search
            # - Do not search down a node until ALL* edges leading into it has been search as well
            # - Do not search down an implicit output untill ALL* implicit outputs have been searched
            # - * All edges dependent on the output
            fully_visited = set()  # set containing nodes that were fully visited
            queue = [output_node]  # list to perform BFS on in order

            # Check if output is a leaf node.
            # If so, do not perform BFS on it as there is nothing to search through.
            if not output_descendants:
                queue = []
            current_op_num = 0
            while queue:
                # search downstream of node 'current'
                # Node 'current' HAS to be a variable
                # queue should never contain variables that have already appeared queue before
                current = queue.pop(0)
                fully_visited.add(current)

                # Get the operation which outputs variable 'current'
                # variable 'successor's --> operation 'middle_operation' --> variable 'current'
                middle_operations = self.rev_graph.successors(current)
                middle_operation_list = list(middle_operations)
                if len(middle_operation_list) != 1:
                    # dev error. This error should have been caught way earlier. if triggered, something is wrong with compiler.
                    raise ValueError(f'Variable {current} has {len(middle_operation_list)} predecessors.')
                middle_operation = middle_operation_list[0]  # should be of type OperationNode
                backend_op = middle_operation.back_operation
                if not isinstance(middle_operation, OperationNode):
                    # dev error. This error should have been caught way earlier. if triggered, something is wrong with compiler.
                    raise ValueError(f'Variable {current} predecessor {middle_operation} is not an operation!')

                # We only pass through operations once. Therefore, if we already computed it,
                # do not pass through again!
                if middle_operation in processed_operations:
                    continue

                # check if the operation is fully visited.
                # Make sure all outputs of the operation have had their respective paths completed.
                op_fully_visited = True
                for middle_op_predecessor in self.rev_graph.predecessors(middle_operation):

                    # middle_op_predecessor are the outputs of the middle operation

                    # First, if middle_op_predecessor is not connected to the current output, we
                    # will never visit it so ignore it.
                    ignore_pred = True
                    if output_node == middle_op_predecessor:
                        ignore_pred = False
                    elif middle_op_predecessor in output_descendants:
                        ignore_pred = False
                    if ignore_pred:
                        continue

                    # output is now dependent on middle_op_predecessor.
                    # if middle_op_predecessor is NOT in fully_visited, this operation is
                    # not ready to process
                    if middle_op_predecessor not in fully_visited:
                        op_fully_visited = False

                # If this operation's output paths are incomplete, do not compute the partials
                if not op_fully_visited:
                    continue

                # if program reaches here, middle_operation has been fully visited so we now process it
                processed_operations.add(middle_operation)

                # cool print statements:
                current_op_num += 1
                if current_op_num == 1:
                    print_loading(
                        out_str,
                        current_op_num,
                        self.num_ops,
                        False)

                if (current_op_num) % stride == 0:
                    print_loading(
                        out_str,
                        current_op_num,
                        self.num_ops,
                        False)

                # :::::GENERATE CODE FOR MIDDLE_OPERATION:::::
                # Two things to do:
                # 1) Compute partials of the operation.
                # 2) Add partials to path.
                # Note that 1) is only done once per operation; we do not need
                # to compute partials of an operation more than once.

                # 1) Compute partials:
                # We only compute operation partials if the output variables have been fully visited
                # If so...
                # there are two ways to compute partials:
                # 1a) Compute the Jacobian matrix independent of paths_out: called once per script
                # --- Standard Explicit Operations
                # --- CustomExplicitOperations: Compute Derivatives
                # 1b) Compute the mat-mat / mat-vec product as a function of paths_out: called once per output
                # --- ImplicitOperations: Adjoint method
                # --- CustomExplicitOperations: Compute Jacvec Product
                # --- CustomImplicitOperations: Adjoint method (?)

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

                        backend_op.get_partials(partials_dict, partials_block, vars, is_sparse_jac)

                        # # write to script
                        # rev_block.write(partials_block)
                        prerev_vars.update(vars)

                        # write a summary of the operation
                        rev_block.write(backend_op.op_summary_block)
                        # if the input shapes are not matching, reshape (hopefully not needed)
                        if backend_op.needs_input_reshape:
                            rev_block.write(backend_op.reshape_block)
                        # the main evaluation script is written here.
                        rev_block.write(partials_block)
                        # if the input shapes are not matching, reshape it back to original shape (hopefully not needed)
                        if backend_op.needs_input_reshape:
                            rev_block.write(backend_op.unreshape_block)

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
                            rev_block.write(f'{path_in_name} = np.zeros(({output_size}, {np.prod(brack_var.var.shape)}))')

                    # path_out_names are paths from output to the outputs of the jac_function operation
                    # THESE PATHS SHOULD BE FULLY COMPUTED BY NOW (unless the jac_function output is independent of output or is the output)
                    path_out_names = []

                    for jac_function_output_lang in backend_op.ordered_out_names:

                        jac_function_output_id = backend_op.get_output_id(jac_function_output_lang)
                        path_out_name = get_path_name(jac_function_output_id)

                        # This part is complicated. We feed in paths to the jac_function method.
                        # These paths SHOULD be computed by now. However, they may not be in two cases:
                        # 1) jac_function output is independent of output -> path is zero
                        # 2) jac_function output IS output -> path is identity

                        if path_out_name not in initialized_paths:
                            if self.unique_to_node[jac_function_output_id] in output_descendants:
                                raise ValueError(f'path output {jac_function_output_id} of jac_function operation {middle_operation.name} should have been computed but is not.')
                            if jac_function_output_id == out_id:
                                rev_block.write(f'{path_out_name} = np.eye({output_size})')
                            else:
                                implicit_out_size = np.prod(self.unique_to_node[jac_function_output_id].var.shape)
                                rev_block.write(f'{path_out_name} = np.zeros(({output_size}, {implicit_out_size}))')

                        path_out_names.append(path_out_name)

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
                        elif successor in input_ancestors[input_id]:
                            # elif nx.has_path(self.rev_graph, successor, self.unique_to_node[input_id]):
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
                            path_successor = get_path_name(successor.id)
                            path_current = get_path_name(predecessor.id)
                            partials_name = get_deriv_name(predecessor.id, successor.id)

                            # If this is the first iteration in BFS, we need to set seed for output
                            if predecessor == output_node:
                                # The line below had issues with pointers.
                                initialized_path_string = get_init_path_string(partials_name, backend_op, self.sparsity_type)
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
                        path_successor = get_path_name(successor.id)

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

                    # Determine whether this successor is ready to travel down for BFS.
                    # We check by making sure all of successor's direct predecessor variables are in 'fully_visited'
                    queue_successor = True
                    for successors_predecessor_op in self.rev_graph.predecessors(successor):
                        for successors_predecessor in self.rev_graph.predecessors(successors_predecessor_op):
                            # successors_predecessor is a variable upstream

                            # if successors_predecessor not related to output, ignore it
                            if successors_predecessor not in output_descendants:
                                continue

                            # if successors_predecessor is related to output and not visited, wait.
                            if successors_predecessor not in fully_visited:
                                queue_successor = False

                    # If so, add successor to the queue
                    if queue_successor:
                        # if successor.promoted_id[0] != '_':
                        #     print(successor.promoted_id, successor.var.shape)

                        # # successor may be fully visited more than once if MiSo operation
                        if successor in fully_visited:
                            raise ValueError(f'Cannot re-queue successor {successor}.')

                        is_leaf_node = (self.rev_graph.out_degree(successor) == 0)
                        # If successor is a leaf node, we don't need to add it to queue.
                        # Remember queue stores list of nodes to perform BFS on
                        if not is_leaf_node:
                            queue.append(successor)

                # rev_block.write('fma_time += time.time()-start')

            # Finally, set the derivatives
            # There are three possible cases for setting the derivatives:
            # 1) If the input is found in initialized_paths, the derivative is the path
            # 2) If the input is NOT found in initialized_paths AND output == input, the derivative is identity
            # 3) If the input is NOT found in initialized_paths AND output != input (aka all other possibilities), the derivative is zero
            for input_id in input_ids:
                totals_name = get_deriv_name(out_id, input_id, partials=False)
                input_path_name = get_path_name(input_id)

                if input_path_name in initialized_paths:  # case 1:
                    rev_block.write(f'{totals_name} = {input_path_name}.copy()')
                elif input_id == out_id:  # case 2:
                    rev_block.comment(f'{totals_name} = identity')
                    prerev_vars[totals_name] = np.eye(output_size)
                else:  # case 3:
                    rev_block.comment(f'{totals_name} = zero')
                    input_size = np.prod(self.unique_to_node[input_id].var.shape)
                    prerev_vars[totals_name] = np.zeros((output_size, input_size))

            # print statement
            print_loading(
                out_str,
                current_op_num,
                self.num_ops,
                True)

        return rev_block, prerev_vars

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
        return f'DIAG_MULT({path_current},{partials_name})'
    else:
        # standard multiplication
        return f'{path_current}@{partials_name}'


def get_init_path_string(partials_name, backend_op, sparsity_type):
    """
    returns the string to set initial 'seed' derivative of output
    """
    # print(backend_op.elementwise, backend_op)
    if not backend_op.elementwise:
        return f'{partials_name}.copy()'
    else:
        is_sparse = get_operation_sparsity(backend_op, sparsity_type)

        if is_sparse:
            return f'sp.diags({partials_name}, format = \'csc\')'
        else:
            return f'np.diagflat({partials_name})'


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
