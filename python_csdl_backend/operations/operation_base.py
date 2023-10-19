
from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.operation_utils import list_to_argument_str


class OperationBase():

    def __init__(self, operation, nx_inputs, nx_outputs, name, **kwargs):
        """
        Base method for operations.
        get_evaluation must be overwritten.
        One or both of get_partials/get_prepartials must be overwritten.

        Parameters:
        -----------
            inputs: Dict
                {node string --> node object}

            outputs: Dict
                {node string --> node object}

            name: str

        """
        self.name = name+'_eval'
        self.operation = operation
        self.nx_inputs_dict = nx_inputs
        self.nx_outputs_dict = nx_outputs
        self.jac_is_function = False
        self.op_summary_block = CodeBlock(add_name=False)
        self.elementwise = operation.properties['elementwise']
        self.linear = operation.properties['linear']

        # map to landuage variable and representation variable
        self.input_csdl_to_rep = {}
        self.input_rep_to_csdl = {}
        self.input_name_to_unique = {}
        self.output_name_to_unique = {}

        self.map_predecessors()
        self.map_successors()

        self.id_to_input_name = {}
        for input_name, id in self.input_name_to_unique.items():
            self.id_to_input_name[id] = input_name

        self.id_to_output_name = {}
        for output_name, id in self.output_name_to_unique.items():
            self.id_to_output_name[id] = output_name

        if len(self.output_name_to_unique) != len(self.id_to_output_name):
            # dev error, not a user error
            raise ValueError('size mismatch')

        if len(self.input_name_to_unique) != len(self.id_to_input_name):
            # dev error, not a user error
            raise ValueError('size mismatch')

        # Check if any input variables need reshaping.
        self.needs_input_reshape = False
        self.reshape_block = CodeBlock(newline=False, add_name=False)
        self.unreshape_block = CodeBlock(newline=False, add_name=False)
        self.check_input_reshaping()

        # Keep a commented block that describes the operation for easy debugging
        # n_in = list(nx_inputs)
        # n_out = list(nx_outputs)
        # inputs_string = list_to_argument_str(n_in)
        # outputs_string = list_to_argument_str(n_out)
        inputs_string_lang = list_to_argument_str([self.get_lang_input(x) for x in nx_inputs])
        outputs_string_lang = list_to_argument_str([self.get_lang_output(x) for x in nx_outputs])
        inputs_shapes = list_to_argument_str([str(lang_input_var.shape) for lang_input_var in self.input_csdl_to_rep])
        outputs_shapes = list_to_argument_str([str(successor.var.shape) for successor in self.nx_outputs_dict.values()])
        for an_output in nx_outputs.values():
            self.full_namespace = an_output.unpromoted_namespace
        self.op_summary_block.comment(f'op {self.name}')
        # self.op_summary_block.comment(f'REP:  {inputs_string} --> {outputs_string}')
        self.op_summary_block.comment(f'LANG: {inputs_string_lang} --> {outputs_string_lang}')
        self.op_summary_block.comment(f'SHAPES: {inputs_shapes} --> {outputs_shapes}')
        self.op_summary_block.comment(f'full namespace: {self.full_namespace}')

    def map_predecessors(self):
        """
        we are given csdl_operation.dependencies and
        representation_operation.predecessors.

        We need to map between the two. Also, map language names to representation names.

        If rep predecessor is not a connection variable, find the local name of rep predecessor
        and the matching local name of lang predecessor.

        If rep predecessor IS a connection source, find the matching predecessor lang variable object.

        raise error if matching variable is not found.
        """

        # # Map predecessors from user-defined CSDL variable objects to representation variable objects
        # for csdl_pred in self.operation.dependencies:
        #     # Find correct predecessor
        #     pred = list(csdl_pred.rep_node)[0]

        #     # If there is a mismatch somehow, find the correct one
        #     # (Unfortunately this makes it O(n**2))
        #     found = False

        #     print(pred.var.name)
        #     for i in csdl_pred.rep_nodes:
        #         if not hasattr(i, 'id'):
        #             print('\t', i.var)
        #         else:
        #             print('\t', i.var, i.id)

        #     print([(idd, self.nx_inputs_dict[idd].var) for idd in self.nx_inputs_dict.keys()])
        #     # print(pred.var, [node.var for node in csdl_pred.rep_nodes])
            
        #     # print(pred.var, [node.var for node in csdl_pred.rep_nodes])
        #     if pred.id not in self.nx_inputs_dict:
        #         for rep_node in csdl_pred.rep_nodes:

        #             if not hasattr(rep_node, 'id'):
        #                 continue
        #             if rep_node.id in self.nx_inputs_dict:
        #                 pred = rep_node
        #                 found = True
        #                 break
        #     else:
        #         found = True

        #     # Raise error:
        #     if not found:

        #         # for i in pred.connected_to:
        #         #     print(i.id, i, i.var.name)
        #         # for i in pred.declared_to:
        #         #     print(i.id, i, i.var.name)

        #         print('\n\n::::::ERROR::::::')
        #         print('CSDL variable: ', csdl_pred.name)
        #         for rn in csdl_pred.rep_nodes:
        #             if hasattr(rn, 'id'):
        #                 rnid = rn.id
        #             else:
        #                 rnid = 'no id'
        #             s = f'\t{rnid}, {rn}, {rn.var}, {rn.var.name}'
        #             print(s)
        #         print('Found IR node:', pred)

        #         raise KeyError(f'input {pred.name} ({pred.id}) not found in nx_inputs_dict \n {self.nx_inputs_dict}, \n')

        #     # Set correct mapping
        #     self.input_csdl_to_rep[csdl_pred] = pred
        #     self.input_rep_to_csdl[pred] = csdl_pred

        #     # if not hasattr(pred, 'id'):
        #     #     print(csdl_pred.name, [f'\n{rn}, {rn.var}, {rn.var.name}' for rn in csdl_pred.rep_nodes])
        #     self.input_name_to_unique[csdl_pred.name] = pred.id
        # return
        
        # OLD implementation that was less efficient
        for pred in self.nx_inputs_dict.values():
            found_match = False
            # if not part of a connection, find the variable associated with the dependencies by the local name
            # first check if variable is a connection and find the tgt node.
            if len(pred.connected_to) > 0:

                # if node is a connection source, compare the language variable objects

                # check if the input is a source node itself
                for csdl_pred in self.operation.dependencies:
                    if pred.var is csdl_pred:
                        found_match = True
                        self.input_csdl_to_rep[csdl_pred] = pred
                        self.input_rep_to_csdl[pred] = csdl_pred
                        self.input_name_to_unique[csdl_pred.name] = pred.id

                # otherwise, check if the connection targets are a source
                for tgt in pred.connected_to:
                    for csdl_pred in self.operation.dependencies:
                        # print('\t', csdl_pred, csdl_pred.name)
                        if tgt.var.name == csdl_pred.name:
                            if found_match:
                                if csdl_pred in self.input_csdl_to_rep:
                                    continue
                                # This is a dev error. If this has been thrown, user has not necessarily made an error.
                                raise KeyError(f'already found a match')
                            found_match = True
                            self.input_csdl_to_rep[csdl_pred] = pred
                            self.input_rep_to_csdl[pred] = csdl_pred
                            self.input_name_to_unique[csdl_pred.name] = pred.id

            # if not a connection or connection target not found, try to match declared variables.
            if found_match == True:
                continue

            # Loop through predecessors and check if local names match
            for csdl_pred in self.operation.dependencies:

                # check names
                if '.' in csdl_pred.name:
                    csdl_pred_name = csdl_pred.name.split('.')[-1]
                else:
                    csdl_pred_name = csdl_pred.name
                if csdl_pred_name == pred.name:

                    # If we already found a match, raise an error
                    if found_match:

                        if csdl_pred in self.input_csdl_to_rep:
                            continue

                        # This is a dev error. If this has been thrown, user has not necessarily made an error.
                        # If we already found a match, that means two inputs have the same local name (?).
                        # I guess this is actually happen
                        raise KeyError(f'already found a match')

                    # set found_match = True and set mapping
                    found_match = True
                    self.input_csdl_to_rep[csdl_pred] = pred
                    self.input_rep_to_csdl[pred] = csdl_pred
                    self.input_name_to_unique[csdl_pred.name] = pred.id

            # If we haven't found a match after looping through all predecessors, throw error
            if not found_match:
                print('\n\n::::::ERROR::::::')
                print('current operation:', self.operation.name, self.operation)
                print('predecessor (connection source) without match:', pred.namespace, pred.name)
                print('all REP predecessors:')
                for other_pred in self.nx_inputs_dict.values():
                    print('\t', other_pred.namespace, other_pred.name, other_pred.var)
                print('all LANG predecessors:')
                for csdl_pred in self.operation.dependencies:
                    print('\t', csdl_pred.name, csdl_pred)

                print()
                for tgt in pred.connected_to:
                    print(tgt, tgt.var, tgt.var.name)
                    for tgtgt in tgt.connected_to:
                        print(tgtgt, tgtgt.var, tgtgt.var.name)
                print('::::::ERROR::::::\n\n')

                # This is a dev error. If this has been thrown, user has not necessarily made an error.
                raise KeyError(f'Did not find match')

        # Uncomment to print
        # for csdl_pred, rep_pred in input_csdl_to_rep.items():
        #     print(csdl_pred.name, '-->', rep_pred.name)

    def map_successors(self):
        """
        Similar to inputs, map successors to outputs.

        Unlike mapping predecessors, successors will never be merged in the representation
        """

        for successor in self.nx_outputs_dict.values():
            self.output_name_to_unique[successor.name] = successor.id

    def get_input_id(self, lang_input_name):
        """
        given a local input name, return unique ID
        """
        return self.input_name_to_unique[lang_input_name]

    def get_output_id(self, lang_output_name):
        """
        given a local output name, return unique ID
        """
        return self.output_name_to_unique[lang_output_name]

    def get_lang_input(self, input_id):
        """
        given an input id, return variable name
        """
        return self.id_to_input_name[input_id]

    def get_lang_output(self, output_id):
        """
        given an output id, return variable name
        """
        return self.id_to_output_name[output_id]

    def get_evaluation(self, eval_block, vars):
        """
        returns a CodeBlock object that contains script to write:

        <output_name> = <f(inputs)>

        """

        raise NotImplementedError('Evaluation script is not implemented')

    def get_partials(self, partials_dict, partials_block, vars, is_sparse_jac, lazy):
        """
        returns a CodeBlock object that contains script to write:

        <partial_name> = <df/dx(inputs)>

        write the derivative code to partials_block. ANY precomputed variables to be used can be set in vars.
        """

        return None

    def get_accumulation_function(self, input_paths, path_output, partials_block, vars):
        # This method only gets called if self.jac_is_function == True
        raise NotImplementedError('not implemented')

    def determine_sparse(self):
        """
        returns true or false depending on whether the jacobian(s) are sparse or dense.
        True --> use scipy sparse matrix for jacobian
        False --> use numpy array for jacobian

        default is False
        """

        # default return False
        return False

    def determine_sparse_default_elementwise(self, input_size):

        if input_size > 100:
            return True
        else:
            return False

    def check_input_reshaping(self):
        """
        if the REP input and LANG input do not match shapes, we have to reshape.
        """

        # for lang_input_var, rep_input_var in self.input_csdl_to_rep.items():
        # for lang_input_var, rep_input_var in self.input_csdl_to_rep.items():
        for lang_input_var in self.input_csdl_to_rep:
            rep_input_var = self.input_csdl_to_rep[lang_input_var]

            lang_shape = lang_input_var.shape
            rep_shape = rep_input_var.var.shape

            # detect shape mismatch
            if lang_shape != rep_shape:
                in_id = rep_input_var.id
                self.needs_input_reshape = True

                self.reshape_block.write(f'{in_id} = {in_id}.reshape({lang_shape})')
                self.unreshape_block.write(f'{in_id} = {in_id}.reshape({rep_shape})')

    def is_jac_function(self, vjp = False):
        return self.jac_is_function