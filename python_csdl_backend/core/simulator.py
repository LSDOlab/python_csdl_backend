from csdl import SimulatorBase, Model, Operation, ImplicitOperation, GraphRepresentation
# from csdl.core.output import Output
# from csdl.core.input import Input
from python_csdl_backend.core.instructions import Instructions
from python_csdl_backend.core.operation_map import csdl_to_back_map
from python_csdl_backend.core.systemgraph import SystemGraph
from python_csdl_backend.utils.general_utils import get_deriv_name, to_list, lineup_string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import networkx as nx


class Simulator(SimulatorBase):

    def __init__(self,
                 representation,
                 mode='rev',
                 analytics=False,
                 sparsity='auto',
                 display_scripts=False):
        """
        CSDL compiler backend. Evaluates model and derivatives.

        Parameters:
        -----------
            representation: GraphRepresentation (accepts csdl Models as well for now)
                Representation of the CSDL model to evaluate.

            mode: str
                String specifying which method to compute derivatives. (accepts only 'rev')

            analytics: bool
                (EXPERIMENTAL) Writes all nodes and their predecessors to a text file.

            sparsity: str
                (EXPERIMENTAL) Store standard operation partial derivatives as sparse 
                or dense matrices. 'auto', 'dense' or 'sparse'.

            display_scripts: bool
                (EXPERIMENTAL) saves derivative and evaluation scripts as a python 
                files for debugging. These files are not used for computation.

        """
        self.display_scripts = display_scripts
        self.recorder = None

        # check modes
        self.mode = mode
        self.analytics = analytics
        allowed_modes = ['rev']
        if mode not in allowed_modes:
            raise NotImplementedError(f'mode must be one of {allowed_modes}. {mode} given.')

        # check model is Model or GraphRepresentation
        if isinstance(representation, Model):
            self.rep = GraphRepresentation(representation)
        elif isinstance(representation, GraphRepresentation):
            self.rep = representation
        else:
            raise TypeError(f'representation argmument must be a csdl Model or GraphRepresentation object. {Model} given.')

        # Set input variables:
        # Initialize state values dictionary
        # state_vals[<unique_name>] = <np.ndarray()>
        self.state_vals = {}

        # process design variables, objectives, and constraints
        self.opt_bool = False  # Boolean on whether we are running an optimization problem or not
        self.dvs = self.rep.design_variables
        self.obj = self.rep.objective
        self.cvs = self.rep.constraints

        # if an objective is specified, this is an optimization problem
        if self.obj:
            # check to make sure design variables exist:
            if len(self.dvs) == 0:
                raise KeyError('objective was specified but no design variables found')
            self.opt_bool = True

        if self.opt_bool:
            self.dv_keys = [dv_name for dv_name in self.dvs.keys()]
            self.constraint_keys = [c_name for c_name in self.cvs.keys()]

            self.output_keys = self.constraint_keys + [self.obj['name']]

        # start code generation for eval and adjoint.

        # :::::ANALYTICS:::::
        if self.analytics:
            print('PROCESSING GRAPH...')
            filename = 'SUMMARY_GRAPH.txt'
        # :::::ANALYTICS:::::

        # model and graph creation
        self.system_graph = SystemGraph(
            self.rep,
            mode=mode,
            sparsity_type=sparsity,
            dvs=self.dvs,
            objective=self.obj,
            constraints=self.cvs,
            opt_bool=self.opt_bool)

        # :::::ANALYTICS:::::
        if self.analytics:
            # loop through operations and stuff
            operation_analytics = self.system_graph.get_analytics(filename)

            # print some stuff
            isdag = nx.is_directed_acyclic_graph(self.system_graph.eval_graph)
            print('IS DAG?', isdag)
            if not isdag:
                print(nx.find_cycle(self.system_graph.eval_graph))
            for op_type in operation_analytics:
                count = operation_analytics[op_type]['count']
                print(f'operation count: {count}\t type: {op_type}')
            print('GRAPH PROCESSED')
        # :::::ANALYTICS:::::

        # prepare design variable vectors
        if self.opt_bool:
            self.process_optimization_vars()

        #  ----------- create model evaluation script -----------
        self.eval_instructions = Instructions(f'RUN_MODEL{self.rep}')
        # self.eval_instructions.script.write('import numpy as np')

        # This line basically creates the mode evaluation graph
        eval_block, self.preeval_vars, state_vals_extracted, variable_info = self.system_graph.generate_evaluation()
        self.eval_instructions.script.write(eval_block)
        self.eval_instructions.compile()
        self.variable_info = variable_info
        self.ran_bool = False
        if self.display_scripts:
            self.eval_instructions.save()

        #  ----------- create model evaluation script -----------

        # set up derivatives
        for var in state_vals_extracted:
            self.state_vals[var] = state_vals_extracted[var]

        # maps: (ofs, wrts) ---> Instructions object
        self.derivative_instructions_map = {}
        # if mode == 'rev':
        #     self.graph_reversed = self.eval_graph.reverse()

    def _generate_adjoint(self, outputs, inputs):
        '''
        given a list of outputs and inputs, creates the script for the adjoint.
        '''

        # list of outputs and inputs to get derivatives of.
        output_names = to_list(outputs)
        input_names = to_list(inputs)

        # name of instructions
        adj_name = 'GC:'
        for output_name in output_names:
            adj_name += f'{output_name},'
        adj_name += '-->'
        for input_name in input_names:
            adj_name += f'{input_name},'

        # initialize adjoint derivatives instructions to write to
        # This script will be ran every
        adj_instructions = Instructions(adj_name)
        # adj_instructions.script.write('import scipy.sparse as sp')
        # adj_instructions.script.write('import numpy as np')

        # generate reverse is only a function of unique ID's therefore, find the appropriate id's for each input and output
        output_ids = []
        for output_name in output_names:
            output_ids.append(self._find_unique_id(output_name))
        input_ids = []
        for input_name in input_names:
            input_ids.append(self._find_unique_id(input_name))

        # generate script
        rev_script, pre_vars = self.system_graph.generate_reverse(output_ids, input_ids)

        # write the computation steps
        adj_instructions.script.write(rev_script)

        # Save/compile
        if self.display_scripts:
            adj_instructions.save()
        adj_instructions.compile()
        return adj_instructions, pre_vars

    def run(self):
        """
        Run model.
        """
        # Execute compiled code, return all evaluated variables
        new_states = self._run(self.state_vals)
        self.ran_bool = True

        for key in self.state_vals:
            self.state_vals[key] = new_states[key]

    def _run(self, states):

        eval_vars = {**states, **self.preeval_vars}
        new_states = self.eval_instructions.execute(eval_vars)

        if self.recorder:

            save_dict = {}
            for var_name in self.recorder.dash_instance.vars['simulator']['var_names']:
                print(var_name)
                save_dict[var_name] = new_states[self._find_unique_id(var_name)]

            self.recorder.record(save_dict, 'simulator')
        return new_states

    def _generate_totals(self, of, wrt):
        '''
        generate derivatives
        '''
        ofs = to_list(of)
        wrts = to_list(wrt)

        # check to see if variables exist
        self.check_variable_existence(ofs)
        self.check_variable_existence(wrts)

        hash_key = (tuple(ofs), tuple(wrts))

        exec, vars = self._generate_adjoint(ofs, wrts)

        self.derivative_instructions_map[hash_key] = {}
        self.derivative_instructions_map[hash_key]['precomputed_vars'] = vars
        self.derivative_instructions_map[hash_key]['executable'] = exec

    def compute_totals(self, of, wrt, return_format='[(of, wrt)]'):
        '''
        compute the derivatives of 'of' with respect to 'wrt'.

        Parameters:
        -----------
            of: list or str
                name(s) of outputs to take derivatives of.
            wrt: list or str
                name(s) of inputs to take derivatives wrt.
            return_format: str
                (EXPERIMENTAL) what format to return derivative in.
        '''

        # make sure simulat has been ran once
        if self.ran_bool == False:
            raise ValueError('Simulator must be ran before computing derivatives.')

        # key of output/wrt combination.
        ofs = to_list(of)
        wrts = to_list(wrt)
        hash_key = (tuple(ofs), tuple(wrts))

        # check to see if variables exist
        self.check_variable_existence(ofs)
        self.check_variable_existence(wrts)

        # If the has key has not been found, generate totals
        if hash_key not in self.derivative_instructions_map:
            print(hash_key, ' not found, generating...')
            self._generate_totals(ofs, wrts)

        # Execute script
        vars = self.derivative_instructions_map[hash_key]['precomputed_vars']
        adj_exec = self.derivative_instructions_map[hash_key]['executable']
        totals_dict = adj_exec.execute({**self.state_vals, **vars})

        # Return computed totals
        return_dict = {}
        for of_name in ofs:
            for wrt_name in wrts:
                of_id = self._find_unique_id(of_name)
                wrt_id = self._find_unique_id(wrt_name)
                var_local_name = get_deriv_name(of_id, wrt_id, partials=False)
                current_derivative = totals_dict[var_local_name]

                if var_local_name in totals_dict:
                    if return_format == '[(of, wrt)]':
                        return_dict[(of_name, wrt_name)] = current_derivative
                    elif return_format == '[of][wrt]':
                        if of_name not in return_dict:
                            return_dict[of_name] = {}
                        return_dict[of_name][wrt_name] = current_derivative

        return return_dict

    def visualize_implementation(self, depth=False):
        # TODO: TEMPORARY VISUALIZATION:

        raise NotImplementedError('visualize implementation is not yet implemented.')

        self._draw(self.system_graph.eval_graph, depth=depth)
        # self._draw(self.expanded_graph.eval_graph)

        plt.show()

    def _draw(self, graph, style='kk', depth=False):

        colors = []
        for node_name in graph.nodes:
            node = graph.nodes[node_name]['csdl_node']
            if isinstance(node, Output):
                colors.append((1.0, 0.0, 0.0))
            elif isinstance(node, Input):
                colors.append((0.0, 1.0, 0.0))
            elif isinstance(node, ImplicitOperation):
                colors.append((1.0, 1.0, 0.0))
            elif isinstance(node, Operation):
                colors.append((0.0, 0.0, 1.0))
            elif isinstance(node, Subgraph):
                colors.append((0.0, 1.0, 1.0))
            elif isinstance(node, Graph):
                if depth:
                    self._draw(node.eval_graph)
                colors.append((1.0, 0.0, 1.0))
            else:
                colors.append((0.5, 0.5, 0.5))

        plt.figure()

        if style == 'kk':
            nx.draw_kamada_kawai(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'planar':
            nx.draw_planar(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'default':
            nx.draw(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'circular':
            nx.draw_circular(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'shell':
            nx.draw_shell(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'spectral':
            nx.draw_spectral(graph, with_labels=True, arrows=True, node_color=colors)
        elif style == 'spring':
            nx.draw_spring(graph, with_labels=True, arrows=True, node_color=colors)
        else:
            raise ValueError(f'style {style} not recognized. Available styles: \'kk\', \'planar\', \'default\',\'circular\', \'shell\', \'spectral\' and \'spring\'')

    def __getitem__(self, key):
        '''
        Get item.

        Parameters:
        ----------
            key: str
                the promoted/unpromoted name of the variable that the user is trying to get.
        '''
        # The state values of the model are saved with private unique ids. Therefore, we find the unique id
        # of key given by user.

        self.check_variable_existence(key)
        unique_id = self._find_unique_id(key)

        if unique_id:
            return self.state_vals[unique_id]
        else:
            raise KeyError(f'Variable {key} not found.')

    def __setitem__(self, key, val):
        '''
        Set item.

        Parameters:
        -----------
            key: str
                the promoted/unpromoted name of the variable that the user is trying to set.
            val: np.ndarray / float
                the value of variable keyed by user.
        '''
        # The state values of the model are saved with private unique ids. Therefore, we find the unique id
        # of key given by user.

        unique_id = self._find_unique_id(key)
        if not unique_id:
            # If key is not found then return error
            raise KeyError(f'Variable {key} not found.')
        else:
            # set the state values
            try:
                val = val.astype('float64')
                self.state_vals[unique_id] = val.reshape(self.state_vals[unique_id].shape)
            except:
                try:
                    self.state_vals[unique_id] = (np.ones(self.state_vals[unique_id].shape)*val).astype('float64')
                except:
                    raise ValueError(f'Cannot set variable {key} to simulator.')

    def _find_unique_id(self, key):
        """
        Given a language variable name,
        see is variable name exists. If so, return unique id
        else, return none
        """

        if key in self.system_graph.promoted_to_unique:
            return self.system_graph.promoted_to_unique[key]
        elif key in self.system_graph.unpromoted_to_unique:
            return self.system_graph.unpromoted_to_unique[key]
        else:
            return None

    def check_variable_existence(self, vars):
        """
        return error if variable does not exist
        """

        var_list = to_list(vars)

        for var in var_list:
            if not self._find_unique_id(var):
                raise KeyError(f'cannot find variable \'{var}\'')

    def check_totals(self, of=None, wrt=None, compact_print=True):
        """
            checks total derivatives using finite difference.

            Parameters:
            -----------
                of: list or str
                    name(s) of outputs to take derivatives of.
                wrt: list or str
                    name(s) of inputs to take derivatives wrt.
        """

        if self.ran_bool == False:
            raise ValueError('Simulator must be ran before computing derivatives.')

        if (of is None) and (wrt is None):

            if not self.opt_bool:
                raise KeyError('cannot find objective/design/constraint variables to compute derivatives of')

            # build dictionary of variable for derivatives:
            obj_name = self.obj['name']
            obj_node = self.obj['node']

            output_info = {obj_name: {}}
            output_info[obj_name]['shape'] = obj_node.var.shape
            output_info[obj_name]['size'] = np.prod(obj_node.var.shape)

            for output_name in self.cvs:
                c_node = self.cvs[output_name]['node']
                output_info[output_name] = {}
                output_info[output_name]['shape'] = c_node.var.shape
                output_info[output_name]['size'] = np.prod(c_node.var.shape)

            input_info = {}
            all_leaf_dict = self.variable_info['leaf_start']
            for input_name in self.dvs:
                d_node = self.dvs[input_name]['node']
                input_info[input_name] = {}
                input_info[input_name]['shape'] = d_node.var.shape
                input_info[input_name]['size'] = np.prod(d_node.var.shape)

        else:

            # list of user given vars
            of_list = to_list(of)
            wrt_list = to_list(wrt)

            # check to see if user given variables exist
            self.check_variable_existence(of_list)
            self.check_variable_existence(wrt)

            # build dictionary of variable for derivatives:
            output_info = {}
            all_output_dict = self.variable_info['outputs']
            for of_var in of_list:
                if of_var not in all_output_dict:
                    raise KeyError(f'output {of_var} not found.')
                output_info[of_var] = {}
                output_info[of_var]['shape'] = all_output_dict[of_var]['shape']
                output_info[of_var]['size'] = all_output_dict[of_var]['size']

            input_info = {}
            all_leaf_dict = self.variable_info['leaf_start']
            for wrt_var in wrt_list:
                if wrt_var not in all_leaf_dict:
                    raise KeyError(f'input variable {wrt_var} not found. can only check derivatives with respect to inputs')

                input_info[wrt_var] = {}
                input_info[wrt_var]['shape'] = all_leaf_dict[wrt_var]['shape']
                input_info[wrt_var]['size'] = all_leaf_dict[wrt_var]['size']

        # compute analytical
        analytical_derivs = self.compute_totals(
            of=list(output_info.keys()),
            wrt=list(input_info.keys()),
        )

        # compute finite difference
        fd_derivs = {}
        for input_name in input_info:
            input_dict = input_info[input_name]
            fd_derivs.update(self._compute_fd_partial(input_name, input_dict, output_info))

        # compute error
        error_dict = {}
        for input_name in input_info:
            for output_name in output_info:
                analytical_jac = analytical_derivs[output_name, input_name]
                try:
                    analytical_jac = analytical_jac.A
                except:
                    pass
                fd_jac = fd_derivs[output_name, input_name]
                error_jac = analytical_jac - fd_jac

                error_dict[(output_name, input_name)] = {}
                error_dict[(output_name, input_name)]['error_jac'] = error_jac
                error_dict[(output_name, input_name)]['abs_error_norm'] = np.linalg.norm(error_jac)
                error_dict[(output_name, input_name)]['analytical_jac'] = analytical_jac
                error_dict[(output_name, input_name)]['analytical_norm'] = np.linalg.norm(analytical_jac)
                error_dict[(output_name, input_name)]['fd_jac'] = fd_jac

                if (np.linalg.norm(fd_jac) < 1e-10) and (np.linalg.norm(analytical_jac) < 1e-10):
                    error_dict[(output_name, input_name)]['relative_error_norm'] = 0.0  # This is pretty messy
                else:
                    error_dict[(output_name, input_name)]['relative_error_norm'] = np.linalg.norm(error_jac)/np.linalg.norm(fd_jac)

        # print
        max_key_len = 0
        max_rel_len = 0
        max_calc_len = len('calc norm')
        max_abs_len = 0
        for key in error_dict:
            key_str = str(key)
            if len(key_str) > max_key_len:
                max_key_len = len(key_str)

            rel_error_str = str(error_dict[key]['relative_error_norm'])
            if len(rel_error_str) > max_rel_len:
                max_rel_len = len(rel_error_str)

            calc_norm_str = str(error_dict[key]['analytical_norm'])
            if len(calc_norm_str) > max_calc_len:
                max_calc_len = len(calc_norm_str)

            abs_error_str = str(error_dict[key]['abs_error_norm'])
            if len(abs_error_str) > max_abs_len:
                max_abs_len = len(abs_error_str)

        max_key_len += 5
        max_rel_len += 5
        max_calc_len += 5

        if compact_print:
            print()

            error_bar = '-'*(max_key_len+max_rel_len+max_calc_len+max_abs_len)
            of_wrt_str = lineup_string('(of,wrt)', max_key_len)
            rel_error_title = lineup_string('relative error', max_rel_len)
            computed_norm_title = lineup_string('calc norm', max_calc_len)

            print(f'{of_wrt_str}{computed_norm_title}{rel_error_title}absolute error')

            # --------------------------------------
            print(error_bar)
            for key in error_dict:

                key_str = lineup_string(str(key), max_key_len)

                calc_norm = error_dict[key]['analytical_norm']
                calc_norm_str = lineup_string(str(calc_norm), max_calc_len)

                rel_error = error_dict[key]['relative_error_norm']
                rel_error_str = lineup_string(str(rel_error), max_rel_len)

                abs_error = error_dict[key]['abs_error_norm']
                print(f'{key_str}{calc_norm_str}{rel_error_str}{abs_error}')
            print(error_bar)
            # --------------------------------------
            print()
        else:
            print()
            for key in error_dict:
                print(f'\n----------------------------------------------------')
                rel_error = error_dict[key]['relative_error_norm']
                abs_error = error_dict[key]['abs_error_norm']
                print(f'{key}\t{rel_error}\t{abs_error}')
                print('finite difference:')
                print(fd_derivs[key])
                print('analytical:')
                print(analytical_derivs[key])

            print(f'----------------------------------------------------')
            print()

        return error_dict

    def _compute_fd_partial(self, input_name, input_dict, outputs_dict, delta=1e-6):

        input_size = input_dict['size']
        input_shape = input_dict['shape']

        # initialize finite difference jacobian to matrices with value 1e6
        fd_jacs = {}
        for output_name in outputs_dict:
            output_size = outputs_dict[output_name]['size']
            output_shape = outputs_dict[output_name]['shape']
            fd_jacs[(output_name, input_name)] = 1e6*np.ones((output_size, input_size))

        # perform finite difference:
        for col_index in range(input_size):

            # set perturbed inputs
            temp_state = self.state_vals.copy()
            input_id = self._find_unique_id(input_name)
            temp_state[input_id] = temp_state[input_id].flatten()
            temp_state[input_id][col_index] = temp_state[input_id][col_index] + delta
            temp_state[input_id] = temp_state[input_id].reshape(input_shape)

            # compute f(x+h)
            # print(temp_state[input_name])
            new_states = self._run(temp_state)
            # print(new_states[input_name])

            # build finite difference jacobian
            for output_name in outputs_dict:
                output_id = self._find_unique_id(output_name)
                output_size = outputs_dict[output_name]['size']
                output_shape = outputs_dict[output_name]['shape']

                output_val_perturbed = new_states[output_id].flatten()
                output_val_original = self.state_vals[output_id].flatten()

                output_check_derivative = (output_val_perturbed - output_val_original)/delta

                for row_index in range(output_size):
                    fd_jacs[(output_name, input_name)][row_index, col_index] = output_check_derivative[row_index]

        return fd_jacs

    def process_optimization_vars(self):

        self.total_dv_size = 0
        for dv_name, dv_dict in self.dvs.items():
            dv_node = dv_dict['node']
            dv_shape = dv_node.var.shape
            dv_size = np.prod(dv_shape)

            dv_dict['size'] = dv_size
            dv_dict['shape'] = dv_shape
            dv_dict['index_lower'] = self.total_dv_size
            self.total_dv_size += dv_size
            dv_dict['index_upper'] = self.total_dv_size

            if not dv_dict['lower']:
                dv_dict['lower'] = np.ones(dv_shape)*(-1.0e30)
            if not dv_dict['upper']:
                dv_dict['upper'] = np.ones(dv_shape)*(1.0e30)

        self.total_constraint_size = 0
        for c_name, c_dict in self.cvs.items():
            c_node = c_dict['node']
            c_shape = c_node.var.shape
            c_size = np.prod(c_shape)

            c_dict['size'] = c_size
            c_dict['shape'] = c_shape
            c_dict['index_lower'] = self.total_constraint_size
            self.total_constraint_size += c_size
            c_dict['index_upper'] = self.total_constraint_size

            if c_dict['lower'] is None:
                c_dict['lower'] = np.ones(c_shape)*(-1.0e30)
            if c_dict['upper'] is None:
                c_dict['upper'] = np.ones(c_shape)*(1.0e30)

    def get_design_variable_metadata(self):
        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)
        return self.dvs

    def get_constraints_metadata(self):
        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)
        return self.cvs

    def update_design_variables(self, x):

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        for dv_name, dv_dict in self.dvs.items():
            i_lower = dv_dict['index_lower']
            i_upper = dv_dict['index_upper']
            shape = dv_dict['shape']
            new_val = x[i_lower:i_upper]
            dv_id = self._find_unique_id(dv_name)

            self.state_vals[dv_id] = new_val.reshape(shape)

    def compute_total_derivatives(self):
        """
        computes derivatives of objective/constraints wrt design variables.
        """

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        self.optimization_derivatives = self.compute_totals(
            of=self.output_keys,
            wrt=self.dv_keys,
        )

        return self.optimization_derivatives

    def objective(self):
        """
        return objective variable value.
        """
        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        obj_name = self.obj['name']
        return self[obj_name]

    def constraints(self):
        """
        return constraint variable value.
        """

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        constraint_vec = np.zeros(self.total_constraint_size)
        for c_name, c_dict in self.cvs.items():
            i_lower = c_dict['index_lower']
            i_upper = c_dict['index_upper']
            shape = c_dict['shape']
            c_id = self._find_unique_id(c_name)
            c_val = self.state_vals[c_id].flatten()
            constraint_vec[i_lower:i_upper] = c_val

        return constraint_vec

    def design_variables(self):

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        dv_vec = np.zeros(self.total_dv_size)
        for dv_name, dv_dict in self.dvs.items():
            i_lower = dv_dict['index_lower']
            i_upper = dv_dict['index_upper']
            shape = dv_dict['shape']
            dv_id = self._find_unique_id(dv_name)
            dv_val = self.state_vals[dv_id].flatten()
            dv_vec[i_lower:i_upper] = dv_val

        return dv_vec

    def objective_gradient(self):

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        obj_gradient = np.zeros(self.total_dv_size)
        obj_name = self.obj['name']
        for dv_name, dv_dict in self.dvs.items():
            i_lower = dv_dict['index_lower']
            i_upper = dv_dict['index_upper']

            if sp.issparse(self.optimization_derivatives[obj_name, dv_name]):
                obj_gradient[i_lower:i_upper] = (self.optimization_derivatives[obj_name, dv_name].toarray()).flatten()
            else:
                obj_gradient[i_lower:i_upper] = (self.optimization_derivatives[obj_name, dv_name]).flatten()

        return obj_gradient

    def constraint_jacobian(self):

        # return error if not optimization problem
        self.check_if_optimization(self.opt_bool)

        constraint_jac = np.zeros((self.total_constraint_size, self.total_dv_size))
        for c_name, c_dict in self.cvs.items():
            i_lower_c = c_dict['index_lower']
            i_upper_c = c_dict['index_upper']
            for dv_name, dv_dict in self.dvs.items():
                i_lower_dv = dv_dict['index_lower']
                i_upper_dv = dv_dict['index_upper']
                constraint_jac[i_lower_c:i_upper_c, i_lower_dv:i_upper_dv] = self.optimization_derivatives[c_name, dv_name]
        return constraint_jac

    def check_partials(self,
                       out_stream=None,
                       includes=None,
                       excludes=None,
                       compact_print=False,
                       abs_err_tol=1e-6,
                       rel_err_tol=1e-6,
                       method='fd',
                       step=None,
                       form='forward',
                       step_calc='abs',
                       force_dense=True,
                       show_only_incorrect=False,
                       ):
        """
        Checks totals. SHOULD ONLY BE USED FOR UNIT TESTING IN CSDL. arguments are not used.
        """
        warnings.warn('check_partials redirects to check_totals and should only be used for unit tests. use check_totals instead')

        if not self.variable_info['outputs']:
            raise KeyError(f'No outputs found')
        if not self.variable_info['leaf_start']:
            raise KeyError(f'No leaf nodes found')

        all_of_names = [out_name for out_name in self.variable_info['outputs'].keys()]
        all_wrt_names = [in_name for in_name in self.variable_info['leaf_start'].keys()]
        warnings.warn(
            f'checking totals with empty \'of\' and \'wrt\' arguments computes and checks derivatives of ALL {len(all_of_names)} model outputs with respect to ALL {len(all_wrt_names)} model inputs. Specify variables to check derivatives in a more efficient manner')

        check_totals = self.check_totals(
            of=all_of_names,
            wrt=all_wrt_names,
            compact_print=compact_print,
        )

        return check_totals

    def assert_check_partials(self, result, atol=1e-8, rtol=1e-8):
        """
        Asserts check partials. SHOULD ONLY BE USED FOR UNIT TESTING IN CSDL. arguments not used.
        """

        for key in result:
            np.testing.assert_almost_equal(
                result[key]['relative_error_norm'],
                0.0,
                decimal=5)

    def add_recorder(self, recorder):
        """
        For dashboard.
        """
        self.recorder = recorder

    def check_if_optimization(self, opt_bool):
        """
        raise error if opt_bool == False
        """

        if not opt_bool:
            raise KeyError('given representation does not specify design variables and an objective.')