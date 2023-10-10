from python_csdl_backend.operations.implicit.wrappers.wrapper_base import ImplicitWrapperBase
import numpy as np


class ImplicitSimWrapper(ImplicitWrapperBase):

    def __init__(self, op, ins, outs):
        from python_csdl_backend.core.simulator import Simulator

        out_res_map: Dict[str, Output] = op.out_res_map  # CSDL NATIVE STATES
        out_in_map: Dict[str, Variable] = op.out_in_map  # keys contain state names, values are lists of non-state input variables
        res_out_map: Dict[str, Variable] = op.res_out_map  # CSDL NATIVE RESIDUALS
        expose: List[str] = op.expose  # EXPOSED VARIABLES

        # create simulator of implicit model

        implicit_checkpoints = 0
        if implicit_checkpoints:
            self.sim = Simulator(op.rep, checkpoints = 1, save_vars=set().union(set(out_res_map.keys()), set(out_in_map.keys()), set(res_out_map.keys()), set(expose)))
        else:
            name = ''
            for state in out_res_map:
                name += state + '__'
            self.sim = Simulator(op.rep, analytics=0,display_scripts=0, name = name)

        # dictionary of information on state
        # Try to keep as many things out of memory after init
        self.states = {}
        self.total_state_size = 0
        for state in out_res_map:
            # Why is there a residual for the exposed variables?....
            # do not make exposed variables states
            if state in expose:
                continue

            # state name is key
            self.states[state] = {}

            # Keep csdl variable of output (not sure if needed)
            self.states[state]['out'] = out_res_map[state]

            self.states[state]['index_lower'] = self.total_state_size
            self.total_state_size += np.prod(out_res_map[state].shape)
            self.states[state]['index_upper'] = self.total_state_size
            self.states[state]['shape'] = out_res_map[state].shape

            # Keep state initial guess for implicit operation
            for state_out in op.outs:
                if state_out.name == state:
                    state_val = state_out.val
            self.states[state]['initial_val'] = state_val

        # ------- <ACTUALLY DON'T THINK THIS IS CORRECT> -------
        # # dictionary of information on residual
        # # find order of residuals:
        # # The implicit model is fully explicit.
        # # Therefore, the jacobian dr/dx can be formed as block tri.
        # # We just need to figure out the topological order of residual outputs to formulate this.
        # temp_res_list = list(res_out_map.keys())
        # ordered_res_list = self.sim.expanded_graph._find_node_order(temp_res_list)
        # self.ordered_residuals = {}
        # for res in ordered_res_list:
        #     self.ordered_residuals[res] = {}
        # ------- <ACTUALLY DON'T THINK THIS IS CORRECT> -------

        self.residuals = {}
        for res in res_out_map:
            # Why is there a residual for the exposed variables?....
            # do not make a residual for expose variables
            if res_out_map[res].name in expose:
                continue

            self.residuals[res] = {}
            self.residuals[res]['state'] = res_out_map[res].name
            self.states[res_out_map[res].name]['residual'] = res

        # dictionary of information on exposed vars
        self.exposed = {}
        for exp in expose:
            self.exposed[exp] = {}

        self.ordered_inputs = ins
        self.ordered_outs = outs

        # dictionary of information on inputs
        self.inputs = {}
        for input in self.ordered_inputs:
            self.inputs[input] = {}
            self.inputs[input]['shape'] = self.sim[input].shape
            self.inputs[input]['size'] = np.prod(self.inputs[input]['shape'])
            
        # list of what to compute derivatives of
        self.of_list = list(self.residuals.keys()) + list(self.exposed.keys())
        self.wrt_list = list(self.states.keys()) + self.ordered_inputs

        #  generate code for the derivatives of implicit model
        self.sim._generate_totals(self.of_list, self.wrt_list)

    def run(self):
        self.sim.run()

    def set_input(self, input_name, val):
        self.sim[input_name] = val

    def set_state(self, state_name, val):
        self.sim[state_name] = val

    def get_residual(self, res_name):
        return self.sim[res_name]

    def get_state(self, state_name):
        return self.sim[state_name]

    def compute_totals(self):
        return self.sim.compute_totals(self.of_list, self.wrt_list)
