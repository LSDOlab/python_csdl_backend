from python_csdl_backend.core.codeblock import CodeBlock


class PathSet():

    def __init__(self, name: str, node_sequence: list, evalgraph):
        self.name = name  # Name of pathset
        self.node_sequence = node_sequence  # Sequence of this path
        self.accumulated = False  # Boolean of whther jac accumulation has been performed
        self.codeblock = CodeBlock(self.name)  # Codeblock to store the code generation
        self.eval_graph = evalgraph
        self.prerev_vars = {}

        # for bookkeeping:
        self.num_total_paths = 1  # Number of total paths in this pathset
        self.paths_dict = {self.name: self}  # Dictionary of paths in this pathset

    def write_to(self, codeblock: CodeBlock, prerev_vars):
        # Write to a given codeblock
        codeblock.write(self.codeblock)
        prerev_vars.update(self.prerev_vars)

    def join_end(self, pathset):
        '''
        Concatenate a pathset to the end of this pathset.
        Because we are adding a pathset to the end of this pathset,
        The first node of the concatenation pathset must match the end node
        of this pathset

        write to self.codeblock:
            <self pathset name> = <self pathset name>*<pathset name>

        pathset must be accumulated (?) before adding.
        '''
        # Raise error if end and start nodes do not match
        if pathset.get_end_node() != self.get_start_node():
            raise ValueError(f'A pathset joined to the end of another pathset must have matching end and start nodes. {self.get_end_node()} expected, {pathset.get_start_node()} given')

        # Write to codeblock
        self.codeblock.write(f'{self.pathset.name} = {self.pathset.name}@{pathset.name}')

    def join_start(self, pathset):
        '''
        *****THIS METHOD SHOULD NEVER BE USED*****
        Concatenate a pathset to the begining of this pathset.
        Because we are adding a pathset to the begining of this pathset,
        The last node of the concatenation pathset must match the starting node
        of this pathset

        write to self.codeblock:
            <self pathset name> = <pathset name>*<self pathset name>

        pathset must be accumulated (?) before adding.
        '''

        # Raise error if end and start nodes do not match
        if pathset.get_end_node() != self.get_start_node():
            raise ValueError(f'A pathset joined to the start of another pathset must have matching end and start nodes. {self.get_end_node()} expected, {pathset.get_start_node()} given')

        # Write to codeblock
        self.codeblock.write(f'{self.pathset.name} = {pathset.name}@{self.pathset.name}')

    def add_alternate_paths(self, pathset):
        '''
        Add another 'parallel' pathset to this path.
        Because the added pathset is parallel, the end and start nodes of
        this pathset and added pathset must be the same.

        write to self.codeblock:
            <self pathset name> = <self pathset name> + <pathset name>

        pathset must be accumulated (?) before adding.
        '''

        # Raise error if end and start nodes do not match
        if pathset.get_end_node() != self.get_end_node():
            raise ValueError(f'Pathset end nodes must match when adding an alternate pathset. {self.get_end_node()} expected, {pathset.get_end_node()} given')
        if pathset.get_start_node() != self.get_start_node():
            raise ValueError(f'Pathset start nodes must match when adding an alternate pathset. {self.get_start_node()} expected, {pathset.get_start_node()} given')

        # Write to codeblock
        self.codeblock.write(f'{self.pathset.name} = {self.pathset.name} + {pathset.name}')

        # For bookeeping:
        self.num_total_paths += 1
        self.paths_dict[pathset.name] = pathset

    def accumulate_jacobians(self, name=None):
        '''
        Perform chain rule on this path.
        write to self.codeblock:
            <self pathset name> = <self pathset name> @ partial_1
            <self pathset name> = <self pathset name> @ partial_2
            <self pathset name> = <self pathset name> @ partial_3
            .
            .
            .
            <self pathset name> = <self pathset name> @ partial_n

        Accumulation can only be done once.
        '''

        if not name:
            name = self.name

        for i, node in enumerate(self.node_sequence):

            csdl_node = self.eval_graph.nodes[node]['csdl_node']

            # If the node is a CSDL variable, we don't need to do anything
            if not isinstance(csdl_node, (Operation)):
                continue

            # If the node is an operation, compute the partials:
            if isinstance(csdl_node, Operation):
                backend_op = self.eval_graph.nodes[node]['operation_lite']

                # If the partials have been computed, we don't need to actually write the partials to compute it.
                partials_name = get_deriv_name(self.node_sequence[i-1], self.node_sequence[i+1])

                if not backend_op.computed_jac:
                    partials_block = CodeBlock(backend_op.name + f' {partials_name}')
                    vars = {}

                    backend_op.get_partials(path[i+1], path[i-1], partials_name, partials_block, vars)

                    self.codeblock.write(partials_block)
                    self.prerev_vars.update(vars)

                    backend_op.computed_jac = True

            if i == 0:
                self.codeblock.write(f'{self.name} = {partials_name}')
            else:
                self.codeblock.write(f'{self.name} = {self.name}@{partials_name}')

            # update codeblock

    def get_end_node(self):
        '''
        returns the last node in this pathset (furthest from model output)
        '''
        return self.node_sequence[-1]

    def get_start_node(self):
        '''
        returns the first node in this pathset (closest to model output)
        '''
        return self.node_sequence[0]
