from python_csdl_backend.core.codeblock import CodeBlock
from python_csdl_backend.utils.general_utils import analyze_dict_memory
import numpy as np
import scipy.sparse as sp
import scipy
import gc

class SingleInstruction():

    def __init__(self, name):
        """ 
        Contains and executes the generated code for each function. 
        Script attribute is a CodeBlock object to write low level code to.

        Parameters:
        ----------
            name: str
                name of the code subblock
        """

        self.name = 'Instructions_' + name
        self.script = CodeBlock(name)
    
    # @profile
    def compile(self):
        """ 
        Converts script to string and compiles.
        """
        self.codeobject = compile(self.script.to_string(), self.name, 'exec', optimize=2)

    def save(self):
        """ 
        Saves the codeblock script.
        """
        self.script.save()
    # @profile
    def execute(self, inputs, sim_name = ''):
        """
        Executes compiled script. Returns dictionary of all local variable values.

        Parameters:
        -----------
            inputs: dict
                dictionary containing input variables
        """

        globals = {}  # Not entirely sure what this is.
        inputs['np'] = np
        inputs['sp'] = sp
        # inputs['scipy'] = scipy

        locals = inputs  # Set local variables

        # Execute compiled code
        exec(self.codeobject, globals, locals)

        # Return local variables
        return locals


class MultiInstructions():
    
    def __init__(self, name):
        """ 
        Container for multiple ordered separate instructions.
        """

        self.name = 'MultiInstructions_' + name
        self.ordered_instructions = []

    def add_single_instruction(self, single_instruction, variables_to_delete):

        instructions_dict = {}
        instructions_dict['single_instruction'] = single_instruction
        instructions_dict['variables_to_delete'] = variables_to_delete
        self.ordered_instructions.append(instructions_dict)

    def execute(self, states, sim_name = ''):
        locals_temp = states
        for i, instruction_dict in enumerate(self.ordered_instructions):
            instruction = instruction_dict['single_instruction']

            locals = instruction.execute(locals_temp)
            
            # print('ran', i, instruction.name, len(states))
            delete_vars = instruction_dict['variables_to_delete']
            for var in delete_vars:
                if var not in locals:
                    raise ValueError('Variable ' + var + ' not in locals.')
                locals[var] = None

            locals_temp = locals

            # gc.collect()
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-= UNCOMMENT FOR MEMORY DEBUGGING -=-=-=-=-=-=-=-=-=-=-=-=-=-=
            # analyze_dict_memory(locals, instruction.name, sim_name = sim_name)
            
        return locals
    
    def save(self):
        """ 
        Saves the codeblock script.
        """
        for instruction_dict in self.ordered_instructions:
            instruction_dict['single_instruction'].save()