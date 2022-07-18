from python_csdl_backend.core.codeblock import CodeBlock
import numpy as np
import scipy.sparse as sp


class Instructions():

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

    def execute(self, inputs):
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

        locals = inputs  # Set local variables

        # Execute compiled code
        exec(self.codeobject, globals, locals)

        # Return local variables
        return locals
