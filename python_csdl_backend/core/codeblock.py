
class CodeBlock():
    def __init__(self, name='codeblock', newline = True, add_name = True):
        """ 
        This class handles a block of code for code generation. Basically a glorified string. 
        Code blocks consists of a list of sub-blocks which are either new lines of code(string) 
        or another codeblock which has it's own sub-blocksBetter than just string as it 
        allows code.write(<code>) instead of string += ' \n < code >' for the code generation.

        Parameters:
        ----------
            name: str
                name of the code subblock
        """

        self.name = name  # name of the code block
        self.sub_blocks = []  # list of blocks containing subblocks and string.
        self.indent_blocks = []  # list of number of indents to each correponding subblock of code.
        self.indent_num = 0  # keep track of current indentation
        self.indent_string = ' '  # indent is a just a space

        # Initialize code
        if newline:
            self.newline()
        if add_name:
            self.comment(name)  # comment name of code

    def write(self, code, linebreak=True):
        """
        Adds string to code or code to the codeblock. All the code is stored a list of new lines of string. 
        to_string must be called to generate the code.

        Parameters:
        ----------
            code: Str() or CodeBlock()
        """

        # Raise error if attempting to write non-string or non-codeblock object
        if not (isinstance(code, str) or isinstance(code, CodeBlock)):
            raise TypeError(f'Code to write must be of type string or CodeBlock. Code type is {type(code)}')

        if linebreak:
            # write a new sub block of code to generate
            self.sub_blocks.append(code)
            self.indent_blocks.append(self.indent_num)
        else:
            # if code argument is a string and the last entry of sub_block is also a string,
            # write code to last string entry
            if (isinstance(code, str)) and (isinstance(self.sub_blocks[-1], str)):
                self.sub_blocks[-1] = self.sub_blocks[-1] + code
            else:
                # Raise error otherwise
                error_string = 'Adding code with linebreak = False requires previous and current subblock entry to be a String. '
                error_string += f'Previous subblock code is {type(self.sub_blocks[-1])} and current code is {type(code)}'
                raise TypeError(error_string)

    def comment(self, comment, linebreak=True):
        """
        Adds string to code as comment

        Parameters:
        ----------
            comment: Str
                Comment to write to code.
        """

        if not (isinstance(comment, str)):
            error_string = f'Can only write comments that are strings. Comment of type {type(comment)} was given.'
            raise TypeError(error_string)

        self.write('# '+comment, linebreak=linebreak)

    def newline(self):
        """
        Adds an empty new line 
        """
        self.write('')

    def indent(self, indent_num=1):
        """
        adds num indents to all following writes to code until unindent is called.

        Parameters:
        -----------
            num: Int
                number of indentations
        """
        if not (isinstance(indent_num, int)):
            error_string = f'Indent number must be an integer. indent_num of type {type(indent_num)} was given.'
            raise TypeError(error_string)

        # Add to current indent_number
        self.indent_num += indent_num

    def unindent(self):
        """
        unindents all following writes to code
        """
        self.indent_num = 0

    def _indent_all(self, indent_num):
        """
        Adds an indentation to entire code block. Called only when to_string is called and 
        not to be used externally.
        """
        for i in range(len(self.indent_blocks)):
            self.indent_blocks[i] += indent_num

    def to_string(self):
        """
        to_string concatenates self.sub_blocks to return a string of the code built from the CodeBlock object.
        Loops through self.sub_blocks and recursively calls to_string if it encounters a CodeBlock object.

        THIS METHOD SHOULD BE CALLED ONLY AFTER ALL CODE IS WRITTEN TO CODEBLOCK OBJECT.
        """

        # Initialize the string to return
        return_string = ''

        for i, sub_block in enumerate(self.sub_blocks):
            # If string, join with return_string
            # else, call to_string of subblock code
            current_indent = self.indent_blocks[i]

            if (isinstance(sub_block, str)):
                # Add a new line with indents then the string of code.
                return_string += '\n' + self.indent_string*current_indent + sub_block
            else:
                # Recursively add string if element is code_block
                sub_block._indent_all(current_indent)
                return_string += sub_block.to_string()

        # Return string
        return return_string

    def print(self):
        """
        prints the code. (calls self.to_string() to generate string)
        """
        print(self.to_string())

    def save(self):
        """
        saves code as python file.
        """
        name = self.name[0:150]
        file_name = name + '.py'
        with open(file_name, "w") as text_file:
            text_file.write(self.to_string())

    def purge_strings(self):
        """
        deletes all strings in sub_blocks
        """
        self.sub_blocks = []
