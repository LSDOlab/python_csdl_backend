def build_ladder(num, siso_op, tiso_op, out_stride = 5, shape = (2,1)):
    import csdl
    import numpy as np
    outputs = []
    inputs = []
    class Ladder(csdl.Model):
        def define(self):

            x_0_o = self.create_input(name='x_0_o', shape = shape, val=np.ones(shape))
            x_0_p = self.create_input(name='x_0_p', shape = shape, val=2.1*np.ones(shape))
            inputs.append(x_0_o.name)
            inputs.append(x_0_p.name)

            o_list = [x_0_o]
            p_list = [x_0_p]
            for i in range(1,num):

                o_current = tiso_op(o_list[-1],p_list[-1], self)
                p_current = siso_op(p_list[-1])

                self.register_output(f'x_{i}_o', o_current)
                self.register_output(f'x_{i}_p', p_current)

                if (i % out_stride == 0) or (i == num-1):
                    outputs.append(o_current.name)
                    outputs.append(p_current.name)

                o_list.append(o_current)
                p_list.append(p_current)
            
            
            outputs.pop() # Do not take derivatives of last output, just for fun
            # add in random operations that are not part of outputs or inputs, just for fun
            self.register_output('x_ignore', csdl.sin(csdl.cos(x_0_o)))
        
    return Ladder, outputs, inputs


def build_braid(tito_op_list_1 , tito_op_list_2, out_stride , shape = (2,)):
    import csdl
    import numpy as np
    outputs = []
    inputs = []
    class Braid(csdl.Model):
        def define(self):

            x_0_o_a = self.create_input(name='x_0_o_a', shape = shape, val=np.ones(shape))
            x_0_o_b = self.create_input(name='x_0_o_b', shape = shape, val=0.5*np.ones(shape))

            x_0_p_a = self.create_input(name='x_0_p_a', shape = shape, val=2.1*np.ones(shape))
            x_0_p_b = self.create_input(name='x_0_p_b', shape = shape, val=3.1*np.ones(shape))
            
            inputs.append(x_0_o_a.name)
            inputs.append(x_0_o_b.name)

            inputs.append(x_0_p_a.name)
            inputs.append(x_0_p_b.name)

            o_list_a = [x_0_o_a]
            o_list_b = [x_0_o_b]

            p_list_a = [x_0_p_a]
            p_list_b = [x_0_p_b]

            for i in range(len(tito_op_list_1)):

                o_current1, o_current2 = tito_op_list_1[i](o_list_a[-1],p_list_a[-1], self)
                p_current1, p_current2 = tito_op_list_2[i](o_list_b[-1],p_list_b[-1], self)

                self.register_output(f'x_{i+1}_o_a', o_current1)
                self.register_output(f'x_{i+1}_o_b', o_current2)

                self.register_output(f'x_{i+1}_p_a', p_current1)
                self.register_output(f'x_{i+1}_p_b', p_current2)

                if (i % out_stride == 0) or (i == len(tito_op_list_1)-1):
                    outputs.append(o_current1.name)
                    outputs.append(o_current2.name)
                    outputs.append(p_current1.name)
                    outputs.append(p_current2.name)

                o_list_a.append(o_current1)
                o_list_b.append(o_current2)

                p_list_a.append(p_current1)
                p_list_b.append(p_current2)
            
            self.register_output('final_sum', o_list_a[-1] + o_list_b[-1] + p_list_a[-1] + p_list_b[-1])
            outputs.append('final_sum')
        
    return Braid, outputs, inputs

def build_embarassing(num, num_paths, siso_op, out_stride = 5, shape = (2,1)):
    import csdl
    import numpy as np
    outputs = []
    inputs = []
    class EmbarassinglyParallel(csdl.Model):
        def define(self):
            
            all_vars = []

            for j in range(num_paths):
                x_0_o = self.create_input(name=f'x_0_{j}', shape = shape, val=(j+1)*np.ones(shape))
                
                inputs.append(x_0_o.name)

                o_list = [x_0_o]

                for i in range(1,num):

                    o_current = siso_op(o_list[-1], self)

                    self.register_output(f'x_{i}_{j}', o_current)

                    if (i % out_stride == 0) or (i == num-1):
                        outputs.append(o_current.name)

                    o_list.append(o_current)
                
                all_vars.append(o_list)
                
            sum_final = self.register_output('final_sum', csdl.sum(*[all_vars[i][-1] for i in range(num_paths)]))
            self.register_output('final_sum_norm', csdl.pnorm(sum_final))
    outputs.append('final_sum')
    return EmbarassinglyParallel, outputs, inputs

