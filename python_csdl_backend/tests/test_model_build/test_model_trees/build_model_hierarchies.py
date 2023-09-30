
import csdl
import numpy as np




def build_model(
        num, # Hierarchy size 
        num_calcs=5, # Number of variables per model 
        num_named_variables=1, # Number of variables per model 
        build_type = 'declared',
        conn_dict = None,
        namespace = '',
    ):
    """
    build_type options:
        'declared w/ promote' : Auto promote variables
        'declared': Do not auto promote variables
        'connected': Apply a connection without
    """

    if conn_dict is None:
        conn_dict = {}
        conn_dict['sources'] = set()
        conn_dict['targets'] = set()
        conn_dict['targets'] = set()

    # print(conn_dict)
    # for i in conn_dict['sources']:
    #     print(i)
    # print('\n\n\n\n')
    # for i in conn_dict['targets']:
    #     print(i)
    class SmallModel(csdl.Model):
        def define(self):

            x = self.declare_variable('x', val=np.ones(2,))
            x1 = x

            for i in range(num_calcs-2):
                x1 = x1*0.5
            
            self.register_output('y', 1.0*x1)


            promotes = []
            for i in range(num_named_variables):
                x = self.declare_variable(f'x_{i}_d', val=np.ones(2,))
                self.register_output(f'x_{i}', x*2.0)
                promotes.append(f'x_{i}_d')

                conn_dict['sources'].add(f'{namespace}x_{i}')
                conn_dict['targets'].add(f'{namespace}x_{i}_d')

            if num > 0:
                p_arg = []
                if 'promote' in build_type:
                    p_arg = promotes

                self.add(
                    build_model(
                            num = num-1,
                            num_calcs=num_calcs,
                            num_named_variables=num_named_variables,
                            build_type=build_type,
                            conn_dict=conn_dict,
                            namespace = namespace+f'level_{num}_0.'
                        ),
                    name = f'level_{num}_0',
                    promotes=p_arg,
                )
                self.add(
                    build_model(
                            num = num-1,
                            num_calcs=num_calcs,
                            num_named_variables=num_named_variables,
                            build_type=build_type,
                            conn_dict=conn_dict,
                            namespace = namespace+f'level_{num}_1.'
                        ),
                    name = f'level_{num}_1',
                    promotes=[],
                )

    return SmallModel()