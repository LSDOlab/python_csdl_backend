from python_csdl_backend.tests.create_single_test import run_test
import csdl
import numpy as np


class Implicit(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')

    def define(self):
        num = self.parameters['num']
        shape_param = (1, num)

        # first implicit model:
        io0 = csdl.Model()
        x0_i = io0.declare_variable('x_0', shape=shape_param)  # state x
        y0_i = io0.declare_variable('y_0')  # state y
        a0_i = io0.declare_variable('d')  # input d
        c0_i = io0.declare_variable('c_0')  # input b

        r_x0 = -x0_i + x0_i**2 - csdl.expand(a0_i, shape_param)  # residual x
        x0_scalar = csdl.reshape(x0_i[0, 2], (1,))
        r_y0 = x0_scalar**2 - c0_i*y0_i + csdl.sin(y0_i)  # residual y
        f0_i = x0_scalar*y0_i + c0_i  # random exposed output

        io0.register_output('r_x0', r_x0)  # state x
        io0.register_output('r_y0', r_y0)  # state y
        io0.register_output('exposed_f0', f0_i)  # output f1

        solve_implicit_0 = self.create_implicit_operation(io0)
        solve_implicit_0.declare_state('x_0', residual='r_x0', val=np.ones(shape_param)*0.2)
        solve_implicit_0.declare_state('y_0', residual='r_y0', val=0.3)
        solve_implicit_0.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
        # solve_implicit_0.linear_solver = csdl.ScipyKrylov()
        solve_implicit_0.linear_solver = csdl.DirectSolver()

        c_0 = self.create_input('c_0', val=3.0)
        d = self.create_input('d', val=3.0)

        x0, y0, f0 = solve_implicit_0(d, c_0, expose=['exposed_f0'])

        # second implicit model:
        io1 = csdl.Model()
        x1_i = io1.declare_variable('x_1', shape=shape_param)  # state x
        y1_i = io1.declare_variable('y_1', shape=(2, 3))  # state y
        a1_i = io1.declare_variable('a_1', shape=shape_param)  # input a
        b1_i = io1.declare_variable('b_1', shape=(2, 3))  # input a

        r_x1 = -x1_i + x1_i**2 - a1_i  # residual x
        x1_scalar = csdl.reshape(x1_i[0, 2], (1,))
        r_y1 = csdl.expand(x1_scalar, (2, 3))**2 - b1_i*y1_i + csdl.sin(y1_i)  # residual y

        y1_scalar = csdl.reshape(y1_i[1, 1], (1,))
        f1_i = x1_i + csdl.expand(y1_scalar, shape_param)  # random exposed output
        f2_i = x1_i[0, 0] + a1_i[0, 0] + f1_i[0, 0] + y1_i[0, 0]  # random exposed output

        io1.register_output('r_x1', r_x1)  # state x
        io1.register_output('r_y1', r_y1)  # state y
        io1.register_output('exposed_f1', f1_i)  # output f1
        io1.register_output('exposed_f2', f2_i)  # output f2

        solve_implicit_1 = self.create_implicit_operation(io1)
        solve_implicit_1.declare_state('x_1', residual='r_x1', val=np.ones(shape_param)*0.2)
        solve_implicit_1.declare_state('y_1', residual='r_y1', val=np.ones((2, 3))*0.3)
        solve_implicit_1.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
        # solve_implicit_1.linear_solver = csdl.ScipyKrylov()
        solve_implicit_1.linear_solver = csdl.DirectSolver()

        # inputs:
        b_1 = self.create_input('b_1', val=np.ones((2, 3))*3)
        self.register_output('a_1', csdl.expand(d, shape_param)*csdl.expand(y0, shape_param)**2+csdl.expand(f0, shape_param))
        a_1 = self.declare_variable('a_1', shape=shape_param)

        x1, y1, f1, f2 = solve_implicit_1(a_1, b_1, expose=['exposed_f1', 'exposed_f2'])

        # third implicit model:
        io2 = csdl.Model()
        x2_i = io2.declare_variable('x_2')  # state x
        a2_i = io2.declare_variable('a_2')  # input a

        r_x2 = -(csdl.sin(x2_i))*(x2_i**2) - 0.2*a2_i  # residual x

        io2.register_output('r_x2', r_x2)  # state x

        solve_implicit_2 = self.create_implicit_operation(io2)
        solve_implicit_2.declare_state('x_2', residual='r_x2', val=0.2, bracket=(-20.0, 20.0))
        # solve_implicit_2.declare_state('x_2', residual='r_x2', val=0.2)
        # solve_implicit_2.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False)
        # solve_implicit_2.linear_solver = csdl.ScipyKrylov()

        self.register_output('a_2', csdl.reshape(x1[0, 0]+y1[1, 1] + a_1[0, 0], (1,)) + y0)
        a_2 = self.declare_variable('a_2', shape=(1, ))
        x2 = solve_implicit_2(a_2)

        # outputs:
        output1 = x1[0, 0]+y1[1, 1]+f1[0, 0] + f2 + csdl.expand(d, (1, 1)) + csdl.expand(x2, (1, 1)) + csdl.expand(f0, (1, 1))
        self.register_output('output1', output1)


def test_implicit_complex():

    vals_dict = {}
    totals_dict = {}
    run_test(
        Implicit(num=10),
        outs=['x_1', 'x_2', 'y_1', 'exposed_f1',  'exposed_f2', 'output1'],
        ins=['b_1', 'd', 'c_0'],
        name='test_implicit_complex',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


def test_implicit_complex_one_out():

    vals_dict = {}
    totals_dict = {}
    run_test(
        Implicit(num=10),
        outs=['output1'],
        ins=['b_1', 'd', 'c_0'],
        name='test_implicit_complex_one_out',
        vals_dict=vals_dict,
        totals_dict=totals_dict,
    )


if __name__ == '__main__':
    # test_implicit_complex()
    # run_test(
    #     Implicit(num=10),
    #     outs=['exposed_f1'],
    #     ins=['b_1'],
    #     name='test_implicit_complex',
    #     vals_dict={},
    #     totals_dict={},
    # )
    # exit()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    outs=['y_1']
    # outs = ['y_0','x_1', 'x_2', 'y_1', 'exposed_f1',  'exposed_f2', 'exposed_f0', 'output1']
    # ins=['c_0']
    # ins=['d', 'c_0']
    ins=['c_0','d']


    from python_csdl_backend import Simulator as LiteSimulator
    sim_lite = LiteSimulator(Implicit(num=10), algorithm='Sync Points Coarse', display_scripts=0)
    sim_lite.run()
    sim_lite.check_partials(compact_print=1)
    # sim_lite.check_totals(of=outs, wrt=ins, compact_print=1)
    # sim_lite.check_partials(compact_print=1)
    # sim_lite.check_totals(of=outs, wrt=ins, compact_print=0)
    # print(sim_lite.compute_totals(of=outs, wrt=ins))


#     run_test(Implicit(num=10), ['output1'], ['b_1', 'd', 'c_0'])

#     from csdl_om import Simulator as OmSimulator
#     from python_csdl_backend import Simulator as LiteSimulator
#     import pytest
#     import numpy as np
#     import time
#     import matplotlib.pyplot as plt

#     outs = ['y_0']
#     ins = ['d', 'c_0']
#     outs = ['x_2']
#     ins = ['c_0']

#     sim_om = OmSimulator(Implicit(num=10))
#     sim_om.run()
#     to = sim_om.prob.compute_totals(of=outs, wrt=ins)
#     # sim_om.prob.check_totals(of=outs, wrt=ins)
#     # exit()
#     # sim_om.prob.check_totals(outs, ins, compact_print=True)
#     # sim_om.prob.check_totals(outs, ins)
#     # exit()

#     sim_lite = LiteSimulator(Implicit(num=10))
#     sim_lite.eval_instructions.save()
#     # sim_lite.visualize_implementation()
#     # exit()
#     sim_lite.run()
#     sim_lite.check_partials(of=outs, wrt=ins, compact_print=False)
#     tl = sim_lite.compute_totals(of=outs, wrt=ins, save_script=True)

#     for output in outs:
#         print('\noutput:    ', output)
#         print('\tcsdl_lite: ', sim_lite[output])
#         print('\tcsdl_om:   ', sim_om[output])

#     for key in tl:
#         print('\ntotals:    ', key)
#         print('\tcsdl_lite: ', tl[key])
#         print('\tcsdl_om:   ', to[key])

#     # exit()

#     # sim_lite.generate_totals(outs, ins, save_script=True)
#     # tots = sim_lite.compute_totals(outs, ins)

#     # for key in tots:
#     #     print()
#     #     print(key)
#     #     print(tots[key])

#     # def main(model_class, outs, ins, num):
#     #     # CSDL OM

#     #     model = model_class()
#     #     sim_om = OmSimulator(model, mode='rev')
#     #     # sim_om.visualize_implementation()

#     #     start = time.time()
#     #     sim_om.run()
#     #     to = time.time() - start

#     #     start = time.time()
#     #     x1 = sim_om.prob.compute_totals(outs, ins)
#     #     tot = time.time() - start

#     #     # CSDL LITE
#     #     sim_lite = LiteSimulator(model)
#     #     sim_lite.eval_instructions.save()
#     #     # sim_lite.visualize_implementation()

#     #     # import cProfile
#     #     # profiler = cProfile.Profile()
#     #     # profiler.enable()

#     #     start = time.time()
#     #     sim_lite.run()
#     #     tl = time.time() - start

#     #     sim_lite.generate_totals(outs, ins)
#     #     start = time.time()
#     #     x = sim_lite.compute_totals(outs, ins)
#     #     tlt = time.time() - start

#     #     for key in x:
#     #         print(key)
#     #         print(np.linalg.norm(x[key].A))
#     #         print(np.linalg.norm(x1[key]))

#     #     # profiler.disable()
#     #     # profiler.dump_stats('output')
#     #     return tl, to, tlt, tot

#     # num_vec = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60, 100, 200, 400,  600,  1000,  2000]
#     # num_vec = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60, 100, 200]

#     # tl_vec = []
#     # to_vec = []
#     # p_vec = []

#     # tlt_vec = []
#     # tot_vec = []
#     # pt_vec = []
#     # for num in num_vec:
#     #     tl, to, tlt, tot = main(Implicit, ['f'], ['a', 'b', 'c'], num)
#     #     tl_vec.append(tl)
#     #     to_vec.append(to)
#     #     p_vec.append(to/tl)

#     #     tlt_vec.append(tlt)
#     #     tot_vec.append(tot)
#     #     pt_vec.append(tot/tlt)
#     #     print('================', num, '================')

#     # plt.figure()
#     # plt.loglog(num_vec, tl_vec)
#     # plt.loglog(num_vec, to_vec)
#     # plt.grid()
#     # plt.xlabel('size')
#     # plt.ylabel('run time(sec)')

#     # plt.figure()
#     # plt.loglog(num_vec, p_vec)
#     # plt.grid()
#     # plt.xlabel('size')
#     # plt.ylabel('run (om time)/(lite time)')

#     # plt.figure()
#     # plt.loglog(num_vec, tlt_vec)
#     # plt.loglog(num_vec, tot_vec)
#     # plt.grid()
#     # plt.xlabel('size')
#     # plt.ylabel('total time(sec)')

#     # plt.figure()
#     # plt.loglog(num_vec, pt_vec)
#     # plt.grid()
#     # plt.xlabel('size')
#     # plt.ylabel('total (om time)/(lite time)')

#     # plt.show()
# -----------------------------------------------------------------------------------------------------
# ('a_1', 'b_1')            0.0                       0.0                        0.0
# ('a_1', 'c_0')            0.7435521166544086        3.5454483321843245e-06     2.636216265315936e-06
# ('a_1', 'd')              4.849301255981369         4.6146122036185166e-07     2.2377655081294247e-06
# ('a_2', 'b_1')            0.42757048003552706       3.212301689326796e-07      1.3734849341195954e-07
# ('a_2', 'c_0')            0.5969224248976637        1.692648354976332e-06      1.0103780503367688e-06
# ('a_2', 'd')              1.9275239097745946        3.8961750870761744e-07     7.50997356302463e-07
# ('exposed_f0', 'b_1')     0.0                       0.0                        0.0
# ('exposed_f0', 'c_0')     1.4551691528390092        1.1817913811433432e-07     1.7197061596974095e-07
# ('exposed_f0', 'd')       0.6315692370949402        0.6973414235518538         1.4551690424104
# ('exposed_f1', 'b_1')     1.3520965771638171        3.207108516321819e-07      4.3363190568070816e-07
# ('exposed_f1', 'c_0')     0.03923265463252868       3.577654752854237e-06      1.403603911521345e-07
# ('exposed_f1', 'd')       0.25586768852334474       0.13294687455111956        0.03923267039743068
# ('exposed_f2', 'b_1')     0.604675971736617         3.2019153433222344e-07     1.9361206517123136e-07
# ('exposed_f2', 'c_0')     0.2599447342254244        3.5485225821886003e-06     9.224164863042006e-07
# ('exposed_f2', 'd')       1.6953086379968936        0.13294677677771982        0.25994461819483106
# ('output1', 'b_1')        1.3515837239306974        3.154750212596401e-07      4.263907695280774e-07
# ('output1', 'c_0')        1.1711662384012278        7.092938331866996e-07      8.307015797459627e-07
# ('output1', 'd')          2.223127237608862         3.7925266124723375e-07     8.431272409126223e-07
# ('x_0', 'b_1')            0.0                       0.0                        0.0
# ('x_0', 'c_0')            0.0                       0.0                        0.0
# ('x_0', 'd')              0.877058019307029         7.726617019913851e-08      6.77669089582067e-08
# ('x_1', 'b_1')            0.0                       0.0                        0.0
# ('x_1', 'c_0')            0.1824973376893732        3.5298025023435096e-06     6.441772854283838e-07
# ('x_1', 'd')              1.1902118883775035        0.13294669890298774        0.18249713293828512
# ('x_2', 'b_1')            0.0005406061704163821     1.386145032086045e-05      7.49348170385293e-09
# ('x_2', 'c_0')            0.0007547292463520234     1.1283638697687124e-05     8.515996038991475e-09
# ('x_2', 'd')              0.0024370983683501077     4.2789295016067284e-07     1.0428176568794134e-09
# ('y_0', 'b_1')            0.0                       0.0                        0.0
# ('y_0', 'c_0')            0.3493841454015935        3.781666215409524e-07      1.3212537192108798e-07
# ('y_0', 'd')              0.31312773931532983       0.5273627358503771         0.349384007147816
# ('y_1', 'b_1')            1.047329505163903         3.2071085163218186e-07     3.358898298171136e-07
# ('y_1', 'c_0')            0.17175131352348483       1.0954174813714903         0.4275705910020349
# ('y_1', 'd')              1.1201284237255305        0.13294672998197524        0.17175116713541475
# -----------------------------------------------------------------------------------------------------