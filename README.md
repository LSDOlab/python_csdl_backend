# python_csdl_backend

Repository for new CSDL backend.

replace csdl_om with python_csdl_backend

```Python
import csdl
import python_csdl_backend


class Sample(csdl.Model):

    def define(self):
        x1 = self.create_input('x1', val=1.0)
        x2 = self.create_input('x2', val=5.0)

        y1 = csdl.sin(x1)
        y2 = -x2
        y3 = csdl.exp(y2)

        y4 = y1+y3

        f = y4+3

        self.register_output('f', f)


# Simulator should be identical for both backends
sim = python_csdl_backend.Simulator(Sample())

# run
sim.run()

# Print outputs
print('\noutputs:')
print('x1 = ', sim['x1'])
print('x2 = ', sim['x2'])
print('f = ', sim['f'])

# compute totals:
totals = sim.compute_totals(of=['f'], wrt=['x2', 'x1'])
print('\nderivatives:')
for key in totals:
    print(key, '=', totals[key])

# check totals:
# sim.check_partials(compact_print=True) # should use check_totals normally.


```