# Python CSDL Backend

Repository for new CSDL backend.

# Installation

Install with `pip install -e .` in root directory.

Requires most recent `csdl`: https://github.com/LSDOlab/csdl

Requires most recent `csdl_om`: 
https://github.com/LSDOlab/csdl_om 
-  Needed to prevent stray `import csdl_om` statements throwing an error
- NOTE: this version of `csdl_om` does not work with caddee. To switch to `csdl_om`, follow the following instructions.

To go back to a working version of CSDL:
- in CSDL root: `git checkout 98d7f6d43a74042a5f8ded2a2b20d431f0c1c342`
- in CSDL_OM root:  `git checkout dev_caddee`

# Features
List of features implemented:
https://docs.google.com/spreadsheets/d/1WWdCow1ZNf7Tx5pk0ql7nIZ5cehOLWBe8dN6LLhGJ0A/edit?usp=sharing 

Note there is currently no way to visualize model implementation.

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


```