# Python CSDL Backend

Repository for new CSDL backend.

# Changelog
- Added optional automatic CPU parallelization using MPI.
    -
  - Performs static list scheduling that generates blocking MPI code for model evaluation and derivatives. May or may not improve performance depending on model structure.
  - To run without parallelization (same as before):

    ```Python
    import csdl
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(Sample())
    ```
    Terminal:
    ```Bash
    python sample.py
    ```
  - To run with parallelization (**new**):
    ```Python
    import csdl
    import python_csdl_backend
    from mpi4py import MPI # conda install mpi4py
    comm = MPI.COMM_WORLD
    sim = python_csdl_backend.Simulator(Sample(), comm = comm)
    ```
    Terminal:
    ```Bash
    mpirun -n {number of processors} python sample.py
    ```
- Added optional automatic checkpointing for adjoint computation.
    -
  - Using checkpointing can reduce peak memory usage but can be ~2x slower than without checkpointing.
  - To compute derivatives using checkpointing, specify additional arguments to `Simulator` constructor:
    ```Python
    import csdl
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(
        Sample(),
        checkpoints: bool = True, # Required
        save_vars: set() = {'sample_model.variable_of_interest'}, # Optional
        checkpoint_stride: int = 150_000_000, # Optional
    )
    ```
  - `save_vars`: Checkpointing may delete variables of interest during computation which can prevent postprocessing (printing/plotting values after computation). Specify names of variables you wish to permanently allocate memory for in `save_vars`. Has no effect when `checkpoints = False`.
  - `checkpoint_stride`: Rough estimate of how large a checkpoint interval is in bytes. If left empty, a stride is automatically chosen.
  - When `checkpoints = True`, additional memory-saving measures are performed such as lazy evaluation of linear partial derivatives as opposed to pre-computation.    

- Other
    - 
    - Visualize checkpoints and parallelization: 
        ```Python
        python_csdl_backend.Simulator(Sample(), visualize_schedule = True)
        ```
    - Significantly reduced memory usage for derivative computation (even without checkpointing) by deallocating partial derivatives and propagated adjoints as they are processed. 
        - **May result in reduced performance for small models compared to previous version.**
    - Added new tests with more complicated models. 
        - Run parallel tests using (in root directory):
            ```Bash
            mpirun -n {number of processors} pytest
            ```
        - Also tests now verify each model with:
            - [ ] checkpointing, [ ] parallelization (previous implementation)
            - [x] checkpointing, [ ] parallelization
            - [ ] checkpointing, [x] parallelization
            - [x] checkpointing, [x] parallelization

# Installation

Install with `pip install -e .` in root directory. run `pytest` in root directory.

Requires most recent `csdl`: https://github.com/LSDOlab/csdl

Requires most recent `csdl_om`: 
https://github.com/LSDOlab/csdl_om 
-  Needed to prevent stray `import csdl_om` statements throwing an error
- NOTE: this version of `csdl_om` does not work with caddee. To switch to `csdl_om`, follow the following instructions.

To go back to a working version of `csdl_om`:
- in CSDL root: `git checkout dev_caddee`
- in CSDL_OM root:  `git checkout dev_caddee`
- (`git checkout master` in both to go back to most recent commit)

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
