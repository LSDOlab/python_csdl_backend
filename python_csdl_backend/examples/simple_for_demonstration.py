if __name__ == '__main__':

    import csdl
    import python_csdl_backend

    # Base model
    model = csdl.Model()

    # First level model
    model1 = csdl.Model()
    model.add(model1, 'ModelA', promotes = [])

    # Second level model
    model2 = csdl.Model()
    model2.create_input('x0', val=3)
    model1.add(model2, 'ModelB', promotes = ['x0'])
    model1.create_input('x1', val=2)

    # declare variable
    x0 = model.declare_variable('ModelA.x0')
    x1 = model.declare_variable('x1')
    model.connect('ModelA.x1', 'x1')
    model.register_output('y', x0**2 + x1**2)

    # run model
    sim = python_csdl_backend.Simulator(model, analytics=1)
    sim.run()
    # should be 3^2 + 2^2 = 13
    print(sim['x1'])
    print(sim['ModelA.x0'])
    print(sim['y'])
