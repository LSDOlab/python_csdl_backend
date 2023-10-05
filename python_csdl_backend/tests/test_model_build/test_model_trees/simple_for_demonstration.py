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
    model2.create_input('x', val=3)
    model1.add(model2, 'ModelB', promotes = ['x'])
    # model1.declare_variable('x')

    # declare variable
    x = model.declare_variable('ModelA.x')
    model.register_output('y', x*2)

    # run model
    sim = python_csdl_backend.Simulator(model, analytics=1)
    sim.run()
    print(sim['y'])
