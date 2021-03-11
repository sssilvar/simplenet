def test_training_routine():
    from simplenet.models import Net
    from simplenet.routines import training_routine

    model = Net()
    training_routine(model)
    return True
