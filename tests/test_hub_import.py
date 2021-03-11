def test_load_from_hub():
    import os
    import torch

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
    model, training_routine = torch.hub.load(ROOT, 'simplenet', source='local')
    training_routine(model, batch_size=48, lr=1e-3, epochs=20)
    torch.save(model.state_dict(), '/tmp/trained_model.pt')
    return True


def test_load_from_hub_remote():
    import torch

    model, training_routine = torch.hub.load('sssilvar/simplenet', 'simplenet', source='github')
    training_routine(model, batch_size=48, lr=1e-3, epochs=20)
    torch.save(model.state_dict(), '/tmp/trained_model.pt')
    return True
