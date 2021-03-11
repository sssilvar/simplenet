def test_load_from_hub():
    import os
    import torch

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
    model, training_routine = torch.hub.load(ROOT, 'simplenet', source='local')
    training_routine(model)
    torch.save(model.state_dict(), '/tmp/trained_model.pt')
    return True