from simplenet.models import Net
from simplenet.routines import training_routine

dependencies = ['torch', 'torchvision']


def simplenet():
    return Net(), training_routine

