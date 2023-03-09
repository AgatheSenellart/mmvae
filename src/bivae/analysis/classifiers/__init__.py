from .classifier_mnist import MnistClassifier, load_pretrained_mnist,load_pretrained_fashion
from .classifier_SVHN import SVHNClassifier, load_pretrained_svhn

__all__ = [
           'load_pretrained_mnist', 
           'load_pretrained_fashion', 
           'load_pretrained_svhn', 
           'MnistClassifier',
           'SVHNClassifier'
           ]