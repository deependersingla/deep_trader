import chainer
import chainer.functions as F
import chainer.links as L
from analog_accuracy import *
import pdb


class MnistMLP(chainer.Chain):

    """A network for multi-layer perceptron.
    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units,200),
            l4=L.Linear(200,100),
            l5=L.Linear(100,50),
            l6=L.Linear(50, 10),
            l7=L.Linear(10, n_out),
        )

    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.sigmoid(self.l7(h6))
        ## for data here
        #self.l6.W.data[0]
        #self.l6.b
        # t.data like this others also can be calcualted
        self.loss = F.mean_squared_error(h7, t)
        self.accuracy = analog_accuracy(h7, t)
        return self.loss