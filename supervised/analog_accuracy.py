import numpy
import pdb

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class AnalogAccuracy(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types
        
        # type_check.expect(
        #     x_type.dtype == numpy.float32,
        #     x_type.ndim >= 2,
        #     t_type.dtype == numpy.float32,
        #     t_type.ndim == 1,
        #     t_type.shape[0] == x_type.shape[0],
        # )
        # for i in range(2, x_type.ndim.eval()):
        #     type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        accuracy_percent = 0.1
        y, t = inputs
        result_list  = []
        for i in range(t.shape[0]):
            value = t[i]
            compare_value = y[i]
            high = value * (1.0 + accuracy_percent)
            low = value * (1.0 - accuracy_percent)
            if low <= compare_value <= high:
                array_value = 1
            else:
                array_value = 0
            result_list.append(array_value)
            numpy_array = numpy.asarray(result_list)
            result_mean = numpy.mean(numpy_array)
        return xp.asarray(result_mean),


def analog_accuracy(y, t):
    """Computes muticlass classification accuracy of the minibatch.

    Args:
        y (Variable): Variable holding a matrix whose (i, j)-th element
            indicates the score of the class j at the i-th example.
        t (Variable): Variable holding an int32 vector of groundtruth labels.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return AnalogAccuracy()(y, t)