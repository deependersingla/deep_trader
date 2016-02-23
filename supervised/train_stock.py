#!/usr/bin/env python
"""
This is a supervised Feef forward network
"""
from __future__ import print_function
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net
from sklearn.cross_validation import train_test_split
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
# optimization
parser.add_argument('--opt', type=str, default='MomentumSGD',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad'])
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()

batchsize = 200
n_epoch = 500
n_units = 500

# Prepare dataset
print('load STOCK dataset')
mnist = data.load_stock_data()
mnist['data'] = mnist['data'].astype(np.float32)
#mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

#N = 300000
#x_train, x_test = np.split(mnist['data'],   [N])
#y_train, y_test = np.split(mnist['target'], [N])
# np.where(y_train==1)[0].shape
#pdb.set_trace()
x_train, x_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], test_size=0.30, random_state=123)
#pdb.set_trace();
print("Buying percentage test={}, train={}".format(np.where(y_train==1)[0].shape[0]*100/y_train.shape[0],np.where(y_test==1)[0].shape[0]*100/y_test.shape[0]))
print("Shorting percentage test={}, train={}".format(np.where(y_train==2)[0].shape[0]*100/y_train.shape[0],np.where(y_test==2)[0].shape[0]*100/y_test.shape[0]))
print("Holding percentage test={}, train={}".format(np.where(y_train==0)[0].shape[0]*100/y_train.shape[0],np.where(y_test==0)[0].shape[0]*100/y_test.shape[0]))
#running only for 1
#temp_1_test = np.where(y_test==1)[0]
#y_test =  y_test[temp_1_test]
#x_test = x_test[temp_1_test]
N = x_train.shape[0]
N_test = y_test.size

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    model = L.Classifier(net.MnistMLP(61, n_units, 3))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(61, n_units, 3))
    xp = cuda.cupy

# Setup optimizer
if 'opt' in args:
#Todo can also pass arguments to each optimizer, see https://github.com/mitmul/chainer-cifar10/blob/master/train.py#L62
    if args.opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
    elif args.opt == 'Adam':
        optimizer = optimizers.Adam(alpha=args.alpha)
else:
    optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ), remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

# Save the model and the optimizer
print('save the model')
serializers.save_hdf5('mlp.model', model)
print('save the optimizer')
serializers.save_hdf5('mlp.state', optimizer)
#python train_stock.py --gpu=0